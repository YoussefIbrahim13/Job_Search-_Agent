"""
backend/agents/recruitment_agent.py
====================================
LangGraph-based Recruitment AI Agent using local Ollama model.

DESIGN PRINCIPLES (unchanged)
------------------------------
1. Sequential-only tool execution — one tool call per LLM step.
2. Hard iteration cap — enforced by the graph router.
3. Structured JSON output — validated and sanitised post-processing.
4. Single source of truth — all agent logic lives here.

Graph topology
--------------
    START → llm_node ◄──────────────────────────────┐
                │                                    │
             _route                                  │
          ┌────┼─────────────────┐                   │
     tool_node │           coerce_internship ─────► llm_node
               │
          graceful_exit → END

CHANGES IN THIS REVISION (volume optimisation pass)
-----------------------------------------------------
FIX E — Router Short-Circuit Bug, corrected at the actual cause:
  DIAGNOSIS CORRECTION: the originally-suspected cause ("the function scans
  the entire ToolMessage.content") was not accurate — _extract_search_state
  already sliced to content[:200] in the prior revision. The real cause is
  that the first ~200 characters of EVERY tavily_job_search return value are
  the fixed pipeline-status banner, which itself contains the literal text
  "INTERNSHIP/TRAINEE" as part of its standing reminder:

      "✓ You MUST still run the INTERNSHIP/TRAINEE variant query."

  Because _extract_search_state pattern-matched against the *tool response*
  text, this banner caused internship_query_done to flip True after the
  FIRST query regardless of what was actually searched — a full-time-only
  query would trigger it just as easily as a real internship query, since
  the banner is identical on every successful response. Re-truncating the
  response text differently would not have fixed this, because the false-
  positive substring is inside the part of the response that any reasonable
  truncation window would still include.

  CORRECT FIX: stop inferring query intent from response content entirely.
  Instead, inspect the actual tool-call ARGUMENTS the model sent — these
  live on the preceding AIMessage.tool_calls[i]["args"]["query"], not on
  the ToolMessage. This is the ground truth of what was searched, and it
  cannot contain banner boilerplate because it is the model's OUTGOING
  argument, not the tool's incoming response.

  _extract_search_state() is rewritten to walk paired (AIMessage, ToolMessage)
  sequences: for every AIMessage with a tavily_job_search tool_call, look at
  its args["query"] (the real, sent query) to decide whether that specific
  call was an internship-flavoured search. The corresponding ToolMessage is
  used only to confirm the call actually completed (i.e. wasn't an error/
  exception path) — never to source the keyword match itself.

FIX F — Query Hygiene Prompt Update (stray "Go" / operational tokens):
  System prompt REQUIREMENT 3 ("QUERY TOKEN HYGIENE") is expanded with an
  explicit rule against leaking bare operational/filler words (especially
  "Go") in front of an unrelated stack, mirroring the tools.py docstring so
  both layers agree. The STEP 1/2 examples are updated to use a C#/.NET
  example explicitly, since that was the stack observed corrupted by the
  leaked "Go" token in production logs.

Earlier changes (preserved without modification):
  - FIX A: Negative-keyword semantic leakage retired from prompt language;
    handled at the tool layer via Tavily exclude_domains.
  - FIX B: Open-web cognitive drift eliminated; APPROVED BOARDS ONLY
    templates sourced from APPROVED_SEARCH_BOARDS in tools.py.
  - FIX 1: Technology-stack injection + _has_tech_signal guard in tools.py.
  - FIX 2: _BLACKLISTED_DOMAINS CSR zombie filter in tools.py +
           _is_blacklisted_domain() re-applied in _validate_and_fix_output.
  - FIX 3: AgentState search-tracking fields, coerce_internship node,
           pipeline-status banner in tool output.
  - Zero hallucination ABSOLUTE PROHIBITIONS block.
  - No hardcoded year anywhere; recency enforced at tool layer.
  - Jobs with invalid application_link are DROPPED, not kept.
"""

from __future__ import annotations

import functools
import json
import logging
import re
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_groq import ChatGroq


# from langchain_ollama import ChatOllama




from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.agents.tools import (
    APPROVED_SEARCH_BOARDS,
    get_tools,
    _is_blacklisted_domain,
    _is_content_pollution_domain,
)
from backend.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

_BAD_URL_RE = re.compile(
    r"/search|\?q=|-jobs-in-|/find-jobs|keyword=|/jobs/?$|/pulse/",
    re.IGNORECASE,
)
_FAKE_LINK_RE = re.compile(r"/jobs/view/\d+$")
_AGGREGATOR_TITLE_RE = re.compile(
    r"\d+\+?\s*(jobs?|vacancies|positions?|openings?)", re.IGNORECASE
)
_FAKE_SKILL_RE = re.compile(
    r"^(skill\d*|n/?a|none|tbd|example|placeholder)$", re.IGNORECASE
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_JOB_BOARD_NAMES_RE = re.compile(
    r"\b(indeed|linkedin|glassdoor|wuzzuf|bayt|monster|ziprecruiter)\b",
    re.IGNORECASE,
)
_PLACEHOLDER_COMPANY_RE = re.compile(
    r"^(company\s*(name)?|example\s*(corp(oration)?)?|acme|"
    r"your\s*company|n/?a|unknown\s*company|placeholder|"
    r"company\s*\d+|org\s*\d+)$",
    re.IGNORECASE,
)
_FAKE_SALARY_RE = re.compile(
    r"(competitive|negotiable|market\s*rate|tbd|attractive)",
    re.IGNORECASE,
)
_VALID_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)

# FIX E: this pattern is now applied to the OUTGOING tool-call query
# argument (the model's real intent), never to the tool's response text.
_INTERNSHIP_QUERY_RE = re.compile(
    r"\b(intern(ship)?|trainee|graduate\s+program|entry.level)\b",
    re.IGNORECASE,
)

_TAVILY_TOOL_NAME = "tavily_job_search"


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    messages              — Full conversation history.
    iterations            — Incremented on each LLM call; enforces cap.
    final_output          — Set by graceful_exit_node.
    queries_executed      — Count of completed tavily_job_search calls.
    internship_query_done — True once an intern/trainee query has been executed.
    coercion_injected     — True after the coercion message has been injected once.
    """
    messages:              Annotated[Sequence[BaseMessage], add_messages]
    iterations:            int
    final_output:          Optional[Dict[str, Any]]
    queries_executed:      int
    internship_query_done: bool
    coercion_injected:     bool


# ---------------------------------------------------------------------------
# FIX E — Search-state extractor (corrected: reads outgoing tool-call args)
# ---------------------------------------------------------------------------
#
# ROOT CAUSE RECAP: the previous version matched _INTERNSHIP_QUERY_RE against
# `ToolMessage.content[:200]`. Every successful tavily_job_search response
# begins with a fixed pipeline-status banner that itself contains the words
# "INTERNSHIP/TRAINEE" as a standing reminder to run the next query. That
# means the regex matched on the FIRST query's response too — full-time or
# not — because the false-positive text lives inside the banner, which is
# always present at the very start of the response regardless of truncation
# width. Changing the truncation window cannot fix a false positive that is
# guaranteed to be inside that window on every single call.
#
# CORRECTED APPROACH: never pattern-match tool RESPONSES for intent. Pattern-
# match the tool-call ARGUMENTS instead — i.e. what the model actually typed
# as the `query` parameter when it invoked tavily_job_search. This is the
# one place in the message history that reflects the model's real search
# intent and contains no boilerplate of any kind.
#
# We pair each AIMessage's tool_calls with the ToolMessage that answers it
# (matched by tool_call_id) so we can also confirm the call actually
# completed without raising — a defensive check, not the keyword source.

def _extract_search_state(messages: Sequence[BaseMessage]) -> tuple[int, bool]:
    """
    Walk the full message history and determine, from GROUND TRUTH tool-call
    arguments (never from response text), how many tavily_job_search calls
    have completed and whether any of them was an internship/trainee query.

    Returns (queries_executed, internship_query_done).
    """
    # Build a lookup of completed ToolMessages by tool_call_id so we can
    # confirm a given tool call actually returned (as opposed to, say, an
    # in-flight call with no response yet in a partial state snapshot).
    completed_tool_call_ids: set[str] = {
        getattr(msg, "tool_call_id", None)
        for msg in messages
        if isinstance(msg, ToolMessage) and getattr(msg, "tool_call_id", None)
    }

    executed    = 0
    intern_done = False

    for msg in messages:
        if not isinstance(msg, AIMessage):
            continue

        tool_calls = getattr(msg, "tool_calls", None) or []
        for tc in tool_calls:
            if tc.get("name") != _TAVILY_TOOL_NAME:
                continue

            tool_call_id = tc.get("id")
            if tool_call_id not in completed_tool_call_ids:
                # Call was made but hasn't completed yet — don't count it.
                continue

            executed += 1

            # GROUND TRUTH: the model's own outgoing query argument.
            sent_query = str((tc.get("args") or {}).get("query", ""))
            if _INTERNSHIP_QUERY_RE.search(sent_query):
                intern_done = True

    return executed, intern_done


# ---------------------------------------------------------------------------
# Approved board reference string (injected into prompts)
# ---------------------------------------------------------------------------
#
# Built from the canonical APPROVED_SEARCH_BOARDS dict in tools.py so the
# prompt and the tool enforcement always agree on which boards are valid.

def _build_approved_boards_prompt_block() -> str:
    """
    Format APPROVED_SEARCH_BOARDS as a human-readable prompt block with
    site: tokens the model can copy directly into query strings.
    """
    lines = ["APPROVED SITE TOKENS — use EXACTLY one (or two with OR) per query:"]
    # Group by category
    global_boards  = ["linkedin", "indeed", "glassdoor"]
    mena_boards    = ["wuzzuf", "bayt", "akhtaboot"]
    remote_boards  = ["weworkremotely", "remoteok", "himalayas"]
    tech_boards    = ["wellfound", "dice"]

    groups = [
        ("Global generalist", global_boards),
        ("MENA / Egypt / Gulf", mena_boards),
        ("Remote-focused", remote_boards),
        ("Tech-specialist", tech_boards),
    ]
    for label, keys in groups:
        tokens = [APPROVED_SEARCH_BOARDS[k] for k in keys if k in APPROVED_SEARCH_BOARDS]
        if tokens:
            lines.append(f"  {label}:")
            for t in tokens:
                lines.append(f"    {t}")
    return "\n".join(lines)


_APPROVED_BOARDS_BLOCK = _build_approved_boards_prompt_block()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are a recruitment assistant. Your job is to find REAL, CURRENTLY OPEN job
positions (including internships) and score how well each one matches the
user's search query.

═══════════════════════════════════════════════════════════════════════
ABSOLUTE PROHIBITIONS — violating ANY of these is a CRITICAL FAILURE
═══════════════════════════════════════════════════════════════════════

1. NEVER invent, fabricate, or hallucinate ANY field — not company names,
   job titles, locations, salaries, skills, experience requirements, or URLs.
2. NEVER output placeholder text such as "Company Name", "Example Corp",
   "skill1", "Competitive salary", "TBD", "N/A", or similar.
3. NEVER reuse, guess, or construct a URL. application_link MUST be the exact
   URL string returned by the search tool — copied character-for-character.
   If the tool did not return a direct application URL, set application_link
   to null.
4. NEVER include a job whose application_link is null, "#", or a job-board
   homepage. Drop it from the results entirely.
5. Only include listings that appear to be currently open and recently posted.
   Discard anything whose snippet says it is closed, filled, or expired.
6. If a field is not explicitly stated in the search snippet, use ONLY:
     • strings  → "Not specified"
     • lists    → []
   Do NOT guess, infer, or fill in from general knowledge.

═══════════════════════════════════════════════════════════════════════
REQUIREMENT 1 — TECHNOLOGY STACK IN EVERY QUERY
═══════════════════════════════════════════════════════════════════════

EVERY query passed to tavily_job_search MUST include at least one concrete
technology keyword (Python, React, Django, Node.js, Flutter, Spring Boot,
C#, ASP.NET…). A query with NO technology keyword is a CRITICAL FAILURE
flagged by the tool.

Forbidden:
  ✗ "Back-End Developer jobs Cairo"
  ✗ "Software Engineer internship Dubai"
  ✗ "Junior Developer remote hiring now"

Required — technology FIRST, then role, then site: token:
  ✓ "Python Django Back-End Developer jobs site:wuzzuf.net"
  ✓ "React Node.js Software Engineer internship site:linkedin.com/jobs"
  ✓ "Flutter Android mobile developer jobs site:indeed.com OR site:bayt.com"
  ✓ "C# ASP.NET Backend Developer jobs site:glassdoor.com"

The PRIMARY TECHNOLOGY STACK is given explicitly in the user message. Use
ONLY that stack's own keywords — never a different language's keyword.

═══════════════════════════════════════════════════════════════════════
REQUIREMENT 2 — APPROVED BOARDS ONLY — ABSOLUTELY NO OPEN-WEB QUERIES
═══════════════════════════════════════════════════════════════════════

EVERY query MUST include a site: clause from the list below.
A query WITHOUT a site: clause is FORBIDDEN. It produces context pollution
from forums, articles, and Q&A boards that are structurally incapable of
being job postings. Do not omit the site: clause for any reason.

{_APPROVED_BOARDS_BLOCK}

You MAY combine up to two approved boards with OR:
  site:linkedin.com/jobs OR site:wuzzuf.net
  site:indeed.com OR site:bayt.com
You MUST NOT use any domain not on the approved list.
You MUST NOT omit site: entirely — this is unconditional.

═══════════════════════════════════════════════════════════════════════
REQUIREMENT 3 — QUERY TOKEN HYGIENE
═══════════════════════════════════════════════════════════════════════

The query argument you pass to tavily_job_search must be a SEARCH TOKEN
STRING ONLY, made up of nothing but: the candidate's own technology/stack
keywords, the role title, an optional modifier word (jobs / internship /
intern / trainee), and the site: clause. Do not include any of the
following:
  • Conversational prefixes: "Search for", "Please find", "I need to look up"
  • Explanations: "This query is for", "The goal of this search is"
  • Negative exclusion clauses: -"closed", -"expired", -"no longer accepting"
    (zombie filtering is handled automatically by the tool — do NOT add these)
  • Bare operational/filler verbs of any kind: "Go" (as in "go ahead and..."),
    "Run", "Execute", "Now", "Ok", "Fetch", "Lookup" — placed in front of an
    UNRELATED stack. This is a CRITICAL FAILURE because search engines read
    leaked words as additional technology keywords. A leaked "Go" in front
    of a C#/.NET query, for example, will be read as the Go/Golang language
    and will corrupt the results away from the .NET stack you actually want.
    NEVER write "Go <some other language> ..." — if the candidate's stack is
    C#/.NET, write "C# ASP.NET ..." and nothing else in front of it. ("Go" is
    only acceptable when the candidate's ACTUAL stack is Go/Golang itself.)
  • Any language other than the search terms and site: tokens themselves

Correct: "Python Django developer jobs site:wuzzuf.net OR site:bayt.com"
Correct: "C# ASP.NET Backend Developer jobs site:linkedin.com/jobs"
Wrong:   "Search for Python Django developer jobs on wuzzuf or bayt"
Wrong:   "Python developer jobs -'closed' -'expired' site:linkedin.com"
Wrong:   "Go ASP.NET Backend Developer jobs site:linkedin.com/jobs"
           ← the leaked "Go" turns a .NET search into a Golang search.

═══════════════════════════════════════════════════════════════════════
SCOPE — ALWAYS include internships
═══════════════════════════════════════════════════════════════════════

Search for BOTH full-time/part-time AND internship/trainee roles.
Your second mandatory query MUST target internships explicitly — it must
contain the word "internship", "intern", or "trainee" in the query you send.

═══════════════════════════════════════════════════════════════════════
MANDATORY SEQUENTIAL SEARCH CHECKLIST — follow IN ORDER, no skipping
═══════════════════════════════════════════════════════════════════════

  ☐ STEP 1 — Full-time / senior query (MANDATORY):
      "<PRIMARY_TECH> <role> jobs <site:BOARD_A>"
      Example: "C# ASP.NET Backend Developer jobs site:wuzzuf.net OR site:bayt.com"
      Use a global or regional board for Step 1.

  ──── WAIT for STEP 1 results before proceeding ────────────────────────

  ☐ STEP 2 — Internship / trainee query (MANDATORY regardless of Step 1):
      "<PRIMARY_TECH> <role> internship <site:BOARD_B>"
      Example: "C# ASP.NET developer internship site:linkedin.com/jobs"
      Use a DIFFERENT board than Step 1. The query text itself MUST contain
      "internship", "intern", or "trainee" — this is how completion of this
      mandatory step is verified, so do not skip the word.

  ──── WAIT for STEP 2 results before proceeding ────────────────────────

  ☐ STEP 3 — Optional: if fewer than 4 valid listings found after Steps 1+2,
      run one more query targeting a different approved board.

  ☐ STEP 4 — Emit final JSON ONLY after Steps 1 AND 2 are both complete.
      ══ STOP GATE ══ If Step 2 is not done, you CANNOT emit JSON. ══════

═══════════════════════════════════════════════════════════════════════
EXTRACTION RULES — verbatim only, no hallucination
═══════════════════════════════════════════════════════════════════════

For EACH job found, copy ONLY what is explicitly in the snippet:
  company_name      → as written; "Unknown" only if genuinely absent
  job_title         → exact title from the listing
  location          → exact location; "Not specified" if absent
  experience_needed → only if explicitly stated; else "Not specified"
  salary_range      → only if a real number/currency appears; else "Not specified"
                       NEVER use "Competitive", "Negotiable", or similar
  required_skills   → skills/technologies explicitly mentioned; [] if none
  application_link  → EXACT URL from the search tool — unmodified.
                       null if no direct URL was returned.
  source            → domain extracted from application_link

═══════════════════════════════════════════════════════════════════════
FILTER BEFORE SCORING
═══════════════════════════════════════════════════════════════════════

DISCARD any job where:
  • application_link is null, "#", a homepage, or a search-results page
  • job_title looks like "50+ Jobs in Cairo" (aggregator listing)
  • company_name is a job board name ("LinkedIn", "Indeed", etc.)
  • the snippet says the listing is closed, filled, or expired
  • the URL is from a forum, blog, Q&A site, or social network
    (these are structurally incapable of being job postings)

═══════════════════════════════════════════════════════════════════════
SCORING (match_score 0–100)
═══════════════════════════════════════════════════════════════════════

TITLE_MATCH (0–50 pts):
  50 → identical or near-identical to searched title
  35 → same role family
  20 → adjacent role or different seniority
  10 → internship when full-time was searched (still relevant)
   5 → loosely related

LOCATION_MATCH (0–30 pts):
  30 → exact location match
  15 → same country / region
   5 → remote
   0 → different country, no remote option

INFO_QUALITY (0–20 pts):
  +5 salary has actual number/currency
  +5 experience explicitly stated
  +5 required_skills has ≥ 3 real skills from snippet
  +5 direct application URL confirmed

match_score = TITLE_MATCH + LOCATION_MATCH + INFO_QUALITY  [clamped 5–98]
Every job MUST have a DIFFERENT score.
match_reason: one sentence citing the actual title and location.

═══════════════════════════════════════════════════════════════════════
FINAL OUTPUT
═══════════════════════════════════════════════════════════════════════

Output ONLY the JSON object below. No markdown fences, no preamble,
no explanation, no text before or after the JSON.

{{
  "job_title": "<user searched title>",
  "location": "<user searched location>",
  "total_found": <integer — count of jobs AFTER filtering>,
  "agent_summary": "<2 sentences: what was searched, what was found, internships included>",
  "search_queries_used": ["<query 1>", "<query 2>"],
  "jobs": [
    {{
      "company_name": "<verbatim from search or 'Unknown'>",
      "job_title": "<verbatim from search>",
      "match_score": <integer>,
      "location": "<verbatim from search or 'Not specified'>",
      "experience_needed": "<verbatim or 'Not specified'>",
      "salary_range": "<verbatim number/currency or 'Not specified'>",
      "required_skills": ["<only explicitly found skills>"],
      "match_reason": "<one specific sentence>",
      "source": "<domain>",
      "application_link": "<EXACT URL from tool>"
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Three-strategy JSON parser for raw LLM output.
    Strategy 1 — direct parse (cleaned string).
    Strategy 2 — outermost balanced { … } block.
    Strategy 3 — bracket-completion on truncated JSON.
    """
    text = _THINK_BLOCK_RE.sub("", text)
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    if start != -1:
        partial   = text[start:]
        partial   = _TRAILING_COMMA_RE.sub(r"\1", partial)
        opens_sq  = partial.count("[") - partial.count("]")
        opens_cu  = partial.count("{") - partial.count("}")
        candidate = partial + "]" * max(opens_sq, 0) + "}" * max(opens_cu, 0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from LLM output: %.300s", text)
    return None


# ---------------------------------------------------------------------------
# Output normalisation
# ---------------------------------------------------------------------------

def _normalise_skills(skills: Any) -> List[str]:
    """Coerce required_skills to List[str] regardless of LLM output format."""
    if skills is None:
        return []
    if isinstance(skills, list):
        return [str(s).strip() for s in skills if str(s).strip()]
    if isinstance(skills, str):
        parts = [p.strip() for p in skills.split(",") if p.strip()]
        return parts if parts else []
    return [str(skills).strip()]


def _validate_and_fix_output(
    raw: Dict[str, Any],
    *,
    cap_score: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate and sanitise the LLM JSON output.

    FIX A addition: _is_content_pollution_domain() is now applied to every
    application_link as an additional post-processing gate, catching any
    forum/blog/social URL the model hallucinated from memory.

    FIX 2 (preserved): _is_blacklisted_domain() applied as a post-processing
    gate for CSR zombie aggregator URLs.
    """
    job_title = raw.get("job_title", "Unknown Position")
    location  = raw.get("location",  "Various")
    jobs_raw  = raw.get("jobs", [])

    if not isinstance(jobs_raw, list):
        jobs_raw = [jobs_raw] if isinstance(jobs_raw, dict) else []

    jobs_fixed: List[Dict[str, Any]] = []

    for job in jobs_raw:
        if not isinstance(job, dict):
            continue

        title = str(job.get("job_title", ""))
        if _AGGREGATOR_TITLE_RE.search(title):
            logger.debug("Dropping aggregator listing: %r", title)
            continue

        company = str(job.get("company_name", ""))
        if _JOB_BOARD_NAMES_RE.search(company):
            logger.debug("Dropping job-board company: %r", company)
            continue

        if _PLACEHOLDER_COMPANY_RE.fullmatch(company.strip()):
            logger.debug("Dropping placeholder company: %r", company)
            continue

        link = str(job.get("application_link", "") or "").strip()

        # FIX 2 (preserved): CSR zombie domain gate
        if _is_blacklisted_domain(link):
            logger.debug("Post-proc drop [blacklist]: %r link=%r", title, link)
            continue

        # FIX A: content-pollution domain gate
        if _is_content_pollution_domain(link):
            logger.debug("Post-proc drop [pollution]: %r link=%r", title, link)
            continue

        if (
            not link
            or link in ("#", "null", "None", "N/A", "n/a")
            or not _VALID_URL_RE.match(link)
            or _FAKE_LINK_RE.search(link)
            or _BAD_URL_RE.search(link)
        ):
            logger.debug("Post-proc drop [bad-link]: %r link=%r", title, link)
            continue

        skills = _normalise_skills(job.get("required_skills"))
        job["required_skills"] = [s for s in skills if not _FAKE_SKILL_RE.match(s)]

        try:
            score = int(job.get("match_score", 50))
        except (TypeError, ValueError):
            score = 50
        score = max(5, min(score, 98))
        if cap_score is not None:
            score = min(score, cap_score)
        job["match_score"] = score

        salary = str(job.get("salary_range", "")).strip()
        if (
            not salary
            or salary.lower() in ("", "n/a", "none", "null", "not specified")
            or _FAKE_SALARY_RE.search(salary)
        ):
            job["salary_range"] = "Not specified"

        defaults: Dict[str, str] = {
            "company_name":      "Unknown",
            "job_title":         job_title,
            "location":          "Not specified",
            "experience_needed": "Not specified",
            "salary_range":      "Not specified",
            "match_reason":      "Matches the search criteria.",
            "source":            "Web",
        }
        for key, default in defaults.items():
            if not job.get(key):
                job[key] = default

        if job.get("source") in ("", "Web", None):
            try:
                from urllib.parse import urlparse
                job["source"] = urlparse(link).netloc or "Web"
            except Exception:
                job["source"] = "Web"

        jobs_fixed.append(job)

    return {
        "job_title":           job_title,
        "location":            location,
        "total_found":         len(jobs_fixed),
        "agent_summary":       raw.get("agent_summary", "Search complete."),
        "search_queries_used": raw.get("search_queries_used", []),
        "jobs":                jobs_fixed,
    }


# ---------------------------------------------------------------------------
# LLM + tool binding
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_llm_with_tools() -> Any:
    
    settings = get_settings()

    llm = ChatGroq(
        model=settings.groq_model,         
        api_key=settings.groq_api_key,    
        temperature=0.0,                 
        max_tokens=2000,                  
    )  

    



    # llm = ChatOllama(
    #     model=settings.ollama_model,
    #     base_url=settings.ollama_base_url,
    #     temperature=0.0,
    #     num_predict=2000,
    #     format="json",
    # )


    tools = get_tools()
    return llm.bind_tools(tools, tool_choice="auto")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> Dict[str, Any]:
    """Call the LLM, increment the iteration counter, refresh search state."""
    settings       = get_settings()
    llm_with_tools = _get_llm_with_tools()

    logger.info(
        "LLM node — iter %d/%d | queries=%d | intern_done=%s",
        state["iterations"] + 1, settings.max_agent_iterations,
        state["queries_executed"], state["internship_query_done"],
    )

    response: AIMessage = llm_with_tools.invoke(state["messages"])

    if getattr(response, "tool_calls", None):
        logger.info("Tool calls: %s", [tc["name"] for tc in response.tool_calls])
    else:
        logger.info("No tool call — model producing final answer.")

    # FIX E: search-state is now computed from outgoing tool-call args, not
    # from response banner text — see _extract_search_state docstring.
    queries_executed, internship_query_done = _extract_search_state(
        list(state["messages"]) + [response]
    )

    return {
        "messages":              [response],
        "iterations":            state["iterations"] + 1,
        "queries_executed":      queries_executed,
        "internship_query_done": internship_query_done,
    }


def graceful_exit_node(state: AgentState) -> Dict[str, Any]:
    """Extract and validate JSON from the most recent AIMessage."""
    last_ai_text: str = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_text = getattr(msg, "content", "") or ""
            break

    raw = _extract_json(last_ai_text)

    if raw:
        result = _validate_and_fix_output(raw)
    else:
        logger.warning("graceful_exit: could not parse JSON from last AIMessage.")
        result = {
            "job_title":           "Unknown",
            "location":            "Various",
            "total_found":         0,
            "agent_summary":       "The agent could not produce a structured response.",
            "search_queries_used": [],
            "jobs":                [],
        }

    return {"final_output": result}


# ---------------------------------------------------------------------------
# FIX 3 (preserved): Coercion message factory and node
# ---------------------------------------------------------------------------

def _build_internship_coercion_message(queries_executed: int) -> HumanMessage:
    """
    Short, high-urgency message injected when the model tries to exit
    before completing its mandatory internship query.
    """
    # Pick a concrete example board for the internship query that differs
    # from the most commonly used Step 1 boards (wuzzuf/bayt) to prompt
    # board variation as required by REQUIREMENT 2.
    return HumanMessage(
        content=(
            f"⛔ STOP — MANDATORY STEP INCOMPLETE ⛔\n\n"
            f"You have completed {queries_executed} query/queries but have NOT run "
            f"the mandatory INTERNSHIP / TRAINEE query (Step 2 of your checklist).\n\n"
            f"You are NOT permitted to emit the final JSON until Step 2 is done.\n\n"
            f"ACTION REQUIRED — call tavily_job_search NOW with a query that "
            f"contains the word 'internship', 'intern', or 'trainee', e.g.:\n"
            f'  "<PRIMARY_TECH> <role> internship site:linkedin.com/jobs"\n'
            f"  OR\n"
            f'  "<PRIMARY_TECH> <role> intern site:wuzzuf.net OR site:akhtaboot.com"\n\n'
            f"Rules:\n"
            f"• Technology keyword MUST be in the query — use only the\n"
            f"  candidate's OWN stack, never a different language's keyword.\n"
            f"• site: clause MUST be from the approved board list.\n"
            f"• No conversational filler or stray operational words (e.g. a\n"
            f"  leaked 'Go', 'Run', 'Now') in the query string.\n\n"
            f"After that tool call returns, combine all results and emit the final JSON."
        )
    )


def coerce_internship_node(state: AgentState) -> Dict[str, Any]:
    """Inject the internship coercion message and set coercion_injected=True."""
    coercion_msg = _build_internship_coercion_message(state["queries_executed"])
    logger.info("coerce_internship_node: injecting coercion message.")
    return {
        "messages":          [coercion_msg],
        "coercion_injected": True,
    }


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    """
    Route after each LLM call.

    Order:
    1. Iteration cap → graceful_exit.
    2. Tool call present → tool_node.
    3. Internship query not done + coercion not yet sent → coerce_internship.
    4. Otherwise → graceful_exit.
    """
    settings = get_settings()

    if state["iterations"] >= settings.max_agent_iterations:
        logger.info("Iteration cap (%d) → graceful_exit.", state["iterations"])
        return "graceful_exit"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        logger.info("Tool call → tool_node.")
        return "tool_node"

    if (
        not state["internship_query_done"]
        and state["queries_executed"] >= 1
        and not state["coercion_injected"]
        and state["iterations"] < settings.max_agent_iterations - 1
    ):
        logger.warning(
            "Early exit attempt after %d queries without internship query → coerce.",
            state["queries_executed"],
        )
        return "coerce_internship"

    logger.info("No tool call → graceful_exit.")
    return "graceful_exit"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph exactly once.

    Topology:
        START → llm_node ← tool_node ← llm_node
                         ← coerce_internship
                         → graceful_exit → END
    """
    tools     = get_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("llm_node",          llm_node)
    graph.add_node("tool_node",         tool_node)
    graph.add_node("graceful_exit",     graceful_exit_node)
    graph.add_node("coerce_internship", coerce_internship_node)

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges(
        "llm_node",
        _route,
        {
            "tool_node":         "tool_node",
            "graceful_exit":     "graceful_exit",
            "coerce_internship": "coerce_internship",
        },
    )
    graph.add_edge("tool_node",         "llm_node")
    graph.add_edge("coerce_internship", "llm_node")
    graph.add_edge("graceful_exit",     END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Internal graph runner
# ---------------------------------------------------------------------------

def _invoke_graph(user_message: str, cv_text: str = "") -> Dict[str, Any]:
    """Run the compiled graph from a fresh initial state."""
    graph = _get_graph()

    system_content = _SYSTEM_PROMPT
    if cv_text:
        system_content += (
            f"\n\n=== CANDIDATE CV CONTENT ===\n{cv_text}\n=== END OF CV ===\n\n"
            "STRICT RULE: Extract skills ONLY from the CV above. "
            "Do NOT add any technology not present verbatim in the CV."
        )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ],
        "iterations":            0,
        "final_output":          None,
        "queries_executed":      0,
        "internship_query_done": False,
        "coercion_injected":     False,
    }

    logger.info("Starting recruitment agent graph…")
    final_state = graph.invoke(initial_state)
    logger.info(
        "Graph finished | iterations=%d | queries=%d | intern_done=%s",
        final_state.get("iterations", 0),
        final_state.get("queries_executed", 0),
        final_state.get("internship_query_done", False),
    )

    return final_state.get("final_output") or {
        "job_title":           "Unknown",
        "location":            "Various",
        "total_found":         0,
        "agent_summary":       "Agent produced no output.",
        "search_queries_used": [],
        "jobs":                [],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_cv_analysis(cv_text: str, detected_title: str = "") -> Dict[str, Any]:
    """
    Analyse a candidate's CV and find matching real-world jobs (incl. internships).

    FIX B: user message provides only approved-board query examples with
    explicit site: clauses. "No site: restriction" guidance is removed entirely.
    The query examples in the message match REQUIREMENT 2 in the system prompt.
    """
    # Extract primary technologies from CV text
    _TECH_VOCAB: List[str] = [
        "Python", "Java", "Kotlin", "Swift", "Go", "Rust", "C++", "C#",
        "Ruby", "PHP", "Scala", "TypeScript", "JavaScript",
        "React", "Vue", "Angular", "Svelte", "Flutter", "Android", "iOS",
        "Next.js", "Nuxt", "Node.js", "Django", "Flask", "FastAPI",
        "Spring Boot", "Spring", "Laravel", "Rails", "Express", "NestJS",
        ".NET", "ASP.NET", "SQL", "PostgreSQL", "MySQL", "MongoDB", "Redis",
        "Elasticsearch", "Apache Spark", "Kafka", "TensorFlow", "PyTorch",
        "pandas", "scikit-learn", "NumPy", "LLM", "NLP", "Machine Learning",
        "Deep Learning", "Data Science", "AWS", "Azure", "GCP", "Docker",
        "Kubernetes", "Terraform", "CI/CD", "DevOps", "MLOps",
        "React Native", "Xamarin",
    ]
    cv_lower   = cv_text.lower()
    found_tech = [t for t in _TECH_VOCAB if t.lower() in cv_lower]
    stack_str  = (
        ", ".join(found_tech[:10]) if found_tech
        else "Not yet identified — read the CV carefully before building queries."
    )

    title_hint = f" The candidate's likely title is '{detected_title}'." if detected_title else ""

    user_message = (
        f"A candidate uploaded their CV.{title_hint}\n\n"
        f"══════════════════════════════════════════════════════\n"
        f"PRIMARY TECHNOLOGY STACK (from CV — USE IN EVERY QUERY):\n"
        f"  {stack_str}\n"
        f"══════════════════════════════════════════════════════\n\n"
        f"RULES:\n"
        f"• Include at least one technology from the stack above in EVERY query.\n"
        f"• Use ONLY the stack listed above — never substitute a different\n"
        f"  language's keyword and never let a stray operational word (e.g.\n"
        f"  a leaked 'Go') slip in front of it.\n"
        f"• Every query MUST include a site: clause from the approved board list.\n"
        f"• Do NOT use open-web queries (no site: restriction). They are FORBIDDEN.\n"
        f"• Do NOT add exclusion clauses (-'closed' etc.) — filtering is automatic.\n"
        f"• Query argument must be search tokens only — no conversational preamble.\n\n"
        f"MANDATORY SEARCH STEPS:\n"
        f"☐ Step 1 (full-time): e.g. '{found_tech[0] if found_tech else '<TECH>'} "
        f"developer jobs site:wuzzuf.net OR site:bayt.com'\n"
        f"☐ Step 2 (internship): e.g. '{found_tech[0] if found_tech else '<TECH>'} "
        f"developer internship site:linkedin.com/jobs'\n"
        f"  Use a DIFFERENT approved board in Step 2 than in Step 1. The query\n"
        f"  text MUST contain 'internship', 'intern', or 'trainee'.\n"
        f"☐ Step 3 (if <4 results): try another approved board.\n"
        f"☐ Step 4: emit final JSON ONLY after Steps 1 and 2 are both done.\n\n"
        f"CV Content:\n{cv_text}"
    )

    return _invoke_graph(user_message, cv_text=cv_text)


def run_targeted_search(job_title: str, location: str) -> Dict[str, Any]:
    """
    Search for real open job listings (incl. internships) for a title and location.

    FIX B: query examples always include an approved site: token.
    FIX F: explicit warning against leaking a stray "Go" or other operational
    token in front of the title's own (possibly non-Go) stack.
    """
    # Extract technology from the title string itself
    _INLINE_TECH_RE = re.compile(
        r"\b(Python|Java(?:Script)?|Kotlin|Swift|Go|Rust|C\+\+|C#|Ruby|PHP|"
        r"Scala|TypeScript|React|Vue|Angular|Flutter|Android|iOS|Node\.?js|"
        r"Django|Flask|FastAPI|Spring(?:\s+Boot)?|Laravel|Rails|Express|"
        r"NestJS|\.NET|ASP\.NET|SQL|PostgreSQL|MySQL|MongoDB|Redis|"
        r"TensorFlow|PyTorch|pandas|AWS|Azure|GCP|Docker|Kubernetes)\b",
        re.IGNORECASE,
    )
    title_techs   = _INLINE_TECH_RE.findall(job_title)
    deduped_techs = list(dict.fromkeys(t.strip() for t in title_techs))
    tech_str      = (
        ", ".join(deduped_techs) if deduped_techs
        else "None in title — use the most common stack for this role type."
    )
    # Best single tech token for query examples
    tech_token = deduped_techs[0] if deduped_techs else "<PRIMARY_TECH>"

    user_message = (
        f"Find REAL, CURRENTLY OPEN '{job_title}' jobs AND internships "
        f"in '{location}'.\n\n"
        f"══════════════════════════════════════════════════════\n"
        f"TECHNOLOGY KEYWORDS FOR QUERIES:\n"
        f"  {tech_str}\n"
        f"══════════════════════════════════════════════════════\n\n"
        f"RULES:\n"
        f"• Every query MUST include a technology keyword AND a site: clause.\n"
        f"• Use ONLY the keyword(s) above — never a different language, and\n"
        f"  never a leaked operational word (e.g. 'Go', 'Run', 'Now') placed\n"
        f"  in front of the stack. A stray 'Go' in front of a C#/.NET query,\n"
        f"  for example, will be read as the Go/Golang language and will\n"
        f"  corrupt the results away from the stack you actually want.\n"
        f"• site: clause MUST be from the approved board list in the system prompt.\n"
        f"• Do NOT use open-web queries without a site: clause — FORBIDDEN.\n"
        f"• Do NOT add -'closed' or similar exclusion text — filtering is automatic.\n"
        f"• Query argument must be search tokens only — no conversational preamble.\n\n"
        f"MANDATORY SEARCH STEPS:\n"
        f"☐ Step 1 (full-time):\n"
        f"  '{tech_token} {job_title} jobs site:wuzzuf.net OR site:bayt.com'\n"
        f"  OR '{tech_token} {job_title} jobs site:linkedin.com/jobs'\n"
        f"☐ Step 2 (internship — MANDATORY even if Step 1 returned results):\n"
        f"  '{tech_token} {job_title} internship site:linkedin.com/jobs'\n"
        f"  OR '{tech_token} {job_title} intern site:wuzzuf.net OR site:akhtaboot.com'\n"
        f"  Use a DIFFERENT approved board than Step 1. The query text MUST\n"
        f"  contain 'internship', 'intern', or 'trainee'.\n"
        f"☐ Step 3 (if <4 valid listings): try site:indeed.com or site:glassdoor.com.\n"
        f"☐ Step 4: emit final JSON ONLY after Steps 1 and 2 are both complete.\n\n"
        f"Only include roles in the '{job_title}' domain (or internships thereof).\n"
        f"Return an empty jobs array rather than inventing listings."
    )

    result = _invoke_graph(user_message)
    return _validate_and_fix_output(result, cap_score=75)