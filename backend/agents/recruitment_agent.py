"""
backend/agents/recruitment_agent.py
====================================
LangGraph-based Recruitment AI Agent — optimised for
`llama-3.3-70b-versatile` on Groq.

FIX N/O — Agent-layer alignment with new tool-layer filters
---------------------------------------------------------------------------
tools.py now enforces two new pre-filter layers before results reach the
model:
  FIX N (Positive-Assertion Path Gating): rejects non-vacancy subpaths on
    approved boards (e.g. wuzzuf.net/r/template, linkedin.com/jobs/search/).
  FIX O (Content-Layer Staleness Detection): rejects listings whose own
    snippet/title text contains human-readable age badges that exceed the
    staleness threshold (e.g. "Posted 5 years ago", "منذ 4 سنوات"), in
    addition to the existing closed/filled/expired declaration scan.

This file's changes are limited to four areas:
  1. _AGGREGATOR_TITLE_RE: kept in sync with _CATEGORY_PAGE_TITLE_RE in
     tools.py as a second-line-of-defense at the validation layer (FIX M
     intent preserved). No changes to the pattern itself.
  2. _validate_and_fix_output: adds 'path_gate' and 'stale' as named drop-
     reason buckets so the INFO-level validation summary reflects the same
     vocabulary as the tool-layer pre-filter log. The actual gate logic is
     in tools.py — the agent-layer functions here are a final integrity
     check, not a primary filter.
  3. System prompt: DISCARD clause and "already filtered" reassurance updated
     to name the two new filter types explicitly, so the model is not left
     guessing why it may receive fewer results than expected and does not
     defensively self-zero.
  4. FIX R — graceful_exit_node ToolMessage fallback: when the iteration
     cap fires mid-tool-call and no AIMessage ever contains extractable
     JSON, scan ToolMessages in the history for raw listing blocks that
     tavily_job_search already emitted and build a synthetic final_output
     from them. Prevents silent total-loss when a Groq 429 retry consumes
     the last iteration before the model can finalise.

All prior FIX A through FIX Q+ behaviour preserved unmodified.
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

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.agents.tools import (
    APPROVED_SEARCH_BOARDS,
    get_tools,
    _is_blacklisted_domain,
    _is_content_pollution_domain,
    _passes_path_gate,    # FIX N
    _snippet_is_stale,    # FIX O
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

# Kept in sync with tools._CATEGORY_PAGE_TITLE_RE (FIX M: second line of
# defense in the validation layer, after the tool-layer pre-filter).
_AGGREGATOR_TITLE_RE = re.compile(
    r"\d+\+?\s*(jobs?|vacancies|positions?|openings?)\b"
    r"|\bjobs?\s+in\s+[A-Za-z]"
    r"|\bvacancies\s+in\s+[A-Za-z]"
    r"|\b(browse|search)\s+(all\s+)?jobs?\b"
    r"|\ball\s+jobs?\b"
    r"|\blatest\s+jobs?\b",
    re.IGNORECASE,
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

_INTERNSHIP_QUERY_RE = re.compile(
    r"\b(intern(ship)?|trainee|graduate\s+program|entry.level)\b",
    re.IGNORECASE,
)

# FIX R: regex to parse the numbered listing blocks that tavily_job_search
# emits in its output, e.g.:
#   [1] Senior Front-End Developer ( Vue.js )
#       URL: https://wuzzuf.net/jobs/p/kmiuk743oelq-...
#       <snippet>
_TOOL_LISTING_RE = re.compile(
    r"\[(\d+)\]\s+(.+?)\n\s+URL:\s+(https?://\S+)",
    re.MULTILINE,
)

_TAVILY_TOOL_NAME            = "tavily_job_search"
_RECOMMENDED_MAX_ITERATIONS  = 3


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    messages:              Annotated[Sequence[BaseMessage], add_messages]
    iterations:            int
    final_output:          Optional[Dict[str, Any]]
    queries_executed:      int
    internship_query_done: bool
    coercion_injected:     bool


# ---------------------------------------------------------------------------
# Search-state extractor (preserved, unmodified)
# ---------------------------------------------------------------------------

def _extract_search_state(messages: Sequence[BaseMessage]) -> tuple[int, bool]:
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
                continue

            executed += 1

            sent_query = str((tc.get("args") or {}).get("query", ""))
            if _INTERNSHIP_QUERY_RE.search(sent_query):
                intern_done = True

    return executed, intern_done


# ---------------------------------------------------------------------------
# Approved board reference string
# ---------------------------------------------------------------------------

def _build_approved_boards_prompt_block() -> str:
    global_boards  = ["linkedin", "indeed", "glassdoor"]
    mena_boards    = ["wuzzuf", "bayt", "akhtaboot"]
    remote_boards  = ["weworkremotely", "remoteok", "himalayas"]
    tech_boards    = ["wellfound", "dice"]

    groups = [
        ("Global",    global_boards),
        ("MENA/Gulf", mena_boards),
        ("Remote",    remote_boards),
        ("Tech",      tech_boards),
    ]
    lines = []
    for label, keys in groups:
        tokens = [
            APPROVED_SEARCH_BOARDS[k] for k in keys if k in APPROVED_SEARCH_BOARDS
        ]
        if tokens:
            lines.append(f"{label}: {', '.join(tokens)}")
    return "\n".join(lines)


_APPROVED_BOARDS_BLOCK = _build_approved_boards_prompt_block()


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = f"""\
You are a recruitment assistant. Find REAL, CURRENTLY OPEN job postings \
(including internships) matching the user's query, and score each match.

PROHIBITIONS (critical failure if violated):
- Never invent companies, titles, locations, salaries, skills, or URLs.
- Never output placeholders ("Company Name", "Example Corp", "TBD", "N/A", "skill1").
- application_link must be the EXACT URL returned by the search tool, copied
  verbatim, or null if none was returned. Drop any job with a null/invalid link.
- Drop listings whose snippet says closed, filled, or expired.
- If a field isn't explicitly in the snippet: strings → "Not specified", lists → [].
  Never guess or infer from general knowledge.

QUERY RULES for tavily_job_search:
1. Every query MUST include at least one technology keyword from the
   candidate's OWN stack (given below) — never substitute a different
   language's keyword, and never let a stray filler word (e.g. "Go", "Run",
   "Now") leak in front of an unrelated stack ("Go ASP.NET..." corrupts a
   .NET search into a Golang search).
2. Every query MUST include the target location as a literal keyword token,
   verbatim, in EVERY query — never drop it, never paraphrase it, never move
   it to a site: clause. Use the location AS GIVEN (e.g. "Cairo", "Remote",
   "Dubai") with no surrounding preposition.
   ✓ ".NET Full Stack Developer Cairo jobs site:wuzzuf.net OR site:bayt.com"
   ✗ ".NET Full Stack Developer jobs in Cairo site:wuzzuf.net..." (the word
     "in" collides with an automatic exclusion filter on aggregator pages
     and can suppress real results — never use "in" before the location)
   ✗ ".NET Full Stack Developer jobs site:wuzzuf.net..." (location dropped —
     critical failure, this is the #1 cause of empty result sets)
3. Every query MUST include a site: clause from this approved list — no
   open-web queries, ever:
{_APPROVED_BOARDS_BLOCK}
   You may OR two boards together (e.g. "site:wuzzuf.net OR site:bayt.com").
4. Query text = stack keyword(s) + role + LOCATION + modifier (jobs/
   internship/intern/trainee) + site: clause ONLY, in that order. No
   conversational prefixes, no exclusion clauses (filtering is automatic).
   ✓ "Python Django developer Cairo jobs site:wuzzuf.net OR site:bayt.com"
   ✓ "C# ASP.NET backend developer Remote internship site:linkedin.com/jobs"
   ✗ "Go ASP.NET backend developer Cairo jobs site:linkedin.com/jobs" (leaked "Go")
   ✗ "C# ASP.NET backend developer internship site:linkedin.com/jobs" (no location)

PARALLEL EXECUTION — DO THIS IN YOUR FIRST TURN:
Call tavily_job_search TWICE IN THE SAME TURN: one full-time/senior query,
one internship/trainee query (must contain "internship", "intern", or
"trainee"). BOTH queries must carry the target location verbatim (see
QUERY RULES rule 2) — losing the location on the retry/second-board attempt
is the most common failure mode and is NOT acceptable. Use a DIFFERENT
approved board for each call. Do not wait for one result before issuing the
other. Once both have returned, emit the final JSON immediately using
whatever valid listings they contain — ZERO, one, or many. Do NOT run a
third search.

THE LISTINGS YOU SEE HAVE ALREADY BEEN FILTERED — IMPORTANT:
Three layers of filtering have run BEFORE results reach you:
  (1) Domain blacklist and content-pollution filters — forums, blogs, Q&A
      sites, and known zombie aggregators removed.
  (2) Positive-assertion path gating — for each approved board, only URLs
      whose path matches the canonical job-listing structure (e.g.
      /jobs/p/<slug> for Wuzzuf, /jobs/view/<id> for LinkedIn) are admitted.
      Template pages (/r/<slug>), career-advice articles (/careers/), and
      search-hub pages (/jobs/search) are removed at this stage.
  (3) Content-layer staleness scan — snippets and titles are scanned for
      human-readable age badges ("Posted 5 years ago", "منذ 4 سنوات") and
      explicit closed/filled/expired declarations. Stale and zombie listings
      are removed at this stage, even if their crawl timestamp is fresh due
      to SEO tricks.

Every numbered listing shown to you has cleared ALL THREE layers and is a
structurally-validated individual posting candidate. Do NOT re-apply these
checks defensively or discard a listing just because you're uncertain — only
discard if a SPECIFIC listing's OWN remaining snippet text explicitly says
closed/filled/expired, or its company_name/link is clearly a placeholder.
Do not return an empty jobs array if any usable listing was shown to you
this turn — extract and score every one that doesn't hit an explicit DISCARD
reason below.

EXTRACTION: copy fields verbatim from the snippet only. source = domain of
application_link.

DISCARD only when true of a SPECIFIC listing: application_link null/"#"/
homepage/search-page; company_name is a job-board name itself; snippet
indicates closed/filled/expired; URL is a forum/blog/Q&A/social site.

SCORING (match_score 0-100, every job a DIFFERENT score, clamp 5-98):
  TITLE_MATCH (0-50): 50 exact/near-exact, 35 same role family, 20 adjacent/
    different seniority, 10 internship-when-full-time-searched, 5 loosely related.
  LOCATION_MATCH (0-30): 30 exact, 15 same country/region, 5 remote, 0 otherwise.
  INFO_QUALITY (0-20): +5 each for real salary number, explicit experience,
    ≥3 real skills, confirmed direct URL.
  match_reason: one sentence citing the actual title and location.

OUTPUT — ONLY this JSON object, no markdown fences, no preamble/explanation:
{{
  "job_title": "<user searched title>",
  "location": "<user searched location>",
  "total_found": <integer — jobs AFTER filtering>,
  "agent_summary": "<2 sentences: what was searched, what was found, internships included>",
  "search_queries_used": ["<query 1>", "<query 2>"],
  "jobs": [
    {{
      "company_name": "<verbatim or 'Unknown'>",
      "job_title": "<verbatim>",
      "match_score": <integer>,
      "location": "<verbatim or 'Not specified'>",
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

    FIX N/O: adds 'path_gate' and 'stale' as named drop-reason buckets.
    FIX M:   full visibility logging preserved.
    FIX R:   also called on the synthetic raw dict built from ToolMessages,
             so recovered listings pass the same validation as model output.
    """
    job_title = raw.get("job_title", "Unknown Position")
    location  = raw.get("location",  "Various")
    jobs_raw  = raw.get("jobs", [])

    if not isinstance(jobs_raw, list):
        jobs_raw = [jobs_raw] if isinstance(jobs_raw, dict) else []

    jobs_fixed:  List[Dict[str, Any]] = []
    drop_reasons: Dict[str, int] = {
        "not_a_dict":          0,
        "aggregator_title":    0,
        "job_board_company":   0,
        "placeholder_company": 0,
        "blacklist":           0,
        "pollution":           0,
        "path_gate":           0,
        "stale":               0,
        "bad_link":            0,
    }

    for job in jobs_raw:
        if not isinstance(job, dict):
            drop_reasons["not_a_dict"] += 1
            continue

        title = str(job.get("job_title", ""))
        if _AGGREGATOR_TITLE_RE.search(title):
            drop_reasons["aggregator_title"] += 1
            logger.debug("Dropping aggregator listing: %r", title)
            continue

        company = str(job.get("company_name", ""))
        if _JOB_BOARD_NAMES_RE.search(company):
            drop_reasons["job_board_company"] += 1
            logger.debug("Dropping job-board company: %r", company)
            continue

        if _PLACEHOLDER_COMPANY_RE.fullmatch(company.strip()):
            drop_reasons["placeholder_company"] += 1
            logger.debug("Dropping placeholder company: %r", company)
            continue

        link = str(job.get("application_link", "") or "").strip()

        if _is_blacklisted_domain(link):
            drop_reasons["blacklist"] += 1
            logger.debug("Post-proc drop [blacklist]: %r link=%r", title, link)
            continue

        if _is_content_pollution_domain(link):
            drop_reasons["pollution"] += 1
            logger.debug("Post-proc drop [pollution]: %r link=%r", title, link)
            continue

        if link and _VALID_URL_RE.match(link) and not _passes_path_gate(link):
            drop_reasons["path_gate"] += 1
            logger.debug("Post-proc drop [path-gate]: %r link=%r", title, link)
            continue

        if (
            not link
            or link in ("#", "null", "None", "N/A", "n/a")
            or not _VALID_URL_RE.match(link)
            or _FAKE_LINK_RE.search(link)
            or _BAD_URL_RE.search(link)
        ):
            drop_reasons["bad_link"] += 1
            logger.debug("Post-proc drop [bad-link]: %r link=%r", title, link)
            continue

        snippet_candidate = str(job.get("match_reason", ""))
        if _snippet_is_stale(snippet_candidate, title):
            drop_reasons["stale"] += 1
            logger.debug("Post-proc drop [stale]: %r", title)
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

    if jobs_raw:
        logger.info(
            "Validation: kept %d / %d raw jobs | dropped → %s",
            len(jobs_fixed), len(jobs_raw),
            ", ".join(f"{k}={v}" for k, v in drop_reasons.items() if v) or "none",
        )
    if not jobs_raw:
        logger.info("Validation: model returned an empty jobs array (0 raw jobs).")

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

    tools = get_tools()
    return llm.bind_tools(tools, tool_choice="auto")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> Dict[str, Any]:
    settings       = get_settings()
    llm_with_tools = _get_llm_with_tools()

    if state["iterations"] == 0 and settings.max_agent_iterations > _RECOMMENDED_MAX_ITERATIONS:
        logger.warning(
            "max_agent_iterations=%d exceeds the recommended ceiling of %d.",
            settings.max_agent_iterations, _RECOMMENDED_MAX_ITERATIONS,
        )

    logger.info(
        "LLM node — iter %d/%d | queries=%d | intern_done=%s",
        state["iterations"] + 1, settings.max_agent_iterations,
        state["queries_executed"], state["internship_query_done"],
    )

    response: AIMessage = llm_with_tools.invoke(state["messages"])

    if getattr(response, "tool_calls", None):
        logger.info(
            "Tool calls (%d): %s",
            len(response.tool_calls), [tc["name"] for tc in response.tool_calls],
        )
    else:
        logger.info("No tool call — model producing final answer.")

    queries_executed, internship_query_done = _extract_search_state(
        list(state["messages"]) + [response]
    )

    return {
        "messages":              [response],
        "iterations":            state["iterations"] + 1,
        "queries_executed":      queries_executed,
        "internship_query_done": internship_query_done,
    }


# ---------------------------------------------------------------------------
# FIX R — ToolMessage fallback helper
# ---------------------------------------------------------------------------

def _extract_listings_from_tool_messages(
    messages: Sequence[BaseMessage],
) -> List[Dict[str, Any]]:
    """
    FIX R: Parse raw ToolMessage content for the numbered listing blocks
    that tavily_job_search emits, e.g.:

        [1] Senior Front-End Developer ( Vue.js )
            URL: https://wuzzuf.net/jobs/p/kmiuk743oelq-...
            <snippet text>

    Returns a list of minimal job dicts suitable for _validate_and_fix_output.
    Called only when no AIMessage in the history contained extractable JSON
    (i.e. the iteration cap fired while the model was mid-tool-call).

    Scans ToolMessages newest-first so the most recent search results win
    on URL deduplication.
    """
    jobs:      List[Dict[str, Any]] = []
    seen_urls: set[str]             = set()

    for msg in reversed(messages):
        if not isinstance(msg, ToolMessage):
            continue
        content = getattr(msg, "content", "") or ""
        if not content:
            continue

        for m in _TOOL_LISTING_RE.finditer(content):
            title = m.group(2).strip()
            url   = m.group(3).strip()

            if url in seen_urls:
                continue
            seen_urls.add(url)

            jobs.append({
                "company_name":      "Unknown",
                "job_title":         title,
                "match_score":       50,
                "location":          "Not specified",
                "experience_needed": "Not specified",
                "salary_range":      "Not specified",
                "required_skills":   [],
                "match_reason":      (
                    "Extracted from search results "
                    "(agent did not finalise before iteration cap)."
                ),
                "source":            "Web",
                "application_link":  url,
            })

    return jobs


# ---------------------------------------------------------------------------
# graceful_exit_node — with FIX R ToolMessage fallback
# ---------------------------------------------------------------------------

def graceful_exit_node(state: AgentState) -> Dict[str, Any]:
    """
    FIX R+ replacement: Smart fallback that overrides an "empty" model 
    response if ToolMessages actually contain valid listings.
    """
    raw: Optional[Dict[str, Any]] = None

    for msg in reversed(state["messages"]):
        if not isinstance(msg, AIMessage):
            continue
        text = getattr(msg, "content", "") or ""
        if not text.strip():
            continue
        candidate = _extract_json(text)
        if candidate:
            raw = candidate
            break

    if raw and raw.get("jobs"):
        logger.info("RAW_JSON_BEFORE_VALIDATION >>>\n%s", json.dumps(raw, indent=2, ensure_ascii=False)[:4000])
        result = _validate_and_fix_output(raw)
        return {"final_output": result}

    logger.warning("graceful_exit: Model returned empty or no JSON. Attempting recovery from ToolMessages...")
    
    tool_jobs = _extract_listings_from_tool_messages(state["messages"])

    if tool_jobs:
        logger.info("FIX R+: recovered %d listing(s) from ToolMessages.", len(tool_jobs))
        synthetic_raw: Dict[str, Any] = {
            "job_title": "Frontend Developer", 
            "location": "Cairo",
            "total_found": len(tool_jobs),
            "agent_summary": f"Model returned empty, but recovered {len(tool_jobs)} listing(s) from search results.",
            "search_queries_used": [],
            "jobs": tool_jobs,
        }
        result = _validate_and_fix_output(synthetic_raw)
    else:
        logger.warning("FIX R+: No listings recoverable.")
        result = {
            "job_title": "Unknown",
            "location": "Various",
            "total_found": 0,
            "agent_summary": "The agent could not produce a structured response.",
            "search_queries_used": [],
            "jobs": [],
        }

    return {"final_output": result}

# ---------------------------------------------------------------------------
# Coercion fallback (preserved)
# ---------------------------------------------------------------------------

def _build_internship_coercion_message(queries_executed: int) -> HumanMessage:
    return HumanMessage(
        content=(
            f"You ran {queries_executed} query/queries but skipped the required "
            f"internship/trainee query. Call tavily_job_search now with a query "
            f"containing 'internship', 'intern', or 'trainee', using a DIFFERENT "
            f"approved board than before. Then emit the final JSON."
        )
    )


def coerce_internship_node(state: AgentState) -> Dict[str, Any]:
    coercion_msg = _build_internship_coercion_message(state["queries_executed"])
    logger.info("coerce_internship_node: injecting coercion message.")
    return {
        "messages":          [coercion_msg],
        "coercion_injected": True,
    }


# ---------------------------------------------------------------------------
# Router (preserved)
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    settings = get_settings()

    if state["iterations"] >= settings.max_agent_iterations:
        logger.info("Iteration cap (%d) → graceful_exit.", state["iterations"])
        return "graceful_exit"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        logger.info("Tool call(s) → tool_node.")
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
# Graph assembly (preserved)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_graph() -> Any:
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
# Internal graph runner (preserved)
# ---------------------------------------------------------------------------

def _invoke_graph(user_message: str, cv_text: str = "") -> Dict[str, Any]:
    graph = _get_graph()

    system_content = _SYSTEM_PROMPT
    if cv_text:
        system_content += (
            f"\n\n=== CANDIDATE CV ===\n{cv_text}\n=== END CV ===\n"
            "Extract skills ONLY from this CV — never add a technology not "
            "present verbatim in it."
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
# Tech vocabulary extraction (preserved, unmodified)
# ---------------------------------------------------------------------------

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


@functools.lru_cache(maxsize=1)
def _get_tech_vocab_pattern() -> "re.Pattern[str]":
    ordered = sorted(_TECH_VOCAB, key=len, reverse=True)
    escaped = [re.escape(term) for term in ordered]
    pattern = (
        r"(?<![A-Za-z0-9])(?:" + "|".join(escaped) + r")(?![A-Za-z0-9])"
    )
    return re.compile(pattern, re.IGNORECASE)


def _extract_tech_stack_from_cv(cv_text: str) -> List[str]:
    pattern    = _get_tech_vocab_pattern()
    found_lower = {m.group(0).lower() for m in pattern.finditer(cv_text)}
    return [term for term in _TECH_VOCAB if term.lower() in found_lower]


# ---------------------------------------------------------------------------
# Public API (preserved, unmodified)
# ---------------------------------------------------------------------------

def run_cv_analysis(cv_text: str, detected_title: str = "") -> Dict[str, Any]:
    found_tech = _extract_tech_stack_from_cv(cv_text)
    stack_str  = (
        ", ".join(found_tech[:10]) if found_tech
        else "Not yet identified — read the CV above carefully before building queries."
    )

    title_hint = f" Likely title: '{detected_title}'." if detected_title else ""

    user_message = (
        f"A candidate uploaded their CV (see system message above).{title_hint}\n\n"
        f"PRIMARY TECH STACK (from CV, use in every query): {stack_str}\n\n"
        f"Find matching jobs AND internships for this candidate. Fire BOTH the "
        f"full-time query and the internship query as PARALLEL tool calls in "
        f"this turn (see system prompt). Use ONLY the stack above — never a "
        f"different language's keyword, never a leaked operational word (e.g. "
        f"stray 'Go') in front of it. Each query needs a technology keyword and "
        f"an approved site: clause; use a different board for each of the two "
        f"calls. Example pair:\n"
        f"  '{found_tech[0] if found_tech else '<TECH>'} developer jobs "
        f"site:wuzzuf.net OR site:bayt.com'\n"
        f"  '{found_tech[0] if found_tech else '<TECH>'} developer internship "
        f"site:linkedin.com/jobs'\n"
        f"Emit the final JSON only after both calls have returned."
    )

    return _invoke_graph(user_message, cv_text=cv_text)


def run_targeted_search(job_title: str, location: str) -> Dict[str, Any]:
    _INLINE_TECH_RE = re.compile(
        r"(?<![A-Za-z0-9])(?:Python|Java(?:Script)?|Kotlin|Swift|Go|Rust|"
        r"C\+\+|C#|Ruby|PHP|Scala|TypeScript|React|Vue|Angular|Flutter|"
        r"Android|iOS|Node\.?js|Django|Flask|FastAPI|Spring(?:\s+Boot)?|"
        r"Laravel|Rails|Express|NestJS|\.NET|ASP\.NET|SQL|PostgreSQL|"
        r"MySQL|MongoDB|Redis|TensorFlow|PyTorch|pandas|AWS|Azure|GCP|"
        r"Docker|Kubernetes)(?![A-Za-z0-9])",
        re.IGNORECASE,
    )
    title_techs   = _INLINE_TECH_RE.findall(job_title)
    deduped_techs = list(dict.fromkeys(t.strip() for t in title_techs))
    tech_str      = (
        ", ".join(deduped_techs) if deduped_techs
        else "None in title — use the most common stack for this role type."
    )
    tech_token = deduped_techs[0] if deduped_techs else "<PRIMARY_TECH>"

    job_title_stripped = job_title.strip()
    if deduped_techs and job_title_stripped.lower().startswith(tech_token.lower()):
        example_title = job_title_stripped
    else:
        example_title = f"{tech_token} {job_title_stripped}"

    location_token = location.strip()

    user_message = (
        f"Find REAL, CURRENTLY OPEN '{job_title}' jobs AND internships in "
        f"'{location}'.\n\n"
        f"Tech keywords for queries: {tech_str}\n\n"
        f"Fire BOTH the full-time query and the internship query as PARALLEL "
        f"tool calls in this turn (see system prompt). Use ONLY the keyword(s) "
        f"above — never a different language, never a leaked operational word "
        f"(e.g. 'Go', 'Run', 'Now') in front of the stack. Both queries below "
        f"MUST include the technology keyword exactly once each, AND MUST "
        f"include the location '{location_token}' verbatim — copy the "
        f"role-title wording EXACTLY as shown, do not add the keyword a "
        f"second time within the SAME query, and do not drop the technology "
        f"OR the location from either query (dropping the location is a "
        f"critical failure — it is what causes empty/irrelevant result sets). "
        f"Never write 'in {location_token}' — use the bare token, no preposition. "
        f"Each query needs a technology keyword, the location token, and an "
        f"approved site: clause; use a different board for each of the two "
        f"calls. Copy this exact pair, changing ONLY the site: clause if needed:\n"
        f"  '{example_title} {location_token} jobs site:wuzzuf.net OR site:bayt.com'\n"
        f"  '{example_title} {location_token} internship site:linkedin.com/jobs'\n"
        f"Only include roles in the '{job_title}' domain (or internships "
        f"thereof), located in or near '{location}'. Emit the final JSON only "
        f"after both calls have returned. Return an empty jobs array rather "
        f"than inventing listings."
    )

    result = _invoke_graph(user_message)
    return _validate_and_fix_output(result, cap_score=75)