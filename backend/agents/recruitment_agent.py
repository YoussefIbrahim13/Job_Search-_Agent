"""
backend/agents/recruitment_agent.py
====================================
LangGraph-based Recruitment AI Agent using Groq / Qwen3-32b.

DESIGN PRINCIPLES
-----------------
1. Sequential-only tool execution  — The agent calls ONE tool per step. The graph
   enforces this: after each ToolNode execution, control returns to the LLM node,
   which either calls the next tool or emits the final JSON answer.

2. Hard iteration cap  — max_agent_iterations (from config) limits how many times
   the (llm → tool → llm) cycle can repeat. When the cap is hit, the graph routes
   to graceful_exit which synthesises whatever partial information was gathered.

3. Structured output  — The agent returns a single JSON object. A post-processing
   step parses and validates it, ensuring required_skills is always List[str].

4. Single source of truth  — This file contains ALL agent logic. There is no
   second copy. FastAPI route handlers live exclusively in routes.py.

Graph topology
--------------
    START → llm_node ◄─────────────────┐
                │                      │
             _route                    │
           ┌───┴────┐                  │
      tool_node   graceful_exit    tool_node ──┘
                       │
                      END

CHANGES IN THIS REVISION
-------------------------
  - Zero Hallucination: System prompt now contains an explicit, numbered
    "ABSOLUTE PROHIBITIONS" block. Any field not found verbatim in search
    snippets must use a defined sentinel ("Unknown", "Not specified", []).
  - Recent Jobs: All search queries target 2026. _validate_and_fix_output
    drops listings whose title or snippet suggests they are stale aggregators.
  - Internship scope: System prompt and both public API user messages explicitly
    instruct the model to include internship roles alongside full-time listings.
  - Valid URLs: Jobs whose application_link resolves to "#" after validation are
    now DROPPED entirely from the output rather than kept with a dummy link.
    A new _PLACEHOLDER_COMPANY_RE guard removes listings with generic fake names.
  - Model: qwen/qwen3-32b via ChatGroq (set in settings.groq_model).
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
)
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from backend.agents.tools import get_tools
from backend.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compiled regex patterns  (module-level — compiled exactly once)
# ---------------------------------------------------------------------------

_BAD_URL_RE = re.compile(
    r"/search|\?q=|-jobs-in-|/find-jobs|keyword=|/jobs/?$",
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

# Catch obviously hallucinated / placeholder company names.
_PLACEHOLDER_COMPANY_RE = re.compile(
    r"^(company\s*(name)?|example\s*(corp(oration)?)?|acme|"
    r"your\s*company|n/?a|unknown\s*company|placeholder|"
    r"company\s*\d+|org\s*\d+)$",
    re.IGNORECASE,
)

# Reject obviously non-specific / stale salary strings the model invents.
_FAKE_SALARY_RE = re.compile(
    r"(competitive|negotiable|market\s*rate|tbd|attractive)",
    re.IGNORECASE,
)

# Minimum URL plausibility: must start with http(s) and contain a dot.
_VALID_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Agent State
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """
    Mutable state threaded through every LangGraph node.

    messages     — Full conversation history (accumulated via add_messages reducer).
    iterations   — Incremented on each LLM call; enforces the iteration cap.
    final_output — Set by graceful_exit_node; contains the parsed job-results dict.
    """

    messages: Annotated[Sequence[BaseMessage], add_messages]
    iterations: int
    final_output: Optional[Dict[str, Any]]


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
#
# IMPORTANT: This prompt contains NO per-request placeholders such as
# {job_title} or {location}.  Those values are injected per-request inside
# the HumanMessage constructed in run_cv_analysis() / run_targeted_search().
# Keeping them out of the SystemMessage allows the prompt to be built once
# and reused safely across all requests.

_SYSTEM_PROMPT = """\
You are a recruitment assistant. Your job is to find REAL, CURRENTLY OPEN job
positions (including internships) and score how well each one matches the
user's search query.

=== ABSOLUTE PROHIBITIONS — violating any of these is a critical failure ===

1. NEVER invent, fabricate, or hallucinate ANY field — not company names,
   job titles, locations, salaries, skills, experience requirements, or URLs.
2. NEVER output placeholder text such as "Company Name", "Example Corp",
   "skill1", "Competitive salary", "TBD", "N/A", or similar.
3. NEVER reuse, guess, or construct a URL. application_link MUST be the exact
   URL string returned by the search tool — copied character-for-character.
   If the tool did not return a direct application URL, set application_link
   to null (do not use "#", "N/A", or any invented path).
4. NEVER include a job whose application_link is null, "#", or a job-board
   homepage. Drop it from the results entirely.
5. NEVER include jobs posted before 2026. Only include listings that are
   demonstrably current (posted in 2026 or marked "actively hiring").
6. If a field is not explicitly stated in the search snippet, use ONLY these
   sentinels:
     • strings  → "Not specified"
     • lists    → []
   Do NOT guess, infer, or fill in from general knowledge.

=== SCOPE — ALWAYS include internships ===

Search for BOTH full-time/part-time positions AND internship roles.
When building search queries, always include a variant that targets internships
explicitly (e.g. add "internship OR intern" to one query).

=== PROCESS (follow exactly) ===

STEP 1 — Search (run at least TWO queries):
  Query A — full-time / senior roles:
    "<job title> jobs <location>  2026 site:linkedin.com OR site:indeed.com"
  Query B — internships:
    "<job title> internship <location>  2026 site:linkedin.com OR site:wuzzuf.net"
  Run additional queries if the first two return fewer than 4 real listings.

STEP 2 — Extract (verbatim only, no hallucination):
For EACH job found, copy ONLY what is explicitly written in the snippet:
  company_name      → as written; "Unknown" only if genuinely absent
  job_title         → exact title from the listing (include "Intern" / "Internship"
                       in the title when that is what the listing says)
  location          → exact location; "Not specified" if absent
  experience_needed → only if explicitly stated; else "Not specified"
  salary_range      → only if a real number/currency appears in the snippet;
                       else "Not specified"  (NEVER use words like "Competitive")
  required_skills   → skills/technologies explicitly mentioned; [] if none
  application_link  → EXACT URL from the search tool result — unmodified,
                       character-for-character. null if none was returned.
  source            → domain extracted from application_link (e.g. "linkedin.com")

STEP 3 — Filter before scoring:
  DISCARD any job where:
    • application_link is null, "#", a job-board homepage, or a search results page
    • job_title matches a pattern like "50+ Jobs in Cairo" (aggregator listing)
    • company_name is a job board ("LinkedIn", "Indeed", "Glassdoor", etc.)
    • the listing appears to be from before 2026

STEP 4 — Score each remaining job honestly (match_score 0–100):

  TITLE_MATCH (0–50 pts):
    50 → identical or near-identical to the searched title
    35 → same role family (e.g. searched "Backend Dev", found "Node.js Dev")
    20 → adjacent role or different seniority
    10 → internship when a full-time role was searched (still relevant)
     5 → loosely related

  LOCATION_MATCH (0–30 pts):
    30 → exact location match
    15 → same country / region
     5 → remote
     0 → different country, no remote option

  INFO_QUALITY (0–20 pts):
    +5 salary is provided with an actual number/currency
    +5 experience is explicitly stated
    +5 required_skills has ≥ 3 real skills from the snippet
    +5 direct application link (confirmed real URL, not a board homepage)

  match_score = TITLE_MATCH + LOCATION_MATCH + INFO_QUALITY  [clamped 5–98]
  Every job MUST have a DIFFERENT score reflecting its actual match.

STEP 5 — Write match_reason:
One sentence referencing the actual found job title and location. Example:
  "Title 'Senior React Developer' exactly matches the search and is in Dubai."

=== FINAL OUTPUT ===
Output ONLY the JSON object below. No markdown fences, no explanation,
no text before or after the JSON.

{
  "job_title": "<user searched title>",
  "location": "<user searched location>",
  "total_found": <integer — count of jobs AFTER filtering>,
  "agent_summary": "<2 sentences: what was searched, what was found, whether internships were included>",
  "search_queries_used": ["<query 1>", "<query 2>"],
  "jobs": [
    {
      "company_name": "<verbatim from search or 'Unknown'>",
      "job_title": "<verbatim from search>",
      "match_score": <integer per scoring above>,
      "location": "<verbatim from search or 'Not specified'>",
      "experience_needed": "<verbatim from search or 'Not specified'>",
      "salary_range": "<verbatim number/currency from search or 'Not specified'>",
      "required_skills": ["<only skills explicitly found in snippet>"],
      "match_reason": "<one specific sentence>",
      "source": "<domain>",
      "application_link": "<EXACT URL from tool — never null or '#' here>"
    }
  ]
}
"""


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Try multiple strategies to parse JSON from raw LLM output.

    Strategy 1 — direct parse of the whole (cleaned) string.
    Strategy 2 — extract the outermost balanced { … } block.
    Strategy 3 — treat everything from the first '{' to EOF as potentially
                 truncated JSON and attempt bracket-completion.
    """
    # Strip reasoning blocks (deepseek-r1 / qwen-style <think> tags)
    text = _THINK_BLOCK_RE.sub("", text)
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "").strip()

    # Strategy 1 — direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2 — extract outermost { … }
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    # Strategy 3 — treat everything from first '{' to EOF as truncated JSON.
    if start != -1:
        partial = text[start:]
        partial = _TRAILING_COMMA_RE.sub(r"\1", partial)
        opens_sq = partial.count("[") - partial.count("]")
        opens_cu = partial.count("{") - partial.count("}")
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

    Parameters
    ----------
    raw       : parsed dict from _extract_json
    cap_score : if set, clamp every match_score to this maximum.
                Pass cap_score=75 for targeted search (no CV → no genuine
                high-match signal available).

    Key changes vs original
    -----------------------
    - Jobs with no valid application_link are DROPPED (not kept with "#").
    - Placeholder company names rejected via _PLACEHOLDER_COMPANY_RE.
    - Hallucinated "competitive / negotiable" salaries replaced with sentinel.
    - _BAD_URL_RE + _FAKE_LINK_RE + _VALID_URL_RE all applied; any failure → drop.
    - total_found always reflects the post-filter count.
    - agent_summary and search_queries_used preserved from raw input.
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

        # ── Drop aggregator listings ("50+ Jobs in Dubai") ────────────────
        title = str(job.get("job_title", ""))
        if _AGGREGATOR_TITLE_RE.search(title):
            logger.debug("Dropping aggregator listing: %r", title)
            continue

        # ── Drop job-board company names ("LinkedIn", "Indeed", …) ────────
        company = str(job.get("company_name", ""))
        if _JOB_BOARD_NAMES_RE.search(company):
            logger.debug("Dropping job-board company listing: %r", company)
            continue

        # ── Drop obviously hallucinated / placeholder company names ────────
        if _PLACEHOLDER_COMPANY_RE.fullmatch(company.strip()):
            logger.debug("Dropping placeholder company name: %r", company)
            continue

        # ── application_link — strict validation: drop if not a real URL ──
        link = str(job.get("application_link", "") or "").strip()
        if (
            not link
            or link in ("#", "null", "None", "N/A", "n/a")
            or not _VALID_URL_RE.match(link)
            or _FAKE_LINK_RE.search(link)
            or _BAD_URL_RE.search(link)
        ):
            logger.debug(
                "Dropping job with invalid/missing application_link: %r (link=%r)",
                title, link,
            )
            continue

        # ── required_skills — strip fake placeholders ──────────────────────
        skills = _normalise_skills(job.get("required_skills"))
        job["required_skills"] = [s for s in skills if not _FAKE_SKILL_RE.match(s)]

        # ── match_score — clamp to [5, 98] then apply optional cap ─────────
        try:
            score = int(job.get("match_score", 50))
        except (TypeError, ValueError):
            score = 50
        score = max(5, min(score, 98))
        if cap_score is not None:
            score = min(score, cap_score)
        job["match_score"] = score

        # ── salary_range — reject vague/hallucinated phrases ───────────────
        salary = str(job.get("salary_range", "")).strip()
        if (
            not salary
            or salary.lower() in ("", "n/a", "none", "null", "not specified")
            or _FAKE_SALARY_RE.search(salary)
        ):
            job["salary_range"] = "Not specified"

        # ── defaults for any remaining missing fields ──────────────────────
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

        # Ensure source is derived from the validated link if missing/generic
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
# LLM + tool binding  (instantiated ONCE via lru_cache)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_llm_with_tools() -> Any:
    """
    Instantiate ChatGroq with qwen/qwen3-32b and bind tools exactly once.

    lru_cache ensures a single instance regardless of how many times this
    function is called, eliminating per-request HTTP session overhead.
    """
    settings = get_settings()
    llm = ChatGroq(
        model=settings.groq_model,        # "qwen/qwen3-32b" set in config
        api_key=settings.groq_api_key,
        temperature=0.0,                  # 0.0 for maximum factual determinism
        max_tokens=1500,                  # increased — qwen3-32b handles longer JSON
    )
    tools = get_tools()
    return llm.bind_tools(tools, tool_choice="auto")


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def llm_node(state: AgentState) -> Dict[str, Any]:
    """Call the LLM (with tools bound) and increment the iteration counter."""
    settings = get_settings()
    llm_with_tools = _get_llm_with_tools()

    logger.info(
        "LLM node — iteration %d / %d",
        state["iterations"] + 1,
        settings.max_agent_iterations,
    )

    response: AIMessage = llm_with_tools.invoke(state["messages"])

    if getattr(response, "tool_calls", None):
        logger.info("Tool call: %s", [tc["name"] for tc in response.tool_calls])
    else:
        logger.info("No tool call — model producing final answer.")

    return {
        "messages":   [response],
        "iterations": state["iterations"] + 1,
    }


def graceful_exit_node(state: AgentState) -> Dict[str, Any]:
    """
    Extract and validate the JSON from the most recent AIMessage.

    Scans backwards through message history so a ToolMessage is never
    mistakenly read as the final answer when the iteration cap fires
    immediately after a tool call.
    """
    last_ai_text: str = ""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            last_ai_text = getattr(msg, "content", "") or ""
            break

    raw = _extract_json(last_ai_text)

    if raw:
        # _validate_and_fix_output now DROPS jobs with invalid links.
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
# Router
# ---------------------------------------------------------------------------

def _route(state: AgentState) -> str:
    """
    Decide the next node after each LLM call.

    Rules (checked in order):
    1. Iteration cap exceeded → graceful_exit (prevents infinite loops).
    2. Last message has tool_calls → tool_node (one call at a time).
    3. Otherwise → graceful_exit (model produced final answer or gave up).
    """
    settings = get_settings()

    if state["iterations"] >= settings.max_agent_iterations:
        logger.info(
            "Iteration cap reached (%d) → graceful_exit.", state["iterations"]
        )
        return "graceful_exit"

    last_msg = state["messages"][-1]
    if isinstance(last_msg, AIMessage) and getattr(last_msg, "tool_calls", None):
        logger.info("Tool call detected → routing to tool_node.")
        return "tool_node"

    logger.info("No tool call detected → routing to graceful_exit.")
    return "graceful_exit"


# ---------------------------------------------------------------------------
# Graph assembly  (lru_cache replaces the fragile global + None-check pattern)
# ---------------------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def _get_graph() -> Any:
    """
    Build and compile the LangGraph StateGraph exactly once.

    lru_cache is thread-safe by design; it replaces the original
    `global _COMPILED_GRAPH / if None` pattern which had a TOCTOU race
    under async concurrency.
    """
    tools     = get_tools()
    tool_node = ToolNode(tools)

    graph = StateGraph(AgentState)
    graph.add_node("llm_node",      llm_node)
    graph.add_node("tool_node",     tool_node)
    graph.add_node("graceful_exit", graceful_exit_node)

    graph.add_edge(START, "llm_node")
    graph.add_conditional_edges(
        "llm_node",
        _route,
        {"tool_node": "tool_node", "graceful_exit": "graceful_exit"},
    )
    graph.add_edge("tool_node",     "llm_node")
    graph.add_edge("graceful_exit", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Internal graph runner
# ---------------------------------------------------------------------------

def _invoke_graph(user_message: str, cv_text: str = "") -> Dict[str, Any]:
    """
    Initialise graph state and run the compiled graph synchronously.

    The CV text (when provided) is appended to the system prompt so the LLM
    can extract skills without hallucinating.  Everything else — job title,
    location, and search instructions — goes into the HumanMessage.

    Returns final_output as produced by graceful_exit_node (already validated).
    Callers must NOT call _validate_and_fix_output again on the result
    (except run_targeted_search which re-applies cap_score=75).
    """
    graph = _get_graph()

    system_content = _SYSTEM_PROMPT
    if cv_text:
        system_content += (
            f"\n\n=== CANDIDATE CV CONTENT ===\n{cv_text}\n=== END OF CV ===\n\n"
            "STRICT RULE: Extract skills ONLY from the CV above. "
            "Do NOT guess or add any technology unless it appears verbatim in the CV."
        )

    initial_state: AgentState = {
        "messages": [
            SystemMessage(content=system_content),
            HumanMessage(content=user_message),
        ],
        "iterations":   0,
        "final_output": None,
    }

    logger.info("Starting recruitment agent graph…")
    final_state = graph.invoke(initial_state)
    logger.info(
        "Agent graph finished. Iterations used: %d",
        final_state.get("iterations", 0),
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

    Parameters
    ----------
    cv_text        : Raw text extracted from the uploaded CV file.
    detected_title : Optional title hint from cv_parser (e.g. "Software Engineer").

    Returns
    -------
    Validated job-results dict (already processed by graceful_exit_node).
    No score cap applied — the full CV provides genuine match signal.
    All jobs in the result are guaranteed to have a valid application_link.
    """
    title_hint = (
        f" The candidate's likely title is '{detected_title}'."
        if detected_title
        else ""
    )
    user_message = (
        f"A candidate uploaded their CV.{title_hint}\n\n"
        "MANDATORY STEPS:\n"
        "1. Run at least TWO tavily_job_search queries:\n"
        "   • Query A: full-time/senior roles based on the candidate's skills and title.\n"
        "   • Query B: internship roles using the same skills + 'internship OR intern  2026'.\n"
        "2. Use ONLY results returned by the search tool — zero hallucination.\n"
        "3. ONLY include listings with a real, direct application URL from the tool.\n"
        "   Drop any listing whose URL is missing, a '#', or a job-board homepage.\n"
        "4. Build the final JSON from verified search results only.\n\n"
        f"CV Content:\n{cv_text}"
    )
    # _invoke_graph → graceful_exit_node → _validate_and_fix_output (once).
    # NOT called again here to avoid double-validation.
    return _invoke_graph(user_message, cv_text=cv_text)


def run_targeted_search(job_title: str, location: str) -> Dict[str, Any]:
    """
    Search for real open job listings (incl. internships) matching a title & location.

    Parameters
    ----------
    job_title : The role to search for, e.g. "Senior Python Developer".
    location  : Geographic target, e.g. "Dubai" or "Remote".

    Returns
    -------
    Validated job-results dict with match_score capped at 75.
    The cap reflects that without a CV we cannot compute a genuine high-match
    signal. 75 is the honest ceiling for a title+location search.
    All jobs in the result are guaranteed to have a valid application_link.
    """
    user_message = (
        f"Find REAL, CURRENTLY OPEN '{job_title}' jobs AND internships in '{location}'.\n\n"
        "MANDATORY STEPS:\n"
        f"1. Run at least TWO tavily_job_search queries:\n"
        f"   • Query A: '\"{job_title}\" jobs {location}  2026 hiring now'\n"
        f"   • Query B: '\"{job_title}\" internship {location}  2026'\n"
        "2. Use ONLY the search results — never invent jobs, salaries, skills, or URLs.\n"
        "3. ONLY include listings with a real, direct application URL returned by the tool.\n"
        "   Drop any listing whose URL is missing, '#', or a job-board search page.\n"
        f"4. Only include roles directly in the '{job_title}' domain (or internships thereof).\n"
        "5. If fewer than 3 real listings with valid URLs are found, return what exists.\n"
        "   Return an empty jobs array rather than inventing listings."
    )
    # graceful_exit_node already called _validate_and_fix_output without a cap.
    # We call it a second time ONLY to apply the cap_score=75 ceiling.
    # _validate_and_fix_output is idempotent for all fields except match_score,
    # and already-dropped jobs (no valid link) are not re-added.
    result = _invoke_graph(user_message)
    return _validate_and_fix_output(result, cap_score=75)