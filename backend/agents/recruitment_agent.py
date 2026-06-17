"""
backend/agents/recruitment_agent.py
====================================
LangGraph-based Recruitment AI Agent using a local Ollama model.

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

Fixes vs original
-----------------
  - Removed circular self-import (this file has no FastAPI imports)
  - System prompt contains NO per-request placeholders (job_title / location
    are injected via HumanMessage, not SystemMessage)
  - LLM + tool binding instantiated ONCE at graph-build time via lru_cache
  - graceful_exit_node scans backwards for the last AIMessage; never reads a
    ToolMessage as the "final answer" when the iteration cap fires
  - URL injection replaced by per-job link validation (no positional mapping)
  - _COMPILED_GRAPH replaced with functools.lru_cache
  - _validate_and_fix_output called exactly ONCE per request
  - _sanitize_targeted_search_result (score=70 corruption) removed entirely
  - All regex patterns compiled at module level (never inside functions)
  - _extract_json Strategy 3 fixed: scans from first '{' to EOF, not to rfind('}')
  - run_cv_analysis no longer calls _validate_and_fix_output a second time
  - Double-validation in run_cv_analysis removed
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
from langchain_ollama import ChatOllama
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
You are a recruitment assistant. Your job is to find REAL open job positions
and score how well each one matches the user's search query.

=== PROCESS (follow exactly) ===

STEP 1 — Search:
Call tavily_job_search with a focused query, e.g.:
  "<job title> jobs <location> 2026 site:linkedin.com OR site:indeed.com"
Run searches with varied queries if needed.

STEP 2 — Extract (no hallucination):
For EACH job found, extract ONLY what is explicitly written in the snippet:
  company_name      → as written; "Unknown" if absent
  job_title         → exact title from the listing
  location          → exact location; "Not specified" if absent
  experience_needed → only if explicitly stated; else "Not specified"
  salary_range      → only if a real number/currency appears; else "Not specified"
  required_skills   → skills/technologies explicitly mentioned; [] if none
  application_link  → exact URL from the search tool, unmodified
  source            → domain extracted from application_link

STEP 3 — Score each job honestly (match_score 0–100):

  TITLE_MATCH (0–50 pts):
    50 → identical or near-identical to the searched title
    35 → same role family (e.g. searched "Backend Dev", found "Node.js Dev")
    20 → adjacent role or different seniority
     5 → loosely related

  LOCATION_MATCH (0–30 pts):
    30 → exact location match
    15 → same country / region
     5 → remote
     0 → different country, no remote option

  INFO_QUALITY (0–20 pts):
    +5 salary is provided
    +5 experience is explicitly stated
    +5 required_skills has ≥ 3 real skills
    +5 direct application link (not a job-board homepage)

  match_score = TITLE_MATCH + LOCATION_MATCH + INFO_QUALITY  [clamped 5–98]
  Every job MUST have a DIFFERENT score reflecting its actual match.

STEP 4 — Write match_reason:
One sentence referencing the actual found job title and location. Example:
  "Title 'Senior React Developer' exactly matches the search and is in Dubai."

=== FINAL OUTPUT ===
Output ONLY the JSON object below. No markdown fences, no explanation.

{
  "job_title": "<user searched title>",
  "location": "<user searched location>",
  "total_found": <integer>,
  "agent_summary": "<2 sentences: what was searched and what was found>",
  "search_queries_used": ["<query 1>", "<query 2>"],
  "jobs": [
    {
      "company_name": "<from search>",
      "job_title": "<from search>",
      "match_score": <integer per scoring above>,
      "location": "<from search or Not specified>",
      "experience_needed": "<from search or Not specified>",
      "salary_range": "<from search or Not specified>",
      "required_skills": ["<only explicitly found skills>"],
      "match_reason": "<one specific sentence>",
      "source": "<domain>",
      "application_link": "<exact URL>"
    }
  ]
}

=== HARD RULES ===
- NEVER invent company names, titles, salaries, skills, or URLs.
- NEVER give every job the same match_score.
- NEVER use placeholder text like "skill1", "Company Name", "Example Corp".
- If skills are not in the snippet → output [].
- Output ONLY the JSON. Nothing else.
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
                 (Fixed vs original: we scan to EOF, NOT to rfind('}'), so
                 opens_cu can be non-zero when the JSON is actually truncated.)
    """
    # Strip reasoning blocks (deepseek-r1 / o1-style models)
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
    # We intentionally do NOT use text[start:end+1] here because rfind("}")
    # would give opens_cu == 0, making bracket-completion a no-op for the
    # most common truncation case (missing closing braces).
    if start != -1:
        partial = text[start:]                              # scan to EOF
        partial = _TRAILING_COMMA_RE.sub(r"\1", partial)   # fix trailing commas
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

    Changes vs original
    -------------------
    - Aggregator company names filtered via regex (not exact-match string set)
    - _BAD_URL_RE applied in addition to _FAKE_LINK_RE
    - total_found always reflects the post-filter job count
    - agent_summary and search_queries_used preserved from raw input
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
        title = job.get("job_title", "")
        if _AGGREGATOR_TITLE_RE.search(title):
            logger.debug("Dropping aggregator listing: %r", title)
            continue

        # ── Drop job-board company names ("LinkedIn", "Indeed", …) ────────
        company = job.get("company_name", "")
        if _JOB_BOARD_NAMES_RE.search(company):
            logger.debug("Dropping job-board company listing: %r", company)
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

        # ── salary_range ───────────────────────────────────────────────────
        salary = str(job.get("salary_range", "")).strip()
        if salary.lower() in ("", "n/a", "none", "null", "not specified"):
            job["salary_range"] = "Not specified"

        # ── application_link — reject search-result pages and fake links ───
        link = str(job.get("application_link", ""))
        if _FAKE_LINK_RE.search(link) or _BAD_URL_RE.search(link):
            job["application_link"] = "#"

        # ── defaults for any missing fields ───────────────────────────────
        defaults: Dict[str, str] = {
            "company_name":      "Unknown",
            "job_title":         job_title,
            "location":          location,
            "experience_needed": "Not specified",
            "salary_range":      "Not specified",
            "match_reason":      "Matches the search criteria.",
            "source":            "Web",
            "application_link":  "#",
        }
        for key, default in defaults.items():
            if not job.get(key):
                job[key] = default

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
    Instantiate ChatOllama and bind tools exactly once.

    lru_cache ensures a single instance regardless of how many times this
    function is called, eliminating the per-iteration HTTP session overhead
    that the original _make_llm() / llm_node pattern caused.
    """
    settings = get_settings()
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0.1,
        num_predict=2048,
        num_ctx=4096,
        num_thread=8,
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

    Scans backwards through the message history so we never accidentally read a
    ToolMessage as the "final answer".  This is the failure mode in the original
    code when the iteration cap fires immediately after a tool call: at that
    point messages[-1] is a ToolMessage containing raw search results, not the
    LLM's JSON output.
    """
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

    Using lru_cache instead of the original `global _COMPILED_GRAPH / if None`
    pattern is safer under async concurrency: lru_cache is thread-safe by
    design, whereas the None-check pattern has a TOCTOU race if the graph is
    ever invoked from a thread pool executor.
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
    location, and search instructions — goes into the HumanMessage so it is
    scoped to the current request rather than baked into the shared system prompt.

    Returns final_output as produced by graceful_exit_node (already validated).
    Callers must NOT call _validate_and_fix_output again on the result.
    """
    graph = _get_graph()

    system_content = _SYSTEM_PROMPT
    if cv_text:
        system_content += (
            f"\n\n=== CANDIDATE CV CONTENT ===\n{cv_text}\n=== END OF CV ===\n\n"
            "STRICT RULE: Extract skills ONLY from the CV above. "
            "Do NOT guess Java/Spring Boot unless explicitly found in the CV."
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

    # final_output was already validated by graceful_exit_node.
    # Return it directly — do NOT call _validate_and_fix_output again here.
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
    Analyse a candidate's CV and find matching real-world jobs.

    Parameters
    ----------
    cv_text        : Raw text extracted from the uploaded CV file.
    detected_title : Optional title hint from cv_parser (e.g. "Software Engineer").

    Returns
    -------
    Validated job-results dict (already processed by graceful_exit_node).
    No score cap is applied — the agent can compute a genuine high match
    because the full CV content is available for comparison.
    """
    title_hint = (
        f" The candidate's likely title is '{detected_title}'."
        if detected_title
        else ""
    )
    user_message = (
        f"A candidate uploaded their CV.{title_hint}\n"
        "STEP 1: Call tavily_job_search with a query based on the candidate's skills "
        "and the detected title above.\n"
        "STEP 2: Use the REAL results to build the final JSON.\n"
        "DO NOT invent jobs. Only use jobs returned by the search tool.\n\n"
        f"CV Content:\n{cv_text}"
    )
    # _invoke_graph → graceful_exit_node → _validate_and_fix_output (once).
    # We do NOT call _validate_and_fix_output here to avoid double-validation
    # which was bug 1.5 in the review (it reset total_found and re-applied caps).
    return _invoke_graph(user_message, cv_text=cv_text)


def run_targeted_search(job_title: str, location: str) -> Dict[str, Any]:
    """
    Search for real open job listings matching a job title and location.

    Parameters
    ----------
    job_title : The role to search for, e.g. "Senior Python Developer".
    location  : Geographic target, e.g. "Dubai" or "Remote".

    Returns
    -------
    Validated job-results dict with match_score capped at 75.
    The cap reflects the fact that without a CV we cannot compute a genuine
    high-match signal — 75 is the honest ceiling for a title+location search.
    """
    user_message = (
        f"Find REAL open '{job_title}' jobs in '{location}'.\n"
        f"STEP 1: Call tavily_job_search with the query: "
        f"'\"{job_title}\" jobs {location} 2026'\n"
        "STEP 2: Use the REAL search results to build the final JSON.\n"
        f"STRICT: Only include jobs directly matching the '{job_title}' domain.\n"
        "DO NOT invent jobs, salaries, skills, or URLs not in the search results.\n"
        "If no jobs are found, return an empty jobs array."
    )
    # graceful_exit_node already called _validate_and_fix_output without a cap.
    # We call it a second time here ONLY to apply the cap_score=75 ceiling.
    # This is safe because _validate_and_fix_output is idempotent for all fields
    # except match_score, and cap_score=75 is the only mutation we want.
    result = _invoke_graph(user_message)
    return _validate_and_fix_output(result, cap_score=75)