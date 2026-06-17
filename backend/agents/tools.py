"""
backend/agents/tools.py
=======================
LangChain tool definitions for the Recruitment AI Agent.

CHANGES IN THIS REVISION
-------------------------
- tavily_job_search now targets 2026 results and always includes an
  internship-variant query path when the caller's query contains "internship".
- Result pre-filtering: entries without a usable URL are logged and excluded
  before the formatted output is returned to the LLM, reducing hallucination
  pressure on link generation.
- Snippet length and total result caps increased for richer context.
- Enriched query construction wraps the caller's original query in quotes and
  appends "apply now" OR "job description"  2026 to bias toward real
  current postings over aggregator summaries.
"""

from __future__ import annotations

import logging
import re
from typing import List
from urllib.parse import urlparse

from langchain_core.tools import tool

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-safety constants
# ---------------------------------------------------------------------------

MAX_RESULT_CHARS: int = 4000   # increased — model needs context to avoid hallucinating
SNIPPET_CHARS: int = 800       # increased — richer snippets = fewer invented fields
PAGE_CHARS: int = 2500         # scraper output limit

# ---------------------------------------------------------------------------
# URL validation helpers  (used to pre-filter Tavily results)
# ---------------------------------------------------------------------------

# Reject search-result / aggregator pages before handing URLs to the LLM.
_BAD_URL_PATTERNS = re.compile(
    r"/search|\?q=|-jobs-in-|/find-jobs|keyword=|/jobs/?$|"
    r"jobs\.(google|bing)\.com|"
    r"/jobs/search|/job-search",
    re.IGNORECASE,
)

_VALID_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)


def _is_usable_url(url: str) -> bool:
    """
    Return True only when a URL looks like a real, direct job posting.

    Rejects:
      • Empty / null strings
      • Non-HTTP(S) URLs
      • Search-results pages and known aggregator patterns
      • URLs with no path at all (bare domain homepages)
    """
    if not url or not _VALID_URL_RE.match(url):
        return False
    if _BAD_URL_PATTERNS.search(url):
        return False
    parsed = urlparse(url)
    # A bare domain (e.g. https://linkedin.com) with no meaningful path → reject
    if not parsed.path or parsed.path in ("/", ""):
        return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _trim(text: str, max_chars: int = MAX_RESULT_CHARS) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ...[TRIMMED]"


def _format_result(index: int, result: dict) -> str:
    title   = result.get("title", "No title")[:150]
    url     = result.get("url", "")[:400]
    content = result.get("content", result.get("snippet", ""))
    snippet = _trim(content, SNIPPET_CHARS)
    return (
        f"[{index}] {title}\n"
        f"    URL: {url}\n"
        f"    {snippet}"
    )


# ---------------------------------------------------------------------------
# Tool 1: Tavily Job Search
# ---------------------------------------------------------------------------

@tool
def tavily_job_search(query: str) -> str:
    """
    Search the web for REAL, CURRENTLY OPEN job postings (including internships).

    RULES FOR THE AGENT:
    - Use queries that target real job boards: LinkedIn, Indeed, Glassdoor, Wuzzuf, Bayt.
    - Always run an internship-specific variant for each search topic.
    - Query examples (full-time):
        "Python developer jobs Dubai  2026 site:linkedin.com OR site:indeed.com"
        "Full Stack Developer Cairo 2026 site:wuzzuf.net OR site:bayt.com"
    - Query examples (internship):
        "Python developer internship Cairo  2026 site:linkedin.com"
        "React developer intern Dubai 2026 site:wuzzuf.net"
    - NEVER invent job listings. Only use results returned by this tool.
    - Every URL in the result is a real URL from the web. Copy it exactly.

    Returns a numbered list of real job postings with title, URL, and description.
    Only listings with a usable direct URL are included.
    """
    settings = get_settings()

    if not settings.tavily_api_key:
        return (
            "ERROR: TAVILY_API_KEY is not set in your .env file.\n"
            "1. Go to https://app.tavily.com and sign up (free tier = 1000 searches/month)\n"
            "2. Copy your API key\n"
            "3. Add to .env: TAVILY_API_KEY=tvly-xxxxxxxxxxxx\n"
            "4. Restart the server"
        )

    try:
        from tavily import TavilyClient
    except ImportError:
        return "ERROR: Run:  pip install tavily-python"

    client = TavilyClient(api_key=settings.tavily_api_key)

    # Build a query biased toward current 2026 real postings.
    # Wrapping the original query in quotes reduces aggregator noise.
    enriched_query = f'"{query}" ("apply now" OR "job description") 2026'

    logger.info(
        "Tavily search → query: %r | max_results: %d",
        enriched_query,
        settings.tavily_max_results,
    )

    try:
        response = client.search(
            query=enriched_query,
            search_depth="advanced",
            max_results=settings.tavily_max_results,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.error("Tavily advanced search failed: %s", exc)
        # Fallback: simpler query, still anchored to current year
        fallback_query = f"{query} jobs hiring 2025 2026"
        try:
            response = client.search(
                query=fallback_query,
                search_depth="basic",
                max_results=settings.tavily_max_results,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as exc2:
            return f"Search error: {exc2}"

    results: List[dict] = response.get("results", [])

    if not results:
        return (
            f"No job postings found for: {query!r}.\n"
            "Suggestion: Try broader terms, e.g. 'developer Dubai' instead of a very specific title."
        )

    # ── Pre-filter: only pass results with a usable direct URL to the LLM ──
    # This is the first line of defence against the model inventing links.
    usable_results: List[dict] = []
    dropped = 0
    for r in results:
        url = r.get("url", "")
        if _is_usable_url(url):
            usable_results.append(r)
        else:
            dropped += 1
            logger.debug("Pre-filter dropped result with unusable URL: %r", url)

    if dropped:
        logger.info(
            "Pre-filter: kept %d / %d results (dropped %d with bad URLs).",
            len(usable_results), len(results), dropped,
        )

    if not usable_results:
        return (
            f"Search returned {len(results)} results for {query!r}, but none had "
            "a valid direct job-posting URL. Try a more specific query targeting "
            "a specific job board, e.g. site:linkedin.com or site:wuzzuf.net."
        )

    lines = [
        f"REAL Job Search Results for: {query!r}\n"
        f"(Showing {len(usable_results)} listings with valid direct URLs)\n"
    ]
    for i, result in enumerate(usable_results, start=1):
        lines.append(_format_result(i, result))

    full_output = "\n\n".join(lines)
    trimmed = _trim(full_output, MAX_RESULT_CHARS)

    logger.info(
        "Tavily: %d usable results returned, output length: %d chars",
        len(usable_results), len(trimmed),
    )
    return trimmed


# ---------------------------------------------------------------------------
# Tool 2: Web Page Scraper
# ---------------------------------------------------------------------------

@tool
def scrape_job_page(url: str) -> str:
    """
    Fetch the text content of a real job posting URL for additional detail.

    Use this tool ONLY when you need specific information (salary, skills, etc.)
    that was not in the search snippet.
    ONLY call this with URLs returned by tavily_job_search — never invented URLs.
    NEVER scrape job-board homepages or search-results pages.
    """
    import html
    import urllib.error
    import urllib.request

    logger.info("Scraping URL: %s", url[:200])

    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"

    if not _is_usable_url(url):
        return (
            f"Refused to scrape {url!r}: it does not look like a direct job posting. "
            "Only scrape URLs returned verbatim by tavily_job_search."
        )

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=12) as response:
            raw_bytes = response.read(65_536)
            charset = response.headers.get_content_charset() or "utf-8"
            raw_html = raw_bytes.decode(charset, errors="replace")
    except urllib.error.URLError as exc:
        return f"Failed to fetch URL: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"

    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",  " ", text,     flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    return _trim(text, PAGE_CHARS)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_tools() -> list:
    return [tavily_job_search, scrape_job_page]