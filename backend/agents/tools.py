"""
backend/agents/tools.py
=======================
LangChain tool definitions for the Recruitment AI Agent.
"""

from __future__ import annotations

import logging
import re
from typing import List

from langchain_core.tools import tool

from backend.core.config import get_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-safety constants  (increased for better results)
# ---------------------------------------------------------------------------

MAX_RESULT_CHARS: int = 4000   # was 2000 — model needs more context to act
SNIPPET_CHARS: int = 600       # was 300 — richer snippets = fewer hallucinations
PAGE_CHARS: int = 2000         # was 1500


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
    title = result.get("title", "No title")[:150]
    url = result.get("url", "")[:300]
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
    Search the web for REAL, currently open job postings matching the query.

    IMPORTANT RULES:
    - Always search for REAL jobs with actual application URLs.
    - Use queries that target real job boards: LinkedIn, Indeed, Glassdoor, Wuzzuf, Bayt.
    - Query examples:
        "Python developer jobs Dubai site:linkedin.com OR site:indeed.com"
        "Full Stack Developer Cairo site:wuzzuf.net OR site:bayt.com"
        "React developer remote 2024 site:linkedin.com/jobs"
    - NEVER invent job listings. Only use results returned by this tool.
    - If results have real URLs, use them exactly as returned.

    Returns a numbered list of real job postings with title, URL, and description.
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

    enriched_query = f'"{query}" "apply" OR "job description" 2026'

    # Build a richer query targeting real job boards

    logger.info("Tavily search → query: '%s' | max_results: %d", enriched_query, settings.tavily_max_results)

    
    try:
        response = client.search(
            query=enriched_query,
            search_depth="advanced",       # advanced gives better real job results
            max_results=settings.tavily_max_results,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.error("Tavily search failed: %s", exc)
        # Fallback: try without site restriction
        try:
            response = client.search(
                query=query + " jobs hiring now",
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
            f"No job postings found for: '{query}'.\n"
            "Suggestion: Try broader terms, e.g. 'developer Dubai' instead of very specific titles."
        )

    lines = [f"REAL Job Search Results for: '{query}'\n"]
    for i, result in enumerate(results, start=1):
        lines.append(_format_result(i, result))

    full_output = "\n\n".join(lines)
    trimmed = _trim(full_output, MAX_RESULT_CHARS)

    logger.info("Tavily returned %d results, output length: %d chars", len(results), len(trimmed))
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
    Only call this with URLs returned by tavily_job_search — never invented URLs.
    """
    import urllib.request
    import urllib.error
    import html

    logger.info("Scraping URL: %s", url[:150])

    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            raw_bytes = response.read(65_536)
            charset = response.headers.get_content_charset() or "utf-8"
            raw_html = raw_bytes.decode(charset, errors="replace")
    except urllib.error.URLError as exc:
        return f"Failed to fetch URL: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"

    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    return _trim(text, PAGE_CHARS)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_tools() -> list:
    return [tavily_job_search, scrape_job_page]