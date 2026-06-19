"""
backend/agents/tools.py
=======================
LangChain tool definitions for the Recruitment AI Agent.

CHANGES IN THIS REVISION (volume optimisation pass)
-----------------------------------------------------
FIX C — Context Window Truncation Bottleneck:
  Problem: MAX_RESULT_CHARS was capped at 4,000, which silently cut off the
  back half of a 10-result Tavily response before the LLM ever saw it. Since
  truncation happened AFTER formatting (i.e. after the pipeline banner +
  warnings + all numbered results were joined into one string), a single
  query frequently surfaced only the first 1-2 listings even when the
  pre-filter loop had approved 6-8 clean URLs.

  Fix:
    1. MAX_RESULT_CHARS raised 4,000 → 10,000 and SNIPPET_CHARS raised
       800 → 1,200 so more listings survive intact.
    2. The truncation point is now applied PER-RESULT-BLOCK awareness: the
       formatter assembles all usable results first, and only trims the
       trailing overflow at a results-block boundary (never mid-listing),
       via _join_results_within_budget(). This guarantees every listing that
       makes it into the output is complete — never a half-cut job entry
       that confuses the LLM's JSON extraction.
    3. Tavily max_results raised at the call site is left to config
       (settings.tavily_max_results) but the pre-filter loop no longer stops
       conceptually at "however many happened to survive" — it explicitly
       logs and preserves order so the richest, most-complete set of clean
       listings is what gets formatted and (if needed) trimmed last.

FIX D — Query Token Contamination ("stray operational/stack-noise tokens"):
  Problem: small-model tool-call arguments sometimes contain a stray bare
  word like "Go" prepended/appended to an unrelated stack (e.g. an ASP.NET
  search), because "Go" is a common operational/filler token in English
  ("go ahead and search...") that survives the conversational-filler regex
  (which only strips known PHRASE patterns, not single stray tokens). Tavily
  then reads "Go" as the Go programming language and skews results toward
  Golang content, starving the actual requested stack (e.g. C#/.NET) of
  result slots.

  Fix: _strip_stray_stack_noise() — a second-pass sanitiser step that runs
  AFTER _QUERY_FILLER_RE. It removes specific single-token operational/filler
  words (go, search, find, please, lookup, query, fetch, run, execute, ok,
  okay, now) when they appear as a standalone word OUTSIDE of a quoted
  phrase or adjacent to other tech tokens in a way that would change the
  query's stack semantics. Because "Go" is ALSO a legitimate language name,
  the function only strips it when it is NOT immediately followed by a
  language-indicating context (e.g. "Go developer", "Golang", "Go (lang)")
  — i.e. it strips "Go" only when it looks like a leaked operational verb at
  the START of the string (the position these leaks always occur in), never
  when it appears as a genuine mid-query tech token.

Earlier changes (preserved without modification):
  - FIX A: Negative-keyword semantic leakage → moved to Tavily
    exclude_domains; zombie detection moved to post-fetch snippet scanning.
  - FIX B: Open-web cognitive drift → APPROVED_SEARCH_BOARDS allowlist +
    _sanitise_query() filler-prefix stripping.
  - FIX 1: _has_tech_signal() technology-free query guard + warning output.
  - FIX 2: _BLACKLISTED_DOMAINS + _is_blacklisted_domain() CSR zombie filter.
  - FIX 3: per-result pipeline-status banner and closing reminder.
  - Dynamic recency via Tavily start_date (rolling RECENCY_WINDOW_DAYS).
  - Pre-filtering of results without a valid direct URL.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import List
from urllib.parse import urlparse

from langchain_core.tools import tool

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-safety constants
# ---------------------------------------------------------------------------
#
# FIX C: both ceilings raised so a full batch of approved-board results can
# reach the LLM intact instead of being chopped mid-listing.


MAX_RESULT_CHARS: int = 3_000  
SNIPPET_CHARS: int    = 700     
PAGE_CHARS: int       = 2_500



# MAX_RESULT_CHARS: int = 10_000
# SNIPPET_CHARS: int    = 1_200
# PAGE_CHARS: int       = 2_500


# ---------------------------------------------------------------------------
# Recency configuration
# ---------------------------------------------------------------------------

RECENCY_WINDOW_DAYS: int = 60


def _compute_recency_cutoff(days_back: int = RECENCY_WINDOW_DAYS) -> str:
    """
    Return today's date minus ``days_back`` days as YYYY-MM-DD.
    Recomputed on every call so a long-running server never drifts.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return cutoff.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# FIX 2 (preserved): Domain-level blacklist for CSR zombie aggregators
# ---------------------------------------------------------------------------
#
# These domains render "closed/no longer accepting" status via client-side
# JavaScript — invisible to raw HTML scraping. Domain-level blocking is the
# only reliable remedy.
#
# Rationale per entry:
#   founditgulf.com        — CSR "No longer accepting" banner; high zombie rate
#   gulftalent.com         — Same CSR banner; listings indexed for years post-close
#   naukrigulf.com         — Keeps expired postings searchable for SEO
#   tanqeeb.com            — Regional aggregator; expired listings not purged
#   drjobs.ae / drjobs.com — Confirmed stale listing aggregator
#   laimoon.com            — Gulf aggregator with zombie listing problem
#   monsterindia.com       — Shares expired-listing behaviour with Gulf mirrors

_BLACKLISTED_DOMAINS: frozenset[str] = frozenset({
    "founditgulf.com",
    "gulftalent.com",
    "naukrigulf.com",
    "tanqeeb.com",
    "drjobs.ae",
    "drjobs.com",
    "laimoon.com",
    "monsterindia.com",
})


# ---------------------------------------------------------------------------
# FIX A — Content pollution domains (forums, Q&A, social, blogs, news)
# ---------------------------------------------------------------------------
#
# CAUSE: The zombie-job exclusion phrases were injected directly into the
# Tavily query string. Search engines semantically index pages that *discuss*
# those phrases — HR advice articles on "what to do when a job closes",
# Reddit threads asking "how do I know if applications are closed", etc. —
# and surface them as high-relevance hits because they contain the exact
# quoted text. The result was a context window flooded with non-job content.
#
# FIX: These domains are now excluded at the Tavily API level via the
# `exclude_domains` parameter (see _build_tavily_exclude_domains()), which
# prevents them from appearing in results at all — no query-text interaction.
# They are ALSO enforced at the URL pre-filter stage as a belt-and-suspenders
# guard in case the API parameter is ignored for any result.
#
# MAINTENANCE: Add entries here freely; all consumers read from this set.
# Use the registered domain only (no www., no path). Subdomains are caught
# automatically by _is_content_pollution_domain().

_CONTENT_POLLUTION_DOMAINS: frozenset[str] = frozenset({
    # Q&A boards
    "quora.com",
    "stackoverflow.com",
    "stackexchange.com",
    "answers.com",
    "yahoo.com",          # covers answers.yahoo.com
    "ask.com",
    # Social networks / discussion
    "reddit.com",
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com/pulse", # blog sub-path — but we block at domain level;
                          # linkedin.com/jobs/* is allowed via the trusted-hub
                          # whitelist in the approved boards list.
                          # NOTE: linkedin.com itself is in APPROVED_SEARCH_BOARDS;
                          # we do NOT add it here — job listing paths are fine.
                          # Only the /pulse blog path pollutes; handled by
                          # the URL pattern filter (_BAD_URL_PATTERNS) instead.
    "medium.com",
    "substack.com",
    "hashnode.com",
    "dev.to",
    # General news / magazine / educational
    "wikipedia.org",
    "wikihow.com",
    "thoughtco.com",
    "thebalancemoney.com",
    "thebalancecareers.com",
    "investopedia.com",
    "businessinsider.com",
    "forbes.com",
    "techcrunch.com",
    "towardsdatascience.com",
    "analyticsvidhya.com",
    "geeksforgeeks.org",
    "tutorialspoint.com",
    "javatpoint.com",
    "w3schools.com",
    # HR / career advice blogs (not job listing sites)
    "thebalance.com",
    "livecareer.com",
    "resumegenius.com",
    "zety.com",
    "novoresume.com",
    "resumelab.com",
    "jobscan.co",
    "themuse.com",
    "careerbuilder.com",  # search pages only — listed here because it pollutes;
                          # actual job postings on careerbuilder go through
                          # the URL-pattern filter which accepts /job/* paths
})


def _is_content_pollution_domain(url: str) -> bool:
    """
    Return True if *url* comes from a known content-pollution domain
    (forum, Q&A board, social network, blog, or news site).

    Uses the same netloc-normalisation logic as _is_blacklisted_domain:
    strips leading 'www.' and checks for exact match or subdomain suffix.

    Note on linkedin.com: it is intentionally NOT in _CONTENT_POLLUTION_DOMAINS
    because linkedin.com/jobs/* pages are legitimate job listings and are in
    APPROVED_SEARCH_BOARDS. The /pulse blog sub-path is caught separately by
    _BAD_URL_PATTERNS in _is_usable_url() (path contains '/pulse').
    """
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        for blocked in _CONTENT_POLLUTION_DOMAINS:
            if netloc == blocked or netloc.endswith("." + blocked):
                return True
        return False
    except Exception:
        return False


def _is_blacklisted_domain(url: str) -> bool:
    """
    Return True if *url* belongs to any domain in _BLACKLISTED_DOMAINS.
    Used by both the pre-filter and by recruitment_agent._validate_and_fix_output.
    """
    try:
        netloc = urlparse(url).netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        for blocked in _BLACKLISTED_DOMAINS:
            if netloc == blocked or netloc.endswith("." + blocked):
                return True
        return False
    except Exception:
        return False


# ---------------------------------------------------------------------------
# FIX A — Tavily exclude_domains builder
# ---------------------------------------------------------------------------

def _build_tavily_exclude_domains() -> List[str]:
    """
    Return a deduplicated list of all domains Tavily should never return.

    Combines:
      • _BLACKLISTED_DOMAINS  (CSR zombie aggregators)
      • _CONTENT_POLLUTION_DOMAINS (forums, Q&A, social, blogs, news)

    Called fresh on each search so new entries added to either frozenset
    are picked up immediately without restarting the server.
    """
    return sorted(_BLACKLISTED_DOMAINS | _CONTENT_POLLUTION_DOMAINS)


# ---------------------------------------------------------------------------
# Zombie-job content scanner (snippet-level — replaces query-string exclusion)
# ---------------------------------------------------------------------------

_ZOMBIE_CONTENT_SNIPPETS_EN: tuple[str, ...] = (
    "no longer accepting applications",
    "position filled",
    "job expired",
    "this job is no longer available",
    "applications closed",
    "this position has been filled",
    "vacancy closed",
)

_ZOMBIE_CONTENT_SNIPPETS_AR: tuple[str, ...] = (
    "لم نعد نقبل استمارات",
    "تم غلق هذا العرض",
    "انتهت فترة التقديم",
    "الوظيفة مغلقة",
    "تم شغل الوظيفة",
)

# Pre-compile for efficiency — called once per result snippet
_ZOMBIE_SNIPPET_RE = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in _ZOMBIE_CONTENT_SNIPPETS_EN + _ZOMBIE_CONTENT_SNIPPETS_AR
    ),
    re.IGNORECASE,
)


def _snippet_is_zombie(snippet: str) -> bool:
    """Return True if the snippet text indicates the listing is closed/filled."""
    return bool(_ZOMBIE_SNIPPET_RE.search(snippet))


# ---------------------------------------------------------------------------
# FIX B — Approved search board registry
# ---------------------------------------------------------------------------

APPROVED_SEARCH_BOARDS: dict[str, str] = {
    # ── Global generalist ─────────────────────────────────────────────────
    "linkedin":       "site:linkedin.com/jobs",
    "indeed":         "site:indeed.com",
    "glassdoor":      "site:glassdoor.com",
    # ── MENA / Egypt / Gulf regional ──────────────────────────────────────
    "wuzzuf":         "site:wuzzuf.net",
    "bayt":           "site:bayt.com",
    "akhtaboot":      "site:akhtaboot.com",
    # ── Remote-focused ────────────────────────────────────────────────────
    "weworkremotely": "site:weworkremotely.com",
    "remoteok":       "site:remoteok.com",
    "himalayas":      "site:himalayas.app",
    # ── Tech-specialist ───────────────────────────────────────────────────
    "wellfound":      "site:wellfound.com",
    "dice":           "site:dice.com",
}

# Flat list of site: tokens for use in prompt templates and logging
APPROVED_SITE_TOKENS: List[str] = list(APPROVED_SEARCH_BOARDS.values())


# ---------------------------------------------------------------------------
# FIX 1 (preserved): Technology-signal guard
# ---------------------------------------------------------------------------

_TECH_SIGNAL_TERMS: frozenset[str] = frozenset({
    "python", "java", "kotlin", "swift", "go", "golang", "rust", "c++", "c#",
    "ruby", "php", "scala", "typescript", "javascript",
    "react", "vue", "angular", "svelte", "flutter", "android", "ios",
    "next.js", "nextjs", "nuxt",
    "node", "nodejs", "django", "flask", "fastapi", "spring", "laravel",
    "rails", "express", "nest", "nestjs", "dotnet", ".net",
    "sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch",
    "spark", "kafka", "tensorflow", "pytorch", "pandas", "scikit",
    "machine learning", "deep learning", "nlp", "llm", "data science",
    "aws", "azure", "gcp", "docker", "kubernetes", "k8s", "terraform",
    "ci/cd", "devops", "mlops",
    "react native", "xamarin",
    "asp.net", "c#",
})


def _has_tech_signal(query: str) -> bool:
    """Return True if *query* contains at least one concrete technology keyword."""
    lower = query.lower()
    for term in _TECH_SIGNAL_TERMS:
        if len(term) <= 2:
            if re.search(rf"\b{re.escape(term)}\b", lower):
                return True
        else:
            if term in lower:
                return True
    return False


# ---------------------------------------------------------------------------
# FIX B — Query sanitiser (filler-phrase stripping)
# ---------------------------------------------------------------------------

# Patterns that must be stripped from query prefixes.
# Order matters: match longest/most-specific patterns first.
_QUERY_FILLER_RE = re.compile(
    r"^\s*("
    r"please\s+(search\s+(for|about)|find|look\s+up|look\s+for)|"
    r"search\s+(for|about|the\s+web\s+for|for\s+jobs?\s+(related\s+to|about))|"
    r"find\s+(me\s+)?(jobs?|listings?|openings?|postings?|roles?|positions?)"
    r"(\s+(for|related\s+to|about|in|on))?|"
    r"find\s+(me\s+)?|"                              # bare "find me" / "find"
    r"look\s+(up|for)\s+(jobs?|listings?|postings?|roles?)?|"
    r"i\s+(need|want)\s+to\s+(find|search(\s+for)?|look(\s+up)?)|"
    r"can\s+you\s+(find|search(\s+for)?|look\s+up)(\s+jobs?\s*(related\s+to|for|in|about)?)?|"
    r"get\s+(me\s+)?(jobs?|listings?|results?\s+for)|"
    r"query\s*:\s*|search\s*:\s*|"
    r"use\s+tavily\s+to\s+(find|search(\s+for)?)|"
    r"call\s+tavily[_\s]job[_\s]search\s+with\s+"
    r")\s*",
    re.IGNORECASE,
)

# Strip trailing punctuation / quotes that sometimes wrap the query
_QUERY_TRAILING_RE = re.compile(r'[\s"\'.,;:!?]+$')

# Maximum query length sent to Tavily — guards against prompt-injected blobs
MAX_QUERY_CHARS: int = 300


# ---------------------------------------------------------------------------
# FIX D — Stray operational/stack-noise token stripper
# ---------------------------------------------------------------------------
#
# CAUSE: _QUERY_FILLER_RE only matches known multi-word PHRASE patterns
# anchored at the start of the string. It does not catch a single leaked
# operational token such as a bare "Go" that the model sometimes prepends
# (e.g. "Go ASP.NET Backend Developer jobs site:linkedin.com/jobs"). Tavily
# then reads "Go" as the Go/Golang programming language, polluting the
# stack signal and starving the genuinely requested stack (C#/.NET) of
# relevant results.
#
# DANGER: "Go" is ALSO a real, legitimate technology keyword (Golang). We
# must NEVER strip it when it is genuinely meant as the language. The
# distinguishing signal is structural, not lexical:
#   • A genuine Golang query says things like "Go developer", "Golang",
#     "Go backend engineer", "Go (Golang)" — "Go" functions as a NOUN/STACK
#     token sitting among other stack/role tokens, often paired with
#     "developer", "engineer", "backend", or appearing as "Golang" outright.
#   • A LEAKED operational "Go" sits at the very START of the query,
#     immediately followed by an unrelated, clearly-different stack token
#     (e.g. "Go ASP.NET", "Go React", "Go Java") — i.e. "Go" followed by
#     another tech keyword that is NOT itself Go/Golang.
#
# RULE: Strip a leading standalone "Go" token ONLY when:
#   1. It is the first word of the query, AND
#   2. The very next token is a DIFFERENT recognised technology keyword
#      (from _TECH_SIGNAL_TERMS, excluding go/golang itself), AND
#   3. The string "golang" does NOT appear anywhere in the query (if it
#      does, the model clearly meant the Go language and disambiguated
#      itself — leave it alone).
#
# This makes the strip conservative: it only fires in the exact leak
# pattern this bug report describes, and never touches a real "Go developer"
# or "Golang" query.

_LEADING_GO_TOKEN_RE = re.compile(r"^\s*go\b[\s,:-]*", re.IGNORECASE)

# Other common single-token operational/filler leaks. These are stripped
# unconditionally when they appear as a STANDALONE leading word, because
# unlike "Go" none of them double as legitimate technology names.
_LEADING_OPERATIONAL_TOKENS_RE = re.compile(
    r"^\s*(ok|okay|now|please|go\s+ahead\s+and|execute|run|fetch|lookup)\b[\s,:-]*",
    re.IGNORECASE,
)


def _next_token(text: str) -> str:
    """Return the first whitespace-delimited token of *text*, lowercased."""
    match = re.match(r"\s*([^\s]+)", text)
    return match.group(1).lower() if match else ""


def _strip_stray_stack_noise(query: str) -> str:
    """
    Remove a leaked leading operational token (most commonly a stray "Go")
    that contaminates the technology-stack signal sent to Tavily.

    Applied AFTER _QUERY_FILLER_RE / _QUERY_TRAILING_RE, as a second,
    narrower pass. Pure function — does not mutate input.
    """
    q = query.strip()
    if not q:
        return q

    lower_full = q.lower()

    # Step 1 — generic single-token operational leaks (never legitimate
    # tech names, safe to strip unconditionally when leading).
    new_q = _LEADING_OPERATIONAL_TOKENS_RE.sub("", q).strip()
    if new_q != q:
        logger.info(
            "_strip_stray_stack_noise: removed leading operational token | "
            "original=%r | cleaned=%r", q[:120], new_q[:120],
        )
        q = new_q
        lower_full = q.lower()

    # Step 2 — the ambiguous "Go" case. Only strip when structurally certain
    # it's a leak, not a genuine Golang reference.
    if "golang" in lower_full:
        return q  # model explicitly disambiguated — never touch it

    if _LEADING_GO_TOKEN_RE.match(q):
        remainder = _LEADING_GO_TOKEN_RE.sub("", q)
        following_token = _next_token(remainder)
        # Normalise trailing punctuation off the token for comparison
        # (e.g. "ASP.NET," -> "asp.net"), but keep internal punctuation
        # like the '.' in "asp.net" or '#' in "c#" since those are part
        # of the term itself in _TECH_SIGNAL_TERMS.
        following_token_clean = following_token.rstrip(".,;:!?")

        # Use substring-style matching (same approach as _has_tech_signal)
        # rather than exact set membership, so "ASP.NET" (a multi-char term
        # that may appear with case differences) is reliably recognised.
        is_known_tech_term = any(
            following_token_clean == term or following_token_clean in term or term in following_token_clean
            for term in _TECH_SIGNAL_TERMS
        )
        is_different_tech = is_known_tech_term and following_token_clean not in ("go", "golang")

        if is_different_tech:
            logger.info(
                "_strip_stray_stack_noise: removed leaked leading 'Go' token "
                "before unrelated stack token %r | original=%r | cleaned=%r",
                following_token_clean, q[:120], remainder.strip()[:120],
            )
            q = remainder.strip()
        # else: "Go" stays — could be "Go developer jobs..." (genuine Golang)

    return q


def _sanitise_query(raw_query: str) -> str:
    """
    Strip conversational filler prefixes, stray operational/stack-noise
    tokens, and trailing artifacts from a query string before it is sent
    to Tavily.

    Steps applied in order:
      1. Strip leading/trailing whitespace.
      2. Remove known filler PHRASE prefix patterns (_QUERY_FILLER_RE).
      3. FIX D: remove stray single-token operational/stack-noise leaks
         (e.g. a leaked leading "Go" that corrupts the stack signal),
         via _strip_stray_stack_noise().
      4. Strip residual leading/trailing punctuation or quotes.
      5. Collapse internal whitespace runs to a single space.
      6. Truncate to MAX_QUERY_CHARS to prevent prompt-injection blobs.

    Returns the cleaned query string. If the result is empty (the entire
    query was filler), returns the original raw_query stripped of whitespace
    as a fallback — it is better to send an imperfect query than nothing.
    """
    original = raw_query

    q = raw_query.strip()
    q = _QUERY_FILLER_RE.sub("", q)
    q = _strip_stray_stack_noise(q)          # FIX D
    q = _QUERY_TRAILING_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = q[:MAX_QUERY_CHARS]

    if not q:
        # Fallback: return stripped original so the search still runs
        logger.warning(
            "_sanitise_query: entire query was filler → returning stripped original: %r",
            original[:120],
        )
        return original.strip()[:MAX_QUERY_CHARS]

    if q != original.strip():
        logger.info(
            "_sanitise_query: cleaned query | original=%r | cleaned=%r",
            original[:120], q[:120],
        )

    return q


# ---------------------------------------------------------------------------
# URL validation helpers
# ---------------------------------------------------------------------------

_BAD_URL_PATTERNS = re.compile(
    r"/search|\?q=|-jobs-in-|/find-jobs|keyword=|/jobs/?$|"
    r"jobs\.(google|bing)\.com|"
    r"/jobs/search|/job-search|/pulse/",   # /pulse/ catches LinkedIn blog posts
    re.IGNORECASE,
)

_VALID_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)


def _is_usable_url(url: str) -> bool:
    """
    Return True only when a URL looks like a real, direct job posting.

    Rejects (in order):
      1. Empty / non-HTTP(S) strings
      2. CSR zombie-aggregator domains (_BLACKLISTED_DOMAINS)
      3. Content-pollution domains — forums, Q&A, social, blogs (_CONTENT_POLLUTION_DOMAINS)
      4. Search-results page patterns and known aggregator URL shapes
      5. Bare domain homepages (path is '/' or empty)
    """
    if not url or not _VALID_URL_RE.match(url):
        return False
    if _is_blacklisted_domain(url):
        return False
    if _is_content_pollution_domain(url):
        return False
    if _BAD_URL_PATTERNS.search(url):
        return False
    parsed = urlparse(url)
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
# FIX C — Listing-boundary-safe budget joiner
# ---------------------------------------------------------------------------
#
# CAUSE: the previous implementation joined ALL formatted blocks (banner +
# warnings + every numbered result) into one string and THEN called _trim(),
# which truncates blindly at a raw character offset. With 8-10 formatted
# listings this regularly sliced a job entry in half mid-snippet, and the
# half-entry after the cut was often unparseable noise the LLM had to wade
# through (or worse, hallucinate a "completion" for).
#
# FIX: assemble the fixed-cost header blocks (banner/warnings/intro) first,
# then add complete per-result blocks ONE AT A TIME, stopping BEFORE adding
# a block that would breach the budget. This guarantees every listing that
# reaches the LLM is whole, and we keep as many complete listings as the
# budget allows rather than an arbitrary character cut.

def _join_results_within_budget(
    header_blocks: List[str],
    result_blocks: List[str],
    footer_block: str,
    max_chars: int = MAX_RESULT_CHARS,
) -> tuple[str, int]:
    """
    Join header + as many complete result_blocks as fit + footer, never
    exceeding max_chars and never truncating a result block mid-way.

    Returns (joined_text, number_of_result_blocks_included).
    """
    header_text = "\n\n".join(b for b in header_blocks if b)
    footer_text = footer_block or ""

    reserved = len(header_text) + len(footer_text) + 8  # join-newline slack
    budget   = max(max_chars - reserved, 0)

    included: List[str] = []
    used = 0
    for block in result_blocks:
        block_cost = len(block) + 2  # account for the "\n\n" join separator
        if used + block_cost > budget and included:
            # Stop here — adding this block would breach budget, but we
            # already have at least one complete listing included.
            break
        if used + block_cost > budget and not included:
            # Even the FIRST listing doesn't fit in budget (pathological
            # case of a single huge snippet) — include it anyway, trimmed,
            # rather than returning zero listings.
            block = _trim(block, max(budget, 500))
        included.append(block)
        used += block_cost
        if used >= budget:
            break

    parts = [header_text] if header_text else []
    parts.extend(included)
    if footer_text:
        parts.append(footer_text)

    return "\n\n".join(parts), len(included)


# ---------------------------------------------------------------------------
# Tool 1: Tavily Job Search
# ---------------------------------------------------------------------------

@tool
def tavily_job_search(query: str) -> str:
    """
    Search for REAL, CURRENTLY OPEN job postings on APPROVED premium job boards.

    ══════════════════════════════════════════════════════════════════════
    CRITICAL RULES — read completely before constructing each query call
    ══════════════════════════════════════════════════════════════════════

    RULE 1 — TECHNOLOGY STACK IS MANDATORY IN EVERY QUERY:
      Every query MUST include at least one concrete technology keyword from
      the candidate's stack (Python, React, Django, Node.js, Flutter, C#,
      ASP.NET, etc.). A stack-free query is a CRITICAL FAILURE.

    RULE 2 — APPROVED BOARDS ONLY — NO OPEN-WEB QUERIES:
      Every query MUST include a site: clause from the approved list below.
      Queries WITHOUT a site: clause are FORBIDDEN and will produce context
      pollution. You must pick one approved board per query and vary the
      board across your two required query calls.

      APPROVED SITE TOKENS (use exactly as written — one per query):
        Global generalist  : site:linkedin.com/jobs
                             site:indeed.com
                             site:glassdoor.com
        MENA / Egypt / Gulf: site:wuzzuf.net
                             site:bayt.com
                             site:akhtaboot.com
        Remote-focused     : site:weworkremotely.com
                             site:remoteok.com
                             site:himalayas.app
        Tech-specialist    : site:wellfound.com
                             site:dice.com

      You MAY combine up to two approved boards using OR in one query:
        site:linkedin.com/jobs OR site:wuzzuf.net
        site:indeed.com OR site:bayt.com

      You may NOT use any domain not on the list above.
      You may NOT omit the site: clause entirely.

    RULE 3 — QUERY FORMAT (mandatory structure — STACK TOKENS ONLY):
      "<TECH_1> <TECH_2> <role> <modifier> <site:TOKEN>"
      where modifier is one of: jobs, internship, intern, trainee

      The query argument must contain ONLY: technology/stack keywords, the
      role title, the modifier word, and the site: clause. It must NEVER
      contain operational/conversational words such as "Go", "go ahead",
      "search", "find", "please", "now", "ok", "run", "execute" used as
      filler — these are not part of the stack and corrupt the search
      engine's interpretation of what technology you are looking for.
      (Note: "Go" as the actual Golang programming language is fine —
      e.g. "Go Gin backend developer jobs site:dice.com" — the prohibition
      is on "Go" used as a leaked operational verb in front of an UNRELATED
      stack, e.g. "Go ASP.NET developer..." which corrupts a .NET search.)

      Required examples (copy this exact structure):
        ✓ "Python Django Back-End Developer jobs site:wuzzuf.net OR site:bayt.com"
        ✓ "React Node.js Software Engineer internship site:linkedin.com/jobs"
        ✓ "Flutter Android developer jobs hiring now site:indeed.com"
        ✓ "C# ASP.NET Backend Developer jobs site:glassdoor.com OR site:wellfound.com"
        ✓ "pandas scikit-learn Data Analyst intern site:linkedin.com/jobs OR site:wuzzuf.net"

      Forbidden examples:
        ✗ "Back-End Developer jobs Cairo"            ← no tech, no site:
        ✗ "Python developer jobs"                    ← no site:
        ✗ "Python Django developer site:quora.com"   ← non-approved domain
        ✗ "Search for Python jobs on LinkedIn"       ← conversational filler
        ✗ "Go ASP.NET Backend Developer jobs site:linkedin.com/jobs"
              ← leaked operational "Go" corrupts a C#/.NET search into a
                Golang search. Write: "C# ASP.NET Backend Developer jobs
                site:linkedin.com/jobs"

    RULE 4 — TWO QUERIES MINIMUM BEFORE FINAL JSON:
      Call A: full-time/senior roles. Call B: internship/trainee roles.
      Even if Call A returns a full set of results, Call B is MANDATORY.
      Use a DIFFERENT approved board in Call B than you used in Call A.

    RULE 5 — NEVER INVENT RESULTS:
      Use ONLY results returned by this tool. Copy URLs character-for-character.

    RULE 6 — RECENCY IS AUTOMATIC:
      Do NOT add a specific year. The tool enforces a rolling ~60-day recency
      window via Tavily's start_date parameter.

    Returns a numbered list of job postings from approved boards only.
    Forum posts, Q&A results, blog articles, and non-approved domains are
    excluded at the Tavily API level before results reach this function.
    Zombie listings are detected by snippet content and dropped silently.
    Listings are never truncated mid-entry — every listing shown is complete.
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

    # ── FIX B + FIX D: Sanitise query before anything else ──────────────────
    clean_query = _sanitise_query(query)

    # ── FIX 1 (preserved): Technology-signal guard ──────────────────────────
    tech_warning = ""
    if not _has_tech_signal(clean_query):
        tech_warning = (
            "\n\n⚠️  QUERY QUALITY WARNING — NO TECHNOLOGY KEYWORD DETECTED ⚠️\n"
            f"Your query '{clean_query}' contains no concrete technology term.\n"
            "This violates RULE 1. Results below are likely irrelevant.\n"
            "YOU MUST retry with the candidate's primary technology stack included.\n"
            "Example: 'Python Django developer jobs site:wuzzuf.net'\n"
            "Do not emit final JSON until you have run a technology-enriched query.\n"
            "────────────────────────────────────────────────────────────────────\n"
        )
        logger.warning(
            "tavily_job_search: tech-free query after sanitisation → %r", clean_query
        )

    # ── FIX B: Enforce approved-board site: clause ──────────────────────────
    board_warning = ""
    has_site_clause = bool(re.search(r"\bsite:", clean_query, re.IGNORECASE))
    if not has_site_clause:
        board_warning = (
            "\n\n⚠️  OPEN-WEB QUERY WARNING — NO site: CLAUSE DETECTED ⚠️\n"
            f"Your query '{clean_query}' has no site: restriction.\n"
            "Open-web queries are FORBIDDEN (RULE 2). They cause context pollution.\n"
            "YOU MUST retry with an approved board site: token, e.g.:\n"
            "  site:linkedin.com/jobs\n"
            "  site:wuzzuf.net OR site:bayt.com\n"
            "  site:indeed.com\n"
            "────────────────────────────────────────────────────────────────────\n"
        )
        logger.warning(
            "tavily_job_search: no site: clause after sanitisation → %r", clean_query
        )

    # ── FIX A: Use Tavily exclude_domains instead of query-string exclusion ─
    exclude_domains = _build_tavily_exclude_domains()
    recency_cutoff  = _compute_recency_cutoff()

    # Minimal positive emphasis — only the "apply now" OR "hiring now" hint
    # is kept, since it genuinely biases toward active postings without
    # introducing the semantic leakage that the negative phrases caused.
    enriched_query = f'{clean_query} ("apply now" OR "hiring now" OR "job description")'

    logger.info(
        "Tavily search | query: %r | start_date: %s | max_results: %d | "
        "exclude_domains count: %d",
        enriched_query, recency_cutoff,
        settings.tavily_max_results, len(exclude_domains),
    )

    try:
        response = client.search(
            query=enriched_query,
            search_depth="advanced",
            max_results=settings.tavily_max_results,
            start_date=recency_cutoff,
            exclude_domains=exclude_domains,
            include_answer=False,
            include_raw_content=False,
        )
    except Exception as exc:
        logger.error("Tavily advanced search failed: %s", exc)
        # Fallback: basic depth, same exclusions
        try:
            response = client.search(
                query=clean_query,
                search_depth="basic",
                max_results=settings.tavily_max_results,
                start_date=recency_cutoff,
                exclude_domains=exclude_domains,
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as exc2:
            return f"Search error: {exc2}"

    results: List[dict] = response.get("results", [])

    if not results:
        return (
            tech_warning + board_warning +
            f"No job postings found for: {clean_query!r}.\n"
            "Suggestion: Try a different approved board, e.g. "
            "site:indeed.com or site:wuzzuf.net OR site:bayt.com.\n"
            "REMINDER: You MUST still run the internship query (Step 2) "
            "before emitting final JSON."
        )

    # ── Pre-filter loop: URL validity + zombie snippet detection ────────────
    # FIX C: this loop already walks the ENTIRE results list regardless of
    # how many items are dropped along the way — it does not stop early.
    # The volume bottleneck was never here; it was in the post-hoc _trim()
    # call slicing the assembled string. That is fixed below via
    # _join_results_within_budget(), which is budget-aware and listing-
    # boundary-safe instead of a blind character cut.
    usable_results:      List[dict] = []
    dropped_blacklist:   int        = 0
    dropped_pollution:   int        = 0
    dropped_bad_url:     int        = 0
    dropped_zombie_snip: int        = 0

    for r in results:
        url     = r.get("url", "")
        snippet = r.get("content", r.get("snippet", ""))

        if _is_blacklisted_domain(url):
            dropped_blacklist += 1
            logger.debug("Pre-filter [blacklist]    → %r", url)
            continue

        if _is_content_pollution_domain(url):
            dropped_pollution += 1
            logger.debug("Pre-filter [pollution]    → %r", url)
            continue

        if not _is_usable_url(url):
            dropped_bad_url += 1
            logger.debug("Pre-filter [bad-url]      → %r", url)
            continue

        # FIX A: snippet-level zombie detection now happens here, not in the
        # query string. This correctly targets result *content* rather than
        # search index semantics.
        if _snippet_is_zombie(snippet):
            dropped_zombie_snip += 1
            logger.debug("Pre-filter [zombie-snip]  → %r", url)
            continue

        usable_results.append(r)

    total_dropped = dropped_blacklist + dropped_pollution + dropped_bad_url + dropped_zombie_snip
    if total_dropped:
        logger.info(
            "Pre-filter: kept %d / %d | blacklist=%d pollution=%d bad-url=%d zombie-snippet=%d",
            len(usable_results), len(results),
            dropped_blacklist, dropped_pollution, dropped_bad_url, dropped_zombie_snip,
        )

    if not usable_results:
        return (
            tech_warning + board_warning +
            f"Search returned {len(results)} results for {clean_query!r}, but "
            "none passed the quality filter (all were from blocked domains, "
            "had unusable URLs, or their snippets indicated the listing is closed).\n"
            "Try a different approved board or broaden the role title slightly.\n"
            "REMINDER: You MUST still run the internship query (Step 2) "
            "before emitting final JSON."
        )

    # ── FIX 3 (preserved): Pipeline-status banner ───────────────────────────
    pipeline_reminder = (
        "╔══════════════════════════════════════════════════════════════╗\n"
        "║  SEARCH PIPELINE STATUS — ACTION REQUIRED BEFORE FINAL JSON  ║\n"
        "╠══════════════════════════════════════════════════════════════╣\n"
        "║  These are PARTIAL results from ONE query.                   ║\n"
        "║  ✗ Do NOT emit the final JSON yet.                          ║\n"
        "║  ✓ You MUST still run the INTERNSHIP/TRAINEE variant query.  ║\n"
        "║  ✓ Use a DIFFERENT approved board for the internship query.   ║\n"
        "║  ✓ Only emit final JSON after ALL required queries are done.  ║\n"
        "╚══════════════════════════════════════════════════════════════╝\n"
    )

    intro_block = (
        f"Job Search Results for: {clean_query!r}\n"
        f"(Found {len(usable_results)} approved-board listings after filtering, "
        f"posted on or after {recency_cutoff})\n"
    )

    header_blocks = [pipeline_reminder, tech_warning, board_warning, intro_block]

    result_blocks = [
        _format_result(i, result) for i, result in enumerate(usable_results, start=1)
    ]

    footer_block = (
        "\n─── END OF THIS QUERY'S RESULTS ───\n"
        "NEXT STEP: Run the internship/trainee query (Step 2) on a DIFFERENT\n"
        "approved board before producing the final JSON answer.\n"
    )

    # FIX C: budget-aware join — keeps complete listings, never slices one
    # in half, and reports how many of the filtered listings actually made
    # it into the LLM context so volume loss (if any) is visible in logs.
    full_output, included_count = _join_results_within_budget(
        header_blocks, result_blocks, footer_block, max_chars=MAX_RESULT_CHARS
    )

    if included_count < len(usable_results):
        logger.warning(
            "tavily_job_search: budget allowed only %d / %d filtered listings "
            "into the output (MAX_RESULT_CHARS=%d). Consider raising the "
            "ceiling further if this recurs often.",
            included_count, len(usable_results), MAX_RESULT_CHARS,
        )
    else:
        logger.info(
            "tavily_job_search: all %d filtered listings included in output "
            "(%d chars, budget %d).",
            included_count, len(full_output), MAX_RESULT_CHARS,
        )

    return full_output


# ---------------------------------------------------------------------------
# Tool 2: Web Page Scraper
# ---------------------------------------------------------------------------

@tool
def scrape_job_page(url: str) -> str:
    """
    Fetch the text content of a real job posting URL for additional detail.

    Use this tool ONLY for specific details (salary, skills, requirements)
    not present in the search snippet.
    ONLY call this with URLs returned verbatim by tavily_job_search.
    NEVER scrape homepages, search-results pages, blacklisted domains,
    content-pollution domains (forums, Q&A, blogs), or any URL not from
    an approved job board.
    """
    import html
    import urllib.error
    import urllib.request

    logger.info("Scraping URL: %s", url[:200])

    if not url.startswith(("http://", "https://")):
        return "Invalid URL: must start with http:// or https://"

    if _is_blacklisted_domain(url):
        return (
            f"Refused: {url!r} is a blacklisted CSR zombie-aggregator domain. "
            "Do not use any listing from this domain."
        )

    if _is_content_pollution_domain(url):
        return (
            f"Refused: {url!r} is a content-pollution domain (forum, Q&A, blog, "
            "or social network). Only scrape URLs from approved job boards."
        )

    if not _is_usable_url(url):
        return (
            f"Refused: {url!r} does not look like a direct job posting. "
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
            charset   = response.headers.get_content_charset() or "utf-8"
            raw_html  = raw_bytes.decode(charset, errors="replace")
    except urllib.error.URLError as exc:
        return f"Failed to fetch URL: {exc}"
    except Exception as exc:
        return f"Unexpected error: {exc}"

    text = re.sub(r"<script[^>]*>.*?</script>", " ", raw_html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>",   " ", text,    flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    return _trim(text, PAGE_CHARS)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_tools() -> list:
    return [tavily_job_search, scrape_job_page]