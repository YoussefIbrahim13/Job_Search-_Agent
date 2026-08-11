"""
backend/agents/tools.py
=======================
LangChain tool definitions for the Recruitment AI Agent.

WHERE THE FILTERS WENT
---------------------------------------------------------------------------
The filter chain described below under FIX A / FIX N / FIX O — domain
blacklists, board path gates, staleness and category detection — now lives in
`backend/sources/filters.py`. It answers "is this web page an individual,
currently-open job posting?", a question only a provider returning web pages
ever raises, so it belongs with the source layer alongside its other consumer,
the Tavily adapter.

The move was verbatim, verified by diffing every function's output over a
4,214-case corpus before and after. Everything is re-exported here unchanged,
so the notes below still describe the behaviour accurately — only the file has
changed. Read `sources/filters.py` for the current implementation.

CHANGES IN THIS REVISION
---------------------------------------------------------------------------
FIX N — Positive-Assertion Path Gating (replaces domain-only allowlist):
  For our core approved boards (Wuzzuf, Bayt, LinkedIn, Glassdoor,
  Akhtaboot, WeWorkRemotely, Himalayas, Wellfound, Dice), the URL path is
  validated against each board's canonical job-listing URL structure BEFORE
  the result is admitted to the pipeline.

  The old model trusted any URL whose netloc matched an approved domain —
  which silently admitted templates, blog/advice articles, career-hub pages,
  and company profile pages that happen to contain tech keywords. The new
  model makes a positive assertion: this URL path looks like a discrete job
  posting on THIS board. If it does not match, it is dropped with a
  'path-gate' reason in the pre-filter log, regardless of domain trust.

  FIX N — Amendment (live-run corrections from path-gate debug logs):
    LinkedIn: the original pattern only matched /jobs/view/<numeric-id>
      (pure numeric). LinkedIn also serves listings as /jobs/view/<slug>-<id>
      and from country-coded subdomains (pk., eg., bg., etc.). Updated to
      assert that the path ends with a 7+ digit numeric ID, covering both
      forms while excluding /jobs/search/ and /jobs/collections/.
    Indeed: removed from _BOARD_PATH_PATTERNS. Tavily returns Indeed's
      /q-<query>-jobs.html search-results pages exclusively — real listing
      paths (/viewjob, /rc/clk) are almost never surfaced. Indeed junk is
      now caught by a new /q-*-jobs.html clause in _BAD_URL_PATTERNS instead.

  Architecture:
    _BOARD_PATH_PATTERNS: dict[str, re.Pattern]
      Maps normalised netloc (without www.) to a compiled regex matching only
      canonical listing path structures for that board. Boards not in the dict
      (remoteok.com, indeed.com) fall through to _is_usable_url() heuristics.
    _passes_path_gate(url): bool
      Returns True if the URL (a) matches the board's canonical path, or
      (b) belongs to no board in the dict (pass-through). Returns False only
      for board URLs whose paths fail the positive assertion.

FIX O — Content-Layer Staleness / Zombie Detection:
  The previous _snippet_is_zombie() only matched explicit closed/filled/
  expired declarations. It was blind to SEO-spoofed zombie postings that
  carry a human-readable "Posted 5 years ago" / "منذ 4 سنوات" age badge in
  their snippet or title — because those pages are genuinely re-crawled daily
  (a "Similar Jobs" widget changes), so Tavily's start_date filter passes them.

  The fix reads the first-party age metadata FROM THE SNIPPET AND TITLE TEXT
  directly. Two regexes:
    _STALE_AGE_SNIPPET_RE — matches English/Arabic age strings that exceed
      the configured staleness threshold (STALENESS_MONTHS_THRESHOLD = 3):
      "3 months ago", "1 year ago", "2 years ago", "منذ سنة", "منذ 3 سنوات",
      "منذ 6 أشهر", etc.  Only ages ABOVE the threshold are matched; the
      pattern is parameterised so the threshold is a single constant.
    _snippet_is_stale(snippet, title): bool
      Runs both the existing zombie-declaration check AND the new age-metadata
      check. Either signal is sufficient to reject.

  STALENESS_MONTHS_THRESHOLD = 3:
    Any posting whose snippet or title declares it was posted >= 3 months ago
    is rejected. This is intentionally more conservative than RECENCY_WINDOW_DAYS
    (which governs Tavily's crawl-date filter) because crawl dates are
    unreliable for the SEO-spoofing case.

FIX Q++ — Narrow Closed/Age-Badge Probe (replaces FIX Q+ Tier 1 fail-open):
  FIX Q+'s Tier 1 gate skipped the live probe ENTIRELY for wuzzuf.net
  /jobs/p/ and linkedin.com /jobs/view/ URLs, fail-opening (treating them as
  "not stale") because the full-page sidebar-truncated staleness check had
  a confirmed 100% false-positive rate on those two domains (company
  "About Us" blurbs on Wuzzuf, repost badges on LinkedIn's hero section).

  Production screenshots subsequently showed the cost of that trade-off:
  genuinely CLOSED / multi-month-old listings on both domains were passing
  straight through, because Tavily's snippet text never carried the
  closure/age badge (it carried "Similar Jobs" sidebar bleed instead), and
  the Tier 1 fail-open meant nothing else ever looked at the live page.

  Fix — replace blanket fail-open with a NARROW, POSITIVE-ASSERTION probe:
    - Fetch only the first _HEAD_PROBE_BYTES (8KB) of raw HTML — the
      closure badge / posted-X-ago string both render immediately under
      the job title on both boards, well before any sidebar/footer markup.
    - Check that narrow head-slice for ONLY the literal, unambiguous
      closure markers ("Closed" badge on Wuzzuf, "no longer accepting
      applications" / Arabic equivalents on LinkedIn) — strings that do
      NOT appear in company bios or repost badges, so the original
      false-positive sources are not reintroduced.
    - Also run the existing age-string regexes (via _snippet_is_stale)
      against that SAME narrow head-slice only, catching "posted 3 months
      ago" / "منذ سنة" badges near the title without ever reading the
      sidebar/footer region where the bio/repost noise lives.

  Tier 2 (all other boards — glassdoor, akhtaboot, remoteok, etc.) is
  UNCHANGED: full 64KB fetch + dual boundary truncation (sidebar keyword +
  structural HTML tag), as in FIX Q+.

  New / changed helpers:
    _HEAD_PROBE_BYTES            — narrow head-slice fetch size (8KB)
    _WUZZUF_CLOSED_BADGE_RE      — literal "Closed" badge pattern
    _LINKEDIN_CLOSED_BADGE_RE    — literal closure-declaration patterns
    _has_explicit_closed_badge() — positive-assertion badge check
    _is_canonical_listing_url()  — now means "use the narrow head-probe
                                    strategy", not "skip the probe"
    _verify_live_url_is_stale()  — two strategies: narrow head-probe for
                                    wuzzuf/linkedin, unchanged Tier 2 probe
                                    for everything else

All prior FIX A through FIX Q behaviour is preserved unmodified. New logic
is additive or replaces only the live-probe path.
"""

from __future__ import annotations

import logging
import re
import html
import threading
import urllib.error
import urllib.request
import concurrent.futures

from datetime import datetime, timedelta, timezone
from typing import Dict, List
from urllib.parse import urlparse

from langchain_core.tools import tool

from backend.agents.prompts import (
    APPROVED_SEARCH_BOARDS as _APPROVED_SEARCH_BOARDS,
    TAVILY_TOOL_DESCRIPTION,
)
from backend.core.config import get_settings
from backend.core.resilience import tavily_retry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-safety constants
# ---------------------------------------------------------------------------

# Character budget for the listing block handed to the model per tool call.
#
# At 3,000 this was the real bottleneck on recall, not a token-safety measure:
# with SNIPPET_CHARS at 600 it admitted roughly 5 listings, and live runs were
# observed discarding surviving results ("budget allowed only 3 / 6 filtered
# listings"). Those listings had already passed every filter — they were dropped
# purely for want of room.
#
# 9,000 chars is ~2,250 tokens per tool result. With two parallel calls and
# max_tokens raised to 4,096, that sits comfortably inside the 70B model's
# context while roughly tripling how many vetted listings it gets to score.
MAX_RESULT_CHARS: int = 9_000
SNIPPET_CHARS: int    = 600
PAGE_CHARS: int       = 1_800


# ---------------------------------------------------------------------------
# Request-scoped filter statistics
# ---------------------------------------------------------------------------
#
# Why this exists: a search can legitimately return nothing, and "0 FOUND" with
# no explanation is indistinguishable from a broken app. A real run for
# ".net / cairo" examined 10 listings and rejected all of them — 5 category
# pages and 5 verified-closed postings — yet the user was told only that the
# agent "could not produce a structured response", which was both unhelpful and
# untrue. These counters let the response say what actually happened.
#
# Thread-local rather than global: each HTTP request runs the whole graph in its
# own thread via asyncio.to_thread, so concurrent searches must not share
# counters. The tool body increments these on the request thread; the live-probe
# worker pool only returns values back to it.

_stats = threading.local()


def reset_filter_stats() -> None:
    """Begin a new request's tally. Call once before invoking the graph."""
    _stats.data = {
        "examined": 0, "kept": 0, "category": 0,
        "closed": 0, "path_gate": 0, "blocked": 0,
    }


def record_filter_stats(**counts: int) -> None:
    """Accumulate one tool call's drop counts into the current request."""
    data = getattr(_stats, "data", None)
    if data is None:
        return  # not inside a tracked request (e.g. a direct unit-test call)
    for key, value in counts.items():
        if key in data:
            data[key] += value


def get_filter_stats() -> Dict[str, int]:
    """Return the current request's tally (empty when untracked)."""
    return dict(getattr(_stats, "data", {}) or {})


# ---------------------------------------------------------------------------
# FIX Q++ amendment — Live-probe concurrency cap
# ---------------------------------------------------------------------------
# Diagnostic logging (see _verify_live_url_is_stale) confirmed that firing
# one concurrent request per result (previously max_workers=len(usable_
# results), i.e. up to 8) caused multiple Wuzzuf probes to time out under
# contention — 4/8 in one observed run — even though each URL fetches fine
# in isolation. Capping concurrency reduces per-host request pressure;
# combined with the raised per-request timeout in _verify_live_url_is_stale,
# this gives the live probe enough headroom to actually complete instead of
# silently fail-opening every timed-out URL as "not stale."
_LIVE_PROBE_MAX_WORKERS: int = 4


# ---------------------------------------------------------------------------
# Recency configuration
# ---------------------------------------------------------------------------

# Absolute crawl-date cutoff (YYYY-MM-DD) handed to Tavily's `start_date`.
# 30 days == "posted within the past month". Kept in lock-step with the
# relative `time_range` window (settings.tavily_time_range, default "month")
# so the two freshness filters reinforce rather than contradict each other.
RECENCY_WINDOW_DAYS: int = 30


def _compute_recency_cutoff(days_back: int = RECENCY_WINDOW_DAYS) -> str:
    """Return today's date minus ``days_back`` days as YYYY-MM-DD."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return cutoff.strftime("%Y-%m-%d")


def _current_year() -> int:
    """Current UTC year — injected into every query as a recency signal so
    the search engine biases toward freshly-posted listings and away from
    multi-year-old archived pages that share the same URL structure."""
    return datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# Search-result filter chain
# ---------------------------------------------------------------------------
#
# These live in backend/sources/filters.py now. They answer "is this web page an
# individual, currently-open job posting?", which is a question only a provider
# returning web pages ever raises — so they belong with the source layer, and
# the Tavily adapter is their only other consumer.
#
# Re-exported here unchanged so the rest of this module, and anything importing
# from it, is unaffected by the move.
from backend.sources.filters import (  # noqa: F401
    STALENESS_MONTHS_THRESHOLD,
    _arabic_digit_to_int,
    _BAD_URL_PATTERNS,
    _BLACKLISTED_DOMAINS,
    _BOARD_PATH_PATTERNS,
    _build_tavily_exclude_domains,
    _CATEGORY_PAGE_TITLE_RE,
    _CATEGORY_PAGE_URL_RE,
    _CONTENT_POLLUTION_DOMAINS,
    _is_blacklisted_domain,
    _is_category_page,
    _is_content_pollution_domain,
    _is_usable_url,
    _normalise_netloc,
    _passes_path_gate,
    _snippet_is_stale,
    _snippet_is_zombie,
    _VALID_URL_RE,
    _ZOMBIE_DECLARATION_RE,
)
































# ---------------------------------------------------------------------------
# FIX B — Approved search board registry
# ---------------------------------------------------------------------------

# The registry itself now lives in backend/agents/prompts.py, alongside the
# board grouping used to render it into both the tool description and the agent
# system prompt. Keeping the data next to the text that describes it is what
# stops a newly-added board from being filtered-for but never searched-for.
#
# Re-exported here so existing `from backend.agents.tools import
# APPROVED_SEARCH_BOARDS` imports keep working.
APPROVED_SEARCH_BOARDS = _APPROVED_SEARCH_BOARDS
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
# FIX B — Query sanitiser
# ---------------------------------------------------------------------------

_QUERY_FILLER_RE = re.compile(
    r"^\s*("
    r"please\s+(search\s+(for|about)|find|look\s+up|look\s+for)|"
    r"search\s+(for|about|the\s+web\s+for|for\s+jobs?\s+(related\s+to|about))|"
    r"find\s+(me\s+)?(jobs?|listings?|openings?|postings?|roles?|positions?)"
    r"(\s+(for|related\s+to|about|in|on))?|"
    r"find\s+(me\s+)?|"
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

_QUERY_TRAILING_RE = re.compile(r'[\s"\'.,;:!?]+$')
MAX_QUERY_CHARS: int = 300


# ---------------------------------------------------------------------------
# FIX D — Stray operational/stack-noise token stripper
# ---------------------------------------------------------------------------

_LEADING_GO_TOKEN_RE = re.compile(r"^\s*go\b[\s,:-]*", re.IGNORECASE)

_LEADING_OPERATIONAL_TOKENS_RE = re.compile(
    r"^\s*(ok|okay|now|please|go\s+ahead\s+and|execute|run|fetch|lookup)\b[\s,:-]*",
    re.IGNORECASE,
)


def _next_token(text: str) -> str:
    match = re.match(r"\s*([^\s]+)", text)
    return match.group(1).lower() if match else ""


def _strip_stray_stack_noise(query: str) -> str:
    q = query.strip()
    if not q:
        return q

    lower_full = q.lower()

    new_q = _LEADING_OPERATIONAL_TOKENS_RE.sub("", q).strip()
    if new_q != q:
        logger.info(
            "_strip_stray_stack_noise: removed leading operational token | "
            "original=%r | cleaned=%r", q[:120], new_q[:120],
        )
        q          = new_q
        lower_full = q.lower()

    if "golang" in lower_full:
        return q

    if _LEADING_GO_TOKEN_RE.match(q):
        remainder = _LEADING_GO_TOKEN_RE.sub("", q)
        following_token       = _next_token(remainder)
        following_token_clean = following_token.rstrip(".,;:!?")

        is_known_tech_term = any(
            following_token_clean == term
            or following_token_clean in term
            or term in following_token_clean
            for term in _TECH_SIGNAL_TERMS
        )
        is_different_tech = (
            is_known_tech_term
            and following_token_clean not in ("go", "golang")
        )

        if is_different_tech:
            logger.info(
                "_strip_stray_stack_noise: removed leaked leading 'Go' token "
                "before unrelated stack token %r | original=%r | cleaned=%r",
                following_token_clean, q[:120], remainder.strip()[:120],
            )
            q = remainder.strip()

    return q


def _sanitise_query(raw_query: str) -> str:
    original = raw_query

    q = raw_query.strip()
    q = _QUERY_FILLER_RE.sub("", q)
    q = _strip_stray_stack_noise(q)
    q = _QUERY_TRAILING_RE.sub("", q)
    q = re.sub(r"\s+", " ", q).strip()
    q = q[:MAX_QUERY_CHARS]

    if not q:
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
# FIX Q+ — Tier 2 live staleness probe helpers (unchanged by FIX Q++)
# ---------------------------------------------------------------------------

# Tier 2 pattern: raw-HTML structural tags that delimit secondary content.
# Searched in the raw HTML (before tag-stripping) so we can obtain a byte
# position and map it proportionally to the stripped-text position.
_STRUCTURAL_BOUNDARY_RAW_RE = re.compile(
    r"<(?:aside|footer|nav)\b",
    re.IGNORECASE,
)

# Tier 2 pattern: keyword boundary in stripped text (FIX Q behaviour,
# preserved and still applied as the primary boundary check).
_SIDEBAR_BOUNDARY_RE = re.compile(
    r"people also viewed|similar jobs|related jobs|"
    r"more jobs like this|jobs you may like|you might also like|"
    r"وظائف مشابهة|وظائف أخرى|قد تهمك أيضا|وظائف ذات صلة",
    re.IGNORECASE,
)


def _truncate_at_sidebar_boundary(text: str, raw_html: str = "") -> str:
    """
    FIX Q+ Tier 2: truncate stripped text at the EARLIER of:
      (a) a sidebar/widget keyword boundary in the stripped text, or
      (b) a structural HTML tag boundary (<aside>, <footer>, <nav>)
          detected in the raw HTML and mapped proportionally to the
          stripped text via character-count ratio.

    Falls back to returning the full text if neither boundary is found.
    The optional raw_html argument is new in FIX Q+; callers that pass
    only text (FIX Q call sites) continue to work unchanged.
    """
    cut = len(text)

    # (a) keyword boundary in stripped text
    m = _SIDEBAR_BOUNDARY_RE.search(text)
    if m:
        cut = min(cut, m.start())

    # (b) structural HTML boundary → proportional mapping into stripped text
    if raw_html:
        sm = _STRUCTURAL_BOUNDARY_RAW_RE.search(raw_html)
        if sm:
            raw_pos  = sm.start()
            raw_len  = len(raw_html)
            text_pos = int(raw_pos / max(raw_len, 1) * len(text))
            cut = min(cut, text_pos)

    return text[:cut] if cut < len(text) else text


# ---------------------------------------------------------------------------
# FIX Q++ — Narrow Closed/Age-Badge Probe (replaces FIX Q+ Tier 1 fail-open)
# ---------------------------------------------------------------------------
#
# See module docstring for full rationale. Summary: FIX Q+ skipped the live
# probe entirely for wuzzuf.net /jobs/p/ and linkedin.com /jobs/view/ URLs
# to avoid false positives from company-bio bleed (Wuzzuf) and repost
# badges (LinkedIn). That blanket fail-open let genuinely closed/stale
# listings on both domains through undetected, because Tavily's snippet
# text never carried the closure/age badge for these URLs.
#
# FIX Q++ replaces the fail-open with a NARROW, POSITIVE-ASSERTION probe:
# fetch only the first _HEAD_PROBE_BYTES of raw HTML (the badge/title block
# renders early on both boards, well before sidebar/footer markup), then
# check ONLY for the literal closure marker for that board, plus the
# existing age-string regexes — scoped to that same narrow slice so the
# bio/repost-badge content further down the page is never read.

# The closure badge and the posted-X-ago string both render immediately
# under the job title on Wuzzuf and in LinkedIn's hero section — i.e. very
# early in the raw HTML. 8KB is comfortably past that block on both boards
# (verified against the SURE International Technology / Halr Tech Group
# listings that exposed this gap) while staying well short of where
# "Similar Jobs" / sidebar / footer markup begins.
_HEAD_PROBE_BYTES: int = 8_000

# LinkedIn needs a much larger window. Its closure badge sits at roughly byte
# 31,500-34,700 of the guest job page, far past the 8KB budget above — so the
# probe fetched the page, found nothing, and reported the listing as fresh.
# Measured across 8 live listings: closed ones carried the badge at ~31.5-31.8KB.
# That gap is why two year-old "No longer accepting applications" postings were
# returned to the user at a 90% match score.
#
# Reading further is safe here because the LinkedIn check is a positive
# assertion on a structural CSS class (see _LINKEDIN_CLOSED_BADGE_RE), not
# prose matching — so sidebar/footer bleed cannot trigger it.
_LINKEDIN_HEAD_PROBE_BYTES: int = 65_536

# Literal closure-badge patterns (positive assertion, not prose). These
# match ONLY the specific, unambiguous UI strings each board renders for a
# closed/expired listing — deliberately narrow so they cannot match inside
# a company "About Us" blurb or a LinkedIn repost badge (the two sources of
# false positives that FIX Q+ was built to avoid).
_WUZZUF_CLOSED_BADGE_RE = re.compile(
    r">\s*Closed\s*<",        # the literal "Closed" pill/badge element
    re.IGNORECASE,
)

# The FIRST alternative is the load-bearing one: LinkedIn renders a
#     <figcaption class="closed-job__flavor--closed">
# element only on closed listings. Verified across 8 live listings — present on
# every closed one, absent on every open one, and inversely correlated with the
# apply widget in all 8.
#
# Matching the class rather than the prose matters twice over: it cannot be
# tripped by a job description that happens to discuss applications closing,
# and it is locale-independent, so it works on Arabic-language pages served to
# MENA users without needing a translation for each phrase. The localized
# prose variants are kept as a fallback in case the markup changes.
_LINKEDIN_CLOSED_BADGE_RE = re.compile(
    r"closed-job__flavor--closed|"
    r"no longer accepting applications|"
    r"لم نعد نقبل استمارات|"
    r"تم إغلاق هذا الإعلان|"
    r"تم غلق هذا العرض",
    re.IGNORECASE,
)


def _has_explicit_closed_badge(raw_html_head: str, netloc: str) -> bool:
    """
    Positive-assertion closure check against a narrow head-of-page slice.

    Returns True only when the LITERAL closure marker for that specific
    board is found — never on generic "closed"-adjacent prose, which is
    what made the old full-page snippet-staleness check unsafe to run
    unscoped on these two domains.
    """
    if "wuzzuf.net" in netloc:
        return bool(_WUZZUF_CLOSED_BADGE_RE.search(raw_html_head))
    if "linkedin.com" in netloc:
        return bool(_LINKEDIN_CLOSED_BADGE_RE.search(raw_html_head))
    return False


# Canonical listing URL shapes for wuzzuf and linkedin. When
# _is_canonical_listing_url() returns True, _verify_live_url_is_stale()
# uses the NARROW head-probe strategy (FIX Q++) instead of the full Tier 2
# dual-boundary probe.
_WUZZUF_LISTING_RE = re.compile(r"^/jobs/p/", re.IGNORECASE)
_LINKEDIN_LISTING_RE = re.compile(
    r"^/jobs/view/(?:[^/]+-)?(\d{7,})/?$",
    re.IGNORECASE,
)


def _is_canonical_listing_url(url: str) -> bool:
    """
    FIX Q++ probe-strategy gate.

    True for URL shapes that should use the NARROW head-probe strategy
    instead of the full Tier 2 dual-boundary probe:
      - wuzzuf.net /jobs/p/<slug>
      - *.linkedin.com /jobs/view/<id>

    These are the two domains where full-page sidebar-truncated staleness
    checking produced a 100% false-positive rate (company bio bleed on
    Wuzzuf, repost badge in LinkedIn's hero section). They still get a
    live check under FIX Q++ — just a narrower, badge-only one — rather
    than being fail-opened with no check at all (as under FIX Q+).
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = parsed.path or "/"
    except Exception:
        return False

    if netloc == "wuzzuf.net" or netloc.endswith(".wuzzuf.net"):
        return bool(_WUZZUF_LISTING_RE.match(path))

    if netloc == "linkedin.com" or netloc.endswith(".linkedin.com"):
        return bool(_LINKEDIN_LISTING_RE.match(path))

    return False


# ---------------------------------------------------------------------------
# FIX Q++ — Phase 2 Live Shallow Probing (replaces FIX Q+ probe)
# ---------------------------------------------------------------------------

def _verify_live_url_is_stale(url: str, timeout: float = 6.0) -> bool:
    """
    FIX Q++ probe — two DIFFERENT strategies depending on board, both of
    which now actually inspect the live page (no blanket fail-open):

    Tier 1 (wuzzuf /jobs/p/ and linkedin /jobs/view/):
        Fetch only the first _HEAD_PROBE_BYTES of raw HTML. Check that
        narrow head-slice for:
          (a) a literal closure badge specific to that board, or
          (b) a posted-X-ago string via the existing _snippet_is_stale
              age-regexes, run ONLY against this head-slice.
        Never touches sidebar/footer content, so the original false-
        positive sources (company bio, repost badge further down the
        page) are not reintroduced.

    Tier 2 (all other boards — glassdoor, akhtaboot, remoteok, etc.):
        Unchanged from FIX Q+: fetch first 64KB, apply dual boundary
        truncation (sidebar keyword + structural HTML tag), then run
        _snippet_is_stale on the primary-content window.

    Fails open (returns False) on any network error or timeout, for
    both tiers — unchanged from prior behaviour.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
    except Exception:
        return False

    use_narrow_probe = _is_canonical_listing_url(url)
    if not use_narrow_probe:
        fetch_size = 65_536
    elif "linkedin.com" in netloc:
        # LinkedIn buries its closure badge ~31KB into the page; 8KB misses it.
        fetch_size = _LINKEDIN_HEAD_PROBE_BYTES
    else:
        fetch_size = _HEAD_PROBE_BYTES

    try:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                "Accept": (
                    "text/html,application/xhtml+xml,application/xml;q=0.9,"
                    "image/webp,*/*;q=0.8"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            raw_bytes = response.read(fetch_size)
            charset   = response.headers.get_content_charset() or "utf-8"
            raw_html  = raw_bytes.decode(charset, errors="replace")
    except urllib.error.HTTPError as exc:
        # FIX Q++ diagnostic: an HTTP error (403, 429, etc.) is fundamentally
        # different from "fetched the page and it was clean" — surface it at
        # WARNING so it isn't confused with a verified-fresh result. Body is
        # captured (truncated) since anti-bot blocks often explain themselves
        # there (e.g. Cloudflare challenge, rate-limit message).
        try:
            body_preview = exc.read(300).decode("utf-8", errors="replace")
        except Exception:
            body_preview = "<unreadable>"
        logger.warning(
            "Live probe BLOCKED [http-%s] for %r — fail-open, NOT actually "
            "verified. Response body preview: %r",
            exc.code, url, body_preview,
        )
        return False
    except Exception as exc:
        logger.warning(
            "Live probe FAILED [%s] for %r — fail-open, NOT actually "
            "verified. Error: %s",
            type(exc).__name__, url, exc,
        )
        return False

    if use_narrow_probe:
        # --- Tier 1: narrow head-only badge + age check -------------------
        #
        # The two checks below deliberately run over DIFFERENT windows:
        #
        #   badge → the full fetched slice. It is a positive assertion on a
        #           structural CSS class, so depth cannot make it wrong, and on
        #           LinkedIn the badge only appears ~31KB in.
        #   age   → the first _HEAD_PROBE_BYTES only. These are prose regexes
        #           ("posted 2 years ago"), and past the hero section the page
        #           carries "Similar jobs" / "People also viewed" entries whose
        #           age strings belong to OTHER postings. Running them over the
        #           widened slice would fail this listing for its neighbours'
        #           staleness.
        if _has_explicit_closed_badge(raw_html, netloc):
            logger.info(
                "Live probe [closed-badge] → %r (narrow head-probe, %d bytes)",
                url, len(raw_html),
            )
            return True

        head_text = re.sub(r"<[^>]+>", " ", raw_html[:_HEAD_PROBE_BYTES])
        head_text = html.unescape(head_text)

        if _snippet_is_stale(head_text, title=""):
            logger.info(
                "Live probe [age-badge] → %r (narrow head-probe, %d bytes)",
                url, _HEAD_PROBE_BYTES,
            )
            return True

        logger.debug(
            "Live probe OK (narrow head-probe, no badge found): %r", url
        )
        return False

    # --- Tier 2: unchanged full-page dual-boundary probe ------------------
    text = re.sub(r"<[^>]+>", " ", raw_html)
    text = html.unescape(text)
    text = _truncate_at_sidebar_boundary(text, raw_html)

    return _snippet_is_stale(text, title="")







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
# FIX C — Listing-boundary-safe budget joiner (preserved)
# ---------------------------------------------------------------------------

def _join_results_within_budget(
    header_blocks: List[str],
    result_blocks: List[str],
    footer_block: str,
    max_chars: int = MAX_RESULT_CHARS,
) -> tuple[str, int]:
    header_text = "\n\n".join(b for b in header_blocks if b)
    footer_text = footer_block or ""

    reserved = len(header_text) + len(footer_text) + 8
    budget   = max(max_chars - reserved, 0)

    included: List[str] = []
    used = 0
    for block in result_blocks:
        block_cost = len(block) + 2
        if used + block_cost > budget and included:
            break
        if used + block_cost > budget and not included:
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

# The LLM-facing description is generated in backend/agents/prompts.py from
# the same constants the agent system prompt uses, so the two cannot drift.
# It is passed explicitly rather than written as a docstring because a
# docstring cannot interpolate those shared values.
@tool(description=TAVILY_TOOL_DESCRIPTION)
def tavily_job_search(query: str) -> str:
    """Search approved job boards via Tavily. See prompts.TAVILY_TOOL_DESCRIPTION."""
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

    clean_query = _sanitise_query(query)

    tech_warning = ""
    if not _has_tech_signal(clean_query):
        tech_warning = (
            "\n\n⚠️ NO TECH KEYWORD DETECTED in this query. Retry with the "
            "candidate's primary stack included, e.g. 'Python Django developer "
            "jobs site:wuzzuf.net'.\n"
        )
        logger.warning(
            "tavily_job_search: tech-free query after sanitisation → %r", clean_query
        )

    board_warning = ""
    has_site_clause = bool(re.search(r"\bsite:", clean_query, re.IGNORECASE))
    if not has_site_clause:
        board_warning = (
            "\n\n⚠️ NO site: CLAUSE DETECTED. Retry with an approved board "
            "token, e.g. site:linkedin.com/jobs or site:wuzzuf.net.\n"
        )
        logger.warning(
            "tavily_job_search: no site: clause after sanitisation → %r", clean_query
        )

    exclude_domains = _build_tavily_exclude_domains()
    recency_cutoff  = _compute_recency_cutoff()
    time_range      = settings.tavily_time_range
    current_year    = _current_year()

    # STRICT TIME FILTERING (query layer): append explicit recency tokens so
    # the search engine biases toward freshly-posted roles. The current year
    # plus "hiring now"/"actively hiring" pull active listings to the top and
    # push multi-year-old archived pages down/out, complementing the
    # time_range crawl-date filter applied on the API call.
    #
    # The negative terms -"jobs in" -"browse jobs" -"vacancies in" used to be
    # appended here too. They were removed: the system prompt itself identifies
    # the "jobs in" token collision as a primary cause of empty result sets
    # (a legitimate listing page frequently contains that phrase in nav chrome),
    # and _is_category_page() already rejects true aggregator pages
    # deterministically AFTER retrieval — which is the right place for it, since
    # it inspects the actual URL and title rather than suppressing the query.
    enriched_query = (
        f'{clean_query} '
        f'("hiring now" OR "actively hiring" OR "apply now" OR "{current_year}")'
    )

    # NOTE: Tavily rejects start_date/end_date and time_range together
    # ("When time_range is set, start_date or end_date cannot be set"), so we
    # pass ONLY time_range as the crawl-date filter. recency_cutoff is retained
    # purely for the human-readable "posted within the past ~N days" note.
    logger.info(
        "Tavily search | query: %r | time_range: %s (~since %s) | "
        "max_results: %d | exclude_domains count: %d",
        enriched_query, time_range, recency_cutoff,
        settings.tavily_max_results, len(exclude_domains),
    )

    # Transient failures (429/5xx/timeouts) are retried with backoff before the
    # basic-depth fallback engages, so a momentary blip no longer costs one of
    # the agent's very limited iterations.
    @tavily_retry
    def _search(query: str, depth: str) -> dict:
        return client.search(
            query=query,
            search_depth=depth,
            max_results=settings.tavily_max_results,
            time_range=time_range,
            exclude_domains=exclude_domains,
            include_answer=False,
            include_raw_content=False,
        )

    try:
        response = _search(enriched_query, "advanced")
    except Exception as exc:
        logger.error("Tavily advanced search failed after retries: %s", exc)
        try:
            response = _search(clean_query, "basic")
        except Exception as exc2:
            logger.error("Tavily basic search also failed after retries: %s", exc2)
            return f"Search error: {exc2}"

    results: List[dict] = response.get("results", [])

    if not results:
        return (
            tech_warning + board_warning +
            f"No job postings found for: {clean_query!r}.\n"
            "Try a different approved board, e.g. site:indeed.com or "
            "site:wuzzuf.net OR site:bayt.com."
        )

    # ── Pre-filter loop ──────────────────────────────────────────────────
    usable_results:    List[dict] = []
    dropped_blacklist: int        = 0
    dropped_pollution: int        = 0
    dropped_category:  int        = 0
    dropped_path_gate: int        = 0
    dropped_bad_url:   int        = 0
    dropped_stale:     int        = 0

    for r in results:
        url     = r.get("url", "")
        title   = r.get("title", "")
        snippet = r.get("content", r.get("snippet", ""))

        if _is_blacklisted_domain(url):
            dropped_blacklist += 1
            logger.debug("Pre-filter [blacklist]    → %r", url)
            continue

        if _is_content_pollution_domain(url):
            dropped_pollution += 1
            logger.debug("Pre-filter [pollution]    → %r", url)
            continue

        if _is_category_page(title, url):
            dropped_category += 1
            logger.debug("Pre-filter [category]     → title=%r url=%r", title, url)
            continue

        if not _passes_path_gate(url):
            dropped_path_gate += 1
            logger.info("Pre-filter [path-gate]    → title=%r url=%r", title, url)
            continue

        if not _is_usable_url(url):
            dropped_bad_url += 1
            logger.debug("Pre-filter [bad-url]      → %r", url)
            continue

        if _snippet_is_stale(snippet, title):
            dropped_stale += 1
            logger.debug("Pre-filter [stale]        → title=%r url=%r", title, url)
            continue

        usable_results.append(r)

    # ── Phase 2: Live Shallow Probing (FIX Q++) ──────────────────────────
    # FIX Q++ amendment: cap concurrent probe workers instead of scaling
    # 1:1 with result count. Diagnostic logging showed that firing 5-8
    # simultaneous requests at the same host (especially wuzzuf.net) causes
    # them to starve each other under load — 4/8 Wuzzuf probes timed out
    # in a single run at timeout=2.5s, even though each URL fetches fine
    # in isolation (confirmed via standalone synchronous test). Capping
    # concurrency to _LIVE_PROBE_MAX_WORKERS reduces per-host contention;
    # raising the per-request timeout (see _verify_live_url_is_stale
    # default) gives slower concurrent fetches enough headroom to finish
    # rather than being treated as fail-open network errors.
    dropped_live_stale: int = 0

    if usable_results:
        verified_results: List[dict] = []

        worker_count = min(len(usable_results), _LIVE_PROBE_MAX_WORKERS)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=worker_count
        ) as executor:
            future_to_job = {
                executor.submit(_verify_live_url_is_stale, r.get("url", "")): r
                for r in usable_results
            }

            for future in concurrent.futures.as_completed(future_to_job):
                r              = future_to_job[future]
                is_live_stale  = future.result()

                if is_live_stale:
                    dropped_live_stale += 1
                    logger.info(
                        "Phase 2 Live-verify [stale] → %r", r.get("url")
                    )
                else:
                    verified_results.append(r)

        usable_results = verified_results

    total_dropped = (
        dropped_blacklist + dropped_pollution + dropped_category +
        dropped_path_gate + dropped_bad_url + dropped_stale + dropped_live_stale
    )
    if total_dropped:
        logger.info(
            "Pre-filter: kept %d / %d | blacklist=%d pollution=%d category=%d "
            "path-gate=%d bad-url=%d stale=%d live-stale=%d",
            len(usable_results), len(results),
            dropped_blacklist, dropped_pollution, dropped_category,
            dropped_path_gate, dropped_bad_url, dropped_stale, dropped_live_stale,
        )

    # Accumulate into request-scoped stats so the API can explain an empty
    # result set to the user in terms of what actually happened, instead of
    # the generic "could not produce a structured response".
    record_filter_stats(
        examined=len(results),
        kept=len(usable_results),
        category=dropped_category,
        closed=dropped_stale + dropped_live_stale,
        path_gate=dropped_path_gate,
        blocked=dropped_blacklist + dropped_pollution + dropped_bad_url,
    )

    if not usable_results:
        drop_counts = {
            "path-gate (non-vacancy subpath)": dropped_path_gate,
            "category/listing page":           dropped_category,
            "stale/zombie listing":            dropped_stale,
            "bad URL":                         dropped_bad_url,
            # Was omitted, so a run where every listing was verified CLOSED by
            # the live probe reported some other reason as dominant — the most
            # actionable signal available, misattributed.
            "closed listing (verified live)":  dropped_live_stale,
        }
        dominant      = max(drop_counts, key=drop_counts.get)
        dominant_note = (
            f" The dominant drop reason was '{dominant}' "
            f"({drop_counts[dominant]} result(s))."
        )
        return (
            tech_warning + board_warning +
            f"Search returned {len(results)} results for {clean_query!r}, but "
            "none passed the quality filter (all were from blocked domains, "
            "non-vacancy subpaths, category/listing pages, stale/zombie "
            f"postings, or had unusable URLs).{dominant_note}\n"
            "Try a different approved board or broaden the role title slightly."
        )

    pipeline_reminder = (
        "[Partial results from one query — run both full-time AND "
        "internship/trainee queries before finalizing.]\n"
    )

    # Report the filter that was ACTUALLY applied. This used to claim
    # "posted on/after <recency_cutoff>" — but recency_cutoff is never sent to
    # Tavily (start_date and time_range are mutually exclusive), so the model was
    # being told a date filter had been applied that had not been. time_range is
    # also a CRAWL-date window, not a posting-date one; saying so keeps the model
    # from treating it as a guarantee about when the job was posted.
    intro_block = (
        f"Results for: {clean_query!r} "
        f"({len(usable_results)} listings; crawled within the past {time_range})\n"
    )

    header_blocks = [pipeline_reminder, tech_warning, board_warning, intro_block]
    result_blocks = [
        _format_result(i, result) for i, result in enumerate(usable_results, start=1)
    ]
    footer_block  = ""

    full_output, included_count = _join_results_within_budget(
        header_blocks, result_blocks, footer_block, max_chars=MAX_RESULT_CHARS
    )

    if included_count < len(usable_results):
        logger.warning(
            "tavily_job_search: budget allowed only %d / %d filtered listings "
            "into the output (MAX_RESULT_CHARS=%d).",
            included_count, len(usable_results), MAX_RESULT_CHARS,
        )
    else:
        logger.info(
            "tavily_job_search: all %d filtered listings included in output "
            "(%d chars, budget %d).",
            included_count, len(full_output), MAX_RESULT_CHARS,
        )

    logger.info(
        "RAW_LISTINGS_SENT_TO_LLM (query=%r) >>>\n%s", clean_query, full_output
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

    if not _passes_path_gate(url):
        return (
            f"Refused: {url!r} does not match the canonical job-listing URL "
            "structure for its domain. Only scrape individual posting URLs "
            "returned verbatim by tavily_job_search."
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

    text = re.sub(
        r"<script[^>]*>.*?</script>", " ", raw_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(
        r"<style[^>]*>.*?</style>", " ", text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)

    return _trim(text, PAGE_CHARS)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

def get_tools() -> list:
    return [tavily_job_search, scrape_job_page]