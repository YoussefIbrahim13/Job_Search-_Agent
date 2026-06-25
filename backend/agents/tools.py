"""
backend/agents/tools.py
=======================
LangChain tool definitions for the Recruitment AI Agent.

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
import urllib.error
import urllib.request
import concurrent.futures

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from urllib.parse import urlparse

from langchain_core.tools import tool

from backend.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token-safety constants
# ---------------------------------------------------------------------------

MAX_RESULT_CHARS: int = 3_000
SNIPPET_CHARS: int    = 600
PAGE_CHARS: int       = 1_800


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

RECENCY_WINDOW_DAYS: int = 20


def _compute_recency_cutoff(days_back: int = RECENCY_WINDOW_DAYS) -> str:
    """Return today's date minus ``days_back`` days as YYYY-MM-DD."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days_back)
    return cutoff.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# FIX 2 (preserved): Domain-level blacklist for CSR zombie aggregators
# ---------------------------------------------------------------------------

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

_CONTENT_POLLUTION_DOMAINS: frozenset[str] = frozenset({
    "quora.com", "stackoverflow.com", "stackexchange.com", "answers.com",
    "yahoo.com", "ask.com",
    "reddit.com", "facebook.com", "twitter.com", "x.com", "linkedin.com/pulse",
    "medium.com", "substack.com", "hashnode.com", "dev.to",
    "wikipedia.org", "wikihow.com", "thoughtco.com", "thebalancemoney.com",
    "thebalancecareers.com", "investopedia.com", "businessinsider.com",
    "forbes.com", "techcrunch.com", "towardsdatascience.com",
    "analyticsvidhya.com", "geeksforgeeks.org", "tutorialspoint.com",
    "javatpoint.com", "w3schools.com",
    "thebalance.com", "livecareer.com", "resumegenius.com", "zety.com",
    "novoresume.com", "resumelab.com", "jobscan.co", "themuse.com",
    "careerbuilder.com",
})


def _is_content_pollution_domain(url: str) -> bool:
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


def _build_tavily_exclude_domains() -> List[str]:
    return sorted(_BLACKLISTED_DOMAINS | _CONTENT_POLLUTION_DOMAINS)


# ---------------------------------------------------------------------------
# FIX N — Positive-Assertion Path Gating
# ---------------------------------------------------------------------------
#
# ARCHITECTURE NOTE:
# Each pattern is a POSITIVE ASSERTION of what a real listing URL looks like
# on that board. It must match; if it does not, the result is dropped.
# The patterns are intentionally permissive within the correct path structure
# (e.g. they allow arbitrary slugs/IDs after the canonical prefix) so that
# legitimate posting URLs are never over-eagerly rejected. They are strict
# only about the top-level subpath, which is the reliable discriminator
# between listing pages and non-vacancy content on each board.
#
# Pattern rationale per board:
#   wuzzuf.net      : individual listings live under /jobs/p/<slug> exclusively.
#                     /jobs/ alone (no /p/ segment) is a search/listing hub.
#                     /r/<slug> paths are their public "Job Description Template"
#                     library — not real vacancies.
#   linkedin.com    : individual listings use TWO path forms in the wild:
#                       /jobs/view/<numeric-id>           (pure numeric)
#                       /jobs/view/<slug>-<numeric-id>    (slug + trailing ID)
#                     LinkedIn serves listings from country-coded subdomains
#                     (pk.linkedin.com, eg.linkedin.com, bg.linkedin.com, etc.)
#                     which are handled by the endswith() check in
#                     _passes_path_gate().
#                     /jobs/search/ and /jobs/collections/ are list/hub pages
#                     and do NOT end in a numeric ID, so they are correctly
#                     excluded.
#   bayt.com        : individual listings use /en/jobs/<slug>-<id>/ (English)
#                     or /ar/jobs/ (Arabic). Also /job/<id>/ on some subdomains.
#   indeed.com      : REMOVED from the strict path dict. Tavily returns Indeed's
#                     internal search-results pages (/q-<query>-jobs.html) almost
#                     exclusively — the real listing paths (/viewjob, /rc/clk)
#                     are almost never surfaced. Keeping a strict path gate for
#                     Indeed was blocking 100% of Indeed results while yielding
#                     zero genuine listings. Indeed is now handled by
#                     _BAD_URL_PATTERNS (which blocks /q-*-jobs.html search
#                     pages) and the existing _is_usable_url() heuristics. Any
#                     genuine /viewjob URL that does surface passes through.
#   glassdoor.com   : /job-listing/<slug> is the canonical individual listing.
#                     /Jobs/ (capital J) and /jobs/ are list hubs.
#   akhtaboot.com   : /jobs/<numeric-id>- prefix is canonical for listings.
#   weworkremotely.com : /remote-jobs/<category>/<slug> is the listing path.
#   himalayas.app   : /jobs/<slug> with a non-numeric slug is canonical.
#   wellfound.com   : /jobs/<slug> (previously angel.co/jobs/<slug>).
#   dice.com        : /jobs/detail/<slug> is the canonical path.
#   remoteok.com    : flat slugs directly under / (e.g. /remote-python-jobs-<id>).
#                     No strict path structure to assert; falls through to
#                     _is_usable_url() which already rejects root/search paths.

_BOARD_PATH_PATTERNS: Dict[str, re.Pattern] = {
    # Wuzzuf: must be /jobs/p/<anything>
    "wuzzuf.net": re.compile(
        r"^/jobs/p/[^/]+",
        re.IGNORECASE,
    ),
    # LinkedIn: two canonical forms for individual listings —
    #   pure numeric:  /jobs/view/4396364201
    #   slug + id:     /jobs/view/net-application-developer-intern-at-apexanalytix-4396364201
    # Both are asserted by requiring the path to end with a numeric ID segment.
    "linkedin.com": re.compile(
        r"^/jobs/view/(?:[^/]+-)?(\d{7,})/?$",
        re.IGNORECASE,
    ),
    # Bayt: English /en/jobs/<slug> or Arabic /ar/jobs/<slug>,
    # or legacy /job/<id>
    "bayt.com": re.compile(
        r"^/(en|ar)/jobs/[^/]+/?$"
        r"|^/job/\d+",
        re.IGNORECASE,
    ),
    # Indeed: intentionally absent — see rationale in comment block above.
    # Glassdoor: /job-listing/<slug>
    "glassdoor.com": re.compile(
        r"^/job-listing/[^/]+",
        re.IGNORECASE,
    ),
    # Akhtaboot: /jobs/<numeric-id>-<anything>
    "akhtaboot.com": re.compile(
        r"^/jobs/\d+[^/]*",
        re.IGNORECASE,
    ),
    # WeWorkRemotely: /remote-jobs/<category>/<slug>
    "weworkremotely.com": re.compile(
        r"^/remote-jobs/[^/]+/[^/]+",
        re.IGNORECASE,
    ),
    # Himalayas: /jobs/<slug> (not /companies/, not root)
    "himalayas.app": re.compile(
        r"^/jobs/[^/]+/?$",
        re.IGNORECASE,
    ),
    # Wellfound (formerly AngelList Talent): /jobs/<slug>
    "wellfound.com": re.compile(
        r"^/jobs/[^/]+",
        re.IGNORECASE,
    ),
    # Dice: /jobs/detail/<slug>
    "dice.com": re.compile(
        r"^/jobs/detail/[^/]+",
        re.IGNORECASE,
    ),
}


def _normalise_netloc(netloc: str) -> str:
    """Strip www. prefix and lowercase for consistent dict lookup."""
    netloc = netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    return netloc


def _passes_path_gate(url: str) -> bool:
    """
    Positive-assertion path gate for approved boards.

    Returns:
        True  — URL either (a) belongs to a board in _BOARD_PATH_PATTERNS
                and its path MATCHES the canonical listing pattern, or
                (b) belongs to no board in the dict (pass-through for boards
                without a strict path structure).
        False — URL belongs to a board in the dict but its path does NOT
                match the canonical pattern (e.g. wuzzuf.net/r/template-slug,
                linkedin.com/jobs/search/, indeed.com/jobs).
    """
    try:
        parsed = urlparse(url)
        netloc = _normalise_netloc(parsed.netloc)
        path   = parsed.path or "/"
    except Exception:
        return True  # malformed URL; let the existing _is_usable_url() handle it

    # Find the matching board entry (accounts for subdomains like eg.linkedin.com)
    pattern: Optional[re.Pattern] = None
    for board_domain, board_pattern in _BOARD_PATH_PATTERNS.items():
        if netloc == board_domain or netloc.endswith("." + board_domain):
            pattern = board_pattern
            break

    if pattern is None:
        # Board not in dict — no path assertion applied, pass through.
        return True

    return bool(pattern.search(path))


# ---------------------------------------------------------------------------
# FIX O — Content-Layer Staleness / Zombie Detection
# ---------------------------------------------------------------------------
#
# STALENESS_MONTHS_THRESHOLD controls what "too old" means when reading
# human-readable age strings from snippet/title text. Any posting that
# declares itself posted >= this many months ago is rejected regardless of
# Tavily's crawl timestamp.
#
# Set to 3 months — conservative enough to catch multi-year zombies while
# safe from false-positives on legitimate postings that are a few weeks old.

STALENESS_MONTHS_THRESHOLD: int = 3  # months

_EN_YEARS_RE = re.compile(
    r"\b([1-9][0-9]*)\s+year[s]?\s+ago\b",
    re.IGNORECASE,
)
_EN_MONTHS_RE = re.compile(
    r"\b([1-9][0-9]*)\s+month[s]?\s+ago\b",
    re.IGNORECASE,
)

_AR_YEARS_RE = re.compile(
    r"منذ\s+(?:سنة|سنتين|(?:[٠-٩\d]+)\s*سنوات?)",
    re.IGNORECASE,
)
_AR_MONTHS_RE = re.compile(
    r"منذ\s+(?:شهر(?:ين)?|(?:[٠-٩\d]+)\s*(?:أشهر|شهور|شهر))",
    re.IGNORECASE,
)

_ZOMBIE_CONTENT_SNIPPETS_EN: tuple[str, ...] = (
    "no longer accepting applications",
    "position filled",
    "job expired",
    "this job is no longer available",
    "applications closed",
    "this position has been filled",
    "vacancy closed",
    "closed",
)
_ZOMBIE_CONTENT_SNIPPETS_AR: tuple[str, ...] = (
    "لم نعد نقبل استمارات",
    "تم غلق هذا العرض",
    "انتهت فترة التقديم",
    "الوظيفة مغلقة",
    "تم شغل الوظيفة",
)
_ZOMBIE_DECLARATION_RE = re.compile(
    "|".join(
        re.escape(phrase)
        for phrase in _ZOMBIE_CONTENT_SNIPPETS_EN + _ZOMBIE_CONTENT_SNIPPETS_AR
    ),
    re.IGNORECASE,
)


def _arabic_digit_to_int(text: str) -> int:
    """Convert a string that may contain Arabic-Indic digits to int."""
    arabic_digits = "٠١٢٣٤٥٦٧٨٩"
    western = str.maketrans(arabic_digits, "0123456789")
    return int(text.translate(western))


def _snippet_is_stale(snippet: str, title: str = "") -> bool:
    """
    Return True if the snippet or title contains evidence that this listing
    is a stale zombie, via EITHER:
      (a) an explicit closed/filled/expired declaration, OR
      (b) a human-readable posting-age string that exceeds
          STALENESS_MONTHS_THRESHOLD.
    """
    combined = f"{title} {snippet}"

    # (a) Explicit closed/filled/expired declaration
    if _ZOMBIE_DECLARATION_RE.search(combined):
        return True

    # (b) English: years — any 1+ year is stale
    for m in _EN_YEARS_RE.finditer(combined):
        try:
            years = int(m.group(1))
            if years >= 1:
                return True
        except ValueError:
            pass

    # (b) English: months — stale if >= threshold
    for m in _EN_MONTHS_RE.finditer(combined):
        try:
            months = int(m.group(1))
            if months >= STALENESS_MONTHS_THRESHOLD:
                return True
        except ValueError:
            pass

    # (b) Arabic: years (any match means stale — all forms are >= 1 year)
    if _AR_YEARS_RE.search(combined):
        return True

    # (b) Arabic: months — need to extract the number and check threshold.
    for m in _AR_MONTHS_RE.finditer(combined):
        matched_text = m.group(0)
        if "شهرين" in matched_text:
            if 2 >= STALENESS_MONTHS_THRESHOLD:
                return True
            continue
        if re.search(r"منذ\s+شهر\b", matched_text):
            if 1 >= STALENESS_MONTHS_THRESHOLD:
                return True
            continue
        num_match = re.search(r"([٠-٩\d]+)", matched_text)
        if num_match:
            try:
                months = _arabic_digit_to_int(num_match.group(1))
                if months >= STALENESS_MONTHS_THRESHOLD:
                    return True
            except (ValueError, TypeError):
                pass

    return False


# Keep old name as a thin alias so any legacy imports remain unbroken.
def _snippet_is_zombie(snippet: str) -> bool:
    """Backwards-compatible alias — prefer _snippet_is_stale() for new code."""
    return _snippet_is_stale(snippet)


# ---------------------------------------------------------------------------
# FIX K — Category / aggregator page detector (preserved)
# ---------------------------------------------------------------------------

_CATEGORY_PAGE_TITLE_RE = re.compile(
    r"\d+\+?\s*(jobs?|vacancies|positions?|openings?)\b"
    r"|\bjobs?\s+in\s+[A-Za-z]"
    r"|\bvacancies\s+in\s+[A-Za-z]"
    r"|\b(browse|search)\s+(all\s+)?jobs?\b"
    r"|\ball\s+jobs?\b"
    r"|\blatest\s+jobs?\b"
    r"|\btop\s+\d*\s*jobs?\b",
    re.IGNORECASE,
)

_CATEGORY_PAGE_URL_RE = re.compile(
    r"/category/|/browse|/all-jobs|/jobs-in-|/c/jobs|/job-listings/?$",
    re.IGNORECASE,
)


def _is_category_page(title: str, url: str) -> bool:
    if _CATEGORY_PAGE_TITLE_RE.search(title or ""):
        return True
    if _CATEGORY_PAGE_URL_RE.search(url or ""):
        return True
    return False


# ---------------------------------------------------------------------------
# FIX B — Approved search board registry
# ---------------------------------------------------------------------------

APPROVED_SEARCH_BOARDS: dict[str, str] = {
    "linkedin":       "site:linkedin.com/jobs",
    "indeed":         "site:indeed.com",
    "glassdoor":      "site:glassdoor.com",
    "wuzzuf":         "site:wuzzuf.net",
    "bayt":           "site:bayt.com",
    "akhtaboot":      "site:akhtaboot.com",
    "weworkremotely": "site:weworkremotely.com",
    "remoteok":       "site:remoteok.com",
    "himalayas":      "site:himalayas.app",
    "wellfound":      "site:wellfound.com",
    "dice":           "site:dice.com",
}

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

# Literal closure-badge patterns (positive assertion, not prose). These
# match ONLY the specific, unambiguous UI strings each board renders for a
# closed/expired listing — deliberately narrow so they cannot match inside
# a company "About Us" blurb or a LinkedIn repost badge (the two sources of
# false positives that FIX Q+ was built to avoid).
_WUZZUF_CLOSED_BADGE_RE = re.compile(
    r">\s*Closed\s*<",        # the literal "Closed" pill/badge element
    re.IGNORECASE,
)

_LINKEDIN_CLOSED_BADGE_RE = re.compile(
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
    fetch_size = _HEAD_PROBE_BYTES if use_narrow_probe else 65_536

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
        if _has_explicit_closed_badge(raw_html, netloc):
            logger.info(
                "Live probe [closed-badge] → %r (narrow head-probe, %d bytes)",
                url, len(raw_html),
            )
            return True

        head_text = re.sub(r"<[^>]+>", " ", raw_html)
        head_text = html.unescape(head_text)

        if _snippet_is_stale(head_text, title=""):
            logger.info(
                "Live probe [age-badge] → %r (narrow head-probe, %d bytes)",
                url, len(raw_html),
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
# URL validation helpers
# ---------------------------------------------------------------------------

_BAD_URL_PATTERNS = re.compile(
    r"/search|\?q=|-jobs-in-|/find-jobs|keyword=|/jobs/?$|"
    r"jobs\.(google|bing)\.com|"
    r"/jobs/search|/job-search|/pulse/|"
    # Indeed search-results pages: /q-<query>-jobs.html or /q-<query>-jobs
    r"/q-[^/]+-jobs(?:\.html)?$",
    re.IGNORECASE,
)

_VALID_URL_RE = re.compile(r"^https?://[^\s/$.?#].[^\s]*$", re.IGNORECASE)


def _is_usable_url(url: str) -> bool:
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

      You MAY combine up to two approved boards using OR in one query.
      You may NOT use any domain not on the list above.
      You may NOT omit the site: clause entirely.

    RULE 3 — QUERY FORMAT (mandatory structure — STACK TOKENS ONLY):
      "<TECH_1> <TECH_2> <role> <modifier> <site:TOKEN>"
      where modifier is one of: jobs, internship, intern, trainee

      Required examples:
        ✓ "Python Django Back-End Developer jobs site:wuzzuf.net OR site:bayt.com"
        ✓ "React Node.js Software Engineer internship site:linkedin.com/jobs"
        ✓ "C# ASP.NET Backend Developer jobs site:glassdoor.com OR site:wellfound.com"

      Forbidden examples:
        ✗ "Back-End Developer jobs Cairo"            ← no tech, no site:
        ✗ "Python developer jobs"                    ← no site:
        ✗ "Go ASP.NET Backend Developer jobs site:linkedin.com/jobs"
              ← leaked operational "Go" corrupts a C#/.NET search.

    RULE 4 — FIRE BOTH REQUIRED QUERIES IN PARALLEL, IN YOUR FIRST TURN.

    RULE 5 — NEVER INVENT RESULTS. Copy URLs character-for-character.

    RULE 6 — RECENCY IS AUTOMATIC. Do NOT add a specific year.

    Returns a numbered list of job postings from approved boards only.
    Category/listing pages, templates, blog articles, career-advice pages,
    and non-vacancy subpaths are excluded BEFORE results reach you via a
    multi-layer filter: (1) URL path gating against each board's canonical
    listing structure, (2) snippet/title content scanning for age badges
    (e.g. "Posted 5 years ago") and closed declarations, and (3) a live
    shallow probe for boards where snippet staleness alone is insufficient.
    Every listing shown below is a structurally-validated individual posting
    candidate.
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

    enriched_query = (
        f'{clean_query} ("apply now" OR "hiring now" OR "job description") '
        f'-"jobs in" -"browse jobs" -"vacancies in"'
    )

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

    if not usable_results:
        drop_counts = {
            "path-gate (non-vacancy subpath)": dropped_path_gate,
            "category/listing page":           dropped_category,
            "stale/zombie listing":            dropped_stale,
            "bad URL":                         dropped_bad_url,
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

    intro_block = (
        f"Results for: {clean_query!r} "
        f"({len(usable_results)} listings, posted on/after {recency_cutoff})\n"
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