"""
The search-result filter chain.

WHY THIS LIVES IN `sources` AND NOT `agents`
--------------------------------------------
These patterns exist to answer one question: is this web page an individual,
currently-open job posting? That question only arises for a provider that
returns *web pages*. A JSearch or Remotive record already is a vacancy — the
provider knows it is a job, who posted it, and when.

So the entire chain applies to the Tavily adapter and nowhere else. That is the
single largest reduction in false-positive surface the structured rewrite buys:
not because the filters got better, but because most of what they defend
against stops arriving.

They moved here from `agents/tools.py` verbatim. The move was verified by
snapshotting every function's output over a 4,214-case corpus — real captured
Tavily responses, every board shape in `_BOARD_PATH_PATTERNS`, and the
English/Arabic staleness cases — before and after, and diffing.

WHAT THESE FILTERS ARE FOR
--------------------------
Six layers, applied in order. Order matters: a URL rejected early never reaches
a later stage, so the reported reason is always the *first* rejection.

    blacklist -> pollution -> category -> path_gate -> bad_url -> stale

Two of these carry scars worth remembering, because both were silent:

* A bare "closed" in the zombie list matched inside "undis**closed**", killing
  every listing that declined to state a salary — routine in MENA postings.
* Bayt's path gate knew only `/en/jobs/<slug>`, while Bayt serves
  `/en/<country>/jobs/<slug>-<id>`. Every real Bayt listing was rejected at the
  gate while the category pages around it were correctly caught.

Neither raised an error. Both simply returned fewer jobs. That is the failure
mode of this entire module, and the reason `survives_filter_chain` reports
*which* layer rejected a result rather than a bare boolean — a drop-reason
histogram makes an over-broad pattern visible in a day.
"""

from __future__ import annotations

import re
from typing import Dict, List
from urllib.parse import urlparse

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
    # Image and video platforms. A live Cairo search surfaced an Instagram
    # post titled "Senior Python Developer Nile Bits — Cairo" that cleared the
    # entire filter chain: it reads exactly like a listing and is not one.
    "instagram.com", "tiktok.com", "youtube.com", "pinterest.com",
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
    # Bayt: three shapes.
    #   /en/jobs/<slug>                      — the form originally handled
    #   /en/<country>/jobs/<slug>-<id>       — what Bayt actually serves today
    #   /job/<id>                            — legacy
    #
    # The country-segment form was missing, so every real Bayt listing was
    # rejected at the path gate. Found by running a live Cairo search and
    # inspecting the drop reasons: a genuine "Senior Backend Developer" posting
    # was discarded while the category pages around it were correctly caught.
    # Bayt is one of the two primary MENA boards, so this silently removed a
    # large share of the app's core market — for the agent as well as here.
    #
    # The country form requires a trailing numeric id, which is what separates
    # an individual posting from a listing hub such as
    # /en/egypt/jobs/junior-backend-jobs.
    "bayt.com": re.compile(
        r"^/(en|ar)/jobs/[^/]+/?$"
        r"|^/(en|ar)/[a-z\-]+/jobs/[^/]+-\d{4,}/?$"
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
    # Greenhouse: individual listings live at /<company>/jobs/<numeric-id>
    # across boards.greenhouse.io and job-boards.greenhouse.io. A bare
    # /<company> path is the company's board hub, not a vacancy.
    "greenhouse.io": re.compile(
        r"^/[^/]+/jobs/\d+",
        re.IGNORECASE,
    ),
    # Lever: individual listings are /<company>/<posting-id>. A bare
    # /<company> path is the company's list page; a two-segment path is a
    # discrete posting (optionally followed by /apply).
    "lever.co": re.compile(
        r"^/[^/]+/[^/]+",
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


# Declarations that a listing is dead.
#
# These are REGEX FRAGMENTS, not literal substrings, and every one of them must
# be anchored (\b or a required neighbouring word). They are NOT re.escape'd.
#
# Why this matters: this list previously contained the bare substring "closed",
# escaped into a boundary-free alternation. It therefore matched inside
# "undisclosed", "disclosed", "enclosed", "closed-loop" and "closed captions" —
# and "salary undisclosed" is routine phrasing on Bayt and Wuzzuf, which carry
# most of this app's MENA coverage. It was the single largest source of dropped
# real jobs in the pipeline.
#
# Rule for anyone adding a phrase here: require the surrounding words that make
# it a closure declaration. "closed" alone is a substring of ordinary English;
# "vacancy closed" is a statement about this job.
_ZOMBIE_CONTENT_SNIPPETS_EN: tuple[str, ...] = (
    r"\bno longer accepting applications\b",
    r"\bposition filled\b",
    r"\bjob expired\b",
    r"\bthis job is no longer available\b",
    r"\bapplications?\s+(?:are\s+|is\s+|been\s+)?closed\b",
    r"\bthis position has been filled\b",
    # "<the position|vacancy|job|role|posting> [is|was|has been] closed"
    r"\b(?:position|vacancy|job|role|posting|listing)\s+"
    r"(?:is\s+|was\s+|has\s+been\s+)?closed\b",
    r"\bno longer open\b",
    r"\bexpired\s+(?:job|vacancy|posting|listing)\b",
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
        # EN entries are already anchored regex; AR entries are literals with no
        # word-boundary concept in Arabic script, so they stay escaped.
        _ZOMBIE_CONTENT_SNIPPETS_EN
        + tuple(re.escape(p) for p in _ZOMBIE_CONTENT_SNIPPETS_AR)
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
# Public interface
# ---------------------------------------------------------------------------
#
# The chain is exposed under public names; the underscore-prefixed definitions
# above are the historical spellings, kept so `agents/tools.py` can import them
# unchanged.

is_blacklisted_domain = _is_blacklisted_domain
is_content_pollution_domain = _is_content_pollution_domain
is_category_page = _is_category_page
passes_path_gate = _passes_path_gate
is_usable_url = _is_usable_url
snippet_is_stale = _snippet_is_stale
snippet_is_zombie = _snippet_is_zombie
build_exclude_domains = _build_tavily_exclude_domains
normalise_netloc = _normalise_netloc


class FilterVerdict:
    """
    Why a result was rejected, or `KEPT`.

    Naming the layer rather than returning a bare boolean is what makes
    drop-reason telemetry possible — and that telemetry is what would have
    surfaced the bare-"closed" bug in a day instead of a code audit, because
    the `stale` bucket would have swallowed a visibly implausible share of
    every search.
    """

    KEPT = "kept"
    BLACKLIST = "blacklist"
    POLLUTION = "pollution"
    CATEGORY = "category"
    PATH_GATE = "path_gate"
    BAD_URL = "bad_url"
    STALE = "stale"


def survives_filter_chain(url: str, title: str, snippet: str) -> str:
    """
    Run the ordered chain and return the first rejecting layer, or `KEPT`.

    Order is significant and mirrors the agent's. Reordering changes which
    bucket a drop lands in even when the outcome is unchanged, which is why the
    sequence is pinned by tests rather than left to the reader.
    """
    if is_blacklisted_domain(url):
        return FilterVerdict.BLACKLIST
    if is_content_pollution_domain(url):
        return FilterVerdict.POLLUTION
    if is_category_page(title, url):
        return FilterVerdict.CATEGORY
    if not passes_path_gate(url):
        return FilterVerdict.PATH_GATE
    if not is_usable_url(url):
        return FilterVerdict.BAD_URL
    if snippet_is_stale(snippet, title):
        return FilterVerdict.STALE
    return FilterVerdict.KEPT


__all__ = [
    "FilterVerdict",
    "build_exclude_domains",
    "is_blacklisted_domain",
    "is_category_page",
    "is_content_pollution_domain",
    "is_usable_url",
    "normalise_netloc",
    "passes_path_gate",
    "snippet_is_stale",
    "snippet_is_zombie",
    "survives_filter_chain",
    "STALENESS_MONTHS_THRESHOLD",
]
