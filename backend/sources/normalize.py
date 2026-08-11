"""
Provider-agnostic field normalization shared by every adapter.

Anything here converts messy provider data into the vocabularies
`NormalizedJob` is defined in. Provider-*specific* mapping (JSearch's
"FULLTIME" -> EmploymentType.FULL_TIME) stays in that provider's adapter; only
logic that would otherwise be copy-pasted across adapters belongs here.

The skill and seniority vocabularies deliberately come from `cv_parser`. The
scorer compares a candidate's skills and level against a posting's, so both
sides must be expressed in identical terms — otherwise that comparison needs a
translation table, and translation tables between two independently-evolving
vocabularies rot silently.
"""

from __future__ import annotations

import html
import re
from datetime import datetime, timezone
from typing import Optional

from backend.parsers.cv_parser import _SENIORITY_PATTERNS, _harvest_skills
from backend.sources.schema import Seniority

# Block-level tags whose boundaries carry meaning. Replacing them with a
# newline before stripping keeps "Python</li><li>Django" from becoming
# "PythonDjango", which would hide both skills from the harvester.
_BLOCK_TAG_RE = re.compile(
    r"</?\s*(?:p|div|br|li|ul|ol|tr|td|th|h[1-6]|section|article)\b[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES_RE = re.compile(r"\n\s*\n+")


def skills_from_text(*texts: Optional[str]) -> list[str]:
    """
    Harvest canonical skill names from one or more free-text blocks.

    Uses the same deterministic dictionary the CV parser uses, so a posting's
    "Django" and a CV's "django" both normalize to the identical token and
    overlap becomes a set intersection rather than fuzzy matching.

    Order follows the skills dictionary, not order of appearance, and is stable
    across calls — which matters because these end up in a cache key.
    """
    combined = "\n".join(text for text in texts if text)
    if not combined.strip():
        return []
    return _harvest_skills(combined)


def seniority_from_text(*texts: Optional[str]) -> Optional[Seniority]:
    """
    Infer a seniority band from a job title (and optionally other short text).

    Returns None when no keyword matches. None means "not stated", which the
    scorer treats as neutral — a guess here would silently distort every
    ranking that uses it.

    Pass the title alone where possible. A full description will mention
    "senior" in unrelated contexts ("reports to a senior manager", "work with
    senior stakeholders") and produce confident nonsense.
    """
    combined = " ".join(text for text in texts if text)
    if not combined.strip():
        return None

    # _SENIORITY_PATTERNS is ordered most-senior-first, so the first hit wins
    # and "Senior Engineering Lead" resolves to lead rather than senior.
    for label, pattern in _SENIORITY_PATTERNS:
        if pattern.search(combined):
            try:
                return Seniority(label)
            except ValueError:
                # The parser grew a label the schema does not model yet.
                # Degrade to "unknown" rather than raising mid-search.
                return None
    return None


def seniority_from_months(months: Optional[int]) -> Optional[Seniority]:
    """
    Band a required-experience figure into a seniority level.

    Boundaries mirror `cv_parser._infer_seniority` so a posting requiring five
    years and a candidate with five years land on the same band. Keeping the
    two banding functions numerically identical is the point; if one moves, the
    other must.
    """
    if months is None or months < 0:
        return None

    years = months / 12.0
    if years < 1:
        return Seniority.JUNIOR
    if years < 3:
        return Seniority.JUNIOR
    if years < 6:
        return Seniority.MID
    if years < 10:
        return Seniority.SENIOR
    return Seniority.LEAD


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    """
    Parse a provider ISO-8601 timestamp into an aware UTC datetime.

    Returns None on anything unparseable rather than raising: one malformed
    timestamp must not fail the whole search. A missing date costs the job its
    recency points, which is the correct outcome for a record whose freshness
    we genuinely cannot establish.

    Handles the trailing "Z" that `fromisoformat` rejected before 3.11, since
    providers emit it constantly.
    """
    if not value:
        return None

    text = value.strip()
    if not text:
        return None
    if text.endswith(("Z", "z")):
        text = f"{text[:-1]}+00:00"

    try:
        parsed = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


# Currency symbols and codes seen in job-board salary strings.
_CURRENCY_SYMBOLS: dict[str, str] = {
    "$": "USD", "€": "EUR", "£": "GBP", "₹": "INR", "﷼": "SAR",
    "د.إ": "AED", "ج.م": "EGP",
}
_CURRENCY_CODES: frozenset[str] = frozenset(
    {"USD", "EUR", "GBP", "EGP", "AED", "SAR", "QAR", "KWD", "JOD", "INR", "CAD", "AUD"}
)

# "120,000", "120k", "1.2k" — a number with optional thousands separators and
# an optional k suffix.
_AMOUNT = r"(\d[\d,\.]*)\s*([kK])?"
# Ranges are routinely written "$120,000 - $150,000", repeating the symbol.
# Without allowing it before the second figure the range fails to match and the
# string falls through to the single-amount path, yielding min == max — a
# silently wrong range rather than no range.
_CURRENCY_PREFIX = r"(?:[$€£₹﷼]|\b[A-Z]{3}\b)?\s*"
_SALARY_RANGE_RE = re.compile(
    rf"{_AMOUNT}\s*(?:-|–|—|to)\s*{_CURRENCY_PREFIX}{_AMOUNT}", re.IGNORECASE
)
_SALARY_SINGLE_RE = re.compile(_AMOUNT)

# Plausibility floor per period. An annual figure below ~1000 is a stray number
# ("top 10", "3 years"); an hourly rate of 50 is entirely normal. Using one
# floor for both either admits garbage or discards every hourly rate.
_MIN_PLAUSIBLE_BY_PERIOD: dict[Optional[str], float] = {
    "HOUR": 5.0,
    "DAY": 20.0,
    "WEEK": 50.0,
    "MONTH": 100.0,
    "YEAR": 1000.0,
    None: 100.0,
}

_PERIOD_HINTS: tuple[tuple[re.Pattern, str], ...] = (
    (re.compile(r"\b(?:per\s+hour|/\s*h(?:r|our)?|hourly)\b", re.IGNORECASE), "HOUR"),
    (re.compile(r"\b(?:per\s+day|/\s*day|daily)\b", re.IGNORECASE), "DAY"),
    (re.compile(r"\b(?:per\s+week|/\s*w(?:k|eek)?|weekly)\b", re.IGNORECASE), "WEEK"),
    (re.compile(r"\b(?:per\s+month|/\s*mo(?:nth)?|monthly|شهري)\b", re.IGNORECASE), "MONTH"),
    (re.compile(r"\b(?:per\s+year|/\s*y(?:r|ear)?|yearly|annually|per\s+annum|p\.?a\.?)\b",
                re.IGNORECASE), "YEAR"),
)


def _to_amount(
    digits: str,
    k_suffix: Optional[str],
    period: Optional[str] = None,
) -> Optional[float]:
    """
    Parse '120,000' / '120k' into a float, or None if implausible.

    Note on European formatting: "60.000" meaning sixty thousand parses as
    60.0 and is then rejected by the floor. That is the intended outcome —
    silently reading it as sixty is far worse than declining to parse it.
    """
    cleaned = digits.replace(",", "")
    # A trailing '.' or multiple dots is malformed rather than decimal.
    if cleaned.count(".") > 1:
        return None
    try:
        amount = float(cleaned)
    except ValueError:
        return None
    if k_suffix:
        amount *= 1000

    floor = _MIN_PLAUSIBLE_BY_PERIOD.get(period, 100.0)
    if not (floor <= amount < 100_000_000):
        return None
    return amount


def parse_salary_text(
    value: Optional[str],
) -> tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Best-effort parse of a free-text salary string.

    Returns ``(min, max, currency, period)`` with any component None when it
    could not be established confidently.

    Used by providers that publish salary as prose (Remotive, Jooble) rather
    than numeric fields. Those adapters set ``provides_structured_salary =
    False`` so the scorer knows these figures are inferred and can weight them
    accordingly — the capability flag and this function are two halves of one
    honest answer, not a contradiction.

    Deliberately conservative. A wrong salary is worse than no salary: absence
    scores neutral, whereas a misparsed figure actively mis-ranks the job. When
    in doubt this returns None.
    """
    if not value or not value.strip():
        return (None, None, None, None)

    text = value.strip()

    currency: Optional[str] = None
    for symbol, code in _CURRENCY_SYMBOLS.items():
        if symbol in text:
            currency = code
            break
    if currency is None:
        for token in re.findall(r"\b[A-Z]{3}\b", text):
            if token in _CURRENCY_CODES:
                currency = token
                break

    period: Optional[str] = None
    for pattern, label in _PERIOD_HINTS:
        if pattern.search(text):
            period = label
            break

    salary_min: Optional[float] = None
    salary_max: Optional[float] = None

    range_match = _SALARY_RANGE_RE.search(text)
    if range_match:
        # "120 - 150k": a k suffix on only the second figure applies to both.
        # Resolved before parsing, because 120 would otherwise fail the annual
        # floor and discard an otherwise perfectly good range.
        low_suffix = range_match.group(2) or range_match.group(4)
        low = _to_amount(range_match.group(1), low_suffix, period)
        high = _to_amount(range_match.group(3), range_match.group(4), period)
        if low is not None and high is not None:
            salary_min, salary_max = min(low, high), max(low, high)
    else:
        single = _SALARY_SINGLE_RE.search(text)
        if single:
            amount = _to_amount(single.group(1), single.group(2), period)
            if amount is not None:
                salary_min = salary_max = amount

    if salary_min is None and salary_max is None:
        return (None, None, None, None)

    return (salary_min, salary_max, currency, period)


def parse_unix_timestamp(value) -> Optional[datetime]:
    """
    Convert a Unix epoch seconds value into an aware UTC datetime.

    Some providers (Arbeitnow) publish `created_at` as an integer rather than
    an ISO string. Returns None on anything implausible rather than raising:
    a garbage timestamp costs the job its recency points, which is correct for
    a record whose freshness cannot be established, but must not drop the job.

    Zero is rejected — it is the epoch, and in practice always means "unset"
    rather than 1 January 1970.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None

    # Bounds chosen to reject both "unset" sentinels and millisecond values
    # mistakenly passed as seconds (which land ~50,000 years in the future).
    if not (946_684_800 < seconds < 4_102_444_800):  # 2000-01-01 .. 2100-01-01
        return None

    try:
        return datetime.fromtimestamp(seconds, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def strip_html(value: Optional[str]) -> str:
    """
    Reduce an HTML job description to readable plain text.

    Job boards return descriptions as HTML fragments. Feeding those to the
    skill harvester unmodified means matching against tag names and attribute
    values, and rendering them in a UI means an XSS surface. Both are avoided
    by normalizing to text at the adapter boundary — once, on the way in.

    This is not a sanitizer for untrusted HTML you intend to render; it is a
    text extractor. Nothing here should ever be marked safe for innerHTML.
    """
    if not value:
        return ""

    # Block boundaries become newlines before tags are removed, so list items
    # and paragraphs do not run together into a single unsearchable word.
    text = _BLOCK_TAG_RE.sub("\n", value)
    text = _ANY_TAG_RE.sub(" ", text)
    text = html.unescape(text)

    text = _WHITESPACE_RE.sub(" ", text)
    text = _BLANK_LINES_RE.sub("\n\n", text)
    return text.strip()


def first_non_empty(*values: Optional[str]) -> Optional[str]:
    """First value that is a non-blank string, else None."""
    for value in values:
        if value and str(value).strip():
            return str(value).strip()
    return None
