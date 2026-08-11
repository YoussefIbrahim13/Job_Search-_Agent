"""
The normalized job record every source adapter emits.

FROZEN SCHEMA — read before changing
------------------------------------
`NormalizedJob` is the contract between four subsystems that are built in
parallel: the source adapters (2.2), the deterministic scorer (2.3), the
persistence layer (Phase 4's `JobResult`), and the API responses the frontend
renders (Phase 7). Changing a field name or type here is not a local edit; it
is a migration plus a frontend change plus a re-record of every fixture.

Additive changes (a new optional field with a default) are cheap and fine.
Renames, type changes, and semantic changes are not. Prefer adding.

WHY THIS EXISTS AT ALL
----------------------
The prototype's "job" was an untyped dict assembled by an LLM reading search
snippets. Every field was a string, `posted_date` was whatever prose the model
copied out of a snippet, and "Not specified" was a legitimate value for
anything. That representation cannot support ranking: you cannot decay a score
by recency when the date is the string "2 weeks ago", and you cannot compare
salaries when one is "EGP 40,000 - 60,000 per month" and another is null.

This model exists so that scoring can be arithmetic instead of guesswork.
Everything the scorer needs is a typed field with an explicit "unknown"
representation (`None`), so the scorer can distinguish "this job pays badly"
from "this job did not state its pay" — a distinction the prototype could not
make, and which matters because MENA postings routinely omit salary.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
#
# Every enum carries an explicit UNKNOWN member rather than using None. The
# distinction is deliberate: `None` on an *optional* field means "the provider
# did not tell us", while UNKNOWN on a required enum means the same thing but
# keeps the field non-nullable for consumers that must always branch on it.
# Where both would be redundant the field is simply Optional and there is no
# UNKNOWN member (see Seniority).


class WorkMode(str, Enum):
    """
    Where the work physically happens.

    Authoritative over any boolean remote flag — see `NormalizedJob.is_remote`,
    which is a derived property rather than a stored field.
    """

    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNKNOWN = "unknown"


class EmploymentType(str, Enum):
    """
    Contract shape.

    INTERNSHIP is a first-class member because the prototype spent an entire
    agent-graph branch (the "coercion node") trying to force a second
    internship query through an LLM. With a structured field it is a filter.
    """

    FULL_TIME = "full_time"
    PART_TIME = "part_time"
    CONTRACT = "contract"
    INTERNSHIP = "internship"
    TEMPORARY = "temporary"
    UNKNOWN = "unknown"


class Seniority(str, Enum):
    """
    Career level.

    Values match `cv_parser._SENIORITY_PATTERNS` exactly, plus MID which that
    module produces from years-of-experience banding. Keeping the vocabularies
    identical is what lets the scorer compare a candidate's level against a
    posting's without a translation table — the kind of mapping layer that
    silently rots.
    """

    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"
    PRINCIPAL = "principal"


# Ordered weakest → strongest, for distance comparisons in the scorer.
SENIORITY_ORDER: tuple[Seniority, ...] = (
    Seniority.INTERN,
    Seniority.JUNIOR,
    Seniority.MID,
    Seniority.SENIOR,
    Seniority.LEAD,
    Seniority.PRINCIPAL,
)


class SalaryPeriod(str, Enum):
    """
    The unit a salary figure is quoted in.

    Not cosmetic: MENA boards quote monthly and US boards quote yearly, so
    comparing the raw numbers without this field is off by 12x. `Wuzzuf`
    monthly EGP against a US annual USD figure is the exact comparison the
    scorer has to make.
    """

    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"


# Multipliers to annualize a figure. Hourly/daily/weekly assume a conventional
# full-time schedule; they are estimates by nature, which is why the scorer
# treats salary as a soft signal rather than a hard filter.
_ANNUALIZE: dict[SalaryPeriod, float] = {
    SalaryPeriod.YEAR: 1.0,
    SalaryPeriod.MONTH: 12.0,
    SalaryPeriod.WEEK: 52.0,
    SalaryPeriod.DAY: 260.0,    # 52 weeks x 5 days
    SalaryPeriod.HOUR: 2080.0,  # 260 days x 8 hours
}


# ---------------------------------------------------------------------------
# Slug / URL normalization
# ---------------------------------------------------------------------------

_SLUG_STRIP_RE = re.compile(r"[^a-z0-9]+")

# Tracking parameters that change per-referral without changing the posting.
#
# Split into prefix and exact matches deliberately. A blanket prefix check on
# short tokens like "ref" or "source" would also strip `refnum`, `refid`, and
# `sourceId` — which several ATS platforms use as the *job identifier*. That
# would collapse every posting on such a board to a single canonical key, and
# the symptom (one job shown where there were forty) looks like thin provider
# coverage rather than a URL bug.
_TRACKING_PARAM_PREFIXES: tuple[str, ...] = ("utm_", "gh_")
_TRACKING_PARAM_EXACT: frozenset[str] = frozenset(
    {"gclid", "fbclid", "msclkid", "ref", "referrer", "source", "src", "trk", "trackingid"}
)


def slugify(value: str) -> str:
    """
    Lowercase ASCII slug, used to build a fallback dedup key.

    Arabic and other non-Latin text normalizes to empty rather than to
    mojibake; callers must handle an empty slug, which `canonical_key` does by
    falling back to the URL.
    """
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return _SLUG_STRIP_RE.sub("-", ascii_only.lower()).strip("-")


def normalize_url(url: str) -> str:
    """
    Reduce a posting URL to a stable identity.

    Strips scheme differences, a leading ``www.``, the fragment, tracking query
    parameters, and a trailing slash. Query parameters that are *not* tracking
    are kept, because some boards put the job id there (``?jk=<id>`` on Indeed),
    and discarding it would collapse every posting on that board into one key.
    """
    if not url:
        return ""

    parsed = urlparse(url.strip())
    host = (parsed.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]

    path = (parsed.path or "").rstrip("/")

    kept_params = []
    for pair in (parsed.query or "").split("&"):
        if not pair:
            continue
        key = pair.split("=", 1)[0].lower()
        if key in _TRACKING_PARAM_EXACT:
            continue
        if any(key.startswith(prefix) for prefix in _TRACKING_PARAM_PREFIXES):
            continue
        kept_params.append(pair)

    return urlunparse(("", host, path, "", "&".join(sorted(kept_params)), ""))


# ---------------------------------------------------------------------------
# NormalizedJob
# ---------------------------------------------------------------------------


class NormalizedJob(BaseModel):
    """One job posting, provider-agnostic."""

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=False,
        # Not frozen: `raw` holds a dict, and pydantic's frozen models derive
        # __hash__ from all fields, which would raise on any attempt to hash an
        # instance. Treat instances as immutable by convention; use
        # `canonical_key` when a hashable identity is needed.
        frozen=False,
    )

    # ── Provenance ──────────────────────────────────────────────────────────
    provider: str = Field(
        description="Adapter name that produced this record, e.g. 'jsearch'."
    )
    source_id: str = Field(
        description="The provider's own identifier for the posting. Unique "
                    "within a provider, not across providers."
    )

    # ── Core identity ───────────────────────────────────────────────────────
    title: str
    company: str
    apply_url: str

    # ── Location ────────────────────────────────────────────────────────────
    #
    # `location_raw` is kept verbatim alongside the parsed parts because
    # providers disagree wildly on formatting and the raw string is what a user
    # recognizes. The parsed fields are what the scorer compares.
    location_raw: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    work_mode: WorkMode = WorkMode.UNKNOWN

    # ── Timing ──────────────────────────────────────────────────────────────
    #
    # Always timezone-aware UTC. A naive datetime here would raise TypeError the
    # first time the scorer compared it against `datetime.now(timezone.utc)` —
    # at runtime, in the recency component, on whichever provider happened to
    # return a naive timestamp. The validator below makes that impossible.
    posted_at: Optional[datetime] = None

    # ── Compensation ────────────────────────────────────────────────────────
    #
    # `None` means "not stated", which is NOT the same as zero and must never
    # be scored as such. MENA postings omit salary as a matter of course; a
    # scorer that treats absence as a low salary buries the entire region.
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = Field(
        default=None, description="ISO-4217 code, uppercased, e.g. 'EGP'."
    )
    salary_period: Optional[SalaryPeriod] = None

    # ── Classification ──────────────────────────────────────────────────────
    employment_type: EmploymentType = EmploymentType.UNKNOWN
    seniority: Optional[Seniority] = None

    # ── Content ─────────────────────────────────────────────────────────────
    description: str = ""
    required_skills: list[str] = Field(
        default_factory=list,
        description="Canonical skill names from cv_parser._SKILLS_DICTIONARY. "
                    "Sharing that vocabulary is what makes CV-to-posting skill "
                    "overlap a set intersection instead of fuzzy matching.",
    )

    # ── Escape hatch ────────────────────────────────────────────────────────
    raw: dict[str, Any] = Field(
        default_factory=dict,
        repr=False,
        description="The provider's original record. Kept for debugging and "
                    "for backfilling fields we do not parse yet. Strip before "
                    "persisting at scale — it dominates row size.",
    )

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("title", "company", "provider", "source_id")
    @classmethod
    def _require_non_blank(cls, value: str, info) -> str:
        stripped = (value or "").strip()
        if not stripped:
            raise ValueError(f"{info.field_name} must not be blank")
        return stripped

    @field_validator("apply_url")
    @classmethod
    def _require_http_url(cls, value: str) -> str:
        stripped = (value or "").strip()
        parsed = urlparse(stripped)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            # A job the user cannot open is not a job. The prototype emitted
            # "#" and homepage URLs here and let the UI render them as links.
            raise ValueError(f"apply_url must be an absolute http(s) URL, got {value!r}")
        return stripped

    @field_validator("posted_at")
    @classmethod
    def _coerce_to_utc(cls, value: Optional[datetime]) -> Optional[datetime]:
        if value is None:
            return None
        if value.tzinfo is None:
            # Providers that omit an offset are documented as UTC. Assuming UTC
            # is the safe reading: the alternative — assuming server-local —
            # makes recency scoring vary by deployment region.
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @field_validator("salary_currency")
    @classmethod
    def _normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        stripped = value.strip().upper()
        return stripped or None

    @model_validator(mode="after")
    def _reconcile(self) -> "NormalizedJob":
        # Salary bounds arrive reversed from providers that map "up to X" onto
        # the min field. Swap rather than reject: the range is still usable.
        if (
            self.salary_min is not None
            and self.salary_max is not None
            and self.salary_min > self.salary_max
        ):
            self.salary_min, self.salary_max = self.salary_max, self.salary_min

        # A bare number with no period cannot be compared against anything.
        # Default to the most common quoting convention rather than discarding
        # the figure, but only when a figure actually exists.
        if (self.salary_min is not None or self.salary_max is not None) and self.salary_period is None:
            self.salary_period = SalaryPeriod.YEAR

        return self

    # ── Derived values ──────────────────────────────────────────────────────

    @property
    def is_remote(self) -> bool:
        """
        True when the role is fully remote.

        DELIBERATE DEVIATION from the roadmap, which listed `is_remote` as a
        stored field alongside `work_mode`. Two writable fields encoding the
        same fact can contradict each other — `work_mode=ONSITE` with
        `is_remote=True` is representable, and nothing would catch it. Deriving
        it removes that state entirely.

        No expressiveness is lost: `work_mode` already distinguishes "not
        remote" (ONSITE/HYBRID) from "we were not told" (UNKNOWN), which a
        boolean cannot. Adapters map their provider's boolean into `work_mode`.
        """
        return self.work_mode is WorkMode.REMOTE

    @property
    def annual_salary_range(self) -> Optional[tuple[Optional[float], Optional[float]]]:
        """
        Salary annualized in its original currency, or None if unstated.

        Currency conversion is deliberately NOT done here — it needs live rates
        and belongs to the scorer, which can decide how to treat a stale rate.
        This only removes the period mismatch, which is pure arithmetic.
        """
        if self.salary_min is None and self.salary_max is None:
            return None
        multiplier = _ANNUALIZE[self.salary_period or SalaryPeriod.YEAR]
        return (
            self.salary_min * multiplier if self.salary_min is not None else None,
            self.salary_max * multiplier if self.salary_max is not None else None,
        )

    @property
    def age_days(self) -> Optional[float]:
        """Days since posting, or None when the provider gave no date."""
        if self.posted_at is None:
            return None
        return (datetime.now(timezone.utc) - self.posted_at).total_seconds() / 86400.0

    @property
    def canonical_key(self) -> str:
        """
        Stable cross-provider identity, used to dedup the registry fan-out.

        The apply URL is the strongest signal — the same posting aggregated by
        three providers usually carries the same destination URL. When it is
        missing or too generic to identify a posting, fall back to
        company+title+city, which is weaker (it collapses two genuinely
        different openings with the same title at the same company in the same
        city) but errs toward showing the user one row instead of three
        identical ones.
        """
        # A bare host with no path ("https://acme.com/") identifies a company,
        # not a vacancy. `apply_url` is validated to always carry a host, so the
        # normalized string is never empty and testing it as a whole would make
        # this fallback unreachable — what matters is whether anything
        # *identifying* survives normalization. Tracking-only query strings do
        # not count, which is why this inspects the normalized form rather than
        # the original.
        normalized = normalize_url(self.apply_url)
        normalized_parts = urlparse(normalized)
        if (normalized_parts.path or "").strip("/") or normalized_parts.query:
            return f"url:{normalized}"

        parts = [slugify(self.company), slugify(self.title), slugify(self.city or "")]
        fallback = "|".join(part for part in parts if part)
        if fallback:
            return f"cts:{fallback}"

        # Nothing identifying survived normalization (e.g. an Arabic-only title
        # and company with no usable URL). Fall back to provider identity so
        # the record is still unique rather than colliding with every other
        # unidentifiable record.
        return f"src:{self.provider}:{self.source_id}"

    @property
    def field_richness(self) -> int:
        """
        Count of populated high-value fields.

        The registry uses this to choose a winner when two providers return the
        same posting: prefer the record that actually filled in the fields the
        scorer needs, rather than whichever arrived first.
        """
        candidates = (
            self.posted_at,
            self.city,
            self.country,
            self.salary_min,
            self.salary_max,
            self.seniority,
            self.description or None,
            self.required_skills or None,
            self.work_mode if self.work_mode is not WorkMode.UNKNOWN else None,
            self.employment_type if self.employment_type is not EmploymentType.UNKNOWN else None,
        )
        return sum(1 for value in candidates if value is not None)
