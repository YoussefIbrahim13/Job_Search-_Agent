"""
The search request every source adapter accepts.

FROZEN SCHEMA — see the note at the top of schema.py; the same rules apply.

`SearchCriteria` is the second half of the Phase 2 contract. It is consumed by
every adapter (each translating it into that provider's query dialect), by the
cache (which keys on its normalized form), and by the scorer (which compares
job fields back against the same criteria that produced them).

That last point is the design constraint worth stating explicitly: the object
used to *search* and the object used to *rank* are the same object. The
prototype had no such thing — the query was a string assembled by an LLM and
the ranking criteria lived in a prompt, so nothing guaranteed the jobs were
ranked against what was actually asked for. One object, two consumers, no
opportunity to drift.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.sources.schema import (
    EmploymentType,
    SalaryPeriod,
    Seniority,
    WorkMode,
)

if TYPE_CHECKING:  # pragma: no cover
    from backend.parsers.cv_parser import CandidateProfile


# Default recency window. Deliberately wider than the prototype's Tavily
# `time_range="month"`: that filtered on CRAWL date, which is a different thing
# from posting date and was never a real freshness guarantee. With a structured
# `posted_at` the scorer can decay smoothly instead, so the hard cutoff only
# needs to exclude the genuinely ancient.
DEFAULT_MAX_AGE_DAYS = 45

# Per-provider result ceiling. Free tiers are measured in hundreds of requests
# per month, so this is about controlling cost per call, not page size.
DEFAULT_LIMIT = 25


class SearchCriteria(BaseModel):
    """What the user is looking for, in provider-agnostic terms."""

    model_config = ConfigDict(extra="forbid")

    # ── What ────────────────────────────────────────────────────────────────
    titles: list[str] = Field(
        default_factory=list,
        description="Target job titles, most-preferred first. Adapters that "
                    "accept only one query string use titles[0] and leave the "
                    "rest to the scorer.",
    )
    skills: list[str] = Field(
        default_factory=list,
        description="Canonical skill names, shared vocabulary with "
                    "NormalizedJob.required_skills and cv_parser.",
    )

    # ── Where ───────────────────────────────────────────────────────────────
    locations: list[str] = Field(
        default_factory=list,
        description="Free-text locations as the user gave them ('Cairo', "
                    "'Dubai'). Kept verbatim: the prototype's single largest "
                    "source of empty result sets was dropping or paraphrasing "
                    "this token.",
    )
    work_modes: list[WorkMode] = Field(
        default_factory=list,
        description="Acceptable work modes. Empty means no preference — which "
                    "is NOT the same as listing all of them, since an explicit "
                    "list excludes UNKNOWN postings and an empty one does not.",
    )

    # ── Shape of the role ───────────────────────────────────────────────────
    employment_types: list[EmploymentType] = Field(
        default_factory=list,
        description="Empty means no preference. Listing INTERNSHIP here is "
                    "what replaces the prototype's LLM 'coercion node' — a "
                    "whole agent-graph branch that existed to nag the model "
                    "into running a second internship query.",
    )
    seniority: Optional[Seniority] = None
    years_experience: Optional[float] = None

    # ── Compensation ────────────────────────────────────────────────────────
    salary_min: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[SalaryPeriod] = None

    # ── Filters ─────────────────────────────────────────────────────────────
    max_age_days: int = Field(default=DEFAULT_MAX_AGE_DAYS, ge=1, le=365)
    limit: int = Field(default=DEFAULT_LIMIT, ge=1, le=100)

    # ── Validators ──────────────────────────────────────────────────────────

    @field_validator("titles", "skills", "locations", mode="before")
    @classmethod
    def _clean_string_list(cls, value):
        """
        Strip, drop blanks, and de-duplicate case-insensitively while keeping
        the caller's ordering — order carries preference for `titles`.
        """
        if value is None:
            return []
        if isinstance(value, str):
            value = [value]

        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value:
            text = str(item).strip()
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(text)
        return cleaned

    @field_validator("salary_currency")
    @classmethod
    def _normalize_currency(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return value.strip().upper() or None

    # ── Derived values ──────────────────────────────────────────────────────

    @property
    def remote_ok(self) -> bool:
        """
        True when a fully-remote role would be acceptable.

        Derived rather than stored, for the same reason `NormalizedJob.is_remote`
        is: a separate boolean could contradict `work_modes` and nothing would
        catch it. An empty `work_modes` means no preference, so remote is fine.
        """
        return not self.work_modes or WorkMode.REMOTE in self.work_modes

    @property
    def primary_title(self) -> Optional[str]:
        """The single most-wanted title, for adapters that accept only one."""
        return self.titles[0] if self.titles else None

    @property
    def primary_location(self) -> Optional[str]:
        """The single location to query, for adapters that accept only one."""
        return self.locations[0] if self.locations else None

    def cache_key(self, provider: str) -> str:
        """
        Deterministic cache key for (provider, criteria).

        Built from a canonical JSON form with sorted keys, so two criteria that
        differ only in field ordering — or in the order of a set-like list —
        hit the same cache entry. `titles` and `locations` are NOT sorted: their
        order is semantic (preference, and which one a single-query adapter
        uses), so reordering them is a genuinely different search.

        `limit` is excluded on purpose. A cached 25-result response satisfies a
        later 10-result request, and keying on it would multiply cache misses
        across a quota measured in hundreds of calls per month.
        """
        payload = {
            "titles": self.titles,
            "locations": self.locations,
            "skills": sorted(s.casefold() for s in self.skills),
            "work_modes": sorted(w.value for w in self.work_modes),
            "employment_types": sorted(e.value for e in self.employment_types),
            "seniority": self.seniority.value if self.seniority else None,
            "years_experience": self.years_experience,
            "salary_min": self.salary_min,
            "salary_currency": self.salary_currency,
            "salary_period": self.salary_period.value if self.salary_period else None,
            "max_age_days": self.max_age_days,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
        return f"{provider}:{digest}"

    # ── Construction ────────────────────────────────────────────────────────

    @classmethod
    def from_candidate_profile(
        cls,
        profile: "CandidateProfile",
        *,
        locations: Optional[list[str]] = None,
        work_modes: Optional[list[WorkMode]] = None,
        employment_types: Optional[list[EmploymentType]] = None,
        **overrides,
    ) -> "SearchCriteria":
        """
        Build criteria from a parsed CV.

        This is the bridge from today's `CandidateProfile` to the adapter layer.
        Phase 3 introduces a persisted, user-editable `Profile` and adds a
        `from_profile` alongside this — it does not replace it, because CV-only
        search (upload, get results, no account) stays a supported path.

        Anything the parser could not determine stays `None` here rather than
        being guessed. A wrong `seniority` silently distorts every subsequent
        ranking, whereas a missing one just means that component scores neutral.
        """
        seniority: Optional[Seniority] = None
        if profile.seniority:
            try:
                seniority = Seniority(profile.seniority)
            except ValueError:
                # The parser's vocabulary is meant to match Seniority exactly.
                # If it ever diverges, degrade to "unknown" rather than raising
                # — a CV upload must not 500 because of a new title keyword.
                seniority = None

        return cls(
            titles=[profile.detected_title] if profile.detected_title else [],
            skills=list(profile.skills),
            locations=locations or [],
            work_modes=work_modes or [],
            employment_types=employment_types or [],
            seniority=seniority,
            years_experience=profile.years_experience,
            **overrides,
        )
