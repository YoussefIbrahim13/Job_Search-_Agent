"""
The interface every job source implements.

FROZEN SCHEMA — see the note at the top of schema.py.

Adapters are written in parallel by different people (or in different sessions)
once this file lands, so the contract has to be unambiguous about the parts
that are easy to get subtly wrong: what a failure means, what a capability flag
promises, and who is responsible for filtering.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from backend.sources.criteria import SearchCriteria
from backend.sources.schema import NormalizedJob


class SourceError(Exception):
    """Base class for adapter failures. Never raised directly."""


class ProviderUnavailable(SourceError):
    """
    The provider could not be reached, or returned a server error.

    Transient by assumption. The registry retries per `backend.core.resilience`
    and then falls through to the next adapter.
    """


class ProviderQuotaExceeded(SourceError):
    """
    The provider refused the call for quota or rate-limit reasons.

    Distinct from `ProviderUnavailable` because the response differs: a quota
    failure must trip the circuit breaker for a meaningful cooldown rather than
    being retried, since retrying a 429 against a monthly cap burns the
    remaining budget faster. Phase 6 surfaces this to the user as a specific
    message rather than a generic error.
    """


class ProviderConfigError(SourceError):
    """
    The adapter is missing an API key or is otherwise misconfigured.

    Not transient and not worth retrying. The registry should skip the adapter
    for the whole process lifetime rather than failing every search.
    """


@runtime_checkable
class JobSource(Protocol):
    """
    A source of job postings.

    Implementations live in `backend/sources/<provider>.py`, one per provider.

    RESPONSIBILITY BOUNDARY
    -----------------------
    An adapter's job is translation, not judgement. It converts `SearchCriteria`
    into the provider's query dialect, calls it, and maps the response onto
    `NormalizedJob`. It does not score, rank, filter by relevance, or dedup —
    those happen once, centrally, over the merged result set from all adapters.

    The one exception is the Tavily adapter, which must apply the legacy filter
    chain (`sources/filters.py`) because its input is unstructured search
    results rather than vacancy records. Every other adapter receives data that
    is already known to be a job posting, which is precisely why moving to
    structured providers removes most of the prototype's false-positive surface
    rather than reimplementing it.
    """

    # ── Identity ────────────────────────────────────────────────────────────

    name: str
    """Short lowercase identifier, e.g. 'jsearch'. Used in cache keys, quota
    accounting, logs, and `NormalizedJob.provider`."""

    # ── Capability flags ────────────────────────────────────────────────────
    #
    # These are promises the scorer relies on, not documentation. A source that
    # sets provides_structured_dates=True while returning parsed-from-prose
    # guesses will silently corrupt the recency component for every job it
    # returns, and nothing will look broken.

    provides_structured_dates: bool
    """True only if `posted_at` comes from a real provider timestamp field.
    False for anything inferred from snippet text such as 'Posted 2 weeks ago'."""

    provides_structured_salary: bool
    """True only if salary figures come from numeric provider fields, not from
    a regex over a free-text description."""

    # ── Behaviour ───────────────────────────────────────────────────────────

    def supports(self, criteria: SearchCriteria) -> bool:
        """
        Whether this source can usefully serve these criteria.

        Cheap and synchronous — it gates whether the adapter is called at all,
        so it must not do I/O. Used to skip providers that would waste a quota
        call: a remote-only board asked for onsite Cairo roles, or a provider
        whose country coverage excludes the requested location.

        Returning False is not an error; it means "not my department".
        """
        ...

    async def search(
        self,
        criteria: SearchCriteria,
        limit: int | None = None,
    ) -> list[NormalizedJob]:
        """
        Fetch postings matching `criteria`.

        Args:
            criteria: What to search for.
            limit: Maximum records to return. Falls back to `criteria.limit`.

        Returns:
            Zero or more `NormalizedJob`. An empty list is a valid, successful
            result and must not be raised as an error — "no jobs matched" is
            information, and the registry distinguishes it from "this provider
            is broken" when deciding whether to trip a breaker.

        Raises:
            ProviderUnavailable: transport failure or 5xx.
            ProviderQuotaExceeded: 429 or an explicit quota response.
            ProviderConfigError: missing credentials or invalid configuration.

        Implementations must not raise anything else. An unexpected provider
        response is a `ProviderUnavailable`, not a bare exception — one adapter
        misbehaving must degrade the fan-out, not fail the user's whole search.
        """
        ...
