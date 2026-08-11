"""
Provider fan-out: run every usable source concurrently, merge, dedup.

WHAT THIS IS RESPONSIBLE FOR
----------------------------
Adapters translate. The registry decides *whether* to call each one, survives
the ones that fail, and turns several providers' answers into a single list.
Everything about cost and failure lives here so no adapter has to know it
exists in a fan-out.

Four mechanisms, each earning its place:

* **Cache** — JSearch's free tier is a few hundred requests per *month*. Two
  users running the same search must not spend twice. A cache hit consumes no
  quota and makes no call.
* **Quota accounting** — checked *before* the call, so it protects the budget
  rather than reporting on its destruction after the fact.
* **Circuit breaker** — a provider that just failed three times in a row will
  almost certainly fail the next one too. Without a breaker every search pays
  the full timeout to rediscover the same outage, and the user waits for it.
* **Per-provider timeout** — bounds one provider's total contribution to
  latency across however many HTTP calls it makes internally.

DEGRADED IS THE NORMAL CASE
---------------------------
With free tiers and unauthenticated endpoints, some provider is usually
unavailable. A search where two of four sources answered is a success, not an
error — so `search` returns a `SearchOutcome` carrying per-provider status
alongside the jobs, rather than raising. Phase 6 renders that as "we couldn't
reach X" instead of a blank page, and Phase 4 stores it for cost attribution.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from enum import Enum
from typing import Optional, Sequence

from backend.core.cache import CacheBackend, get_cache
from backend.core.config import get_settings
from backend.sources.base import (
    JobSource,
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.schema import NormalizedJob

logger = logging.getLogger(__name__)

# Sentinel for sorting: a job with no posting date must lose a recency
# tie-break rather than crash the comparison.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


class ProviderStatus(str, Enum):
    """Why a provider did or did not contribute to a search."""

    OK = "ok"
    CACHED = "cached"
    SKIPPED_UNSUPPORTED = "skipped_unsupported"
    SKIPPED_QUOTA = "skipped_quota"
    SKIPPED_BREAKER = "skipped_breaker"
    SKIPPED_DISABLED = "skipped_disabled"
    ERROR_QUOTA = "error_quota"
    ERROR_UNAVAILABLE = "error_unavailable"
    ERROR_CONFIG = "error_config"
    ERROR_TIMEOUT = "error_timeout"
    ERROR_UNEXPECTED = "error_unexpected"


# Statuses that mean "this provider was asked and answered".
_CONTRIBUTING = frozenset({ProviderStatus.OK, ProviderStatus.CACHED})


@dataclass
class ProviderOutcome:
    """What one provider did during one search."""

    provider: str
    status: ProviderStatus
    job_count: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None

    @property
    def contributed(self) -> bool:
        return self.status in _CONTRIBUTING


@dataclass
class SearchOutcome:
    """Merged result of a fan-out, with per-provider detail."""

    jobs: list[NormalizedJob] = field(default_factory=list)
    providers: list[ProviderOutcome] = field(default_factory=list)
    duplicates_merged: int = 0

    @property
    def contributing_providers(self) -> list[str]:
        return [o.provider for o in self.providers if o.contributed]

    @property
    def failed_providers(self) -> list[str]:
        return [o.provider for o in self.providers if o.status.name.startswith("ERROR")]

    @property
    def any_contributed(self) -> bool:
        return any(o.contributed for o in self.providers)

    @property
    def degraded(self) -> bool:
        """
        True when at least one provider was expected to answer and did not.

        Distinct from `any_contributed`: a search can return plenty of jobs and
        still be degraded, which is worth telling the user because the results
        they are looking at are missing a source they would otherwise have had.
        """
        return bool(self.failed_providers)


@dataclass
class _ProviderState:
    """Per-provider runtime state, persisted across searches."""

    consecutive_failures: int = 0
    open_until: Optional[float] = None      # time.monotonic() deadline
    disabled: bool = False                  # config error: never retry
    calls_today: int = 0
    quota_day: Optional[date] = None

    def breaker_is_open(self) -> bool:
        if self.open_until is None:
            return False
        if time.monotonic() >= self.open_until:
            # Half-open: let the next search through. If it fails the breaker
            # simply re-opens, which is cheaper than tracking a separate
            # half-open state for a fan-out that runs a few times a minute.
            self.open_until = None
            return False
        return True


class SourceRegistry:
    """Runs the configured sources and merges their results."""

    def __init__(
        self,
        sources: Sequence[JobSource],
        *,
        cache: Optional[CacheBackend] = None,
        daily_quotas: Optional[dict[str, int]] = None,
    ) -> None:
        # Order is the priority order: it breaks dedup ties and determines the
        # order of otherwise-equal results.
        self._sources = list(sources)
        self._cache = cache if cache is not None else get_cache()
        self._quotas = dict(daily_quotas or {})
        self._state: dict[str, _ProviderState] = {
            source.name: _ProviderState() for source in self._sources
        }

    # ── Public API ──────────────────────────────────────────────────────────

    async def search(
        self,
        criteria: SearchCriteria,
        limit: int | None = None,
    ) -> SearchOutcome:
        """
        Query every usable provider concurrently and merge the results.

        Never raises on provider failure. A provider that errors contributes a
        `ProviderOutcome` explaining why and nothing else; the user still gets
        whatever the others returned.
        """
        effective_limit = limit or criteria.limit

        tasks = [
            self._run_provider(source, criteria, effective_limit)
            for source in self._sources
        ]
        # return_exceptions is belt-and-braces: _run_provider is written not to
        # raise, but a bug there must degrade one provider rather than take
        # down the whole search.
        results = await asyncio.gather(*tasks, return_exceptions=True)

        outcomes: list[ProviderOutcome] = []
        harvested: list[list[NormalizedJob]] = []

        for source, result in zip(self._sources, results):
            if isinstance(result, BaseException):
                logger.exception(
                    "registry: provider %s raised out of its own handler",
                    source.name,
                    exc_info=result,
                )
                outcomes.append(
                    ProviderOutcome(
                        provider=source.name,
                        status=ProviderStatus.ERROR_UNEXPECTED,
                        error=str(result),
                    )
                )
                continue
            outcome, jobs = result
            outcomes.append(outcome)
            harvested.append(jobs)

        merged, duplicates = self._merge(harvested)

        logger.info(
            "registry: %d job(s) from %s (%d duplicate(s) merged)%s",
            len(merged[:effective_limit]),
            ", ".join(o.provider for o in outcomes if o.contributed) or "no providers",
            duplicates,
            f"; failed: {', '.join(o.provider for o in outcomes if o.status.name.startswith('ERROR'))}"
            if any(o.status.name.startswith("ERROR") for o in outcomes)
            else "",
        )

        return SearchOutcome(
            jobs=merged[:effective_limit],
            providers=outcomes,
            duplicates_merged=duplicates,
        )

    # ── Per-provider execution ──────────────────────────────────────────────

    async def _run_provider(
        self,
        source: JobSource,
        criteria: SearchCriteria,
        limit: int,
    ) -> tuple[ProviderOutcome, list[NormalizedJob]]:
        """
        Run one provider, translating every outcome into a status.

        Deliberately never raises: the fan-out's whole purpose is that one
        provider's problem is not the user's problem.
        """
        settings = get_settings()
        state = self._state.setdefault(source.name, _ProviderState())
        name = source.name

        if state.disabled:
            return ProviderOutcome(name, ProviderStatus.SKIPPED_DISABLED), []

        if not source.supports(criteria):
            return ProviderOutcome(name, ProviderStatus.SKIPPED_UNSUPPORTED), []

        # Cache is checked before the breaker and before quota: a cached answer
        # costs nothing and is still correct while a provider is down.
        cache_key = criteria.cache_key(name)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return (
                ProviderOutcome(name, ProviderStatus.CACHED, job_count=len(cached)),
                list(cached),
            )

        if state.breaker_is_open():
            return ProviderOutcome(name, ProviderStatus.SKIPPED_BREAKER), []

        if not self._has_quota(name, state):
            return ProviderOutcome(name, ProviderStatus.SKIPPED_QUOTA), []

        started = time.monotonic()
        try:
            jobs = await asyncio.wait_for(
                source.search(criteria, limit),
                timeout=settings.source_provider_timeout,
            )
        except asyncio.TimeoutError:
            self._record_failure(state, settings)
            return (
                ProviderOutcome(
                    name,
                    ProviderStatus.ERROR_TIMEOUT,
                    duration_ms=self._elapsed_ms(started),
                    error=f"exceeded {settings.source_provider_timeout}s",
                ),
                [],
            )
        except ProviderQuotaExceeded as exc:
            # Not a failure of the provider — it is working and telling us to
            # stop. Open the breaker for a long cooldown instead of counting it
            # toward the failure threshold, because retrying burns budget.
            state.open_until = time.monotonic() + settings.source_quota_cooldown
            logger.warning("registry: %s quota exhausted: %s", name, exc)
            return (
                ProviderOutcome(
                    name, ProviderStatus.ERROR_QUOTA,
                    duration_ms=self._elapsed_ms(started), error=str(exc),
                ),
                [],
            )
        except ProviderConfigError as exc:
            # Never transient. Retrying a bad key every search wastes latency
            # forever, so this provider is out for the life of the process.
            state.disabled = True
            logger.error("registry: %s disabled for this process: %s", name, exc)
            return (
                ProviderOutcome(
                    name, ProviderStatus.ERROR_CONFIG,
                    duration_ms=self._elapsed_ms(started), error=str(exc),
                ),
                [],
            )
        except ProviderUnavailable as exc:
            self._record_failure(state, settings)
            logger.warning("registry: %s unavailable: %s", name, exc)
            return (
                ProviderOutcome(
                    name, ProviderStatus.ERROR_UNAVAILABLE,
                    duration_ms=self._elapsed_ms(started), error=str(exc),
                ),
                [],
            )
        except Exception as exc:  # noqa: BLE001 - a buggy adapter must not 500 the search
            self._record_failure(state, settings)
            logger.exception("registry: %s raised an unexpected error", name)
            return (
                ProviderOutcome(
                    name, ProviderStatus.ERROR_UNEXPECTED,
                    duration_ms=self._elapsed_ms(started), error=str(exc),
                ),
                [],
            )

        # Success: the provider answered, so forget any earlier failures.
        state.consecutive_failures = 0
        state.open_until = None
        self._consume_quota(name, state)
        self._cache.set(cache_key, list(jobs))

        return (
            ProviderOutcome(
                name, ProviderStatus.OK,
                job_count=len(jobs), duration_ms=self._elapsed_ms(started),
            ),
            list(jobs),
        )

    # ── Quota ───────────────────────────────────────────────────────────────

    def _has_quota(self, name: str, state: _ProviderState) -> bool:
        limit = self._quotas.get(name)
        if limit is None:
            return True  # free / unmetered provider
        self._roll_quota_day(state)
        return state.calls_today < limit

    def _consume_quota(self, name: str, state: _ProviderState) -> None:
        if self._quotas.get(name) is None:
            return
        self._roll_quota_day(state)
        state.calls_today += 1

    @staticmethod
    def _roll_quota_day(state: _ProviderState) -> None:
        """
        Reset the daily counter when the UTC date changes.

        UTC rather than local time so the reset point does not move with the
        deployment region — a counter that resets at a different hour than the
        provider's own accounting is worse than no counter.
        """
        today = datetime.now(timezone.utc).date()
        if state.quota_day != today:
            state.quota_day = today
            state.calls_today = 0

    # ── Breaker ─────────────────────────────────────────────────────────────

    @staticmethod
    def _record_failure(state: _ProviderState, settings) -> None:
        state.consecutive_failures += 1
        if state.consecutive_failures >= settings.source_breaker_threshold:
            state.open_until = time.monotonic() + settings.source_breaker_cooldown

    @staticmethod
    def _elapsed_ms(started: float) -> float:
        return round((time.monotonic() - started) * 1000, 1)

    # ── Merge ───────────────────────────────────────────────────────────────

    def _merge(
        self, harvested: Sequence[Sequence[NormalizedJob]]
    ) -> tuple[list[NormalizedJob], int]:
        """
        Collapse per-provider results into one deduplicated list.

        The same posting aggregated by three providers arrives three times with
        the same `canonical_key`. The winner is the record with the most
        populated fields, because the scorer can only use what is actually
        there — a JSearch record with a real date and salary beats an Arbeitnow
        record for the same job with neither.

        Fields are NOT merged across duplicates. A record assembled from three
        providers would have a `raw` payload matching none of them and a
        provenance that cannot be explained to a user or a debugger. Picking one
        coherent record is worth losing a field or two.

        Order is provider priority, then arrival — deterministic, so the same
        inputs always produce the same list. Actual ranking is the scorer's job.
        """
        best: dict[str, NormalizedJob] = {}
        order: list[str] = []
        duplicates = 0

        for jobs in harvested:
            for job in jobs:
                key = job.canonical_key
                incumbent = best.get(key)
                if incumbent is None:
                    best[key] = job
                    order.append(key)
                    continue

                duplicates += 1
                if self._rank(job) > self._rank(incumbent):
                    best[key] = job

        return [best[key] for key in order], duplicates

    @staticmethod
    def _rank(job: NormalizedJob) -> tuple[int, datetime]:
        """
        Sort key for choosing between duplicate records.

        Richness first, then recency. `posted_at` falls back to the epoch so an
        undated record loses the tie-break instead of raising on a None
        comparison — the kind of crash that would only appear once two
        providers happened to return the same job.
        """
        return (job.field_richness, job.posted_at or _EPOCH)


# ---------------------------------------------------------------------------
# Default assembly
# ---------------------------------------------------------------------------


def build_default_registry(
    *, cache: Optional[CacheBackend] = None
) -> SourceRegistry:
    """
    Assemble the standard provider stack in priority order.

    JSearch first: it is the only source with both real posting dates and
    numeric salaries, so its records win dedup ties on merit and its results
    are the ones the scorer can do the most with. The free providers follow and
    are what keep the app answering once its quota is gone.

    Tavily is last, deliberately. It reads the open web rather than a jobs API,
    so its records carry no posting date and a derived company name — the
    weakest in the system. Ordering it last means that when it returns the same
    posting as a structured provider, the structured record wins the dedup tie
    on field richness rather than on arrival order. It earns its place as the
    only coverage for boards no API indexes, which for MENA is not a rounding
    error.

    Adapters are imported lazily so that importing this module — which the API
    layer does at startup — does not drag in every provider's dependencies.
    """
    from backend.sources.arbeitnow import ArbeitnowSource
    from backend.sources.jsearch import JSearchSource
    from backend.sources.remotive import RemotiveSource
    from backend.sources.tavily import TavilySource

    settings = get_settings()

    return SourceRegistry(
        sources=[
            JSearchSource(),
            RemotiveSource(),
            ArbeitnowSource(),
            TavilySource(),
        ],
        cache=cache,
        # Only metered providers appear here; absence means unmetered.
        daily_quotas={"jsearch": settings.jsearch_daily_quota},
    )
