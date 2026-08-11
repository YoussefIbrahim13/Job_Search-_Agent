"""
Tests for the provider fan-out registry.

Driven with fake sources rather than HTTP: what is under test is scheduling,
cost control, and failure survival, none of which involve transport.

The bias here is toward the failure paths. With free tiers and unauthenticated
endpoints, some provider being unavailable is the *normal* case, so the
behaviour that matters most is what happens when things go wrong — one provider
failing must never cost the user the others, and a provider that is down must
not be re-tried on every search at the cost of the full timeout each time.
"""
import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from backend.core.cache import InMemoryTTLCache
from backend.core.config import get_settings
from backend.sources.base import (
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.registry import (
    ProviderStatus,
    SourceRegistry,
    build_default_registry,
)
from backend.sources.schema import NormalizedJob, SalaryPeriod, WorkMode


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_job(provider="fake", url="https://example.com/jobs/1", **overrides):
    base = dict(
        provider=provider,
        source_id=url,
        title="Backend Engineer",
        company="Acme",
        apply_url=url,
    )
    base.update(overrides)
    return NormalizedJob(**base)


class FakeSource:
    """Configurable adapter conforming to the JobSource protocol."""

    provides_structured_dates = True
    provides_structured_salary = True

    def __init__(
        self,
        name="fake",
        jobs=None,
        error=None,
        supported=True,
        delay=0.0,
    ):
        self.name = name
        self._jobs = jobs if jobs is not None else []
        self._error = error
        self._supported = supported
        self._delay = delay
        self.call_count = 0

    def supports(self, criteria) -> bool:
        return self._supported

    async def search(self, criteria, limit=None):
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._error is not None:
            raise self._error
        return list(self._jobs)


@pytest.fixture
def criteria():
    return SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])


@pytest.fixture
def cache():
    return InMemoryTTLCache()


@pytest.fixture(autouse=True)
def _fast_breaker(monkeypatch):
    """Short cooldowns so breaker transitions are testable without long sleeps."""
    monkeypatch.setenv("SOURCE_BREAKER_THRESHOLD", "2")
    monkeypatch.setenv("SOURCE_BREAKER_COOLDOWN", "0.05")
    monkeypatch.setenv("SOURCE_QUOTA_COOLDOWN", "0.05")
    monkeypatch.setenv("SOURCE_PROVIDER_TIMEOUT", "0.5")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Fan-out
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_results_from_every_provider_are_merged(criteria, cache):
    a = FakeSource("a", [make_job("a", "https://a.com/jobs/1")])
    b = FakeSource("b", [make_job("b", "https://b.com/jobs/2")])

    outcome = await SourceRegistry([a, b], cache=cache).search(criteria)

    assert len(outcome.jobs) == 2
    assert set(outcome.contributing_providers) == {"a", "b"}


@pytest.mark.asyncio
async def test_providers_run_concurrently_not_serially(criteria, cache):
    """
    Three providers at 0.15s each must take ~0.15s total, not ~0.45s. Serial
    fan-out would put provider latency on the critical path additively, which
    on a real search is the difference between seconds and tens of seconds.
    """
    sources = [FakeSource(f"s{i}", [make_job(f"s{i}", f"https://s{i}.com/j")], delay=0.15)
               for i in range(3)]

    started = time.monotonic()
    outcome = await SourceRegistry(sources, cache=cache).search(criteria)
    elapsed = time.monotonic() - started

    assert len(outcome.jobs) == 3
    assert elapsed < 0.35, f"fan-out took {elapsed:.2f}s — providers ran serially"


@pytest.mark.asyncio
async def test_unsupported_providers_are_never_called(criteria, cache):
    """A skipped provider costs nothing: no call, no quota, no latency."""
    unsupported = FakeSource("no", [make_job()], supported=False)

    outcome = await SourceRegistry([unsupported], cache=cache).search(criteria)

    assert unsupported.call_count == 0
    assert outcome.providers[0].status is ProviderStatus.SKIPPED_UNSUPPORTED


@pytest.mark.asyncio
async def test_limit_is_applied_to_the_merged_list(criteria, cache):
    a = FakeSource("a", [make_job("a", f"https://a.com/jobs/{i}") for i in range(5)])
    b = FakeSource("b", [make_job("b", f"https://b.com/jobs/{i}") for i in range(5)])

    outcome = await SourceRegistry([a, b], cache=cache).search(criteria, limit=3)
    assert len(outcome.jobs) == 3


# ---------------------------------------------------------------------------
# Failure isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_one_failing_provider_does_not_cost_the_user_the_others(criteria, cache):
    """The entire reason the fan-out exists."""
    good = FakeSource("good", [make_job("good", "https://good.com/j/1")])
    bad = FakeSource("bad", error=ProviderUnavailable("boom"))

    outcome = await SourceRegistry([good, bad], cache=cache).search(criteria)

    assert len(outcome.jobs) == 1
    assert outcome.contributing_providers == ["good"]
    assert outcome.failed_providers == ["bad"]
    assert outcome.degraded is True


@pytest.mark.asyncio
async def test_an_adapter_bug_is_contained(criteria, cache):
    """
    An adapter raising something outside the SourceError hierarchy is a bug,
    not a provider outage — but it still must not 500 the user's search.
    """
    good = FakeSource("good", [make_job("good", "https://good.com/j/1")])
    buggy = FakeSource("buggy", error=TypeError("adapter bug"))

    outcome = await SourceRegistry([good, buggy], cache=cache).search(criteria)

    assert len(outcome.jobs) == 1
    statuses = {o.provider: o.status for o in outcome.providers}
    assert statuses["buggy"] is ProviderStatus.ERROR_UNEXPECTED


@pytest.mark.asyncio
async def test_slow_provider_is_cut_off(criteria, cache):
    """
    A provider that hangs must not hold the search. Timeout is 0.5s here; the
    slow source would take 5s.
    """
    good = FakeSource("good", [make_job("good", "https://good.com/j/1")])
    slow = FakeSource("slow", [make_job("slow", "https://slow.com/j/1")], delay=5.0)

    started = time.monotonic()
    outcome = await SourceRegistry([good, slow], cache=cache).search(criteria)
    elapsed = time.monotonic() - started

    assert elapsed < 2.0
    statuses = {o.provider: o.status for o in outcome.providers}
    assert statuses["slow"] is ProviderStatus.ERROR_TIMEOUT
    assert len(outcome.jobs) == 1


@pytest.mark.asyncio
async def test_empty_results_are_success_not_failure(criteria, cache):
    empty = FakeSource("empty", [])
    outcome = await SourceRegistry([empty], cache=cache).search(criteria)

    assert outcome.jobs == []
    assert outcome.providers[0].status is ProviderStatus.OK
    assert outcome.degraded is False


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_breaker_opens_after_repeated_failures(criteria, cache):
    """
    Threshold is 2 here. After it trips, the provider is not called again —
    otherwise every search pays the full timeout to rediscover the same outage.
    """
    bad = FakeSource("bad", error=ProviderUnavailable("down"))
    registry = SourceRegistry([bad], cache=cache)

    await registry.search(criteria)
    await registry.search(criteria)
    assert bad.call_count == 2

    outcome = await registry.search(criteria)
    assert bad.call_count == 2, "breaker did not stop the third call"
    assert outcome.providers[0].status is ProviderStatus.SKIPPED_BREAKER


@pytest.mark.asyncio
async def test_breaker_closes_after_its_cooldown(criteria, cache):
    bad = FakeSource("bad", error=ProviderUnavailable("down"))
    registry = SourceRegistry([bad], cache=cache)

    await registry.search(criteria)
    await registry.search(criteria)
    await registry.search(criteria)  # skipped
    assert bad.call_count == 2

    await asyncio.sleep(0.08)  # cooldown is 0.05s

    await registry.search(criteria)
    assert bad.call_count == 3, "breaker never re-tried the provider"


@pytest.mark.asyncio
async def test_success_resets_the_failure_count(criteria, cache):
    """
    Otherwise failures accumulate across unrelated searches and a provider that
    is fine trips the breaker after enough scattered blips.
    """
    source = FakeSource("flaky", error=ProviderUnavailable("blip"))
    registry = SourceRegistry([source], cache=cache)

    await registry.search(criteria)          # failure 1 of 2
    source._error = None
    source._jobs = [make_job("flaky", "https://flaky.com/j/1")]
    await registry.search(criteria)          # success -> should reset to 0

    source._error = ProviderUnavailable("blip")
    source._jobs = []
    cache.clear()
    await registry.search(criteria)          # failure: 1 if reset, else 2

    # The fourth search is what discriminates. Up to here the call count is 3
    # either way — without the reset the counter would have reached the
    # threshold on the previous line and the breaker would now be open, so this
    # call would be skipped.
    cache.clear()
    outcome = await registry.search(criteria)

    assert source.call_count == 4, "breaker opened despite an intervening success"
    assert outcome.providers[0].status is not ProviderStatus.SKIPPED_BREAKER


@pytest.mark.asyncio
async def test_quota_refusal_opens_the_breaker_immediately(criteria, cache):
    """
    A 429 is the provider working correctly and saying stop. It must not be
    retried on the next search, and it must not need to happen `threshold`
    times first — each retry burns budget that is already exhausted.
    """
    source = FakeSource("metered", error=ProviderQuotaExceeded("out of quota"))
    registry = SourceRegistry([source], cache=cache)

    first = await registry.search(criteria)
    assert first.providers[0].status is ProviderStatus.ERROR_QUOTA

    second = await registry.search(criteria)
    assert second.providers[0].status is ProviderStatus.SKIPPED_BREAKER
    assert source.call_count == 1


@pytest.mark.asyncio
async def test_config_error_disables_the_provider_permanently(criteria, cache):
    """
    A bad API key will still be bad in five minutes. Retrying it every search
    costs latency forever and can never succeed.
    """
    source = FakeSource("misconfigured", error=ProviderConfigError("no key"))
    registry = SourceRegistry([source], cache=cache)

    first = await registry.search(criteria)
    assert first.providers[0].status is ProviderStatus.ERROR_CONFIG

    await asyncio.sleep(0.08)  # longer than any cooldown

    second = await registry.search(criteria)
    assert second.providers[0].status is ProviderStatus.SKIPPED_DISABLED
    assert source.call_count == 1


# ---------------------------------------------------------------------------
# Cache and quota
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_second_identical_search_is_served_from_cache(criteria, cache):
    source = FakeSource("a", [make_job("a", "https://a.com/j/1")])
    registry = SourceRegistry([source], cache=cache)

    await registry.search(criteria)
    outcome = await registry.search(criteria)

    assert source.call_count == 1
    assert outcome.providers[0].status is ProviderStatus.CACHED
    assert len(outcome.jobs) == 1


@pytest.mark.asyncio
async def test_different_criteria_do_not_share_a_cache_entry(cache):
    source = FakeSource("a", [make_job("a", "https://a.com/j/1")])
    registry = SourceRegistry([source], cache=cache)

    await registry.search(SearchCriteria(titles=["Backend Engineer"]))
    await registry.search(SearchCriteria(titles=["Data Engineer"]))

    assert source.call_count == 2


@pytest.mark.asyncio
async def test_cache_hits_do_not_consume_quota(criteria, cache):
    """
    The point of caching against a few-hundred-per-month tier: a repeat search
    must cost nothing at all.
    """
    source = FakeSource("metered", [make_job("metered", "https://m.com/j/1")])
    registry = SourceRegistry([source], cache=cache, daily_quotas={"metered": 2})

    first = SearchCriteria(titles=["Backend Engineer"])
    second = SearchCriteria(titles=["Data Engineer"])

    await registry.search(first)              # consumes 1 of 2
    cached = await registry.search(first)     # must consume 0
    assert cached.providers[0].status is ProviderStatus.CACHED

    # The discriminating step: a second distinct search must still fit inside
    # the quota. If the cache hit had consumed a call, this would be skipped.
    fresh = await registry.search(second)

    assert fresh.providers[0].status is ProviderStatus.OK, (
        "the cache hit consumed quota, so a genuinely new search was refused"
    )
    assert source.call_count == 2


@pytest.mark.asyncio
async def test_quota_is_enforced_before_the_call(cache):
    """
    Checked before dispatch, so it protects the budget rather than reporting on
    its destruction afterwards.
    """
    source = FakeSource("metered", [make_job("metered", "https://m.com/j/1")])
    registry = SourceRegistry([source], cache=cache, daily_quotas={"metered": 2})

    for i in range(3):
        # Distinct criteria so the cache never serves these.
        await registry.search(SearchCriteria(titles=[f"Role {i}"]))

    assert source.call_count == 2
    outcome = await registry.search(SearchCriteria(titles=["Role 99"]))
    assert outcome.providers[0].status is ProviderStatus.SKIPPED_QUOTA


@pytest.mark.asyncio
async def test_unmetered_providers_have_no_quota_ceiling(cache):
    source = FakeSource("free", [make_job("free", "https://f.com/j/1")])
    registry = SourceRegistry([source], cache=cache)  # no quota entry

    for i in range(10):
        await registry.search(SearchCriteria(titles=[f"Role {i}"]))

    assert source.call_count == 10


@pytest.mark.asyncio
async def test_failed_calls_are_not_cached(criteria, cache):
    """Caching an error would extend one outage across the whole TTL."""
    source = FakeSource("bad", error=ProviderUnavailable("down"))
    registry = SourceRegistry([source], cache=cache)

    await registry.search(criteria)
    assert cache.size == 0


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_same_posting_from_two_providers_appears_once(criteria, cache):
    url = "https://boards.greenhouse.io/acme/jobs/4019283"
    a = FakeSource("a", [make_job("a", url)])
    b = FakeSource("b", [make_job("b", url + "?utm_source=x")])

    outcome = await SourceRegistry([a, b], cache=cache).search(criteria)

    assert len(outcome.jobs) == 1
    assert outcome.duplicates_merged == 1


@pytest.mark.asyncio
async def test_richer_record_wins_a_duplicate(criteria, cache):
    """
    The scorer can only use fields that are actually present, so the record
    with a real date and salary must survive over a bare one — regardless of
    which provider happened to be listed first.
    """
    url = "https://boards.greenhouse.io/acme/jobs/1"
    sparse = FakeSource("sparse", [make_job("sparse", url)])
    rich = FakeSource("rich", [make_job(
        "rich", url,
        posted_at=datetime.now(timezone.utc),
        city="Cairo",
        salary_min=40000, salary_max=60000, salary_period=SalaryPeriod.MONTH,
        work_mode=WorkMode.REMOTE,
    )])

    outcome = await SourceRegistry([sparse, rich], cache=cache).search(criteria)

    assert len(outcome.jobs) == 1
    assert outcome.jobs[0].provider == "rich"


@pytest.mark.asyncio
async def test_recency_breaks_a_richness_tie(criteria, cache):
    url = "https://boards.greenhouse.io/acme/jobs/1"
    now = datetime.now(timezone.utc)
    older = FakeSource("older", [make_job("older", url, posted_at=now - timedelta(days=10))])
    newer = FakeSource("newer", [make_job("newer", url, posted_at=now)])

    outcome = await SourceRegistry([older, newer], cache=cache).search(criteria)
    assert outcome.jobs[0].provider == "newer"


@pytest.mark.asyncio
async def test_undated_duplicate_loses_without_raising(criteria, cache):
    """
    Comparing a datetime against None would raise — a crash that only appears
    once two providers happen to return the same job, which is exactly the case
    dedup exists for.
    """
    url = "https://boards.greenhouse.io/acme/jobs/1"
    undated = FakeSource("undated", [make_job("undated", url)])
    dated = FakeSource("dated", [make_job("dated", url, posted_at=datetime.now(timezone.utc))])

    outcome = await SourceRegistry([undated, dated], cache=cache).search(criteria)
    assert outcome.jobs[0].provider == "dated"


@pytest.mark.asyncio
async def test_duplicate_fields_are_not_frankenstiened(criteria, cache):
    """
    The winner is one coherent record, not a blend. A merged record would carry
    a `raw` payload matching no provider and a provenance nobody could explain.
    """
    url = "https://boards.greenhouse.io/acme/jobs/1"
    with_salary = FakeSource("sal", [make_job(
        "sal", url, salary_min=1000, salary_max=2000, salary_period=SalaryPeriod.MONTH
    )])
    with_date = FakeSource("dat", [make_job(
        "dat", url, posted_at=datetime.now(timezone.utc), city="Cairo", country="EG",
        work_mode=WorkMode.REMOTE, description="x", required_skills=["Python"],
    )])

    outcome = await SourceRegistry([with_salary, with_date], cache=cache).search(criteria)
    winner = outcome.jobs[0]

    assert winner.provider == "dat"
    assert winner.salary_min is None, "fields were merged across providers"


@pytest.mark.asyncio
async def test_merge_order_is_deterministic(criteria, cache):
    a = FakeSource("a", [make_job("a", f"https://a.com/j/{i}") for i in range(3)])
    b = FakeSource("b", [make_job("b", f"https://b.com/j/{i}") for i in range(3)])

    first = await SourceRegistry([a, b], cache=InMemoryTTLCache()).search(criteria)
    second = await SourceRegistry([a, b], cache=InMemoryTTLCache()).search(criteria)

    assert [j.apply_url for j in first.jobs] == [j.apply_url for j in second.jobs]


# ---------------------------------------------------------------------------
# Outcome reporting
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_outcome_reports_per_provider_detail(criteria, cache):
    good = FakeSource("good", [make_job("good", "https://good.com/j/1")])
    bad = FakeSource("bad", error=ProviderUnavailable("down"))
    skipped = FakeSource("skipped", supported=False)

    outcome = await SourceRegistry([good, bad, skipped], cache=cache).search(criteria)

    by_name = {o.provider: o for o in outcome.providers}
    assert by_name["good"].status is ProviderStatus.OK
    assert by_name["good"].job_count == 1
    assert by_name["bad"].error
    assert by_name["skipped"].status is ProviderStatus.SKIPPED_UNSUPPORTED
    assert len(outcome.providers) == 3


@pytest.mark.asyncio
async def test_all_providers_failing_returns_empty_rather_than_raising(criteria, cache):
    """
    A total outage is still an answer the API layer can turn into a useful
    message. Raising here would surface as a 500 for a condition the user can
    do nothing about and that is not a bug.
    """
    sources = [FakeSource(f"s{i}", error=ProviderUnavailable("down")) for i in range(3)]
    outcome = await SourceRegistry(sources, cache=cache).search(criteria)

    assert outcome.jobs == []
    assert outcome.any_contributed is False
    assert len(outcome.failed_providers) == 3


# ---------------------------------------------------------------------------
# Default assembly
# ---------------------------------------------------------------------------


def test_default_registry_prioritises_the_structured_provider():
    """
    JSearch first: it is the only source with both real dates and numeric
    salaries, so its records win dedup ties on merit.
    """
    registry = build_default_registry(cache=InMemoryTTLCache())
    names = [s.name for s in registry._sources]

    assert names[0] == "jsearch"
    assert set(names) == {"jsearch", "remotive", "arbeitnow", "tavily"}


def test_default_registry_puts_the_web_search_provider_last():
    """
    Tavily's records have no posting date and a derived company name — the
    weakest in the system. Ordering it last means a structured provider wins
    the dedup tie when both return the same posting.
    """
    registry = build_default_registry(cache=InMemoryTTLCache())
    names = [s.name for s in registry._sources]
    assert names[-1] == "tavily"


def test_only_metered_providers_have_quotas():
    registry = build_default_registry(cache=InMemoryTTLCache())
    assert "jsearch" in registry._quotas
    assert "remotive" not in registry._quotas
    assert "arbeitnow" not in registry._quotas
