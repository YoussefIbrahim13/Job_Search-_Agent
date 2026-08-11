"""
Tests for the JSearch adapter.

Network is intercepted with respx; nothing here touches RapidAPI.

Two things get equal weight. The first is field mapping — the failures there
are silent and expensive (a monthly salary read as annual is a 12x error where
both numbers look plausible). The second is failure translation: the registry
decides whether to retry, trip a breaker, or skip a provider based purely on
which exception class comes out, so mapping a 429 to the wrong class means
either burning the remaining monthly budget on retries or abandoning a
provider over a transient blip.
"""
import json
from datetime import timezone
from pathlib import Path

import httpx
import pytest
import respx

from backend.core.config import get_settings
from backend.sources.base import (
    JobSource,
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.jsearch import JSearchSource
from backend.sources.schema import EmploymentType, SalaryPeriod, Seniority, WorkMode

FIXTURE = (
    Path(__file__).parent.parent
    / "fixtures" / "providers" / "jsearch" / "cairo_backend.json"
)
PAYLOAD = json.loads(FIXTURE.read_text(encoding="utf-8"))

API_URL = "https://jsearch.p.rapidapi.com/search"


@pytest.fixture(autouse=True)
def _configure_key(monkeypatch):
    """
    Give the adapter a key for the duration of each test.

    Settings is lru_cached, so the cache is cleared around the patch — without
    that, whichever test ran first would freeze the key for the whole session.
    """
    monkeypatch.setenv("JSEARCH_API_KEY", "test-rapidapi-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def criteria() -> SearchCriteria:
    return SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])


async def run_search(criteria, payload=PAYLOAD, status=200, **kwargs):
    """Drive the adapter against a mocked response."""
    with respx.mock:
        route = respx.get(API_URL).mock(
            return_value=httpx.Response(status, json=payload)
        )
        jobs = await JSearchSource().search(criteria, **kwargs)
    return jobs, route


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


def test_adapter_conforms_to_the_source_protocol():
    assert isinstance(JSearchSource(), JobSource)


def test_capability_flags_claim_only_what_the_provider_gives():
    """
    The scorer trusts these. Claiming structured dates while returning parsed
    prose would silently corrupt the recency component for every job.
    """
    source = JSearchSource()
    assert source.provides_structured_dates is True
    assert source.provides_structured_salary is True
    assert source.name == "jsearch"


# ---------------------------------------------------------------------------
# supports()
# ---------------------------------------------------------------------------


def test_unconfigured_adapter_reports_unsupported(monkeypatch, criteria):
    """
    A missing key is not an error. The app must keep working for someone who
    never signed up for RapidAPI, falling through to the free providers.
    """
    monkeypatch.setenv("JSEARCH_API_KEY", "")
    get_settings.cache_clear()
    assert JSearchSource().supports(criteria) is False


def test_empty_criteria_are_unsupported():
    """
    A blank free-text query returns an arbitrary slice of the global job
    market. Declining costs nothing; calling burns a request to produce noise.
    """
    assert JSearchSource().supports(SearchCriteria()) is False
    assert JSearchSource().supports(SearchCriteria(locations=["Cairo"])) is False


def test_titles_or_skills_are_enough_to_search():
    assert JSearchSource().supports(SearchCriteria(titles=["Backend Engineer"]))
    assert JSearchSource().supports(SearchCriteria(skills=["Python"]))


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_query_combines_title_and_location(criteria):
    _, route = await run_search(criteria)
    sent = dict(httpx.URL(str(route.calls[0].request.url)).params)
    assert sent["query"] == "Backend Engineer in Cairo"


@pytest.mark.asyncio
async def test_query_omits_location_when_none_given():
    _, route = await run_search(SearchCriteria(titles=["Backend Engineer"]))
    sent = dict(httpx.URL(str(route.calls[0].request.url)).params)
    assert sent["query"] == "Backend Engineer"


@pytest.mark.asyncio
async def test_credentials_are_sent_as_rapidapi_headers(criteria):
    _, route = await run_search(criteria)
    headers = route.calls[0].request.headers
    assert headers["x-rapidapi-key"] == "test-rapidapi-key"
    assert headers["x-rapidapi-host"] == "jsearch.p.rapidapi.com"


@pytest.mark.parametrize(
    "max_age_days,expected",
    [(1, "today"), (3, "3days"), (7, "week"), (30, "month"), (45, "all"), (365, "all")],
)
@pytest.mark.asyncio
async def test_max_age_maps_onto_provider_buckets(max_age_days, expected):
    criteria = SearchCriteria(titles=["Backend Engineer"], max_age_days=max_age_days)
    _, route = await run_search(criteria)
    sent = dict(httpx.URL(str(route.calls[0].request.url)).params)
    assert sent["date_posted"] == expected


@pytest.mark.asyncio
async def test_internship_filter_is_sent_as_a_structured_parameter():
    """
    This is what replaces the prototype's LLM 'coercion node' — a whole
    agent-graph branch that existed to nag the model into running a second
    internship query.
    """
    criteria = SearchCriteria(
        titles=["Backend Engineer"],
        employment_types=[EmploymentType.INTERNSHIP],
    )
    _, route = await run_search(criteria)
    sent = dict(httpx.URL(str(route.calls[0].request.url)).params)
    assert sent["employment_types"] == "INTERN"


@pytest.mark.asyncio
async def test_remote_only_filter_is_sent_when_remote_is_the_only_mode():
    criteria = SearchCriteria(titles=["Backend Engineer"], work_modes=[WorkMode.REMOTE])
    _, route = await run_search(criteria)
    sent = dict(httpx.URL(str(route.calls[0].request.url)).params)
    assert sent["remote_jobs_only"] == "true"


@pytest.mark.asyncio
async def test_remote_filter_is_not_sent_when_hybrid_is_also_acceptable():
    """
    Narrowing to remote-only here would silently discard the hybrid results the
    user explicitly said they would accept.
    """
    criteria = SearchCriteria(
        titles=["Backend Engineer"],
        work_modes=[WorkMode.REMOTE, WorkMode.HYBRID],
    )
    _, route = await run_search(criteria)
    sent = dict(httpx.URL(str(route.calls[0].request.url)).params)
    assert "remote_jobs_only" not in sent


# ---------------------------------------------------------------------------
# Record mapping
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unmappable_records_are_skipped_without_failing_the_batch(criteria):
    """
    The fixture holds six records: two are unusable (no apply link, no
    employer). One bad posting must not cost the user the other four.
    """
    jobs, _ = await run_search(criteria)
    assert len(jobs) == 4
    assert all(job.company for job in jobs)
    assert all(job.apply_url.startswith("https://") for job in jobs)


@pytest.mark.asyncio
async def test_core_fields_map_across(criteria):
    jobs, _ = await run_search(criteria)
    job = jobs[0]

    assert job.provider == "jsearch"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Sylndr"
    assert job.city == "Cairo"
    assert job.country == "EG"
    assert job.location_raw == "Cairo, Cairo Governorate, EG"
    assert job.employment_type is EmploymentType.FULL_TIME


@pytest.mark.asyncio
async def test_posted_at_is_parsed_as_aware_utc(criteria):
    """
    The whole point of moving to a structured provider: a real timestamp
    instead of the prose 'Posted 2 weeks ago'.
    """
    jobs, _ = await run_search(criteria)
    posted = jobs[0].posted_at
    assert posted is not None
    assert posted.tzinfo is not None
    assert posted.astimezone(timezone.utc).isoformat().startswith("2026-08-05T09:30")


@pytest.mark.asyncio
async def test_unparseable_date_degrades_to_none_rather_than_dropping_the_job(criteria):
    """
    A bad timestamp costs the job its recency points — the right outcome for a
    record whose freshness cannot be established — but the job itself is still
    a real vacancy and must survive.
    """
    jobs, _ = await run_search(criteria)
    contractor = next(j for j in jobs if j.company == "Instabug")
    assert contractor.posted_at is None
    assert contractor.employment_type is EmploymentType.CONTRACT


@pytest.mark.asyncio
async def test_monthly_salary_keeps_its_period(criteria):
    """
    A 12x error that looks plausible in both directions. The period must
    survive mapping or a Cairo monthly figure silently outranks a US annual.
    """
    jobs, _ = await run_search(criteria)
    job = jobs[0]
    assert (job.salary_min, job.salary_max) == (40000, 60000)
    assert job.salary_currency == "EGP"
    assert job.salary_period is SalaryPeriod.MONTH
    assert job.annual_salary_range == (480000, 720000)


@pytest.mark.asyncio
async def test_zero_salary_is_treated_as_unstated(criteria):
    """
    JSearch returns 0 for 'unspecified'. Zero is not an offer, and scoring it
    as one would rank the job as paying nothing rather than as not saying.
    """
    jobs, _ = await run_search(criteria)
    intern = next(j for j in jobs if j.employment_type is EmploymentType.INTERNSHIP)
    assert intern.salary_min is None
    assert intern.salary_max is None
    assert intern.annual_salary_range is None


@pytest.mark.asyncio
async def test_remote_flag_maps_to_remote_work_mode(criteria):
    jobs, _ = await run_search(criteria)
    remote = next(j for j in jobs if j.company == "Doist")
    assert remote.work_mode is WorkMode.REMOTE
    assert remote.is_remote is True


@pytest.mark.asyncio
async def test_not_remote_maps_to_unknown_not_onsite(criteria):
    """
    JSearch does not distinguish onsite from hybrid, so job_is_remote=false
    means only 'not flagged remote'. Claiming ONSITE would assert something the
    provider never said.
    """
    jobs, _ = await run_search(criteria)
    assert jobs[0].work_mode is WorkMode.UNKNOWN
    assert jobs[0].is_remote is False


@pytest.mark.asyncio
async def test_employment_type_reads_both_provider_spellings(criteria):
    """
    JSearch moved from scalar `job_employment_type` to list
    `job_employment_types`. Handling one spelling only would turn every job
    UNKNOWN after a provider-side rename, with nothing in the logs.
    """
    jobs, _ = await run_search(criteria)
    scalar_form = next(j for j in jobs if j.company == "Sylndr")
    list_form = next(j for j in jobs if j.company == "Doist")
    assert scalar_form.employment_type is EmploymentType.FULL_TIME
    assert list_form.employment_type is EmploymentType.FULL_TIME


@pytest.mark.asyncio
async def test_seniority_prefers_structured_experience_over_title_keywords(criteria):
    """60 months banded to MID, even though the title says 'Senior'."""
    jobs, _ = await run_search(criteria)
    assert jobs[0].seniority is Seniority.MID


@pytest.mark.asyncio
async def test_seniority_falls_back_to_title_when_experience_is_absent(criteria):
    jobs, _ = await run_search(criteria)
    lead = next(j for j in jobs if j.company == "Instabug")
    assert lead.seniority is Seniority.LEAD


@pytest.mark.asyncio
async def test_skills_use_the_shared_cv_vocabulary(criteria):
    """
    Skills must come back in the same canonical form the CV parser produces,
    so overlap is a set intersection rather than fuzzy matching.
    """
    from backend.parsers.cv_parser import _harvest_skills

    jobs, _ = await run_search(criteria)
    skills = jobs[0].required_skills
    assert "Python" in skills
    assert "Django" in skills
    assert set(skills) <= set(_harvest_skills(" ".join(skills)))


@pytest.mark.asyncio
async def test_raw_payload_is_retained(criteria):
    jobs, _ = await run_search(criteria)
    assert jobs[0].raw["job_id"]


@pytest.mark.asyncio
async def test_limit_is_honoured(criteria):
    jobs, _ = await run_search(criteria, limit=2)
    assert len(jobs) == 2


# ---------------------------------------------------------------------------
# Failure translation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_quota_exhaustion_is_distinguishable_from_an_outage(criteria):
    """
    A 429 against a monthly cap must trip the breaker and stop. Retrying it
    burns what is left of the budget faster.
    """
    with pytest.raises(ProviderQuotaExceeded):
        await run_search(criteria, payload={}, status=429)


@pytest.mark.parametrize("status", [401, 403])
@pytest.mark.asyncio
async def test_rejected_key_is_a_config_error(criteria, status):
    """Not transient. Retrying a bad key wastes every remaining request."""
    with pytest.raises(ProviderConfigError):
        await run_search(criteria, payload={}, status=status)


@pytest.mark.parametrize("status", [500, 502, 503])
@pytest.mark.asyncio
async def test_server_errors_are_provider_unavailable(criteria, status):
    with pytest.raises(ProviderUnavailable):
        await run_search(criteria, payload={}, status=status)


@pytest.mark.asyncio
async def test_timeout_is_provider_unavailable(criteria):
    with respx.mock:
        respx.get(API_URL).mock(side_effect=httpx.ConnectTimeout("timed out"))
        with pytest.raises(ProviderUnavailable):
            await JSearchSource().search(criteria)


@pytest.mark.asyncio
async def test_non_json_body_is_provider_unavailable(criteria):
    with respx.mock:
        respx.get(API_URL).mock(
            return_value=httpx.Response(200, text="<html>gateway</html>")
        )
        with pytest.raises(ProviderUnavailable):
            await JSearchSource().search(criteria)


@pytest.mark.asyncio
async def test_unexpected_envelope_shape_is_reported_not_silently_empty(criteria):
    """
    'data' arriving as an object rather than a list means the provider changed
    its envelope. Returning [] would look like 'no jobs matched' and hide it.
    """
    with pytest.raises(ProviderUnavailable, match="payload shape"):
        await run_search(criteria, payload={"status": "OK", "data": {"oops": 1}})


@pytest.mark.asyncio
async def test_empty_result_set_is_success_not_an_error(criteria):
    """
    'No jobs matched' is information. Raising here would make the registry trip
    a breaker on a provider that is working perfectly.
    """
    jobs, _ = await run_search(criteria, payload={"status": "OK", "data": []})
    assert jobs == []


@pytest.mark.asyncio
async def test_search_without_a_key_raises_config_error(monkeypatch, criteria):
    monkeypatch.setenv("JSEARCH_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(ProviderConfigError):
        await JSearchSource().search(criteria)
