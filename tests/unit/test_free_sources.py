"""
Tests for the free, unauthenticated adapters: Remotive and Arbeitnow.

These two exist partly to serve users when the paid quota is gone, and partly
to prove `NormalizedJob` generalizes. They differ from JSearch in exactly the
ways that would expose a schema baked around one provider: salary as prose
rather than numbers, no salary field at all, HTML descriptions, a Unix
timestamp instead of ISO, and a board with no server-side query.
"""
import json
from pathlib import Path

import httpx
import pytest
import respx

from backend.sources.arbeitnow import ArbeitnowSource
from backend.sources.base import JobSource, ProviderQuotaExceeded, ProviderUnavailable
from backend.sources.criteria import SearchCriteria
from backend.sources.remotive import RemotiveSource
from backend.sources.schema import (
    EmploymentType,
    SalaryPeriod,
    Seniority,
    WorkMode,
)

FIXTURES = Path(__file__).parent.parent / "fixtures" / "providers"
REMOTIVE_PAYLOAD = json.loads(
    (FIXTURES / "remotive" / "backend_python.json").read_text(encoding="utf-8")
)
ARBEITNOW_PAYLOAD = json.loads(
    (FIXTURES / "arbeitnow" / "page1.json").read_text(encoding="utf-8")
)

REMOTIVE_URL = "https://remotive.com/api/remote-jobs"
ARBEITNOW_URL = "https://www.arbeitnow.com/api/job-board-api"


@pytest.fixture
def criteria() -> SearchCriteria:
    return SearchCriteria(titles=["Backend Engineer"], skills=["Python", "Django"])


# ===========================================================================
# Remotive
# ===========================================================================


async def run_remotive(criteria, payload=REMOTIVE_PAYLOAD, status=200, **kwargs):
    with respx.mock:
        route = respx.get(REMOTIVE_URL).mock(
            return_value=httpx.Response(status, json=payload)
        )
        jobs = await RemotiveSource().search(criteria, **kwargs)
    return jobs, route


def test_remotive_conforms_to_the_protocol():
    assert isinstance(RemotiveSource(), JobSource)


def test_remotive_declares_salary_as_unstructured():
    """
    Salary arrives as prose and is parsed best-effort. The flag is what tells
    the scorer those numbers are inferred — claiming otherwise would present a
    regex guess as a provider guarantee.
    """
    source = RemotiveSource()
    assert source.provides_structured_dates is True
    assert source.provides_structured_salary is False


def test_remotive_declines_when_remote_is_excluded(criteria):
    onsite_only = criteria.model_copy(update={"work_modes": [WorkMode.ONSITE]})
    assert RemotiveSource().supports(onsite_only) is False


def test_remotive_serves_a_location_search_anyway():
    """
    A Cairo-based candidate searching "Cairo" should still see worldwide remote
    roles — those are jobs they can actually take, and for many MENA candidates
    they are the ones that escape the local salary ceiling. Filtering on
    location here would remove the provider's entire value.
    """
    cairo = SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])
    assert RemotiveSource().supports(cairo) is True


@pytest.mark.asyncio
async def test_remotive_maps_core_fields(criteria):
    jobs, _ = await run_remotive(criteria)
    job = jobs[0]

    assert job.provider == "remotive"
    assert job.title == "Senior Backend Engineer"
    assert job.company == "Buffer"
    assert job.employment_type is EmploymentType.FULL_TIME
    assert job.posted_at is not None


@pytest.mark.asyncio
async def test_remotive_listings_are_always_remote(criteria):
    """Every listing on the board is remote by definition."""
    jobs, _ = await run_remotive(criteria)
    assert all(job.work_mode is WorkMode.REMOTE for job in jobs)
    assert all(job.is_remote for job in jobs)


@pytest.mark.asyncio
async def test_remotive_parses_an_annual_salary_range(criteria):
    jobs, _ = await run_remotive(criteria)
    job = jobs[0]
    assert (job.salary_min, job.salary_max) == (120000, 150000)
    assert job.salary_currency == "USD"


@pytest.mark.asyncio
async def test_remotive_parses_an_hourly_rate_with_its_period(criteria):
    """
    Hourly figures are small numbers. A single plausibility floor tuned for
    annual salaries would discard every one of them.
    """
    jobs, _ = await run_remotive(criteria)
    contractor = next(j for j in jobs if j.company == "Toptal")
    assert (contractor.salary_min, contractor.salary_max) == (60, 85)
    assert contractor.salary_period is SalaryPeriod.HOUR


@pytest.mark.asyncio
async def test_remotive_leaves_unparseable_salary_absent():
    """
    "Competitive salary based on experience" carries no figure. Absent scores
    neutral; a guess would actively mis-rank the job.

    Uses criteria matching the platform role's stack, so the keyword gate is
    not what this test ends up measuring.
    """
    platform = SearchCriteria(titles=["Platform Engineer"], skills=["Kubernetes"])
    jobs, _ = await run_remotive(platform)
    vague = next(j for j in jobs if j.company == "Doist")
    assert vague.salary_min is None
    assert vague.salary_max is None


@pytest.mark.asyncio
async def test_remotive_strips_html_from_descriptions(criteria):
    jobs, _ = await run_remotive(criteria)
    description = jobs[0].description
    assert "<" not in description
    assert "&amp;" not in description
    assert "Docker & AWS" in description


@pytest.mark.asyncio
async def test_remotive_harvests_skills_from_tags_and_description(criteria):
    jobs, _ = await run_remotive(criteria)
    skills = jobs[0].required_skills
    assert "Python" in skills
    assert "Django" in skills
    assert "PostgreSQL" in skills


@pytest.mark.asyncio
async def test_remotive_does_not_invent_a_city_from_a_region(criteria):
    """
    "Worldwide" and "Europe" describe where the candidate may live, not where
    the job is. Parsing either into `city` would make the scorer's location
    comparison confidently wrong.
    """
    jobs, _ = await run_remotive(criteria)
    job = jobs[0]
    assert job.location_raw == "Worldwide"
    assert job.city is None
    assert job.country is None


@pytest.mark.asyncio
async def test_remotive_skips_records_without_a_url(criteria):
    jobs, _ = await run_remotive(criteria)
    assert all(job.company != "NoURL Inc" for job in jobs)
    # Three of the five fixture records survive: one has no URL, and the
    # Kubernetes/Go platform role is gated out of a Python/Django search.
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_remotive_gates_irrelevant_jobs_client_side(criteria):
    """
    The bug this fixes was found only by probing the live API. Remotive accepts
    `search` and ignores it, so the adapter was returning the twenty newest
    remote jobs for every query — sales and copywriting roles fed into a Django
    search. Mocked tests could never catch it: a fixture returns whatever it was
    written to return, so a silently-dropped parameter looks like it works.
    """
    jobs, _ = await run_remotive(criteria)
    companies = {job.company for job in jobs}

    assert "Buffer" in companies       # Python/Django
    assert "Doist" not in companies    # Kubernetes/Go platform role


@pytest.mark.asyncio
async def test_remotive_gate_runs_before_the_limit(criteria):
    """
    Truncating first would fill the quota with whatever happened to be newest
    and discard the relevant jobs further down the feed — which is exactly how
    the live failure presented.
    """
    jobs, _ = await run_remotive(criteria, limit=2)
    assert len(jobs) == 2
    assert all("Python" in j.required_skills or "Django" in j.required_skills
               for j in jobs)


@pytest.mark.asyncio
async def test_remotive_maps_internship_type(criteria):
    jobs, _ = await run_remotive(criteria)
    intern = next(j for j in jobs if j.company == "Zapier")
    assert intern.employment_type is EmploymentType.INTERNSHIP


@pytest.mark.asyncio
async def test_remotive_translates_failures(criteria):
    with pytest.raises(ProviderQuotaExceeded):
        await run_remotive(criteria, payload={}, status=429)
    with pytest.raises(ProviderUnavailable):
        await run_remotive(criteria, payload={}, status=503)
    with pytest.raises(ProviderUnavailable, match="payload shape"):
        await run_remotive(criteria, payload={"jobs": "nope"})


@pytest.mark.asyncio
async def test_remotive_empty_feed_is_success(criteria):
    jobs, _ = await run_remotive(criteria, payload={"jobs": []})
    assert jobs == []


# ===========================================================================
# Arbeitnow
# ===========================================================================


async def run_arbeitnow(criteria, payload=ARBEITNOW_PAYLOAD, status=200, **kwargs):
    with respx.mock:
        route = respx.get(ARBEITNOW_URL).mock(
            return_value=httpx.Response(status, json=payload)
        )
        jobs = await ArbeitnowSource().search(criteria, **kwargs)
    return jobs, route


def test_arbeitnow_conforms_to_the_protocol():
    assert isinstance(ArbeitnowSource(), JobSource)


def test_arbeitnow_declares_no_structured_salary():
    """The feed has no salary field at all, so there is nothing to promise."""
    source = ArbeitnowSource()
    assert source.provides_structured_dates is True
    assert source.provides_structured_salary is False


def test_arbeitnow_declines_when_there_is_nothing_to_gate_on():
    """
    With no keywords the gate admits everything and the adapter returns the
    newest page of an unrelated job board. Declining is strictly better.
    """
    assert ArbeitnowSource().supports(SearchCriteria()) is False
    assert ArbeitnowSource().supports(SearchCriteria(locations=["Berlin"])) is False


def test_arbeitnow_generic_title_words_alone_do_not_qualify():
    """
    "Senior Engineer" is entirely stopwords. Gating on it would admit the whole
    board, which is the failure the stopword list exists to prevent.
    """
    assert ArbeitnowSource().supports(SearchCriteria(titles=["Senior Engineer"])) is False
    assert ArbeitnowSource().supports(SearchCriteria(titles=["Backend Engineer"])) is True


@pytest.mark.asyncio
async def test_arbeitnow_gate_keeps_relevant_and_drops_unrelated(criteria):
    """
    The marketing role must not survive a Python search. This is subject-matter
    filtering standing in for a `search` parameter the provider does not offer
    — not relevance scoring, which stays with the scorer.
    """
    jobs, _ = await run_arbeitnow(criteria)
    companies = {job.company for job in jobs}

    assert "Zalando" in companies
    assert "Hellofresh" in companies
    assert "Beiersdorf" not in companies


@pytest.mark.asyncio
async def test_arbeitnow_parses_unix_timestamps(criteria):
    jobs, _ = await run_arbeitnow(criteria)
    job = next(j for j in jobs if j.company == "Zalando")
    assert job.posted_at is not None
    assert job.posted_at.tzinfo is not None
    assert job.posted_at.year == 2025


@pytest.mark.asyncio
async def test_arbeitnow_zero_timestamp_is_treated_as_unset(criteria):
    """
    Epoch zero means "unset" in practice, never 1970. Keeping it would date the
    job 55 years ago and zero its recency score — the job survives, undated.
    """
    jobs, _ = await run_arbeitnow(criteria)
    sap = next(j for j in jobs if j.company == "SAP")
    assert sap.posted_at is None


@pytest.mark.asyncio
async def test_arbeitnow_never_invents_salary(criteria):
    jobs, _ = await run_arbeitnow(criteria)
    assert all(job.salary_min is None and job.salary_max is None for job in jobs)


@pytest.mark.asyncio
async def test_arbeitnow_maps_remote_flag_and_job_types_list(criteria):
    jobs, _ = await run_arbeitnow(criteria)
    remote = next(j for j in jobs if j.company == "Hellofresh")
    onsite = next(j for j in jobs if j.company == "Zalando")

    assert remote.work_mode is WorkMode.REMOTE
    # Not flagged remote means "not told", not "onsite".
    assert onsite.work_mode is WorkMode.UNKNOWN
    assert onsite.employment_type is EmploymentType.FULL_TIME


@pytest.mark.asyncio
async def test_arbeitnow_maps_internship_from_job_types(criteria):
    jobs, _ = await run_arbeitnow(criteria)
    student = next(j for j in jobs if j.company == "Celonis")
    assert student.employment_type is EmploymentType.INTERNSHIP


@pytest.mark.asyncio
async def test_arbeitnow_skips_records_without_a_company(criteria):
    jobs, _ = await run_arbeitnow(criteria)
    assert all(job.company for job in jobs)
    assert len(jobs) == 4


@pytest.mark.asyncio
async def test_arbeitnow_deduplicates_across_pages(criteria):
    """
    The feed is recency-ordered, so the same posting can appear on two pages as
    new jobs shift the window. Without the guard it would be returned twice.
    """
    jobs, route = await run_arbeitnow(criteria)
    urls = [job.apply_url for job in jobs]
    assert len(urls) == len(set(urls))
    # The same page payload is served for every request, so any duplicate
    # would have to come from page walking.
    assert route.call_count >= 1


@pytest.mark.asyncio
async def test_arbeitnow_stops_walking_pages_once_the_limit_is_met(criteria):
    jobs, route = await run_arbeitnow(criteria, limit=2)
    assert len(jobs) == 2
    assert route.call_count == 1


@pytest.mark.asyncio
async def test_arbeitnow_translates_failures(criteria):
    with pytest.raises(ProviderQuotaExceeded):
        await run_arbeitnow(criteria, payload={}, status=429)
    with pytest.raises(ProviderUnavailable):
        await run_arbeitnow(criteria, payload={}, status=500)
    with pytest.raises(ProviderUnavailable, match="payload shape"):
        await run_arbeitnow(criteria, payload={"data": {"oops": 1}})


@pytest.mark.asyncio
async def test_arbeitnow_empty_feed_is_success(criteria):
    jobs, _ = await run_arbeitnow(criteria, payload={"data": []})
    assert jobs == []


# ===========================================================================
# Cross-provider
# ===========================================================================


@pytest.mark.asyncio
async def test_all_three_providers_emit_the_same_shape(criteria):
    """
    The point of adding two more adapters before building the registry: prove
    the schema was not quietly designed around JSearch alone. Three providers
    with genuinely different response formats must be indistinguishable
    downstream apart from `provider` and the capability flags.
    """
    remotive_jobs, _ = await run_remotive(criteria)
    arbeitnow_jobs, _ = await run_arbeitnow(criteria)

    for job in remotive_jobs + arbeitnow_jobs:
        assert job.canonical_key
        assert job.apply_url.startswith("https://")
        assert job.title and job.company
        assert isinstance(job.required_skills, list)
        # age_days must be computable or explicitly unknown — never a crash,
        # which is what a naive datetime would cause inside the scorer.
        assert job.age_days is None or job.age_days >= 0


@pytest.mark.asyncio
async def test_providers_are_distinguishable_by_name(criteria):
    remotive_jobs, _ = await run_remotive(criteria)
    arbeitnow_jobs, _ = await run_arbeitnow(criteria)
    assert {j.provider for j in remotive_jobs} == {"remotive"}
    assert {j.provider for j in arbeitnow_jobs} == {"arbeitnow"}
