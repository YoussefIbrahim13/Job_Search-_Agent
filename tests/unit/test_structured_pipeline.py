"""
Tests for the structured pipeline service and its wiring into the API.

Two questions here, and the second matters more than it looks.

First: does the flag actually route? A flag that silently does nothing is worse
than no flag, because it produces a comparison where both arms are the same
code.

Second: do both paths return the *same shape*? Running old and new side by side
is only informative if the frontend renders them identically — any UI
difference would confound the comparison the flag exists to enable.
"""
from datetime import datetime, timedelta, timezone

import pytest

from backend.core.cache import InMemoryTTLCache
from backend.core.config import get_settings
from backend.ranking.scorer import ScoredJob, score_job
from backend.services import job_search
from backend.sources.criteria import SearchCriteria
from backend.sources.registry import (
    ProviderOutcome,
    ProviderStatus,
    SearchOutcome,
    SourceRegistry,
)
from backend.sources.schema import (
    EmploymentType,
    NormalizedJob,
    SalaryPeriod,
    Seniority,
    WorkMode,
)

NOW = datetime(2026, 8, 11, 12, 0, 0, tzinfo=timezone.utc)

# Keys the existing frontend reads for each job. Any of these disappearing is a
# broken page, not a failed test in the abstract.
LEGACY_JOB_KEYS = {
    "company_name", "job_title", "match_score", "location",
    "experience_needed", "salary_range", "required_skills",
    "match_reason", "source", "application_link",
}
LEGACY_TOP_KEYS = {
    "job_title", "location", "total_found", "agent_summary",
    "search_queries_used", "jobs",
}


def make_job(idx=0, **overrides) -> NormalizedJob:
    base = dict(
        provider="jsearch",
        source_id=str(idx),
        title="Backend Engineer",
        company="Acme",
        apply_url=f"https://wuzzuf.net/jobs/p/{idx}-Backend-Engineer",
        city="Cairo",
        country="EG",
        posted_at=NOW - timedelta(days=2),
        required_skills=["Python", "Django"],
        description="Backend work with Python and Django.",
    )
    base.update(overrides)
    return NormalizedJob(**base)


class FakeSource:
    name = "fake"
    provides_structured_dates = True
    provides_structured_salary = True

    def __init__(self, jobs):
        self._jobs = jobs

    def supports(self, criteria):
        return True

    async def search(self, criteria, limit=None):
        return list(self._jobs)


@pytest.fixture(autouse=True)
def _isolate_registry(monkeypatch):
    """
    The registry is cached across requests so breaker state survives. Tests
    must not inherit each other's provider state.
    """
    monkeypatch.setenv("SEMANTIC_PASS_ENABLED", "false")
    get_settings.cache_clear()
    job_search.reset_registry()
    yield
    job_search.reset_registry()
    get_settings.cache_clear()


def install_registry(jobs):
    registry = SourceRegistry([FakeSource(jobs)], cache=InMemoryTTLCache())
    job_search._registry = registry
    return registry


@pytest.fixture
def criteria():
    return SearchCriteria(
        titles=["Backend Engineer"], locations=["Cairo"], skills=["Python", "Django"]
    )


# ===========================================================================
# Response shape
# ===========================================================================


@pytest.mark.asyncio
async def test_response_carries_every_legacy_key(criteria):
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)

    assert LEGACY_TOP_KEYS <= set(response)
    assert LEGACY_JOB_KEYS <= set(response["jobs"][0])


@pytest.mark.asyncio
async def test_response_is_json_serializable(criteria):
    """FastAPI will encode this; a stray enum or datetime would 500 at runtime."""
    import json

    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)
    json.dumps(response)


@pytest.mark.asyncio
async def test_match_score_is_an_integer_percentage(criteria):
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)
    score = response["jobs"][0]["match_score"]

    assert isinstance(score, int)
    assert 0 <= score <= 100


@pytest.mark.asyncio
async def test_additive_fields_ride_along(criteria):
    """
    Phase 7 makes the breakdown a visible "why did this rank here" feature. It
    is already the fastest way to debug a surprising ordering.
    """
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)
    job = response["jobs"][0]

    assert job["provider"] == "jsearch"
    assert job["score_breakdown"]["components"]
    assert response["pipeline"] == "structured"


# ===========================================================================
# Formatting
# ===========================================================================


@pytest.mark.asyncio
async def test_absent_salary_renders_as_not_specified(criteria):
    """
    Never a zero. The legacy contract uses this literal string and the frontend
    already renders it, so silence stays visibly silence.
    """
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)
    assert response["jobs"][0]["salary_range"] == "Not specified"


@pytest.mark.asyncio
async def test_salary_includes_currency_and_period(criteria):
    """
    The period is the whole point: "EGP 40,000 / month" and "EGP 40,000 / year"
    are wildly different offers, and the prototype rendered both identically.
    """
    install_registry([make_job(0, salary_min=40000, salary_max=60000,
                              salary_currency="EGP",
                              salary_period=SalaryPeriod.MONTH)])
    response = await job_search.search_jobs(criteria)
    salary = response["jobs"][0]["salary_range"]

    assert "EGP" in salary
    assert "40,000" in salary and "60,000" in salary
    assert "month" in salary


@pytest.mark.asyncio
async def test_remote_jobs_are_labelled_remote(criteria):
    install_registry([make_job(0, work_mode=WorkMode.REMOTE, city=None, country=None,
                              location_raw="Worldwide")])
    response = await job_search.search_jobs(criteria)
    assert "Remote" in response["jobs"][0]["location"]


@pytest.mark.asyncio
async def test_internship_is_visible_in_experience(criteria):
    install_registry([make_job(0, seniority=Seniority.INTERN,
                              employment_type=EmploymentType.INTERNSHIP)])
    response = await job_search.search_jobs(criteria)
    assert "internship" in response["jobs"][0]["experience_needed"].lower()


@pytest.mark.asyncio
async def test_source_is_the_board_domain_not_the_adapter(criteria):
    """
    "wuzzuf.net" is what a user recognises. Which adapter fetched it is
    engineering trivia, carried separately as `provider`.
    """
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)

    assert response["jobs"][0]["source"] == "wuzzuf.net"
    assert response["jobs"][0]["provider"] == "jsearch"


# ===========================================================================
# match_reason
# ===========================================================================


@pytest.mark.asyncio
async def test_match_reason_is_populated_without_the_llm(criteria):
    """
    The semantic pass is advisory and degrades on any failure, so the
    deterministic explanation is the normal path — not a fallback that rarely
    runs.
    """
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)
    reason = response["jobs"][0]["match_reason"]

    assert reason
    assert reason != "Not specified"


def test_deterministic_reason_cannot_contradict_the_score(criteria):
    """
    The sentence is derived from the same components the number came from, so
    it cannot claim a match the score did not credit. The prototype's
    `match_reason` was prose written independently of the score, and routinely
    described things the score disagreed with.
    """
    job = make_job(0, city="Berlin", country="DE", required_skills=["Java"],
                   posted_at=NOW - timedelta(days=200))
    item = ScoredJob(job=job, breakdown=score_job(job, criteria, now=NOW))
    reason = job_search._deterministic_reason(item)

    assert "Cairo" not in reason
    assert "Python" not in reason


@pytest.mark.asyncio
async def test_semantic_reason_wins_when_present(criteria):
    install_registry([make_job(0)])
    outcome = SearchOutcome(
        jobs=[make_job(0)],
        providers=[ProviderOutcome("fake", ProviderStatus.OK, job_count=1)],
    )
    item = ScoredJob(job=make_job(0),
                     breakdown=score_job(make_job(0), criteria, now=NOW))
    item.breakdown.semantic_reason = "Same Django job family."

    payload = job_search.job_to_legacy_dict(item)
    assert payload["match_reason"] == "Same Django job family."


# ===========================================================================
# Degradation reporting
# ===========================================================================


@pytest.mark.asyncio
async def test_provider_status_is_reported_in_the_response(criteria):
    """
    A degraded search must be diagnosable from the response alone, not only
    from server logs — the user is the one wondering why results look thin.
    """
    install_registry([make_job(0)])
    response = await job_search.search_jobs(criteria)

    assert response["providers"][0]["provider"] == "fake"
    assert response["providers"][0]["status"] == "ok"
    assert response["degraded"] is False


def test_summary_names_failed_providers(criteria):
    outcome = SearchOutcome(
        jobs=[],
        providers=[
            ProviderOutcome("jsearch", ProviderStatus.ERROR_QUOTA, error="out"),
            ProviderOutcome("remotive", ProviderStatus.OK, job_count=0),
        ],
    )
    summary = job_search._build_summary([], outcome, criteria)

    assert "jsearch" in summary
    assert "incomplete" in summary.lower()


def test_summary_does_not_claim_a_search_that_did_not_happen(criteria):
    """
    The prototype's `agent_summary` was model-authored and described searches
    that never ran — including a date filter the code never applied.
    """
    outcome = SearchOutcome(
        jobs=[], providers=[ProviderOutcome("jsearch", ProviderStatus.SKIPPED_QUOTA)]
    )
    summary = job_search._build_summary([], outcome, criteria)
    assert "no sources" in summary


@pytest.mark.asyncio
async def test_no_results_is_a_successful_empty_response(criteria):
    install_registry([])
    response = await job_search.search_jobs(criteria)

    assert response["jobs"] == []
    assert response["total_found"] == 0
    assert response["agent_summary"]


# ===========================================================================
# Entry points
# ===========================================================================


@pytest.mark.asyncio
async def test_targeted_search_harvests_skills_from_the_title():
    """
    A targeted search has no CV, so the title is the only skill signal. Routing
    it through the shared vocabulary means the skills component compares like
    with like instead of scoring every job neutral.
    """
    install_registry([make_job(0)])
    response = await job_search.run_targeted_search("Python Django Developer", "Cairo")

    assert response["job_title"] == "Python Django Developer"
    skills_component = next(
        c for c in response["jobs"][0]["score_breakdown"]["components"]
        if c["name"] == "skills"
    )
    assert skills_component["applicability"] == "compared"


@pytest.mark.asyncio
async def test_cv_analysis_uses_the_already_parsed_fields():
    """Parsing the same upload twice risks two different answers for one CV."""
    install_registry([make_job(0, seniority=Seniority.SENIOR)])
    response = await job_search.run_cv_analysis(
        "irrelevant raw text",
        detected_title="Backend Engineer",
        preferred_location="Cairo",
        skills=["Python", "Django"],
        seniority=Seniority.SENIOR,
        years_experience=7.0,
    )

    seniority_component = next(
        c for c in response["jobs"][0]["score_breakdown"]["components"]
        if c["name"] == "seniority"
    )
    assert seniority_component["applicability"] == "compared"
    assert seniority_component["score"] == seniority_component["max_score"]


# ===========================================================================
# Route wiring
# ===========================================================================


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from backend.main import app

    return TestClient(app)


def test_flag_off_uses_the_agent(client, monkeypatch):
    """
    A flag that silently does nothing is worse than no flag: the comparison it
    exists to enable would have the same code on both sides.
    """
    monkeypatch.setenv("USE_STRUCTURED_PIPELINE", "false")
    get_settings.cache_clear()

    called = {}

    def fake_agent(job_title, location):
        called["agent"] = True
        return {"job_title": job_title, "location": location, "jobs": [],
                "total_found": 0, "agent_summary": "", "search_queries_used": []}

    monkeypatch.setattr("backend.api.routes.run_targeted_search", fake_agent)

    response = client.post("/api/targeted-search",
                           json={"job_title": "Backend Engineer", "location": "Cairo"})

    assert response.status_code == 200
    assert called.get("agent") is True
    assert "pipeline" not in response.json()


def test_flag_on_uses_the_structured_pipeline(client, monkeypatch):
    monkeypatch.setenv("USE_STRUCTURED_PIPELINE", "true")
    get_settings.cache_clear()
    install_registry([make_job(0)])

    def exploding_agent(*args, **kwargs):
        raise AssertionError("the agent must not run when the flag is on")

    monkeypatch.setattr("backend.api.routes.run_targeted_search", exploding_agent)

    response = client.post("/api/targeted-search",
                           json={"job_title": "Backend Engineer", "location": "Cairo"})

    assert response.status_code == 200
    body = response.json()
    assert body["pipeline"] == "structured"
    assert body["jobs"][0]["company_name"] == "Acme"


def test_both_paths_expose_the_same_job_keys(client, monkeypatch):
    """
    The flag is only informative if the frontend renders both identically —
    otherwise a UI difference confounds the comparison.
    """
    monkeypatch.setenv("USE_STRUCTURED_PIPELINE", "true")
    get_settings.cache_clear()
    install_registry([make_job(0)])

    response = client.post("/api/targeted-search",
                           json={"job_title": "Backend Engineer", "location": "Cairo"})
    structured_keys = set(response.json()["jobs"][0])

    assert LEGACY_JOB_KEYS <= structured_keys


def test_seniority_mapping_degrades_on_an_unknown_label():
    from backend.api.routes import _to_seniority

    assert _to_seniority("senior") is Seniority.SENIOR
    assert _to_seniority("architect") is None
    assert _to_seniority("") is None
    assert _to_seniority(None) is None
