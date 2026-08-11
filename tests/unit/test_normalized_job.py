"""
Contract tests for `NormalizedJob`.

These pin the frozen schema. Phases 3, 4, 6 and 7 all build against this shape,
so a change that breaks one of these is a change that breaks a migration, the
scorer, and the frontend simultaneously — the failure should happen here first.

The emphasis is on the properties that are expensive to discover late: timezone
handling (a naive datetime blows up inside the scorer, not at construction),
salary period (a 12x error that looks plausible), and dedup key stability
(silently shows the user three copies of one job, or collapses two real ones).
"""
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from backend.sources.schema import (
    EmploymentType,
    NormalizedJob,
    SalaryPeriod,
    Seniority,
    WorkMode,
    normalize_url,
    slugify,
)


def make_job(**overrides) -> NormalizedJob:
    """Minimal valid job, overridable per-test."""
    base = {
        "provider": "jsearch",
        "source_id": "abc123",
        "title": "Backend Engineer",
        "company": "Acme",
        "apply_url": "https://boards.greenhouse.io/acme/jobs/4019283",
    }
    base.update(overrides)
    return NormalizedJob(**base)


# ---------------------------------------------------------------------------
# Required fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field", ["provider", "source_id", "title", "company"])
def test_blank_required_fields_are_rejected(field):
    """
    Whitespace-only is as useless as empty. The prototype's placeholder problem
    ("Company Name", "") started as values that were technically present.
    """
    with pytest.raises(ValidationError):
        make_job(**{field: "   "})


@pytest.mark.parametrize(
    "bad_url",
    [
        "#",                       # the prototype emitted this as a real link
        "",
        "not-a-url",
        "/jobs/view/123",          # relative
        "ftp://example.com/job",   # non-http scheme
        "javascript:alert(1)",
    ],
)
def test_unusable_apply_urls_are_rejected(bad_url):
    """A job the user cannot open is not a job."""
    with pytest.raises(ValidationError):
        make_job(apply_url=bad_url)


def test_unknown_fields_are_rejected():
    """
    extra='forbid' turns an adapter typo into an immediate failure rather than
    a silently-dropped field that the scorer then reads as None.
    """
    with pytest.raises(ValidationError):
        make_job(job_title="Backend Engineer")


# ---------------------------------------------------------------------------
# Timezone handling
# ---------------------------------------------------------------------------


def test_naive_posted_at_is_coerced_to_utc():
    """
    A naive datetime would raise TypeError the first time the scorer compared
    it against an aware `now()` — at runtime, inside the recency component,
    only for whichever provider happened to return one.
    """
    job = make_job(posted_at=datetime(2026, 8, 1, 12, 0, 0))
    assert job.posted_at.tzinfo is not None
    assert job.posted_at.utcoffset() == timedelta(0)


def test_aware_posted_at_is_converted_to_utc():
    cairo = timezone(timedelta(hours=3))
    job = make_job(posted_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=cairo))
    assert job.posted_at.utcoffset() == timedelta(0)
    assert job.posted_at.hour == 9


def test_age_days_is_none_without_a_date():
    """
    Absence must stay distinguishable from zero: 'undated' and 'posted today'
    score very differently, and conflating them is how an undated listing wins.
    """
    assert make_job().age_days is None


def test_age_days_measures_from_now():
    job = make_job(posted_at=datetime.now(timezone.utc) - timedelta(days=10))
    assert 9.9 < job.age_days < 10.1


# ---------------------------------------------------------------------------
# Salary
# ---------------------------------------------------------------------------


def test_absent_salary_stays_none():
    """
    Never 0. MENA postings routinely omit salary; scoring absence as zero
    buries the entire region, which is the app's core market.
    """
    job = make_job()
    assert job.salary_min is None
    assert job.salary_max is None
    assert job.annual_salary_range is None


def test_reversed_salary_bounds_are_swapped():
    """Providers that map 'up to X' onto the min field produce these."""
    job = make_job(salary_min=90000, salary_max=60000, salary_period=SalaryPeriod.YEAR)
    assert (job.salary_min, job.salary_max) == (60000, 90000)


def test_salary_without_period_defaults_to_year():
    job = make_job(salary_min=80000)
    assert job.salary_period is SalaryPeriod.YEAR


def test_no_period_is_invented_when_there_is_no_salary():
    assert make_job().salary_period is None


def test_monthly_salary_is_annualized():
    """
    The 12x error this prevents is the dangerous kind: both numbers look
    plausible, so a Cairo monthly EGP figure silently loses to a US annual one.
    """
    job = make_job(
        salary_min=40000, salary_max=60000,
        salary_currency="egp", salary_period=SalaryPeriod.MONTH,
    )
    assert job.annual_salary_range == (480000, 720000)
    assert job.salary_currency == "EGP"


def test_open_ended_salary_range_annualizes_the_stated_bound():
    job = make_job(salary_min=5000, salary_period=SalaryPeriod.MONTH)
    assert job.annual_salary_range == (60000, None)


# ---------------------------------------------------------------------------
# Work mode
# ---------------------------------------------------------------------------


def test_is_remote_is_derived_not_stored():
    """
    Deliberate deviation from the roadmap: storing both `is_remote` and
    `work_mode` allows `ONSITE + is_remote=True`, which nothing would catch.
    """
    assert make_job(work_mode=WorkMode.REMOTE).is_remote is True
    assert make_job(work_mode=WorkMode.ONSITE).is_remote is False
    assert make_job(work_mode=WorkMode.HYBRID).is_remote is False
    assert make_job(work_mode=WorkMode.UNKNOWN).is_remote is False

    with pytest.raises(ValidationError):
        make_job(is_remote=True)


def test_unknown_work_mode_is_the_default():
    """Absence of information, not an assertion of onsite."""
    assert make_job().work_mode is WorkMode.UNKNOWN


# ---------------------------------------------------------------------------
# canonical_key
# ---------------------------------------------------------------------------


def test_same_posting_from_two_providers_shares_a_key():
    a = make_job(provider="jsearch", source_id="1",
                 apply_url="https://boards.greenhouse.io/acme/jobs/4019283")
    b = make_job(provider="jooble", source_id="zzz",
                 apply_url="https://www.Boards.Greenhouse.io/acme/jobs/4019283/"
                           "?utm_source=jooble&gh_src=abc#apply")
    assert a.canonical_key == b.canonical_key


def test_different_postings_do_not_share_a_key():
    a = make_job(apply_url="https://boards.greenhouse.io/acme/jobs/1")
    b = make_job(apply_url="https://boards.greenhouse.io/acme/jobs/2")
    assert a.canonical_key != b.canonical_key


def test_non_tracking_query_params_are_preserved():
    """
    Indeed puts the job id in the query string. Dropping every parameter would
    collapse an entire board into one key — the failure mode is invisible: the
    user just sees one Indeed job.
    """
    a = make_job(apply_url="https://indeed.com/viewjob?jk=aaa111")
    b = make_job(apply_url="https://indeed.com/viewjob?jk=bbb222")
    assert a.canonical_key != b.canonical_key


def test_homepage_url_falls_back_to_company_title_city():
    """
    A bare host identifies a company, not a vacancy, so two different roles at
    the same company must not collapse into one key.
    """
    engineer = make_job(title="Backend Engineer", company="Acme", city="Cairo",
                        apply_url="https://acme.com/")
    designer = make_job(title="Product Designer", company="Acme", city="Cairo",
                        apply_url="https://acme.com")

    assert engineer.canonical_key.startswith("cts:")
    assert designer.canonical_key.startswith("cts:")
    assert engineer.canonical_key != designer.canonical_key


def test_tracking_only_query_does_not_count_as_identifying():
    """
    A homepage decorated with utm_* is still a homepage. Checking the original
    URL rather than the normalized one would wrongly treat it as a posting.
    """
    job = make_job(company="Acme", title="Backend Engineer", city="Cairo",
                   apply_url="https://acme.com/?utm_source=newsletter")
    assert job.canonical_key.startswith("cts:")


def test_unidentifiable_record_falls_back_to_provider_identity():
    """
    Arabic-only company and title slugify to empty. The record must still get a
    unique key rather than colliding with every other unidentifiable record.
    """
    a = make_job(company="شركة", title="مطور", source_id="1",
                 apply_url="https://example.com/")
    b = make_job(company="مؤسسة", title="مهندس", source_id="2",
                 apply_url="https://example.com/")
    assert a.canonical_key == "src:jsearch:1"
    assert a.canonical_key != b.canonical_key


def test_canonical_key_is_stable_across_calls():
    job = make_job()
    assert job.canonical_key == job.canonical_key


# ---------------------------------------------------------------------------
# field_richness
# ---------------------------------------------------------------------------


def test_richer_record_wins_on_field_richness():
    """
    The registry uses this to pick a winner when two providers return the same
    posting; it must prefer the one that filled in what the scorer needs.
    """
    sparse = make_job()
    rich = make_job(
        posted_at=datetime.now(timezone.utc),
        city="Cairo",
        country="Egypt",
        salary_min=40000,
        salary_max=60000,
        salary_period=SalaryPeriod.MONTH,
        seniority=Seniority.SENIOR,
        description="Build things.",
        required_skills=["Python", "Django"],
        work_mode=WorkMode.HYBRID,
        employment_type=EmploymentType.FULL_TIME,
    )
    assert rich.field_richness > sparse.field_richness
    assert sparse.field_richness == 0


def test_unknown_enum_members_do_not_count_as_richness():
    """UNKNOWN means 'not told', so it must not make a record look complete."""
    job = make_job(work_mode=WorkMode.UNKNOWN, employment_type=EmploymentType.UNKNOWN)
    assert job.field_richness == 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_slugify_drops_non_ascii_rather_than_mangling():
    assert slugify("Acme Corp.") == "acme-corp"
    assert slugify("شركة") == ""
    assert slugify("Café Ltd") == "cafe-ltd"


def test_id_like_params_are_not_mistaken_for_tracking():
    """
    Several ATS platforms use `refnum`/`refid`/`sourceId` as the job id. A
    blanket prefix match on "ref"/"source" would strip them and collapse an
    entire board to one canonical key — which looks like thin provider coverage,
    not a URL bug.
    """
    a = make_job(apply_url="https://jobs.example.com/apply?refnum=111")
    b = make_job(apply_url="https://jobs.example.com/apply?refnum=222")
    assert a.canonical_key != b.canonical_key

    c = make_job(apply_url="https://jobs.example.com/apply?sourceId=333")
    d = make_job(apply_url="https://jobs.example.com/apply?sourceId=444")
    assert c.canonical_key != d.canonical_key


def test_exact_tracking_params_are_still_stripped():
    plain = make_job(apply_url="https://jobs.example.com/apply/9")
    tracked = make_job(
        apply_url="https://jobs.example.com/apply/9?ref=newsletter&gclid=xyz&utm_medium=cpc"
    )
    assert plain.canonical_key == tracked.canonical_key


def test_normalize_url_sorts_kept_params_for_stability():
    """Parameter order is not semantic; two orderings must not be two jobs."""
    a = normalize_url("https://indeed.com/viewjob?jk=aaa&vjs=3")
    b = normalize_url("https://indeed.com/viewjob?vjs=3&jk=aaa")
    assert a == b


def test_seniority_vocabulary_matches_the_cv_parser():
    """
    The scorer compares a candidate's level to a posting's directly. If these
    vocabularies diverge, that comparison needs a translation table — the kind
    of mapping layer that rots silently.
    """
    from backend.parsers.cv_parser import _SENIORITY_PATTERNS

    parser_labels = {label for label, _ in _SENIORITY_PATTERNS}
    schema_labels = {member.value for member in Seniority}
    assert parser_labels <= schema_labels, (
        f"cv_parser produces seniority labels the schema cannot represent: "
        f"{parser_labels - schema_labels}"
    )
