"""
Contract tests for `SearchCriteria`.

The cache-key tests matter more than they look. Free provider tiers are
measured in low hundreds of requests per month, so a key that varies when it
shouldn't turns every repeat search into a paid call and exhausts the quota;
a key that collides when it shouldn't serves one user another user's results.
Both failures are invisible in normal use.
"""
import pytest
from pydantic import ValidationError

from backend.parsers.cv_parser import CandidateProfile
from backend.sources.criteria import DEFAULT_MAX_AGE_DAYS, SearchCriteria
from backend.sources.schema import EmploymentType, SalaryPeriod, Seniority, WorkMode


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_string_lists_are_stripped_and_deduped_case_insensitively():
    criteria = SearchCriteria(
        titles=["  Backend Engineer ", "backend engineer", "", "   ", "SRE"],
    )
    assert criteria.titles == ["Backend Engineer", "SRE"]


def test_title_order_is_preserved():
    """Order carries preference, and single-query adapters use titles[0]."""
    criteria = SearchCriteria(titles=["Backend Engineer", "Data Engineer"])
    assert criteria.primary_title == "Backend Engineer"


def test_a_bare_string_is_accepted_as_a_single_item_list():
    """Adapters and callers pass one location constantly; failing on it is noise."""
    assert SearchCriteria(locations="Cairo").locations == ["Cairo"]


def test_currency_is_uppercased():
    assert SearchCriteria(salary_currency=" egp ").salary_currency == "EGP"


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        SearchCriteria(work_mode=WorkMode.REMOTE)  # singular: the field is work_modes


@pytest.mark.parametrize("bad", [0, -1, 400])
def test_max_age_days_is_bounded(bad):
    with pytest.raises(ValidationError):
        SearchCriteria(max_age_days=bad)


def test_defaults_are_sane():
    criteria = SearchCriteria()
    assert criteria.max_age_days == DEFAULT_MAX_AGE_DAYS
    assert criteria.work_modes == []
    assert criteria.employment_types == []
    assert criteria.seniority is None


# ---------------------------------------------------------------------------
# remote_ok
# ---------------------------------------------------------------------------


def test_no_work_mode_preference_permits_remote():
    """Empty means 'no preference', which is not the same as 'exclude remote'."""
    assert SearchCriteria().remote_ok is True


def test_explicit_onsite_excludes_remote():
    assert SearchCriteria(work_modes=[WorkMode.ONSITE]).remote_ok is False


def test_explicit_remote_permits_remote():
    assert SearchCriteria(work_modes=[WorkMode.REMOTE, WorkMode.HYBRID]).remote_ok is True


# ---------------------------------------------------------------------------
# cache_key
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic():
    a = SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])
    b = SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])
    assert a.cache_key("jsearch") == b.cache_key("jsearch")


def test_cache_key_is_namespaced_per_provider():
    """Two providers must never serve each other's cached payloads."""
    criteria = SearchCriteria(titles=["Backend Engineer"])
    assert criteria.cache_key("jsearch") != criteria.cache_key("jooble")


def test_skill_order_does_not_change_the_cache_key():
    """Skills are a set; reordering them is the same search."""
    a = SearchCriteria(skills=["Python", "Django"])
    b = SearchCriteria(skills=["Django", "Python"])
    assert a.cache_key("jsearch") == b.cache_key("jsearch")


def test_title_order_does_change_the_cache_key():
    """
    Order is semantic here — it decides which title a single-query adapter
    actually sends, so the two produce genuinely different result sets.
    """
    a = SearchCriteria(titles=["Backend Engineer", "Data Engineer"])
    b = SearchCriteria(titles=["Data Engineer", "Backend Engineer"])
    assert a.cache_key("jsearch") != b.cache_key("jsearch")


def test_limit_does_not_affect_the_cache_key():
    """
    A cached 25-result response satisfies a later 10-result request. Keying on
    limit would multiply misses against a quota measured in hundreds per month.
    """
    a = SearchCriteria(titles=["Backend Engineer"], limit=10)
    b = SearchCriteria(titles=["Backend Engineer"], limit=25)
    assert a.cache_key("jsearch") == b.cache_key("jsearch")


@pytest.mark.parametrize(
    "field,value",
    [
        ("locations", ["Dubai"]),
        ("work_modes", [WorkMode.REMOTE]),
        ("employment_types", [EmploymentType.INTERNSHIP]),
        ("seniority", Seniority.SENIOR),
        ("years_experience", 5.0),
        ("salary_min", 50000.0),
        ("salary_currency", "USD"),
        ("salary_period", SalaryPeriod.MONTH),
        ("max_age_days", 7),
    ],
)
def test_every_meaningful_field_changes_the_cache_key(field, value):
    """
    Guards the reverse failure: a field omitted from the key means two
    different searches silently share one cached result set.
    """
    base = SearchCriteria(titles=["Backend Engineer"])
    variant = SearchCriteria(titles=["Backend Engineer"], **{field: value})
    assert base.cache_key("jsearch") != variant.cache_key("jsearch"), (
        f"{field} is not part of the cache key, so changing it would serve a "
        f"stale result set from a different search"
    )


# ---------------------------------------------------------------------------
# from_candidate_profile
# ---------------------------------------------------------------------------


def test_from_candidate_profile_maps_the_parsed_fields():
    profile = CandidateProfile(
        raw_text="...",
        detected_title="Backend Engineer",
        skills=["Python", "Django"],
        years_experience=6.0,
        seniority="senior",
    )
    criteria = SearchCriteria.from_candidate_profile(profile, locations=["Cairo"])

    assert criteria.titles == ["Backend Engineer"]
    assert criteria.skills == ["Python", "Django"]
    assert criteria.locations == ["Cairo"]
    assert criteria.seniority is Seniority.SENIOR
    assert criteria.years_experience == 6.0


def test_undetermined_profile_fields_stay_none():
    """
    A wrong seniority silently distorts every subsequent ranking; a missing one
    just scores that component neutral. Never guess.
    """
    profile = CandidateProfile(raw_text="...", detected_title="", skills=[])
    criteria = SearchCriteria.from_candidate_profile(profile)

    assert criteria.titles == []
    assert criteria.seniority is None
    assert criteria.years_experience is None


def test_unmappable_seniority_degrades_instead_of_raising():
    """
    A CV upload must not 500 because the parser grew a title keyword the schema
    does not know about yet.
    """
    profile = CandidateProfile(raw_text="...", seniority="architect")
    criteria = SearchCriteria.from_candidate_profile(profile)
    assert criteria.seniority is None


def test_overrides_reach_the_constructed_criteria():
    profile = CandidateProfile(raw_text="...", detected_title="Backend Engineer")
    criteria = SearchCriteria.from_candidate_profile(
        profile,
        work_modes=[WorkMode.REMOTE],
        employment_types=[EmploymentType.INTERNSHIP],
        max_age_days=7,
    )
    assert criteria.work_modes == [WorkMode.REMOTE]
    assert criteria.employment_types == [EmploymentType.INTERNSHIP]
    assert criteria.max_age_days == 7
