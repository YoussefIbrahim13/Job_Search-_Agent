"""
Tests for the deterministic scorer.

Two things get the most attention here.

First, the three-state rule — "not requested" vs "job silent" vs "compared".
Conflating any two of them is the main way a ranker goes quietly wrong, and the
specific failure that matters for this product is scoring an unstated salary as
zero: MENA postings omit salary routinely, so that single mistake would bury the
app's core market beneath US listings that happen to publish a number.

Second, determinism. The whole argument for moving ranking out of the LLM is
that facts compare identically every time. A test suite that did not pin that
would be conceding the point.
"""
from datetime import datetime, timedelta, timezone

import pytest

from backend.ranking.scorer import (
    DEFAULT_SCORE_THRESHOLD,
    WEIGHT_LOCATION,
    WEIGHT_TITLE,
    WEIGHT_RECENCY,
    WEIGHT_SALARY,
    WEIGHT_SENIORITY,
    WEIGHT_SKILLS,
    Applicability,
    has_relevance_evidence,
    rank_jobs,
    score_job,
    score_location,
    score_title,
    score_recency,
    score_salary,
    score_seniority,
    score_skills,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.schema import (
    NormalizedJob,
    SalaryPeriod,
    Seniority,
    WorkMode,
)

NOW = datetime(2026, 8, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_job(**overrides) -> NormalizedJob:
    base = dict(
        provider="jsearch",
        source_id="1",
        title="Backend Engineer",
        company="Acme",
        apply_url="https://boards.greenhouse.io/acme/jobs/1",
    )
    base.update(overrides)
    return NormalizedJob(**base)


# ===========================================================================
# Title
# ===========================================================================
#
# This component exists because of a live failure. With no title scoring, a
# search for "Python Django Developer" returned "Inside Sales Contractor" at
# 90/100 — remote, fresh, and with nothing anywhere comparing the title to the
# request. Role matching had been left entirely to the LLM, which is designed
# to degrade.


def test_no_target_title_awards_full_marks():
    component = score_title(make_job(title="Anything"), SearchCriteria())
    assert component.applicability is Applicability.NOT_REQUESTED


def test_exact_title_scores_full():
    component = score_title(
        make_job(title="Python Django Developer"),
        SearchCriteria(titles=["Python Django Developer"]),
    )
    assert component.score == WEIGHT_TITLE


def test_unrelated_title_scores_zero():
    """The regression that motivated the component."""
    component = score_title(
        make_job(title="Inside Sales Contractor"),
        SearchCriteria(titles=["Python Django Developer"]),
    )
    assert component.score == 0.0


def test_extra_words_in_the_job_title_are_not_penalised():
    """
    Coverage is measured over what the user asked for, not over the job's
    title. "Senior Python Django Developer, EMEA (m/f/d)" fully satisfies a
    request for "Python Django Developer".
    """
    component = score_title(
        make_job(title="Senior Python Django Developer, EMEA (m/f/d)"),
        SearchCriteria(titles=["Python Django Developer"]),
    )
    assert component.score == WEIGHT_TITLE


def test_seniority_words_do_not_affect_title_match():
    """Seniority has its own component; counting it twice would double-penalise."""
    senior = score_title(make_job(title="Senior Python Developer"),
                         SearchCriteria(titles=["Python Developer"]))
    plain = score_title(make_job(title="Python Developer"),
                        SearchCriteria(titles=["Python Developer"]))
    assert senior.score == plain.score == WEIGHT_TITLE


def test_partial_title_overlap_scores_partially():
    component = score_title(
        make_job(title="Django Developer"),
        SearchCriteria(titles=["Python Django Developer"]),
    )
    assert 0 < component.score < WEIGHT_TITLE


def test_tech_tokens_outweigh_generic_role_nouns():
    """
    "Developer" appears in a third of all postings; "Django" narrows the field
    enormously. Weighting them equally would rank any developer role as a
    partial match for every other.
    """
    tech = score_title(make_job(title="Django Engineer"),
                       SearchCriteria(titles=["Python Django Developer"]))
    role = score_title(make_job(title="Sales Developer"),
                       SearchCriteria(titles=["Python Django Developer"]))
    assert tech.score > role.score


def test_semantically_similar_titles_are_left_to_the_llm():
    """
    Deliberately literal: "Backend Engineer" shares no word with "Python Django
    Developer" and scores zero here. Rescuing it is the semantic pass's job,
    and this component's contract is only that an unrelated title is not
    mistaken for a match when the LLM is unavailable.
    """
    component = score_title(
        make_job(title="Backend Engineer"),
        SearchCriteria(titles=["Python Django Developer"]),
    )
    assert component.score == 0.0
    assert component.applicability is Applicability.COMPARED


def test_a_title_of_only_stopwords_is_treated_as_no_request():
    component = score_title(
        make_job(title="Python Developer"), SearchCriteria(titles=["Senior role"])
    )
    assert component.applicability is Applicability.NOT_REQUESTED


# ===========================================================================
# Normalization over requested components only
# ===========================================================================


def test_unrequested_components_do_not_inflate_the_score():
    """
    Awarding full marks for components the user never asked about was a
    constant offset — harmless for ordering, fatal for a threshold. A search
    with no salary or seniority preference handed every job 10 free points,
    and an entirely irrelevant listing scored 90/100.
    """
    job = make_job(title="Inside Sales Contractor", city="Berlin", country="DE",
                   posted_at=NOW, required_skills=["Excel"])
    criteria = SearchCriteria(titles=["Python Django Developer"],
                              locations=["Cairo"], skills=["Python", "Django"])

    breakdown = score_job(job, criteria, now=NOW)

    unrequested = [c for c in breakdown.components
                   if c.applicability is Applicability.NOT_REQUESTED]
    assert unrequested, "this test needs at least one unrequested component"
    assert breakdown.available < sum(c.max_score for c in breakdown.components)
    assert breakdown.total < 40


def test_score_is_unchanged_by_adding_an_unexpressed_preference():
    """
    The normalization's real guarantee: how much detail the user supplies must
    not shift the absolute scale, or the threshold means something different
    for every search.
    """
    job = make_job(title="Python Django Developer", city="Cairo", country="EG",
                   posted_at=NOW, required_skills=["Python", "Django"])

    lean = score_job(job, SearchCriteria(titles=["Python Django Developer"]), now=NOW)
    detailed = score_job(
        job,
        SearchCriteria(titles=["Python Django Developer"], locations=["Cairo"],
                       skills=["Python", "Django"]),
        now=NOW,
    )
    assert lean.total == pytest.approx(detailed.total, abs=1.0)


# ===========================================================================
# Relevance gate
# ===========================================================================


def test_job_with_no_connection_to_the_query_is_dropped():
    """
    The live failure, end to end: fully remote, freshly posted, and utterly
    unrelated. Not a weak match — not a match.
    """
    criteria = SearchCriteria(titles=["Python Django Developer"],
                              skills=["Python", "Django"])
    junk = make_job(title="Inside Sales Contractor", required_skills=[],
                    work_mode=WorkMode.REMOTE, posted_at=NOW)

    assert has_relevance_evidence(score_job(junk, criteria, now=NOW)) is False
    assert rank_jobs([junk], criteria, threshold=0, now=NOW) == []


def test_shared_skill_alone_keeps_a_job():
    """
    Gating on the title alone would drop "Backend Engineer" for a Django
    search — a legitimate match that shares no title word. Skill evidence is
    what keeps it alive when the LLM is not running.
    """
    criteria = SearchCriteria(titles=["Python Django Developer"],
                              skills=["Python", "Django"])
    job = make_job(title="Backend Engineer", required_skills=["Python", "Django"])

    assert has_relevance_evidence(score_job(job, criteria, now=NOW)) is True


def test_shared_title_term_alone_keeps_a_job():
    criteria = SearchCriteria(titles=["Python Django Developer"],
                              skills=["Kubernetes"])
    job = make_job(title="Django Developer", required_skills=["Terraform"])

    assert has_relevance_evidence(score_job(job, criteria, now=NOW)) is True


def test_semantic_score_rescues_a_job_with_no_deterministic_evidence():
    """Precisely what the LLM's 10 points are for."""
    criteria = SearchCriteria(titles=["Python Django Developer"],
                              skills=["Python", "Django"])
    job = make_job(title="Platform Engineer", required_skills=[])
    breakdown = score_job(job, criteria, now=NOW)

    assert has_relevance_evidence(breakdown) is False
    breakdown.semantic_score = 8.0
    assert has_relevance_evidence(breakdown) is True


def test_gate_does_nothing_when_nothing_was_asked():
    """With no title and no skills there is no evidence to require."""
    job = make_job(title="Anything At All", required_skills=[])
    assert has_relevance_evidence(score_job(job, SearchCriteria(), now=NOW)) is True


def test_gate_can_be_disabled():
    """
    `rank_with_semantic` needs this: a job with no deterministic evidence is
    exactly the case the LLM might rescue, so it must reach the model.
    """
    criteria = SearchCriteria(titles=["Python Django Developer"],
                              skills=["Python", "Django"])
    junk = make_job(title="Inside Sales Contractor", required_skills=[])

    assert rank_jobs([junk], criteria, threshold=0, now=NOW, gate=False)


# ===========================================================================
# Location
# ===========================================================================


def test_no_location_preference_awards_full_marks():
    """
    The component cannot discriminate, so it must not drag every job down for a
    preference the user never expressed.
    """
    component = score_location(make_job(city="Cairo"), SearchCriteria())
    assert component.score == WEIGHT_LOCATION
    assert component.applicability is Applicability.NOT_REQUESTED


def test_exact_city_scores_full():
    component = score_location(
        make_job(city="Cairo", country="EG"),
        SearchCriteria(locations=["Cairo"]),
    )
    assert component.score == WEIGHT_LOCATION


def test_same_country_scores_partial():
    component = score_location(
        make_job(city="Alexandria", country="EG"),
        SearchCriteria(locations=["Cairo"]),
    )
    assert 0 < component.score < WEIGHT_LOCATION


def test_different_country_scores_zero():
    component = score_location(
        make_job(city="Berlin", country="DE"),
        SearchCriteria(locations=["Cairo"]),
    )
    assert component.score == 0.0


def test_remote_fully_satisfies_a_located_search():
    """
    A remote role is one a Cairo candidate can take without moving, so it is a
    better location match than a job in the next city — not a worse one. The
    prototype scored remote at 5 of 30, below "same country", which
    systematically buried the roles most valuable to its users.
    """
    remote = score_location(
        make_job(work_mode=WorkMode.REMOTE), SearchCriteria(locations=["Cairo"])
    )
    same_country = score_location(
        make_job(city="Alexandria", country="EG"), SearchCriteria(locations=["Cairo"])
    )
    assert remote.score == WEIGHT_LOCATION
    assert remote.score > same_country.score


def test_remote_loses_when_the_search_excludes_remote():
    component = score_location(
        make_job(work_mode=WorkMode.REMOTE),
        SearchCriteria(locations=["Cairo"], work_modes=[WorkMode.ONSITE]),
    )
    assert component.score == 0.0


def test_worldwide_counts_as_a_match():
    component = score_location(
        make_job(location_raw="Worldwide"), SearchCriteria(locations=["Cairo"])
    )
    assert component.score == WEIGHT_LOCATION


def test_job_without_location_scores_neutral_not_zero():
    component = score_location(make_job(), SearchCriteria(locations=["Cairo"]))
    assert component.applicability is Applicability.JOB_SILENT
    assert 0 < component.score < WEIGHT_LOCATION


def test_location_falls_back_to_raw_string():
    """Remotive and Arbeitnow populate only `location_raw`."""
    component = score_location(
        make_job(location_raw="Cairo, Egypt"), SearchCriteria(locations=["Cairo"])
    )
    assert component.score == WEIGHT_LOCATION


# ===========================================================================
# Recency
# ===========================================================================


def test_todays_posting_scores_full():
    component = score_recency(make_job(posted_at=NOW), SearchCriteria(), now=NOW)
    assert component.score == pytest.approx(WEIGHT_RECENCY)


def test_recency_decays_with_age():
    fresh = score_recency(
        make_job(posted_at=NOW - timedelta(days=2)), SearchCriteria(), now=NOW
    )
    stale = score_recency(
        make_job(posted_at=NOW - timedelta(days=20)), SearchCriteria(), now=NOW
    )
    assert fresh.score > stale.score > 0


def test_posting_outside_the_window_scores_zero():
    component = score_recency(
        make_job(posted_at=NOW - timedelta(days=50)),
        SearchCriteria(max_age_days=30),
        now=NOW,
    )
    assert component.score == 0.0


def test_undated_posting_scores_neutral():
    """
    This is the Tavily case, and the reason structured providers matter: an
    undated job is unknown, not fresh and not ancient.
    """
    component = score_recency(make_job(), SearchCriteria(), now=NOW)
    assert component.applicability is Applicability.JOB_SILENT
    assert 0 < component.score < WEIGHT_RECENCY


def test_future_dated_posting_does_not_exceed_full_marks():
    """A provider clock quirk must not make a job score above the maximum."""
    component = score_recency(
        make_job(posted_at=NOW + timedelta(days=3)), SearchCriteria(), now=NOW
    )
    assert component.score == WEIGHT_RECENCY


# ===========================================================================
# Salary
# ===========================================================================


def test_no_salary_expectation_awards_full_marks():
    component = score_salary(
        make_job(salary_min=1000, salary_period=SalaryPeriod.MONTH), SearchCriteria()
    )
    assert component.applicability is Applicability.NOT_REQUESTED
    assert component.score == WEIGHT_SALARY


def test_unstated_salary_scores_neutral_never_zero():
    """
    The single most important behaviour in this module. MENA postings omit
    salary as a matter of course; scoring silence as "pays nothing" would rank
    the app's core market below any US listing that publishes a number — an
    ordering produced by disclosure convention, not by pay.
    """
    component = score_salary(
        make_job(),
        SearchCriteria(salary_min=30000, salary_currency="EGP",
                       salary_period=SalaryPeriod.MONTH),
    )
    assert component.applicability is Applicability.JOB_SILENT
    assert component.score == WEIGHT_SALARY * 0.5
    assert component.score > 0


def test_salary_meeting_expectation_scores_full():
    component = score_salary(
        make_job(salary_min=40000, salary_max=60000, salary_currency="EGP",
                 salary_period=SalaryPeriod.MONTH),
        SearchCriteria(salary_min=40000, salary_currency="EGP",
                       salary_period=SalaryPeriod.MONTH),
    )
    assert component.score == WEIGHT_SALARY


def test_salary_below_expectation_scores_proportionally():
    component = score_salary(
        make_job(salary_min=20000, salary_max=25000, salary_currency="EGP",
                 salary_period=SalaryPeriod.MONTH),
        SearchCriteria(salary_min=50000, salary_currency="EGP",
                       salary_period=SalaryPeriod.MONTH),
    )
    assert 0 < component.score < WEIGHT_SALARY


def test_period_mismatch_is_normalized_before_comparing():
    """
    A monthly figure compared against an annual expectation without
    annualizing is a 12x error in which both numbers look plausible.
    """
    component = score_salary(
        make_job(salary_min=10000, salary_currency="USD",
                 salary_period=SalaryPeriod.MONTH),   # 120k/yr
        SearchCriteria(salary_min=100000, salary_currency="USD",
                       salary_period=SalaryPeriod.YEAR),
    )
    assert component.score == WEIGHT_SALARY


def test_cross_currency_comparison_is_flagged_as_approximate():
    """
    FX rates here are static and will drift. The flag is what lets the UI say
    so rather than presenting a converted figure as exact.
    """
    component = score_salary(
        make_job(salary_min=120000, salary_currency="USD",
                 salary_period=SalaryPeriod.YEAR),
        SearchCriteria(salary_min=50000, salary_currency="EGP",
                       salary_period=SalaryPeriod.MONTH),
    )
    assert component.approximate_fx is True
    assert component.score == WEIGHT_SALARY


def test_unknown_currency_declines_to_compare():
    """Guessing a rate would be worse than admitting we cannot compare."""
    component = score_salary(
        make_job(salary_min=100000, salary_currency="XYZ",
                 salary_period=SalaryPeriod.YEAR),
        SearchCriteria(salary_min=50000, salary_currency="USD",
                       salary_period=SalaryPeriod.YEAR),
    )
    assert component.applicability is Applicability.JOB_SILENT
    assert component.score == WEIGHT_SALARY * 0.5


def test_paying_far_above_expectation_is_not_extra_credit():
    """Otherwise ranking collapses into "sort by salary"."""
    met = score_salary(
        make_job(salary_min=100000, salary_currency="USD",
                 salary_period=SalaryPeriod.YEAR),
        SearchCriteria(salary_min=100000, salary_currency="USD",
                       salary_period=SalaryPeriod.YEAR),
    )
    exceeded = score_salary(
        make_job(salary_min=400000, salary_currency="USD",
                 salary_period=SalaryPeriod.YEAR),
        SearchCriteria(salary_min=100000, salary_currency="USD",
                       salary_period=SalaryPeriod.YEAR),
    )
    assert met.score == exceeded.score == WEIGHT_SALARY


# ===========================================================================
# Seniority
# ===========================================================================


def test_no_seniority_given_awards_full_marks():
    component = score_seniority(make_job(seniority=Seniority.SENIOR), SearchCriteria())
    assert component.applicability is Applicability.NOT_REQUESTED
    assert component.score == WEIGHT_SENIORITY


def test_exact_seniority_scores_full():
    component = score_seniority(
        make_job(seniority=Seniority.SENIOR),
        SearchCriteria(seniority=Seniority.SENIOR),
    )
    assert component.score == WEIGHT_SENIORITY


def test_adjacent_seniority_scores_partial():
    component = score_seniority(
        make_job(seniority=Seniority.MID),
        SearchCriteria(seniority=Seniority.SENIOR),
    )
    assert 0 < component.score < WEIGHT_SENIORITY


def test_distant_seniority_scores_zero():
    component = score_seniority(
        make_job(seniority=Seniority.INTERN),
        SearchCriteria(seniority=Seniority.PRINCIPAL),
    )
    assert component.score == 0.0


def test_seniority_mismatch_is_symmetric():
    """
    Neither over- nor under-qualification is the ranker's to forgive. The
    candidate can always apply anyway; the job simply must not outrank one at
    their actual level.
    """
    below = score_seniority(
        make_job(seniority=Seniority.JUNIOR), SearchCriteria(seniority=Seniority.MID)
    )
    above = score_seniority(
        make_job(seniority=Seniority.SENIOR), SearchCriteria(seniority=Seniority.MID)
    )
    assert below.score == above.score


def test_job_without_seniority_scores_neutral():
    component = score_seniority(make_job(), SearchCriteria(seniority=Seniority.SENIOR))
    assert component.applicability is Applicability.JOB_SILENT
    assert component.score == WEIGHT_SENIORITY * 0.5


# ===========================================================================
# Skills
# ===========================================================================


def test_no_skills_given_awards_full_marks():
    component = score_skills(
        make_job(required_skills=["Python"]), SearchCriteria()
    )
    assert component.applicability is Applicability.NOT_REQUESTED
    assert component.score == WEIGHT_SKILLS


def test_full_overlap_scores_full():
    component = score_skills(
        make_job(required_skills=["Python", "Django"]),
        SearchCriteria(skills=["Python", "Django"]),
    )
    assert component.score == WEIGHT_SKILLS


def test_no_overlap_scores_zero():
    component = score_skills(
        make_job(required_skills=["Java", "Spring"]),
        SearchCriteria(skills=["Python", "Django"]),
    )
    assert component.score == 0.0


def test_asymmetric_lists_are_not_punished():
    """
    Jaccard would score a candidate who has every one of a posting's four
    required skills at 4/30, because their CV lists thirty. That reads as a
    poor match when it is a perfect one — hence the overlap coefficient.
    """
    long_cv = [f"Skill{i}" for i in range(26)] + ["Python", "Django", "Docker", "AWS"]
    component = score_skills(
        make_job(required_skills=["Python", "Django", "Docker", "AWS"]),
        SearchCriteria(skills=long_cv),
    )
    assert component.score == WEIGHT_SKILLS


def test_partial_overlap_scores_partially():
    component = score_skills(
        make_job(required_skills=["Python", "Django", "Java", "Spring"]),
        SearchCriteria(skills=["Python", "Django"]),
    )
    assert component.score == WEIGHT_SKILLS  # both wanted skills are covered


def test_skill_matching_is_case_insensitive():
    component = score_skills(
        make_job(required_skills=["python", "DJANGO"]),
        SearchCriteria(skills=["Python", "Django"]),
    )
    assert component.score == WEIGHT_SKILLS


def test_job_without_skills_scores_neutral():
    component = score_skills(make_job(), SearchCriteria(skills=["Python"]))
    assert component.applicability is Applicability.JOB_SILENT
    assert component.score == WEIGHT_SKILLS * 0.5


def test_matched_skills_are_named_in_the_detail():
    """The breakdown is a product feature, not just a debugging aid."""
    component = score_skills(
        make_job(required_skills=["Python", "Django"]),
        SearchCriteria(skills=["Python", "Django"]),
    )
    assert "Python" in component.detail


# ===========================================================================
# Aggregate
# ===========================================================================


def test_breakdown_is_normalized_to_100():
    breakdown = score_job(
        make_job(city="Cairo", country="EG", posted_at=NOW,
                 required_skills=["Python"], seniority=Seniority.SENIOR),
        SearchCriteria(locations=["Cairo"], skills=["Python"],
                       seniority=Seniority.SENIOR),
        now=NOW,
    )
    assert breakdown.total == 100.0


def test_total_is_comparable_before_and_after_the_semantic_pass():
    """
    Normalizing against what was actually available means a job scored before
    the LLM pass is directly comparable to one scored after it. Otherwise every
    job would appear to lose 10 points for a pass that had not run, and any
    threshold would mean two different things depending on timing.
    """
    job = make_job(city="Cairo", country="EG", posted_at=NOW,
                   required_skills=["Python"], seniority=Seniority.SENIOR)
    criteria = SearchCriteria(locations=["Cairo"], skills=["Python"],
                              seniority=Seniority.SENIOR)

    before = score_job(job, criteria, now=NOW)
    after = score_job(job, criteria, now=NOW)
    after.semantic_score = 10.0

    assert before.total == after.total == 100.0


def test_breakdown_serializes_for_the_api():
    breakdown = score_job(make_job(city="Cairo"), SearchCriteria(locations=["Cairo"]),
                          now=NOW)
    payload = breakdown.as_dict()

    assert payload["total"] == breakdown.total
    assert {c["name"] for c in payload["components"]} == {
        "title", "skills", "location", "recency", "salary", "seniority"
    }
    assert all("detail" in c for c in payload["components"])


def test_a_totally_unstated_job_still_scores_around_neutral():
    """
    A job that answers none of the user's criteria is unknown, not bad. It must
    land near the middle so it can still surface when nothing better exists —
    the alternative is an empty result page.
    """
    breakdown = score_job(
        make_job(),
        SearchCriteria(locations=["Cairo"], skills=["Python"],
                       seniority=Seniority.SENIOR, salary_min=30000,
                       salary_currency="EGP", salary_period=SalaryPeriod.MONTH),
        now=NOW,
    )
    assert 45 <= breakdown.total <= 55


# ===========================================================================
# Ranking
# ===========================================================================


def test_jobs_are_ordered_by_score_descending():
    criteria = SearchCriteria(locations=["Cairo"], skills=["Python", "Django"])
    good = make_job(source_id="good", apply_url="https://x.com/good",
                    city="Cairo", country="EG", posted_at=NOW,
                    required_skills=["Python", "Django"])
    # Shares one skill, so it clears the relevance gate and this test stays
    # about ordering rather than accidentally testing the gate.
    poor = make_job(source_id="poor", apply_url="https://x.com/poor",
                    city="Berlin", country="DE",
                    posted_at=NOW - timedelta(days=25),
                    required_skills=["Python", "Java", "Spring", "Kotlin"])

    ranked = rank_jobs([poor, good], criteria, threshold=0, now=NOW)
    assert [s.job.source_id for s in ranked] == ["good", "poor"]


def test_jobs_below_the_threshold_are_dropped():
    criteria = SearchCriteria(locations=["Cairo"], skills=["Python"])
    poor = make_job(city="Berlin", country="DE",
                    posted_at=NOW - timedelta(days=40),
                    required_skills=["Java"])

    assert rank_jobs([poor], criteria, threshold=DEFAULT_SCORE_THRESHOLD, now=NOW) == []


def test_ranking_is_deterministic():
    """
    The whole argument for moving ranking out of the LLM. Identical inputs must
    produce an identical order, or the Phase 5 eval set measures noise.
    """
    criteria = SearchCriteria(locations=["Cairo"], skills=["Python"])
    jobs = [
        make_job(source_id=str(i), apply_url=f"https://x.com/{i}",
                 city="Cairo", country="EG", posted_at=NOW,
                 required_skills=["Python"])
        for i in range(10)
    ]

    first = [s.job.source_id for s in rank_jobs(jobs, criteria, threshold=0, now=NOW)]
    second = [s.job.source_id for s in rank_jobs(list(reversed(jobs)), criteria,
                                                 threshold=0, now=NOW)]
    assert first == second


def test_limit_is_applied_after_sorting():
    criteria = SearchCriteria(locations=["Cairo"])
    jobs = [
        make_job(source_id="near", apply_url="https://x.com/1", city="Cairo",
                 country="EG", posted_at=NOW),
        make_job(source_id="far", apply_url="https://x.com/2", city="Berlin",
                 country="DE", posted_at=NOW),
    ]
    ranked = rank_jobs(jobs, criteria, threshold=0, limit=1, now=NOW)
    assert [s.job.source_id for s in ranked] == ["near"]


def test_empty_input_returns_empty():
    assert rank_jobs([], SearchCriteria(), now=NOW) == []


def test_scoring_does_not_mutate_the_job():
    """
    Registry results may be shared from cache across requests, so scoring must
    be read-only or one user's search would alter another's.
    """
    job = make_job(city="Cairo", posted_at=NOW)
    before = job.model_dump()
    score_job(job, SearchCriteria(locations=["Cairo"]), now=NOW)
    assert job.model_dump() == before
