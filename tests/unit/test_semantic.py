"""
Tests for the LLM semantic ranking pass.

No network: a fake chat model is injected.

The weight here is on degradation. This pass is worth 10 of 100 points, so
every way it can fail — no key, rate limit, malformed JSON, truncated array,
missing ids, out-of-range scores, a model that returns prose — must leave the
deterministic score intact rather than costing the user their results. A
ranking that is 90% as good beats a 500 by an enormous margin.

The second theme is that the model cannot exceed its remit. It is advisory:
clamped to 0-10, incapable of removing a job, and incapable of overriding the
facts that `scorer.py` established.
"""
import json
from datetime import datetime, timezone

import pytest

from backend.core.cache import InMemoryTTLCache
from backend.core.config import get_settings
from backend.ranking.scorer import (
    WEIGHT_SEMANTIC,
    ScoredJob,
    rank_jobs,
    score_job,
)
from backend.ranking.semantic import (
    SemanticPass,
    _extract_json_object,
    _parse_results,
    apply_semantic_scores,
    rank_with_semantic,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.schema import NormalizedJob, Seniority

NOW = datetime(2026, 8, 11, 12, 0, 0, tzinfo=timezone.utc)


def make_job(idx=0, **overrides) -> NormalizedJob:
    base = dict(
        provider="jsearch",
        source_id=str(idx),
        title="Backend Engineer",
        company="Acme",
        apply_url=f"https://boards.greenhouse.io/acme/jobs/{idx}",
        city="Cairo",
        country="EG",
        posted_at=NOW,
        required_skills=["Python", "Django"],
        description="Build backend services with Python and Django.",
    )
    base.update(overrides)
    return NormalizedJob(**base)


@pytest.fixture
def criteria():
    return SearchCriteria(
        titles=["Backend Engineer"],
        skills=["Python", "Django"],
        locations=["Cairo"],
        seniority=Seniority.MID,
    )


@pytest.fixture
def scored(criteria):
    jobs = [make_job(i) for i in range(3)]
    return [ScoredJob(job=j, breakdown=score_job(j, criteria, now=NOW)) for j in jobs]


@pytest.fixture
def cache():
    return InMemoryTTLCache()


class FakeModel:
    """Returns a canned response and records what it was asked."""

    def __init__(self, response, raise_error=None):
        self._response = response
        self._raise = raise_error
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if self._raise is not None:
            raise self._raise

        class _Msg:
            content = self._response

        return _Msg()


def good_response(count=3, score=8):
    return json.dumps(
        {
            "results": [
                {"id": i, "score": score, "reason": f"Django overlap for job {i}."}
                for i in range(count)
            ]
        }
    )


@pytest.fixture(autouse=True)
def _enable_pass(monkeypatch):
    monkeypatch.setenv("SEMANTIC_PASS_ENABLED", "true")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ===========================================================================
# Happy path
# ===========================================================================


def test_scores_and_reasons_are_attached(scored, criteria, cache):
    model = FakeModel(good_response())
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    assert all(s.breakdown.semantic_score == 8 for s in scored)
    assert all(s.breakdown.semantic_reason for s in scored)


def test_one_call_scores_the_whole_batch(scored, criteria, cache):
    """
    Batching is the point. The prototype put ranking inside an agent loop, so
    the whole jobs array came back in one tool-calling response and truncated
    at 6-8 jobs.
    """
    model = FakeModel(good_response())
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)
    assert len(model.calls) == 1


def test_semantic_score_lifts_the_total(scored, criteria, cache):
    before = scored[0].score
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(good_response(score=10)),
                          cache=cache)
    assert scored[0].score >= before


def test_prompt_contains_candidate_and_job_context(scored, criteria, cache):
    model = FakeModel(good_response())
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    user_message = model.calls[0][1]["content"]
    assert "Backend Engineer" in user_message
    assert "Python" in user_message
    assert '"id": 0' in user_message or '"id":0' in user_message


def test_prompt_excludes_facts_the_scorer_already_handles(scored, criteria, cache):
    """
    The model is asked about role family only. Sending it location, salary, and
    dates invites it to score them — which is the unauditable behaviour the
    deterministic scorer exists to replace.
    """
    model = FakeModel(good_response())
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    system_message = model.calls[0][0]["content"]
    assert "do not judge location, salary, recency, or seniority" in system_message.lower()


# ===========================================================================
# Bounds: the model cannot exceed its remit
# ===========================================================================


def test_out_of_range_scores_are_clamped(scored, criteria, cache):
    """
    A model returning 95 for a 0-10 field would swamp the entire deterministic
    score, turning a 10-point advisory input into the only thing that matters.
    """
    response = json.dumps({"results": [{"id": 0, "score": 95, "reason": "x"}]})
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)

    assert scored[0].breakdown.semantic_score == WEIGHT_SEMANTIC


def test_negative_scores_are_clamped_to_zero(scored, criteria, cache):
    response = json.dumps({"results": [{"id": 0, "score": -20, "reason": "x"}]})
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)
    assert scored[0].breakdown.semantic_score == 0.0


def test_overlong_reasons_are_truncated(scored, criteria, cache):
    response = json.dumps({"results": [{"id": 0, "score": 5, "reason": "x" * 5000}]})
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)
    assert len(scored[0].breakdown.semantic_reason) <= 240


def test_unknown_ids_are_ignored(scored, criteria, cache):
    """A hallucinated id must not create or corrupt a job."""
    response = json.dumps({
        "results": [
            {"id": 0, "score": 7, "reason": "ok"},
            {"id": 99, "score": 10, "reason": "does not exist"},
        ]
    })
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)

    assert scored[0].breakdown.semantic_score == 7
    assert len(scored) == 3


def test_duplicate_ids_keep_the_first(scored, criteria, cache):
    response = json.dumps({
        "results": [
            {"id": 0, "score": 3, "reason": "first"},
            {"id": 0, "score": 9, "reason": "second"},
        ]
    })
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)
    assert scored[0].breakdown.semantic_score == 3


def test_the_pass_never_removes_a_job(scored, criteria, cache):
    """It is advisory. Filtering is the threshold's job, applied afterwards."""
    response = json.dumps({"results": [{"id": 0, "score": 0, "reason": "unrelated"}]})
    result = apply_semantic_scores(scored, criteria, chat_model=FakeModel(response),
                                   cache=cache)
    assert len(result) == 3


# ===========================================================================
# Degradation
# ===========================================================================


@pytest.mark.parametrize(
    "response,why",
    [
        ("", "empty response"),
        ("I think job 0 is a good match!", "prose instead of JSON"),
        ("{", "truncated object"),
        ('{"results": "not-an-array"}', "wrong type for results"),
        ('{"nope": []}', "missing results key"),
        ('{"results": []}', "empty results"),
        ('{"results": [{"score": 5}]}', "rows without ids"),
    ],
)
def test_malformed_responses_leave_the_deterministic_score_intact(
    scored, criteria, cache, response, why
):
    before = [s.score for s in scored]
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)

    assert [s.score for s in scored] == before, why
    assert all(s.breakdown.semantic_score is None for s in scored), why


def test_a_raising_model_degrades_quietly(scored, criteria, cache):
    """
    Uses a non-transient error deliberately, so no retry backoff runs and this
    test stays fast. The retry path has its own test below.
    """
    before = [s.score for s in scored]
    model = FakeModel(None, raise_error=ValueError("malformed request"))

    result = apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    assert len(model.calls) == 1, "a non-transient error must not be retried"
    assert [s.score for s in result] == before
    assert all(s.breakdown.semantic_score is None for s in result)


def test_transient_errors_are_retried_before_degrading(scored, criteria, cache):
    """
    A single Groq 429 must not cost the semantic pass — that is what
    `groq_retry` is wired in for, and nothing else in this module would reveal
    whether the decorator is actually applied.

    This test is intentionally slow (~3s): the retry policy uses real
    exponential backoff, and asserting the attempt count is worth the seconds.
    """
    model = FakeModel(None, raise_error=RuntimeError("rate limit exceeded"))

    result = apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    assert len(model.calls) == 3, "transient failure was not retried"
    assert all(s.breakdown.semantic_score is None for s in result)


def test_a_transient_failure_that_recovers_is_not_lost(scored, criteria, cache):
    """The retry has to actually succeed on a later attempt, not just re-run."""

    class FlakyModel:
        def __init__(self):
            self.calls = []

        def invoke(self, messages):
            self.calls.append(messages)
            if len(self.calls) < 2:
                raise RuntimeError("503 service unavailable")

            class _Msg:
                content = good_response()

            return _Msg()

    model = FlakyModel()
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    assert len(model.calls) == 2
    assert scored[0].breakdown.semantic_score == 8


def test_missing_api_key_degrades_rather_than_raising(scored, criteria, cache, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    get_settings.cache_clear()

    result = apply_semantic_scores(scored, criteria, cache=cache)
    assert all(s.breakdown.semantic_score is None for s in result)


def test_truncated_array_is_not_salvaged(scored, criteria, cache):
    """
    Deliberately no brace repair. The prototype's repair silently discarded
    whatever came after the cut, so a truncated response looked like a complete
    one with fewer jobs — a failure that never surfaced.
    """
    truncated = '{"results": [{"id": 0, "score": 8, "reason": "good"}, {"id": 1, "sc'
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(truncated), cache=cache)

    assert all(s.breakdown.semantic_score is None for s in scored)


def test_partial_results_score_only_what_the_model_answered(scored, criteria, cache):
    """A model that skips ids costs those jobs their 10 points, nothing more."""
    response = json.dumps({"results": [{"id": 0, "score": 9, "reason": "match"}]})
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(response), cache=cache)

    assert scored[0].breakdown.semantic_score == 9
    assert scored[1].breakdown.semantic_score is None
    assert scored[2].breakdown.semantic_score is None


def test_disabled_pass_makes_no_call(scored, criteria, cache, monkeypatch):
    monkeypatch.setenv("SEMANTIC_PASS_ENABLED", "false")
    get_settings.cache_clear()

    model = FakeModel(good_response())
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    assert model.calls == []
    assert all(s.breakdown.semantic_score is None for s in scored)


def test_empty_input_makes_no_call(criteria, cache):
    model = FakeModel(good_response())
    assert apply_semantic_scores([], criteria, chat_model=model, cache=cache) == []
    assert model.calls == []


# ===========================================================================
# Cost control
# ===========================================================================


def test_only_the_top_candidates_are_sent(criteria, cache, monkeypatch):
    """
    Cost scales linearly with batch size, and the jobs below the cut were
    already judged poor matches on facts — tokens spent there change nothing.
    """
    monkeypatch.setenv("SEMANTIC_MAX_JOBS", "2")
    get_settings.cache_clear()

    jobs = [make_job(i) for i in range(6)]
    scored = [ScoredJob(job=j, breakdown=score_job(j, criteria, now=NOW)) for j in jobs]

    model = FakeModel(good_response(count=2))
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    user_message = model.calls[0][1]["content"]
    assert '"id": 2' not in user_message and '"id":2' not in user_message
    assert scored[2].breakdown.semantic_score is None


def test_identical_batches_are_served_from_cache(scored, criteria, cache):
    """
    The pass is deterministic at temperature 0, so a repeated identical request
    is pure cost. Two users running the same search must not pay twice.
    """
    model = FakeModel(good_response())
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    fresh = [ScoredJob(job=s.job, breakdown=score_job(s.job, criteria, now=NOW))
             for s in scored]
    apply_semantic_scores(fresh, criteria, chat_model=model, cache=cache)

    assert len(model.calls) == 1
    assert all(s.breakdown.semantic_score == 8 for s in fresh)


def test_different_criteria_do_not_share_a_cache_entry(scored, cache):
    model = FakeModel(good_response())
    apply_semantic_scores(scored, SearchCriteria(titles=["Backend Engineer"]),
                          chat_model=model, cache=cache)
    apply_semantic_scores(scored, SearchCriteria(titles=["Data Engineer"]),
                          chat_model=model, cache=cache)
    assert len(model.calls) == 2


def test_descriptions_are_truncated_in_the_prompt(criteria, cache, monkeypatch):
    monkeypatch.setenv("SEMANTIC_DESCRIPTION_CHARS", "50")
    get_settings.cache_clear()

    job = make_job(0, description="word " * 500)
    scored = [ScoredJob(job=job, breakdown=score_job(job, criteria, now=NOW))]

    model = FakeModel(good_response(count=1))
    apply_semantic_scores(scored, criteria, chat_model=model, cache=cache)

    assert len(model.calls[0][1]["content"]) < 2000


# ===========================================================================
# Parsing helpers
# ===========================================================================


def test_code_fences_are_stripped():
    payload = _extract_json_object('```json\n{"results": []}\n```')
    assert payload == {"results": []}


def test_surrounding_prose_is_tolerated():
    """Models prepend "Here is the JSON:" constantly."""
    payload = _extract_json_object('Sure! {"results": [{"id": 0}]} Hope that helps.')
    assert "results" in payload


def test_parse_rejects_rows_with_non_numeric_scores():
    payload = {"results": [{"id": 0, "score": "high", "reason": "x"},
                           {"id": 1, "score": 5, "reason": "y"}]}
    parsed = _parse_results(payload, expected=2)
    assert 0 not in parsed
    assert parsed[1][0] == 5


def test_parse_rejects_booleans_masquerading_as_numbers():
    """`isinstance(True, int)` is True in Python — a real source of silent bugs."""
    payload = {"results": [{"id": True, "score": 5, "reason": "x"}]}
    with pytest.raises(Exception):
        _parse_results(payload, expected=2)


# ===========================================================================
# Full pipeline
# ===========================================================================


def test_pipeline_reorders_after_semantic_scores_land(criteria, cache):
    """
    Attaching a score changes a job's total. Without the re-sort the pipeline
    would return jobs in deterministic order while displaying semantic-adjusted
    numbers — an ordering that visibly contradicts itself.
    """
    strong = make_job(0, title="Backend Engineer", required_skills=["Python", "Django"])
    weak = make_job(1, title="Backend Engineer", required_skills=["Python", "Django"])

    # Same deterministic score; the model separates them.
    response = json.dumps({
        "results": [
            {"id": 0, "score": 0, "reason": "different job family"},
            {"id": 1, "score": 10, "reason": "same job family"},
        ]
    })

    ranked = rank_with_semantic(
        [strong, weak], criteria, threshold=0,
        chat_model=FakeModel(response), cache=cache, now=NOW,
    )
    assert [s.job.source_id for s in ranked] == ["1", "0"]


def test_pipeline_applies_the_threshold_after_the_semantic_pass(criteria, cache):
    """
    Filtering first would cut a borderline job before the model could explain
    it — and borderline cases are exactly what semantic fit is meant to
    resolve.
    """
    job = make_job(0, city="Berlin", country="DE", required_skills=["Java"])
    response = json.dumps({"results": [{"id": 0, "score": 10, "reason": "actually fits"}]})

    without = rank_jobs([job], criteria, threshold=0, now=NOW)[0].score
    with_semantic = rank_with_semantic(
        [job], criteria, threshold=0, chat_model=FakeModel(response),
        cache=cache, now=NOW,
    )[0].score

    assert with_semantic > without


def test_pipeline_degrades_to_deterministic_ranking(criteria, cache):
    jobs = [make_job(i) for i in range(3)]
    model = FakeModel(None, raise_error=RuntimeError("groq down"))

    ranked = rank_with_semantic(jobs, criteria, threshold=0, chat_model=model,
                                cache=cache, now=NOW)

    assert len(ranked) == 3
    assert all(s.breakdown.semantic_score is None for s in ranked)


def test_breakdown_serializes_the_semantic_fields(scored, criteria, cache):
    apply_semantic_scores(scored, criteria, chat_model=FakeModel(good_response()),
                          cache=cache)
    payload = scored[0].breakdown.as_dict()

    assert payload["semantic_score"] == 8
    assert payload["semantic_reason"]
