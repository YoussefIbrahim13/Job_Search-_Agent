"""
Job ranking.

Deterministic scoring of `NormalizedJob` against `SearchCriteria`. Everything
here is a fact comparison — location, recency, salary, seniority, skills — and
therefore reproducible, free, and explainable. The LLM's contribution is a
separate 10-point semantic pass; see `scorer.ScoreBreakdown.semantic_score`.
"""

from backend.ranking.scorer import (
    DEFAULT_SCORE_THRESHOLD,
    DETERMINISTIC_MAX,
    Applicability,
    ScoreBreakdown,
    ScoreComponent,
    ScoredJob,
    rank_jobs,
    score_job,
)
from backend.ranking.semantic import (
    SemanticPass,
    apply_semantic_scores,
    rank_with_semantic,
)

__all__ = [
    "DEFAULT_SCORE_THRESHOLD",
    "DETERMINISTIC_MAX",
    "Applicability",
    "ScoreBreakdown",
    "ScoreComponent",
    "ScoredJob",
    "SemanticPass",
    "apply_semantic_scores",
    "rank_jobs",
    "rank_with_semantic",
    "score_job",
]
