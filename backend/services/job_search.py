"""
The structured search pipeline, and its translation to the legacy API shape.

WHAT THIS REPLACES
------------------
The LangGraph agent: an LLM loop that constructed search-engine queries, read
prose snippets, and emitted a JSON array it had scored itself. This is the
linear replacement the roadmap calls for —

    criteria -> registry fan-out -> dedup -> deterministic score
             -> batched semantic pass -> threshold + sort

— with no iteration cap, no coercion node, no graceful-exit recovery, and no
brace repair. Those all existed to survive an architecture where the model was
in the control path; none of them has an equivalent here because none of the
failures they handled can occur.

WHY THE OUTPUT SHAPE IS UNCHANGED
---------------------------------
`to_legacy_response` emits exactly the keys the existing frontend reads. That
is deliberate and temporary. Running old and new behind a flag is only
informative if both render identically — any UI difference would confound the
comparison it exists to enable. Phase 7's rewrite is where the richer fields
(`score_breakdown`, `providers`) become the primary contract; until then they
ride along additively, ignored by the current page.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from urllib.parse import urlparse

from backend.core.config import get_settings
from backend.ranking.scorer import Applicability, ScoredJob
from backend.ranking.semantic import rank_with_semantic
from backend.sources.criteria import SearchCriteria
from backend.sources.registry import SearchOutcome, SourceRegistry, build_default_registry
from backend.sources.schema import EmploymentType, NormalizedJob, SalaryPeriod, Seniority

logger = logging.getLogger(__name__)

NOT_SPECIFIED = "Not specified"

_PERIOD_LABEL: dict[SalaryPeriod, str] = {
    SalaryPeriod.YEAR: "year",
    SalaryPeriod.MONTH: "month",
    SalaryPeriod.WEEK: "week",
    SalaryPeriod.DAY: "day",
    SalaryPeriod.HOUR: "hour",
}

# Cached across requests so provider circuit-breaker and quota state survive.
# A registry rebuilt per request would forget that a provider is down and
# re-discover it on every search, at the cost of the full timeout each time —
# which is the exact failure the breaker exists to prevent.
_registry: Optional[SourceRegistry] = None


def get_registry() -> SourceRegistry:
    global _registry
    if _registry is None:
        _registry = build_default_registry()
    return _registry


def reset_registry() -> None:
    """Drop the cached registry. For tests and configuration reloads."""
    global _registry
    _registry = None


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------


def _format_salary(job: NormalizedJob) -> str:
    """
    Human-readable salary, or "Not specified".

    Never fabricates a figure. The legacy contract uses the literal string
    "Not specified", and the frontend already renders it, so silence stays
    visibly silence rather than becoming a zero.
    """
    if job.salary_min is None and job.salary_max is None:
        return NOT_SPECIFIED

    currency = job.salary_currency or ""
    period = _PERIOD_LABEL.get(job.salary_period or SalaryPeriod.YEAR, "year")

    def fmt(value: Optional[float]) -> str:
        return f"{value:,.0f}" if value is not None else ""

    if job.salary_min is not None and job.salary_max is not None:
        if job.salary_min == job.salary_max:
            amount = fmt(job.salary_min)
        else:
            amount = f"{fmt(job.salary_min)} - {fmt(job.salary_max)}"
    else:
        amount = fmt(job.salary_min if job.salary_min is not None else job.salary_max)

    return " ".join(part for part in (currency, amount) if part) + f" / {period}"


def _format_location(job: NormalizedJob) -> str:
    if job.is_remote:
        base = ", ".join(p for p in (job.city, job.country) if p)
        return f"Remote ({base})" if base else "Remote"
    parts = [p for p in (job.city, job.country) if p]
    if parts:
        return ", ".join(parts)
    return job.location_raw or NOT_SPECIFIED


def _format_experience(job: NormalizedJob) -> str:
    if job.seniority is None:
        return NOT_SPECIFIED
    label = job.seniority.value.replace("_", " ").title()
    if job.employment_type is EmploymentType.INTERNSHIP:
        return f"{label} (internship)"
    return label


def _source_domain(job: NormalizedJob) -> str:
    """
    Domain the listing lives on.

    The legacy field means "which board is this on", which is what a user
    recognises — not which adapter fetched it. Provider attribution is carried
    separately as `provider`.
    """
    host = (urlparse(job.apply_url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host or job.provider


def _deterministic_reason(item: ScoredJob) -> str:
    """
    Explain a match from the score breakdown when the LLM did not.

    The semantic pass is advisory and degrades on any failure, so this is the
    normal path whenever Groq is unreachable or disabled. It reads the same
    components the score was built from, which means the sentence can never
    contradict the number beside it — a property the prototype's LLM-authored
    `match_reason` did not have, since it was prose written independently of
    the score.
    """
    strong = [
        component
        for component in item.breakdown.components
        if component.applicability is Applicability.COMPARED
        and component.fraction >= 0.75
    ]
    if strong:
        return "Matches on " + ", ".join(c.detail for c in strong[:3]) + "."

    unknown = [
        c for c in item.breakdown.components
        if c.applicability is Applicability.JOB_SILENT
    ]
    if unknown:
        return (
            "Partial match; the listing does not state "
            + ", ".join(c.name for c in unknown[:3])
            + "."
        )
    return "Meets the search criteria."


def job_to_legacy_dict(item: ScoredJob) -> dict[str, Any]:
    """One scored job in the shape the existing frontend reads."""
    job = item.job
    return {
        "company_name": job.company,
        "job_title": job.title,
        "match_score": int(round(item.score)),
        "location": _format_location(job),
        "experience_needed": _format_experience(job),
        "salary_range": _format_salary(job),
        "required_skills": list(job.required_skills),
        "match_reason": item.breakdown.semantic_reason or _deterministic_reason(item),
        "source": _source_domain(job),
        "application_link": job.apply_url,
        # Additive fields. The current page ignores them; Phase 7 makes the
        # breakdown a visible "why did this rank here" feature, and it is
        # already the fastest way to debug a surprising ordering.
        "provider": job.provider,
        "posted_at": job.posted_at.isoformat() if job.posted_at else None,
        "work_mode": job.work_mode.value,
        "employment_type": job.employment_type.value,
        "score_breakdown": item.breakdown.as_dict(),
    }


def _build_summary(
    ranked: list[ScoredJob], outcome: SearchOutcome, criteria: SearchCriteria
) -> str:
    """
    Two honest sentences about what happened.

    Names the providers that actually answered and says plainly when some did
    not. The prototype's `agent_summary` was model-authored and routinely
    described a search that had not occurred — including claiming a date filter
    the code never applied.
    """
    what = criteria.primary_title or "your profile"
    where = criteria.primary_location or "any location"
    sources = ", ".join(outcome.contributing_providers) or "no sources"

    first = (
        f"Searched {sources} for {what} in {where}; "
        f"{len(ranked)} match(es) met the relevance threshold."
    )
    if outcome.degraded:
        return first + (
            f" Some sources were unavailable ({', '.join(outcome.failed_providers)}), "
            f"so results may be incomplete."
        )
    if not ranked and outcome.any_contributed:
        return first + " Try widening the location or relaxing the filters."
    return first


def to_legacy_response(
    ranked: list[ScoredJob],
    outcome: SearchOutcome,
    criteria: SearchCriteria,
) -> dict[str, Any]:
    """Assemble the full response body."""
    return {
        "job_title": criteria.primary_title or "",
        "location": criteria.primary_location or "",
        "total_found": len(ranked),
        "agent_summary": _build_summary(ranked, outcome, criteria),
        "search_queries_used": [
            f"{criteria.primary_title or ''} {criteria.primary_location or ''}".strip()
        ],
        "jobs": [job_to_legacy_dict(item) for item in ranked],
        # Per-provider detail, so a degraded search is diagnosable from the
        # response alone rather than only from server logs.
        "providers": [
            {
                "provider": o.provider,
                "status": o.status.value,
                "job_count": o.job_count,
                "duration_ms": o.duration_ms,
                "error": o.error,
            }
            for o in outcome.providers
        ],
        "degraded": outcome.degraded,
        "pipeline": "structured",
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


async def search_jobs(criteria: SearchCriteria) -> dict[str, Any]:
    """
    Run the full pipeline and return a legacy-shaped response.

    Raises nothing for provider failure: the registry degrades and the response
    says which sources were missing. An empty `jobs` array with a populated
    `providers` array is a meaningful answer, not an error.
    """
    settings = get_settings()

    outcome = await get_registry().search(criteria)

    ranked = rank_with_semantic(
        outcome.jobs,
        criteria,
        threshold=settings.ranking_score_threshold,
        limit=criteria.limit,
    )

    logger.info(
        "structured search: %d raw -> %d ranked (threshold %.0f) from %s",
        len(outcome.jobs), len(ranked), settings.ranking_score_threshold,
        ", ".join(outcome.contributing_providers) or "no providers",
    )

    return to_legacy_response(ranked, outcome, criteria)


async def run_targeted_search(job_title: str, location: str) -> dict[str, Any]:
    """Structured replacement for `recruitment_agent.run_targeted_search`."""
    criteria = SearchCriteria(
        titles=[job_title],
        locations=[location] if location else [],
        skills=_skills_from_title(job_title),
    )
    return await search_jobs(criteria)


async def run_cv_analysis(
    cv_text: str,
    detected_title: str = "",
    preferred_location: str = "",
    *,
    skills: Optional[list[str]] = None,
    seniority: Optional[Seniority] = None,
    years_experience: Optional[float] = None,
) -> dict[str, Any]:
    """
    Structured replacement for `recruitment_agent.run_cv_analysis`.

    Takes the already-parsed profile fields rather than re-deriving them: the
    CV was parsed in the route, and parsing it twice would risk two different
    answers for one upload.
    """
    from backend.sources.normalize import skills_from_text

    criteria = SearchCriteria(
        titles=[detected_title] if detected_title else [],
        locations=[preferred_location] if preferred_location else [],
        skills=skills if skills is not None else skills_from_text(cv_text),
        seniority=seniority,
        years_experience=years_experience,
    )
    return await search_jobs(criteria)


def _skills_from_title(job_title: str) -> list[str]:
    """
    Harvest technology tokens from a typed job title.

    A targeted search has no CV, so the title is the only skill signal
    available. Running it through the shared vocabulary means "Python Django
    Developer" contributes the same canonical tokens a CV would, and the skills
    component scores against like for like.
    """
    from backend.sources.normalize import skills_from_text

    return skills_from_text(job_title)
