"""
The LLM's 10 points: semantic role fit, and the sentence explaining a match.

WHAT THE MODEL IS AND IS NOT ASKED
----------------------------------
It is asked exactly one question: is this posting the same *kind* of role as
what the candidate does, given the description? That is a judgement about
meaning — "Platform Engineer" against a Django/DRF background, "Software
Engineer II" against five years of backend work — and it is the one thing in
ranking a language model is genuinely better at than a rule.

It is not asked where the job is, how old it is, what it pays, or whether the
candidate has the skills. Those are facts, `scorer.py` compares them, and the
prototype's habit of asking the model to score them from prose is exactly what
made its ranking unauditable and produced the `cap_score=75` hack.

WHY BATCHED AND NON-AGENTIC
---------------------------
One request scores every candidate at once. The prototype made the ranking
model part of an agent loop, which meant tool-calling overhead, an iteration
budget, and a single response carrying the entire jobs array — which truncated
at 6-8 jobs and was rescued by brace-repair that silently discarded the tail.

Here the output is a small fixed-shape array of `{id, score, reason}`, the
token budget is computed from the number of jobs, and there are no tools. If
the response is malformed there is nothing to repair: the pass degrades and the
deterministic score stands.

DEGRADATION IS A FEATURE
------------------------
Every failure path — no API key, rate limit, malformed JSON, missing ids,
out-of-range scores — leaves `semantic_score` as None and the deterministic
score untouched. `ScoreBreakdown.total` normalizes against what was actually
available, so a job scored without this pass remains directly comparable to one
scored with it. The product works with the LLM switched off; it just works
slightly better with it on.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from typing import Any, Optional, Sequence

from backend.core.cache import CacheBackend, get_cache
from backend.core.config import get_settings
from backend.core.resilience import groq_retry
from backend.ranking.scorer import (
    DEFAULT_SCORE_THRESHOLD,
    WEIGHT_SEMANTIC,
    ScoredJob,
    has_relevance_evidence,
    rank_jobs,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.schema import NormalizedJob

logger = logging.getLogger(__name__)


# Output tokens to budget per job, plus a fixed allowance for the envelope.
# One result is roughly {"id":1,"score":8,"reason":"<~20 words>"} — about 45
# tokens. 90 leaves generous headroom without inviting the model to ramble.
_TOKENS_PER_JOB = 90
_TOKEN_OVERHEAD = 200

# Cap on a single reason string. The UI shows one line; anything longer is the
# model ignoring the instruction, and truncating is kinder than discarding.
_MAX_REASON_CHARS = 240

_SYSTEM_PROMPT = """\
You judge whether a job posting is the same KIND of role as a candidate's \
background. You do not judge location, salary, recency, or seniority — those \
are scored separately by other means, and commenting on them is wasted output.

For each job return:
  score  : integer 0-10 for role-family fit
             10 = same role and same technology family
             7-9 = same role family, adjacent stack or specialism
             4-6 = related discipline, materially different focus
             1-3 = same broad industry, different job
             0   = unrelated work
  reason : ONE sentence, under 25 words, naming the concrete overlap or gap. \
Cite only what appears in the posting. Never invent details.

Respond with JSON only, no prose or code fences:
{"results":[{"id":<int>,"score":<int>,"reason":"<one sentence>"}]}

Include every id you were given, exactly once."""


class SemanticPassError(Exception):
    """Raised internally when a pass cannot complete. Never escapes `apply`."""


def _build_candidate_block(criteria: SearchCriteria) -> str:
    parts: list[str] = []
    if criteria.titles:
        parts.append(f"Target roles: {', '.join(criteria.titles[:3])}")
    if criteria.skills:
        # Cap the skill list: a thirty-skill CV crowds out the job descriptions
        # in the prompt, and the deterministic scorer already handles overlap.
        parts.append(f"Skills: {', '.join(criteria.skills[:15])}")
    if criteria.seniority:
        parts.append(f"Level: {criteria.seniority.value}")
    if criteria.years_experience is not None:
        parts.append(f"Experience: {criteria.years_experience:g} years")
    return "\n".join(parts) or "No candidate details provided."


def _build_jobs_block(scored: Sequence[ScoredJob], description_chars: int) -> str:
    entries = []
    for index, item in enumerate(scored):
        job = item.job
        description = " ".join((job.description or "").split())[:description_chars]
        entries.append(
            json.dumps(
                {
                    "id": index,
                    "title": job.title,
                    "company": job.company,
                    "skills": job.required_skills[:12],
                    "description": description,
                },
                ensure_ascii=False,
            )
        )
    return "\n".join(entries)


def _build_prompt(
    scored: Sequence[ScoredJob],
    criteria: SearchCriteria,
    description_chars: int,
) -> str:
    return (
        f"CANDIDATE\n{_build_candidate_block(criteria)}\n\n"
        f"JOBS (one JSON object per line)\n"
        f"{_build_jobs_block(scored, description_chars)}\n\n"
        f"Return one result object for each of the {len(scored)} ids above."
    )


def _cache_key(prompt: str, model: str) -> str:
    digest = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()[:20]
    return f"semantic:{digest}"


def _extract_json_object(text: str) -> dict:
    """
    Pull the JSON object out of a model response.

    Deliberately simple: strip code fences, take the outermost braces, parse.
    No brace repair. A truncated response here means the token budget was wrong
    and the right answer is to degrade visibly, not to salvage a partial array
    and silently drop whatever came after the cut — which is precisely the
    failure mode this design replaced.
    """
    if not text:
        raise SemanticPassError("empty response")

    cleaned = re.sub(r"^\s*```(?:json)?|```\s*$", "", text.strip(), flags=re.MULTILINE)

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise SemanticPassError("no JSON object in response")

    try:
        parsed = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as exc:
        raise SemanticPassError(f"malformed JSON: {exc}") from exc

    if not isinstance(parsed, dict):
        raise SemanticPassError("response was not a JSON object")
    return parsed


def _parse_results(payload: dict, expected: int) -> dict[int, tuple[float, str]]:
    """
    Validate the model's array into {index: (score, reason)}.

    Every value is bounds-checked rather than trusted. A model returning 95 for
    a 0-10 field would otherwise silently outweigh the entire deterministic
    score, turning a 10-point advisory input into the only thing that matters.
    """
    rows = payload.get("results")
    if not isinstance(rows, list):
        raise SemanticPassError("response has no 'results' array")

    out: dict[int, tuple[float, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue

        raw_id = row.get("id")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (int, float)):
            continue
        index = int(raw_id)
        if not (0 <= index < expected) or index in out:
            continue

        raw_score = row.get("score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            continue
        # Clamp rather than reject: an out-of-range score is still a usable
        # signal about direction, and dropping the row would cost the job its
        # reason string too.
        score = max(0.0, min(WEIGHT_SEMANTIC, float(raw_score)))

        reason = row.get("reason")
        reason = reason.strip()[:_MAX_REASON_CHARS] if isinstance(reason, str) else ""

        out[index] = (score, reason)

    if not out:
        raise SemanticPassError("no usable rows in 'results'")
    return out


class SemanticPass:
    """
    Applies LLM semantic scores to already-ranked jobs.

    `chat_model` is injectable so tests can drive this without a network call
    and without patching module globals. Anything with a langchain-style
    `.invoke(str) -> message` works.
    """

    def __init__(
        self,
        chat_model: Any = None,
        cache: Optional[CacheBackend] = None,
    ) -> None:
        self._chat_model = chat_model
        self._cache = cache if cache is not None else get_cache()

    # ── Model access ────────────────────────────────────────────────────────

    def _get_model(self, max_tokens: int) -> Any:
        if self._chat_model is not None:
            return self._chat_model

        settings = get_settings()
        if not settings.groq_api_key:
            raise SemanticPassError("GROQ_API_KEY is not set")

        from langchain_groq import ChatGroq

        # temperature=0 is what makes this reproducible enough to sit in an
        # eval harness. No tools are bound: this is a single scoring call, not
        # an agent turn.
        return ChatGroq(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=0.0,
            max_tokens=max_tokens,
        )

    @staticmethod
    @groq_retry
    def _invoke(model: Any, system: str, user: str) -> str:
        response = model.invoke(
            [{"role": "system", "content": system}, {"role": "user", "content": user}]
        )
        content = getattr(response, "content", response)
        return content if isinstance(content, str) else str(content)

    # ── Public API ──────────────────────────────────────────────────────────

    def apply(
        self,
        scored: Sequence[ScoredJob],
        criteria: SearchCriteria,
    ) -> list[ScoredJob]:
        """
        Attach semantic scores to the top candidates, in place.

        Returns the same list. Jobs beyond `semantic_max_jobs`, and any the
        model did not score, keep `semantic_score = None` and remain ranked on
        facts alone.

        Never raises. Every failure is logged and degrades to the deterministic
        score, because a ranking that is 90% as good is worth immeasurably more
        than a 500.
        """
        settings = get_settings()
        items = list(scored)

        if not items:
            return items
        if not settings.semantic_pass_enabled:
            logger.debug("semantic pass disabled by configuration")
            return items

        # Applied after deterministic ranking, so the jobs below this cut are
        # ones already judged poor matches on facts. Spending tokens on them
        # would not change their position.
        batch = items[: settings.semantic_max_jobs]

        prompt = _build_prompt(batch, criteria, settings.semantic_description_chars)
        cache_key = _cache_key(prompt, settings.groq_model)

        cached = self._cache.get(cache_key)
        if cached is not None:
            self._attach(batch, cached)
            logger.debug("semantic pass served from cache (%d job(s))", len(batch))
            return items

        max_tokens = _TOKEN_OVERHEAD + _TOKENS_PER_JOB * len(batch)

        try:
            model = self._get_model(max_tokens)
            raw = self._invoke(model, _SYSTEM_PROMPT, prompt)
            results = _parse_results(_extract_json_object(raw), len(batch))
        except SemanticPassError as exc:
            logger.warning(
                "semantic pass unavailable, ranking on deterministic score only: %s",
                exc,
            )
            return items
        except Exception as exc:  # noqa: BLE001 - the pass is advisory, never fatal
            logger.warning(
                "semantic pass failed, ranking on deterministic score only: %s", exc
            )
            return items

        self._cache.set(cache_key, results)
        self._attach(batch, results)

        missing = len(batch) - len(results)
        if missing:
            # A model that skips ids is a prompt or budget problem. Those jobs
            # keep a deterministic-only score, which is correct but silent
            # without this line.
            logger.info(
                "semantic pass: %d/%d job(s) unscored by the model", missing, len(batch)
            )

        return items

    @staticmethod
    def _attach(
        batch: Sequence[ScoredJob], results: dict[int, tuple[float, str]]
    ) -> None:
        for index, item in enumerate(batch):
            entry = results.get(index)
            if entry is None:
                continue
            score, reason = entry
            item.breakdown.semantic_score = score
            item.breakdown.semantic_reason = reason or None


def apply_semantic_scores(
    scored: Sequence[ScoredJob],
    criteria: SearchCriteria,
    *,
    chat_model: Any = None,
    cache: Optional[CacheBackend] = None,
) -> list[ScoredJob]:
    """Convenience wrapper. See `SemanticPass.apply`."""
    return SemanticPass(chat_model=chat_model, cache=cache).apply(scored, criteria)


def rank_with_semantic(
    jobs: Sequence[NormalizedJob],
    criteria: SearchCriteria,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: Optional[int] = None,
    chat_model: Any = None,
    cache: Optional[CacheBackend] = None,
    now: Optional[datetime] = None,
) -> list[ScoredJob]:
    """
    The full ranking pipeline: deterministic score, semantic pass, final order.

    Two ordering decisions matter here.

    The deterministic pass runs with **no threshold**, and is used only to
    decide which jobs are worth spending tokens on. Filtering first would let a
    job be cut for facts the model was about to explain — the borderline cases
    are exactly the ones semantic fit is meant to resolve.

    The threshold is applied **after** the semantic scores land, against the
    final totals, and the list is re-sorted. Attaching a score changes a job's
    total, so skipping the re-sort would return jobs in deterministic order
    while displaying semantic-adjusted numbers: an ordering that visibly
    contradicts itself.

    The relevance gate is likewise deferred. A job with no deterministic
    connection to the request — no shared title term, no shared skill — is
    exactly the case the semantic pass exists to rescue ("Platform Engineer"
    for a Django search). Gating before the model runs would drop it unseen.
    """
    scored = rank_jobs(jobs, criteria, threshold=0.0, now=now, gate=False)

    apply_semantic_scores(scored, criteria, chat_model=chat_model, cache=cache)

    kept = [
        item for item in scored
        if has_relevance_evidence(item.breakdown) and item.score >= threshold
    ]
    kept.sort(key=lambda item: (-item.score, item.job.canonical_key))

    return kept[:limit] if limit else kept
