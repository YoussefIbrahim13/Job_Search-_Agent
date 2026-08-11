"""
Deterministic job scoring.

WHY THIS IS NOT THE LLM'S JOB
-----------------------------
The prototype's "ranking" was the order an LLM happened to emit jobs in, from a
prompt that asked it to score things it could not verify — it was handed prose
snippets and asked to judge location match, recency, and salary fit. That is
non-deterministic, unauditable, and wrong at the margins, and the `cap_score=75`
hack existed to paper over it.

Everything in this module is a fact comparison: does this city match that city,
is this date within that window, is this number bigger than that number. Facts
compare identically every time, cost nothing, and can be explained to the user.
The LLM keeps the one job it is genuinely better at — judging whether two
differently-worded roles are the same *kind* of role — and gets 10 of 100
points for it, in a separate batched pass (see `semantic.py`).

THE THREE-STATE RULE
--------------------
Every component distinguishes three situations, and conflating any two of them
is the main way a ranker goes quietly wrong:

1. **The user did not ask.** No location given, no salary expectation. The
   component cannot discriminate, so it awards full marks to everything. A
   constant offset changes no ordering, and it keeps scores from collapsing
   below the threshold for reasons the user never expressed.

2. **The user asked, the job does not say.** The job is not worse for it — it
   is unknown. Awards `NEUTRAL_FRACTION`. This is the rule that keeps MENA
   listings alive: they routinely omit salary, and scoring absence as zero
   would bury the app's core market beneath US postings that happen to publish
   a number.

3. **Both present.** A real comparison.

Every component records which of the three applied, so `ScoreBreakdown` can
explain any score without re-running it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Sequence

from backend.ranking.geo import means_anywhere, places_match
from backend.sources.normalize import skills_from_text
from backend.sources.criteria import SearchCriteria
from backend.sources.schema import (
    SENIORITY_ORDER,
    NormalizedJob,
    SalaryPeriod,
    Seniority,
    WorkMode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
#
# Sum to 90; the LLM semantic pass supplies the remaining 10. Location leads
# because it is the hardest constraint in practice — a perfect role in the
# wrong country is not a candidate, whereas a slightly stale posting still is.

# Sum to 90; the LLM semantic pass supplies the remaining 10.
#
# Title leads, and that is a correction from live results. The first version of
# this module had no title component at all — role matching had been delegated
# entirely to the LLM's 10 points, on the theory that "is this the same kind of
# job" is a semantic question. It is, but the semantic pass is explicitly
# designed to degrade, so whenever it was off or failing there was *zero* role
# matching. A live search for "Python Django Developer" returned "Inside Sales
# Contractor" at 90/100: remote (full location), fresh (full recency), and
# nothing anywhere comparing the title to the request.
#
# Literal token overlap between a requested title and a job title is a fact,
# and facts belong here. The LLM keeps what only it can do — recognising that
# "Platform Engineer" and "Backend Engineer" are the same role despite sharing
# no words.
#
# Location and recency were also cut. On a remote search they award nearly full
# marks to everything, so weight spent there buys almost no discrimination.
WEIGHT_TITLE = 30.0
WEIGHT_SKILLS = 20.0
WEIGHT_LOCATION = 20.0
WEIGHT_RECENCY = 10.0
WEIGHT_SALARY = 5.0
WEIGHT_SENIORITY = 5.0
WEIGHT_SEMANTIC = 10.0

DETERMINISTIC_MAX = (
    WEIGHT_TITLE + WEIGHT_SKILLS + WEIGHT_LOCATION
    + WEIGHT_RECENCY + WEIGHT_SALARY + WEIGHT_SENIORITY
)

# Awarded when the user expressed a preference but the job is silent. Half
# marks: not rewarded for withholding, not buried for it.
NEUTRAL_FRACTION = 0.5

# Tech tokens in a title are worth more than role nouns: "Django" narrows the
# field far more than "Developer", which appears in a third of all postings.
_TECH_TOKEN_WEIGHT = 2.0
_ROLE_TOKEN_WEIGHT = 1.0

# Seniority words are scored by their own component, and generic filler carries
# no signal. Both are stripped before title tokens are compared, so
# "Senior Python Developer" and "Python Developer" are treated as the same role
# rather than as a partial mismatch.
_TITLE_STOPWORDS: frozenset[str] = frozenset(
    {
        "senior", "junior", "lead", "principal", "staff", "mid", "entry",
        "level", "intern", "internship", "trainee", "graduate", "sr", "jr",
        "a", "an", "the", "of", "for", "in", "at", "to", "with", "and", "or",
        "i", "ii", "iii", "iv", "remote", "hybrid", "onsite", "contract",
        "fulltime", "parttime", "permanent", "freelance", "m", "f", "d", "w",
        "x", "all", "genders", "new", "job", "jobs", "role", "position",
    }
)

_TITLE_TOKEN_RE = re.compile(r"[a-z0-9+#\.]{2,}")

# Recency decays linearly to zero across this many days. Independent of
# `criteria.max_age_days`, which is the hard cutoff — a job inside the window
# still scores better for being fresher.
RECENCY_DECAY_DAYS = 30.0

# Default cutoff for `rank_jobs`. Tuned against the Phase 5 eval set once one
# exists; until then it is a starting point, not a finding.
DEFAULT_SCORE_THRESHOLD = 40.0

# Approximate FX rates to USD, for coarse salary comparison only.
#
# These ARE stale and will drift. They are acceptable here for three reasons:
# salary is 15 of 100 points, the comparison is a ratio rather than an exact
# figure, and the alternative — refusing to compare across currencies — would
# score every cross-border remote role as neutral, which is most of the good
# ones for this app's users. Any component computed with these is flagged
# `approximate_fx` so the UI can say so.
#
# Replace with a rates service before salary carries more weight than this.
_APPROX_USD_RATES: dict[str, float] = {
    "USD": 1.0, "EUR": 1.08, "GBP": 1.27, "CAD": 0.73, "AUD": 0.66,
    "EGP": 0.020, "AED": 0.27, "SAR": 0.27, "QAR": 0.27, "KWD": 3.25,
    "BHD": 2.65, "OMR": 2.60, "JOD": 1.41, "INR": 0.012, "TRY": 0.029,
    "MAD": 0.10, "TND": 0.32, "PKR": 0.0036,
}


class Applicability(str, Enum):
    """Which of the three states produced a component's score."""

    COMPARED = "compared"          # user asked, job answered
    NOT_REQUESTED = "not_requested"  # user expressed no preference
    JOB_SILENT = "job_silent"      # user asked, job did not say


@dataclass
class ScoreComponent:
    """One scored dimension, with enough detail to explain itself."""

    name: str
    score: float
    max_score: float
    applicability: Applicability
    detail: str = ""
    approximate_fx: bool = False

    @property
    def fraction(self) -> float:
        return self.score / self.max_score if self.max_score else 0.0


@dataclass
class ScoreBreakdown:
    """Why a job scored what it scored."""

    components: list[ScoreComponent] = field(default_factory=list)
    semantic_score: Optional[float] = None
    semantic_reason: Optional[str] = None

    @property
    def _scoring_components(self) -> list[ScoreComponent]:
        """
        Components that actually discriminated.

        `NOT_REQUESTED` components are excluded from both numerator and
        denominator rather than awarded full marks. Awarding them was a
        constant offset — harmless for ordering, but it inflated absolute
        scores so badly that the threshold stopped filtering: a search with no
        salary or seniority preference handed every job 30 free points out of
        90, and an entirely irrelevant listing scored 90/100.

        Excluding them normalizes over what the user actually expressed, so the
        threshold means the same thing regardless of how much detail they gave.
        """
        return [
            c for c in self.components
            if c.applicability is not Applicability.NOT_REQUESTED
        ]

    @property
    def earned(self) -> float:
        total = sum(c.score for c in self._scoring_components)
        if self.semantic_score is not None:
            total += self.semantic_score
        return total

    @property
    def available(self) -> float:
        total = sum(c.max_score for c in self._scoring_components)
        if self.semantic_score is not None:
            total += WEIGHT_SEMANTIC
        return total

    @property
    def total(self) -> float:
        """
        Score on a 0-100 scale.

        Normalized against what was actually available rather than a fixed 100,
        so a job scored before the semantic pass is directly comparable to one
        scored after it. Without this, every job would appear to lose 10 points
        for a pass that simply had not run yet, and any threshold would mean
        two different things depending on timing.
        """
        if not self.available:
            # The user expressed no preference on anything. Nothing was asked,
            # so nothing failed. Unreachable in practice — recency always
            # scores — but a silent zero here would drop every job.
            return 100.0
        return round(100.0 * self.earned / self.available, 1)

    def component(self, name: str) -> Optional[ScoreComponent]:
        return next((c for c in self.components if c.name == name), None)

    def as_dict(self) -> dict:
        """Serializable form for the API and for `JobResult.score_breakdown`."""
        return {
            "total": self.total,
            "components": [
                {
                    "name": c.name,
                    "score": round(c.score, 1),
                    "max_score": c.max_score,
                    "applicability": c.applicability.value,
                    "detail": c.detail,
                    "approximate_fx": c.approximate_fx,
                }
                for c in self.components
            ],
            "semantic_score": self.semantic_score,
            "semantic_reason": self.semantic_reason,
        }


@dataclass
class ScoredJob:
    """A job with its score. Ordering is by score, descending."""

    job: NormalizedJob
    breakdown: ScoreBreakdown

    @property
    def score(self) -> float:
        return self.breakdown.total


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


def _title_tokens(text: str) -> tuple[set[str], set[str]]:
    """
    Split a job title into (tech tokens, role tokens).

    Tech tokens come from the shared skills vocabulary, so "Django" in a title
    is the same token as "Django" in a CV. Everything left over that is not a
    seniority word or filler is a role token — "developer", "designer",
    "analyst" — which is what distinguishes one job family from another.
    """
    lowered = text.lower()
    tech = {skill.casefold() for skill in skills_from_text(text)}

    role = {
        token
        for token in _TITLE_TOKEN_RE.findall(lowered)
        if token not in _TITLE_STOPWORDS and token not in tech
    }
    return tech, role


def score_title(job: NormalizedJob, criteria: SearchCriteria) -> ScoreComponent:
    """
    Title / role-family fit (0-30), by weighted coverage of the requested title.

    Measures how much of what the user asked for appears in the job's title,
    not the reverse. A posting called "Senior Python Django Developer, EMEA"
    fully covers a request for "Python Django Developer"; the extra words are
    not a penalty.

    Deliberately literal. It cannot tell that "Platform Engineer" and "Backend
    Engineer" are the same role — that is the semantic pass's job, and this
    component scoring zero is exactly the case the LLM exists to rescue. What
    it does guarantee is that when the LLM is unavailable, an unrelated title
    is still recognised as unrelated instead of scoring full marks by default.
    """
    name = "title"

    if not criteria.titles:
        return ScoreComponent(
            name, WEIGHT_TITLE, WEIGHT_TITLE,
            Applicability.NOT_REQUESTED, "no target title given",
        )

    job_tech, job_role = _title_tokens(job.title)

    best_fraction = 0.0
    best_detail = "no shared terms with the requested title"

    for wanted in criteria.titles:
        want_tech, want_role = _title_tokens(wanted)
        possible = (
            _TECH_TOKEN_WEIGHT * len(want_tech) + _ROLE_TOKEN_WEIGHT * len(want_role)
        )
        if not possible:
            continue

        matched_tech = want_tech & job_tech
        matched_role = want_role & job_role
        earned = (
            _TECH_TOKEN_WEIGHT * len(matched_tech)
            + _ROLE_TOKEN_WEIGHT * len(matched_role)
        )
        fraction = earned / possible

        if fraction > best_fraction:
            best_fraction = fraction
            shared = sorted(matched_tech | matched_role)
            best_detail = (
                "title shares " + ", ".join(shared) if shared
                else "no shared terms with the requested title"
            )

    if best_fraction == 0.0 and not any(_title_tokens(t)[0] or _title_tokens(t)[1]
                                        for t in criteria.titles):
        # The requested title was entirely stopwords ("Senior role"), so there
        # was nothing to compare. Not the job's fault.
        return ScoreComponent(
            name, WEIGHT_TITLE, WEIGHT_TITLE,
            Applicability.NOT_REQUESTED, "requested title had no distinctive terms",
        )

    return ScoreComponent(
        name, WEIGHT_TITLE * best_fraction, WEIGHT_TITLE,
        Applicability.COMPARED, best_detail,
    )


def score_location(job: NormalizedJob, criteria: SearchCriteria) -> ScoreComponent:
    """
    Location fit (0-25).

    Remote is treated as satisfying any requested location, not as a separate
    lesser case. A fully-remote role is one a Cairo-based candidate can take
    without moving, which makes it a *better* location match than a job in the
    next city, not a worse one. The prototype scored remote at 5 of 30 —
    below "same country" — which systematically buried the roles most valuable
    to its users.
    """
    name = "location"

    if not criteria.locations and not criteria.work_modes:
        return ScoreComponent(
            name, WEIGHT_LOCATION, WEIGHT_LOCATION,
            Applicability.NOT_REQUESTED, "no location preference given",
        )

    # Remote satisfies any location the user asked for, provided they accept it.
    if job.work_mode is WorkMode.REMOTE and criteria.remote_ok:
        return ScoreComponent(
            name, WEIGHT_LOCATION, WEIGHT_LOCATION,
            Applicability.COMPARED, "fully remote",
        )

    # An explicit onsite-only search should not be won by a remote listing.
    if job.work_mode is WorkMode.REMOTE and not criteria.remote_ok:
        return ScoreComponent(
            name, 0.0, WEIGHT_LOCATION, Applicability.COMPARED,
            "remote, but the search excludes remote",
        )

    if not criteria.locations:
        # Only a work-mode preference was given, and this job is not remote.
        matched = job.work_mode in criteria.work_modes
        return ScoreComponent(
            name,
            WEIGHT_LOCATION if matched else WEIGHT_LOCATION * NEUTRAL_FRACTION,
            WEIGHT_LOCATION,
            Applicability.COMPARED if matched else Applicability.JOB_SILENT,
            f"work mode {job.work_mode.value}",
        )

    candidates = [job.city, job.country, job.location_raw]
    if not any(candidates):
        return ScoreComponent(
            name, WEIGHT_LOCATION * NEUTRAL_FRACTION, WEIGHT_LOCATION,
            Applicability.JOB_SILENT, "job states no location",
        )

    # "Worldwide" / "Anywhere" in a location field means unconstrained, which
    # for a candidate anywhere is as good as a match.
    if any(means_anywhere(value) for value in candidates if value):
        return ScoreComponent(
            name, WEIGHT_LOCATION, WEIGHT_LOCATION,
            Applicability.COMPARED, "open to any location",
        )

    best_city = False
    best_country = False
    for wanted in criteria.locations:
        for value in candidates:
            if not value:
                continue
            city_hit, country_hit = places_match(wanted, value)
            best_city = best_city or city_hit
            best_country = best_country or country_hit
        if best_city:
            break

    if best_city:
        return ScoreComponent(
            name, WEIGHT_LOCATION, WEIGHT_LOCATION,
            Applicability.COMPARED, "same city",
        )
    if best_country:
        # Same market, different city. Still a real option: whether the commute
        # or the move is acceptable is the candidate's call, not the ranker's.
        return ScoreComponent(
            name, WEIGHT_LOCATION * 0.6, WEIGHT_LOCATION,
            Applicability.COMPARED, "same country, different city",
        )

    return ScoreComponent(
        name, 0.0, WEIGHT_LOCATION, Applicability.COMPARED, "different location",
    )


def score_recency(
    job: NormalizedJob,
    criteria: SearchCriteria,
    now: Optional[datetime] = None,
) -> ScoreComponent:
    """
    Recency (0-20), decaying linearly over 30 days.

    This component was impossible before Phase 2: there was no real posting
    date, only the prose the model copied out of a snippet. It is the clearest
    single argument for the structured-provider rewrite.

    Note it decays on *posting* date, not crawl date. The prototype's Tavily
    `time_range` filtered on when the page was last crawled, which SEO-active
    zombie listings pass trivially — a five-year-old posting whose "similar
    jobs" widget updates daily looks fresh to a crawler and ancient to a reader.
    """
    name = "recency"
    now = now or datetime.now(timezone.utc)

    if job.posted_at is None:
        return ScoreComponent(
            name, WEIGHT_RECENCY * NEUTRAL_FRACTION, WEIGHT_RECENCY,
            Applicability.JOB_SILENT, "no posting date available",
        )

    age_days = (now - job.posted_at).total_seconds() / 86400.0

    if age_days < 0:
        # A future-dated posting is a provider quirk, not a fresher job.
        return ScoreComponent(
            name, WEIGHT_RECENCY, WEIGHT_RECENCY,
            Applicability.COMPARED, "posted today",
        )

    if age_days > criteria.max_age_days:
        return ScoreComponent(
            name, 0.0, WEIGHT_RECENCY, Applicability.COMPARED,
            f"posted {age_days:.0f} days ago, outside the "
            f"{criteria.max_age_days}-day window",
        )

    fraction = max(0.0, 1.0 - (age_days / RECENCY_DECAY_DAYS))
    return ScoreComponent(
        name, WEIGHT_RECENCY * fraction, WEIGHT_RECENCY,
        Applicability.COMPARED, f"posted {age_days:.0f} day(s) ago",
    )


def _to_usd(amount: Optional[float], currency: Optional[str]) -> Optional[float]:
    if amount is None or not currency:
        return None
    rate = _APPROX_USD_RATES.get(currency.upper())
    return amount * rate if rate is not None else None


def score_salary(job: NormalizedJob, criteria: SearchCriteria) -> ScoreComponent:
    """
    Salary fit (0-15).

    The important behaviour here is what happens when a job states no salary:
    it scores neutral, never zero. MENA postings omit salary as a matter of
    course, and treating silence as "pays nothing" would rank the app's core
    market below any US listing that happens to publish a number — an
    ordering produced entirely by disclosure convention rather than by pay.
    """
    name = "salary"

    if criteria.salary_min is None:
        return ScoreComponent(
            name, WEIGHT_SALARY, WEIGHT_SALARY,
            Applicability.NOT_REQUESTED, "no salary expectation given",
        )

    annual = job.annual_salary_range
    if annual is None:
        return ScoreComponent(
            name, WEIGHT_SALARY * NEUTRAL_FRACTION, WEIGHT_SALARY,
            Applicability.JOB_SILENT, "salary not stated",
        )

    job_low, job_high = annual
    job_best = job_high if job_high is not None else job_low

    wanted_multiplier = {
        SalaryPeriod.YEAR: 1.0, SalaryPeriod.MONTH: 12.0, SalaryPeriod.WEEK: 52.0,
        SalaryPeriod.DAY: 260.0, SalaryPeriod.HOUR: 2080.0,
    }.get(criteria.salary_period or SalaryPeriod.YEAR, 1.0)
    wanted_annual = criteria.salary_min * wanted_multiplier

    approximate = False
    job_currency = job.salary_currency
    wanted_currency = criteria.salary_currency

    if job_currency and wanted_currency and job_currency != wanted_currency:
        job_usd = _to_usd(job_best, job_currency)
        wanted_usd = _to_usd(wanted_annual, wanted_currency)
        if job_usd is None or wanted_usd is None:
            # An unknown currency is a genuine unknown. Guessing a rate would
            # be worse than declining to compare.
            return ScoreComponent(
                name, WEIGHT_SALARY * NEUTRAL_FRACTION, WEIGHT_SALARY,
                Applicability.JOB_SILENT,
                f"cannot compare {job_currency} against {wanted_currency}",
            )
        job_best, wanted_annual = job_usd, wanted_usd
        approximate = True

    if job_best is None or wanted_annual <= 0:
        return ScoreComponent(
            name, WEIGHT_SALARY * NEUTRAL_FRACTION, WEIGHT_SALARY,
            Applicability.JOB_SILENT, "salary not comparable",
        )

    ratio = job_best / wanted_annual
    if ratio >= 1.0:
        # Paying above the expectation is not extra credit — the expectation is
        # met. Scaling further would rank purely on pay.
        return ScoreComponent(
            name, WEIGHT_SALARY, WEIGHT_SALARY, Applicability.COMPARED,
            "meets or exceeds expectation", approximate_fx=approximate,
        )

    return ScoreComponent(
        name, WEIGHT_SALARY * max(0.0, ratio), WEIGHT_SALARY,
        Applicability.COMPARED,
        f"about {ratio:.0%} of expectation", approximate_fx=approximate,
    )


def score_seniority(job: NormalizedJob, criteria: SearchCriteria) -> ScoreComponent:
    """
    Seniority fit (0-15), by distance along `SENIORITY_ORDER`.

    Symmetric: a senior candidate shown a junior role and a junior candidate
    shown a senior role are both mismatches, and neither is the ranker's to
    forgive. The candidate can always apply anyway; the job simply should not
    outrank one at their actual level.
    """
    name = "seniority"

    if criteria.seniority is None:
        return ScoreComponent(
            name, WEIGHT_SENIORITY, WEIGHT_SENIORITY,
            Applicability.NOT_REQUESTED, "no seniority given",
        )
    if job.seniority is None:
        return ScoreComponent(
            name, WEIGHT_SENIORITY * NEUTRAL_FRACTION, WEIGHT_SENIORITY,
            Applicability.JOB_SILENT, "job states no seniority",
        )

    order = list(SENIORITY_ORDER)
    distance = abs(order.index(job.seniority) - order.index(criteria.seniority))

    fraction = {0: 1.0, 1: 0.6, 2: 0.3}.get(distance, 0.0)
    detail = (
        "exact level" if distance == 0
        else f"{distance} level(s) from {criteria.seniority.value}"
    )
    return ScoreComponent(
        name, WEIGHT_SENIORITY * fraction, WEIGHT_SENIORITY,
        Applicability.COMPARED, detail,
    )


def score_skills(job: NormalizedJob, criteria: SearchCriteria) -> ScoreComponent:
    """
    Skill overlap (0-15), as an overlap coefficient.

    Uses |A∩B| / min(|A|,|B|) rather than Jaccard. Jaccard divides by the union,
    which punishes asymmetry — and asymmetry is the normal case here: a CV lists
    thirty skills, a posting lists four. A candidate matching all four would
    score 4/30 under Jaccard, which reads as a poor match when it is a perfect
    one. The overlap coefficient asks the question actually being posed: of the
    smaller list, how much is covered?

    Both sides come from `cv_parser._SKILLS_DICTIONARY` via
    `normalize.skills_from_text`, so this is a set intersection on identical
    tokens rather than fuzzy string matching.
    """
    name = "skills"

    wanted = {s.casefold() for s in criteria.skills}
    if not wanted:
        return ScoreComponent(
            name, WEIGHT_SKILLS, WEIGHT_SKILLS,
            Applicability.NOT_REQUESTED, "no skills given",
        )

    have = {s.casefold() for s in job.required_skills}
    if not have:
        return ScoreComponent(
            name, WEIGHT_SKILLS * NEUTRAL_FRACTION, WEIGHT_SKILLS,
            Applicability.JOB_SILENT, "job lists no skills",
        )

    shared = wanted & have
    overlap = len(shared) / min(len(wanted), len(have))
    matched = ", ".join(sorted(s for s in job.required_skills
                               if s.casefold() in shared)[:5])

    return ScoreComponent(
        name, WEIGHT_SKILLS * overlap, WEIGHT_SKILLS, Applicability.COMPARED,
        f"{len(shared)} shared skill(s)" + (f": {matched}" if matched else ""),
    )


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------


def score_job(
    job: NormalizedJob,
    criteria: SearchCriteria,
    now: Optional[datetime] = None,
) -> ScoreBreakdown:
    """Score one job against the criteria. Pure: no I/O, no LLM, no clock skew."""
    return ScoreBreakdown(
        components=[
            score_title(job, criteria),
            score_skills(job, criteria),
            score_location(job, criteria),
            score_recency(job, criteria, now=now),
            score_salary(job, criteria),
            score_seniority(job, criteria),
        ]
    )


def has_relevance_evidence(breakdown: ScoreBreakdown) -> bool:
    """
    Whether anything at all connects this job to what was asked for.

    Three independent signals count: a shared title term, a shared skill, or a
    positive semantic score. Any one is enough.

    The failure this prevents is specific and was observed live — a fully
    remote, freshly-posted "Inside Sales Contractor" scoring above a real
    Django role, because location and recency awarded near-full marks and
    nothing else discriminated. Such a job has *no* connection to the request;
    it is not a weak match, it is not a match.

    Requiring all three to be absent, rather than gating on the title alone, is
    what keeps the legitimate case alive: "Backend Engineer" shares no word
    with "Python Django Developer", but a posting whose description mentions
    Django still has skill evidence and survives — and if the semantic pass is
    running, it rescues the rest.

    When the user gave neither a title nor skills, there is nothing to gate on
    and everything passes.
    """
    title = breakdown.component("title")
    skills = breakdown.component("skills")

    gateable = [
        c for c in (title, skills)
        if c is not None and c.applicability is not Applicability.NOT_REQUESTED
    ]
    if not gateable:
        return True

    if any(c.score > 0 and c.applicability is Applicability.COMPARED for c in gateable):
        return True

    # A JOB_SILENT component is an unknown, not a contradiction — it cannot
    # supply evidence, but it must not be the sole reason for dropping a job
    # either. Only reaching here with every gateable signal at zero is decisive.
    if all(c.applicability is Applicability.JOB_SILENT for c in gateable):
        return True

    return bool(breakdown.semantic_score)


def rank_jobs(
    jobs: Sequence[NormalizedJob],
    criteria: SearchCriteria,
    *,
    threshold: float = DEFAULT_SCORE_THRESHOLD,
    limit: Optional[int] = None,
    now: Optional[datetime] = None,
    gate: bool = True,
) -> list[ScoredJob]:
    """
    Score, filter, and order a set of jobs.

    `gate=False` skips the relevance gate. `rank_with_semantic` uses it for the
    first pass, because a job with no deterministic connection to the request
    is precisely the case the LLM might rescue — dropping it before the model
    sees it would remove the semantic pass's whole reason for existing.

    Ties break on `canonical_key` rather than being left to sort stability.
    Two jobs with identical scores would otherwise order by whichever provider
    happened to answer first, making the same search return different orders on
    different runs — the kind of nondeterminism that makes an eval set useless.
    """
    scored = [ScoredJob(job=job, breakdown=score_job(job, criteria, now=now))
              for job in jobs]

    if gate:
        relevant = [s for s in scored if has_relevance_evidence(s.breakdown)]
        if len(relevant) < len(scored):
            logger.info(
                "ranking: dropped %d/%d job(s) with no relevance to the query",
                len(scored) - len(relevant), len(scored),
            )
    else:
        relevant = scored

    kept = [s for s in relevant if s.score >= threshold]
    kept.sort(key=lambda s: (-s.score, s.job.canonical_key))

    if len(kept) < len(relevant):
        logger.info(
            "ranking: %d/%d job(s) scored below the %.0f threshold",
            len(relevant) - len(kept), len(relevant), threshold,
        )

    return kept[:limit] if limit else kept
