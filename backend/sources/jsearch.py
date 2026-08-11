"""
JSearch (RapidAPI) source adapter — the primary structured provider.

WHY THIS PROVIDER
-----------------
JSearch aggregates Google for Jobs, which is the only realistic route to
Egypt/MENA coverage with structured fields at a hobby price point. Wuzzuf,
Bayt, and Akhtaboot postings surface through Google for Jobs; no native API
indexes them directly at this tier.

Concretely, it returns the fields the snippet-parsing path could never produce:
a real `job_posted_at_datetime_utc` instead of the prose "2 weeks ago", and
numeric `job_min_salary`/`job_max_salary` instead of "EGP 40,000 - 60,000 per
month" as an unparsed string. Those two are what make the deterministic scorer
possible at all.

The free tier is small — low hundreds of requests per month — which is why this
adapter fetches one page per search by default and why quota accounting is not
optional.

FIELD-NAME STABILITY
--------------------
Every provider field is read with `.get()` and a fallback, and every record is
mapped inside a try/except that skips the individual record. JSearch has
renamed fields across versions (`job_employment_type` became
`job_employment_types`, a list), so an adapter that assumes a shape will one
day return zero jobs and log nothing. Both spellings are handled here.

The `live` smoke test in tests/live/ is what catches upstream schema drift;
mocked tests by construction cannot.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import httpx

from backend.core.config import get_settings
from backend.sources.base import (
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.http import new_async_client
from backend.sources.normalize import (
    first_non_empty,
    parse_iso_datetime,
    seniority_from_months,
    seniority_from_text,
    skills_from_text,
)
from backend.sources.schema import (
    EmploymentType,
    NormalizedJob,
    SalaryPeriod,
    WorkMode,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://{host}/search"

# JSearch returns roughly this many records per page. Used only to decide how
# many pages to request; the real count is whatever the response contains.
_RESULTS_PER_PAGE = 10

# Hard ceiling on pages per search. Each page is a separate billed request, and
# the free tier is measured in hundreds per month, so breadth is capped rather
# than driven by the caller's `limit`.
_MAX_PAGES = 2

# criteria.max_age_days -> JSearch's `date_posted` enum. Coarse by design: the
# provider offers only these buckets. Exact age filtering happens in the scorer
# against the real `posted_at`, so this only needs to avoid over-fetching.
_DATE_POSTED_BUCKETS: tuple[tuple[int, str], ...] = (
    (1, "today"),
    (3, "3days"),
    (7, "week"),
    (31, "month"),
)

_EMPLOYMENT_TYPE_TO_JSEARCH: dict[EmploymentType, str] = {
    EmploymentType.FULL_TIME: "FULLTIME",
    EmploymentType.PART_TIME: "PARTTIME",
    EmploymentType.CONTRACT: "CONTRACTOR",
    EmploymentType.INTERNSHIP: "INTERN",
    # TEMPORARY and UNKNOWN have no JSearch equivalent; omitting a filter is
    # correct, since a narrower filter would silently drop valid results.
}

_JSEARCH_TO_EMPLOYMENT_TYPE: dict[str, EmploymentType] = {
    "FULLTIME": EmploymentType.FULL_TIME,
    "PARTTIME": EmploymentType.PART_TIME,
    "CONTRACTOR": EmploymentType.CONTRACT,
    "INTERN": EmploymentType.INTERNSHIP,
    "TEMPORARY": EmploymentType.TEMPORARY,
}

_JSEARCH_TO_SALARY_PERIOD: dict[str, SalaryPeriod] = {
    "YEAR": SalaryPeriod.YEAR,
    "YEARLY": SalaryPeriod.YEAR,
    "MONTH": SalaryPeriod.MONTH,
    "MONTHLY": SalaryPeriod.MONTH,
    "WEEK": SalaryPeriod.WEEK,
    "WEEKLY": SalaryPeriod.WEEK,
    "DAY": SalaryPeriod.DAY,
    "DAILY": SalaryPeriod.DAY,
    "HOUR": SalaryPeriod.HOUR,
    "HOURLY": SalaryPeriod.HOUR,
}


def _to_float(value: Any) -> Optional[float]:
    """Coerce a provider number to float, or None. Never raises."""
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    # JSearch occasionally returns 0 for "unspecified". Zero salary is not a
    # real offer, and treating it as one would rank a job as paying nothing
    # rather than as not stating pay — the exact distinction the schema exists
    # to preserve.
    return result if result > 0 else None


class JSearchSource:
    """JSearch adapter. Conforms to `backend.sources.base.JobSource`."""

    name = "jsearch"
    provides_structured_dates = True
    provides_structured_salary = True

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        # An injected client lets the registry share one connection pool across
        # adapters, and lets tests drive respx without patching globals.
        self._client = client

    # ── Capability ──────────────────────────────────────────────────────────

    def supports(self, criteria: SearchCriteria) -> bool:
        """
        Whether calling this provider is worthwhile.

        Cheap and synchronous — it decides whether a billed request happens at
        all. Two reasons to decline:

        * No API key. Not an error: the app must work for someone who has not
          signed up for RapidAPI, falling through to the free providers.
        * Nothing to search for. JSearch's `query` is free text and a blank
          query returns an arbitrary sample of the global job market, which
          would burn a request to produce noise.
        """
        if not get_settings().jsearch_api_key:
            return False
        return bool(criteria.titles or criteria.skills)

    # ── Search ──────────────────────────────────────────────────────────────

    async def search(
        self,
        criteria: SearchCriteria,
        limit: int | None = None,
    ) -> list[NormalizedJob]:
        settings = get_settings()
        if not settings.jsearch_api_key:
            raise ProviderConfigError(
                "JSEARCH_API_KEY is not set; see .env.example"
            )

        effective_limit = limit or criteria.limit
        pages = max(1, min(_MAX_PAGES, math.ceil(effective_limit / _RESULTS_PER_PAGE)))

        params = self._build_params(criteria, pages)
        payload = await self._request(params, settings)

        records = payload.get("data") or []
        if not isinstance(records, list):
            # A shape change here is the difference between "no jobs" and "the
            # provider changed its envelope". Say which.
            raise ProviderUnavailable(
                f"JSearch returned an unexpected payload shape: "
                f"data was {type(records).__name__}, expected list"
            )

        jobs: list[NormalizedJob] = []
        skipped = 0
        for record in records:
            job = self._map_record(record)
            if job is None:
                skipped += 1
                continue
            jobs.append(job)
            if len(jobs) >= effective_limit:
                break

        if skipped:
            # Worth INFO, not DEBUG: a sudden rise here is the signature of an
            # upstream field rename, and it is otherwise invisible.
            logger.info(
                "jsearch: skipped %d/%d unmappable record(s)", skipped, len(records)
            )

        return jobs

    # ── Request construction ────────────────────────────────────────────────

    def _build_params(self, criteria: SearchCriteria, pages: int) -> dict[str, Any]:
        """
        Translate criteria into JSearch's query dialect.

        The `query` field is free text in the shape "<what> in <where>", which
        is what the provider's own examples use. Note the contrast with the
        prototype's search-engine queries, where the word "in" before a
        location actively suppressed results: that was an artifact of a
        keyword-matching web index, and does not apply to a jobs API that
        parses the phrase.
        """
        what = criteria.primary_title or " ".join(criteria.skills[:3])
        where = criteria.primary_location
        query = f"{what} in {where}" if where else what

        params: dict[str, Any] = {
            "query": query,
            "page": 1,
            "num_pages": pages,
            "date_posted": self._date_posted(criteria.max_age_days),
        }

        employment_types = [
            _EMPLOYMENT_TYPE_TO_JSEARCH[t]
            for t in criteria.employment_types
            if t in _EMPLOYMENT_TYPE_TO_JSEARCH
        ]
        if employment_types:
            params["employment_types"] = ",".join(employment_types)

        # Only narrow to remote when remote is the *only* acceptable mode.
        # Sending remote_jobs_only for a criteria that also accepts hybrid
        # would silently discard the hybrid results the user asked for.
        if criteria.work_modes and set(criteria.work_modes) == {WorkMode.REMOTE}:
            params["remote_jobs_only"] = "true"

        return params

    @staticmethod
    def _date_posted(max_age_days: int) -> str:
        for threshold, bucket in _DATE_POSTED_BUCKETS:
            if max_age_days <= threshold:
                return bucket
        return "all"

    async def _request(self, params: dict[str, Any], settings) -> dict[str, Any]:
        """Issue the HTTP call, mapping transport failures onto SourceErrors."""
        url = _BASE_URL.format(host=settings.jsearch_api_host)
        headers = {
            "X-RapidAPI-Key": settings.jsearch_api_key,
            "X-RapidAPI-Host": settings.jsearch_api_host,
        }

        try:
            if self._client is not None:
                response = await self._client.get(url, params=params, headers=headers)
            else:
                # new_async_client reuses a cached SSL context. Constructing a
                # client with httpx's default context costs 1.7-2.0s, which the
                # registry would otherwise pay once per provider per search.
                async with new_async_client(settings.source_http_timeout) as client:
                    response = await client.get(url, params=params, headers=headers)
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(f"JSearch timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"JSearch request failed: {exc}") from exc

        self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnavailable(
                f"JSearch returned non-JSON body: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ProviderUnavailable(
                f"JSearch returned {type(payload).__name__}, expected an object"
            )
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        """
        Map HTTP status onto the error the registry knows how to respond to.

        The quota/unavailable split matters: a 429 against a monthly cap must
        trip the breaker and stop, whereas a 503 should be retried. Collapsing
        them into one error means either retrying a quota failure (burning what
        budget remains faster) or giving up on a transient blip.
        """
        status = response.status_code
        if status == 429:
            raise ProviderQuotaExceeded("JSearch rate limit or quota exhausted")
        if status in (401, 403):
            raise ProviderConfigError(
                f"JSearch rejected the API key (HTTP {status}). "
                f"Check JSEARCH_API_KEY."
            )
        if status >= 500:
            raise ProviderUnavailable(f"JSearch server error (HTTP {status})")
        if status >= 400:
            raise ProviderUnavailable(f"JSearch rejected the request (HTTP {status})")

    # ── Record mapping ──────────────────────────────────────────────────────

    def _map_record(self, record: Any) -> Optional[NormalizedJob]:
        """
        Map one provider record onto `NormalizedJob`.

        Returns None instead of raising when a record cannot be mapped. One
        malformed posting — a missing apply link, a null employer — must not
        cost the user the other nine on the page.
        """
        if not isinstance(record, dict):
            return None

        try:
            apply_url = first_non_empty(
                record.get("job_apply_link"),
                record.get("job_google_link"),
            )
            title = first_non_empty(
                record.get("job_title"), record.get("job_job_title")
            )
            company = first_non_empty(record.get("employer_name"))

            # These three are load-bearing: without them the record is not a
            # usable job, and the schema would reject it anyway.
            if not (apply_url and title and company):
                return None

            city = first_non_empty(record.get("job_city"))
            country = first_non_empty(record.get("job_country"))
            state = first_non_empty(record.get("job_state"))
            location_raw = ", ".join(p for p in (city, state, country) if p) or None

            description = record.get("job_description") or ""

            return NormalizedJob(
                provider=self.name,
                source_id=str(
                    record.get("job_id") or f"{company}:{title}:{apply_url}"
                ),
                title=title,
                company=company,
                apply_url=apply_url,
                location_raw=location_raw,
                city=city,
                country=country,
                work_mode=self._work_mode(record),
                posted_at=parse_iso_datetime(
                    record.get("job_posted_at_datetime_utc")
                ),
                salary_min=_to_float(record.get("job_min_salary")),
                salary_max=_to_float(record.get("job_max_salary")),
                salary_currency=first_non_empty(record.get("job_salary_currency")),
                salary_period=self._salary_period(record),
                employment_type=self._employment_type(record),
                seniority=self._seniority(record, title),
                description=description,
                required_skills=self._skills(record, description),
                raw=record,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate per-record isolation
            logger.debug(
                "jsearch: could not map record %r: %s",
                record.get("job_id", "<no id>"),
                exc,
            )
            return None

    @staticmethod
    def _work_mode(record: dict) -> WorkMode:
        """
        Map JSearch's remote flag onto a work mode.

        Note the asymmetry: `job_is_remote=True` means REMOTE, but False means
        only "not flagged remote" — JSearch does not distinguish onsite from
        hybrid, so claiming ONSITE would be asserting something the provider
        never said. UNKNOWN is the honest mapping, and the schema has that
        member precisely so this case does not have to be faked.
        """
        return WorkMode.REMOTE if record.get("job_is_remote") is True else WorkMode.UNKNOWN

    @staticmethod
    def _salary_period(record: dict) -> Optional[SalaryPeriod]:
        raw = record.get("job_salary_period")
        if not raw:
            return None
        return _JSEARCH_TO_SALARY_PERIOD.get(str(raw).strip().upper())

    @staticmethod
    def _employment_type(record: dict) -> EmploymentType:
        """
        Read the employment type across both provider spellings.

        JSearch moved from a scalar `job_employment_type` to a list
        `job_employment_types`. Supporting both is what stops a provider-side
        rename from silently turning every job UNKNOWN.
        """
        values = record.get("job_employment_types")
        if isinstance(values, list) and values:
            raw = values[0]
        else:
            raw = record.get("job_employment_type")

        if not raw:
            return EmploymentType.UNKNOWN
        return _JSEARCH_TO_EMPLOYMENT_TYPE.get(
            str(raw).strip().upper(), EmploymentType.UNKNOWN
        )

    @staticmethod
    def _seniority(record: dict, title: str):
        """
        Prefer the structured experience requirement over title keywords.

        `required_experience_in_months` is a number the provider asserts;
        a title keyword is an inference. When both are absent the result is
        None, which the scorer treats as neutral rather than guessing.

        Only the title is passed to the text inference — a full description
        mentions "senior" in unrelated contexts ("reports to a senior
        manager") and would produce confident nonsense.
        """
        experience = record.get("job_required_experience")
        if isinstance(experience, dict):
            months = experience.get("required_experience_in_months")
            if isinstance(months, (int, float)) and not isinstance(months, bool):
                banded = seniority_from_months(int(months))
                if banded is not None:
                    return banded
        return seniority_from_text(title)

    @staticmethod
    def _skills(record: dict, description: str) -> list[str]:
        """
        Harvest skills through the shared CV vocabulary.

        `job_required_skills` is usually null in practice, so the description
        is the real source. Both go through `skills_from_text` so a posting's
        "Django" and a CV's "django" normalize identically and overlap is a set
        intersection rather than fuzzy matching.
        """
        provider_skills = record.get("job_required_skills")
        provider_text = (
            " ".join(str(s) for s in provider_skills)
            if isinstance(provider_skills, list)
            else None
        )
        return skills_from_text(provider_text, description)
