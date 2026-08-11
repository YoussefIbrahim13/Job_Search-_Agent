"""
Remotive source adapter — free, unauthenticated, remote-only.

WHY THIS PROVIDER
-----------------
Remotive is a curated remote-work board with a public JSON API that needs no
key and no account. It costs nothing to query and nothing to rate-limit
against, which makes it the natural floor of the provider stack: when JSearch's
monthly quota is exhausted, this still answers.

It matters disproportionately for this app's users. A candidate in Cairo or
Amman competing for local onsite roles is limited by the local market; remote
roles are the ones where their salary ceiling is set by the employer's market
instead. Remote coverage is not a nice-to-have here.

NO SERVER-SIDE SEARCH
---------------------
Remotive accepts `search`, `category`, and `limit` and ignores all three.
Probing the live endpoint showed every combination — including no parameters at
all — returning the identical twenty newest jobs. Trusting the parameter fed
twenty sales and copywriting roles into a Django search. Relevance is therefore
enforced client-side via `sources.relevance`, exactly as for Arbeitnow.

CAPABILITIES
------------
`publication_date` is a real timestamp, so recency scoring is trustworthy.
Salary is a free-text string ("$120,000 - $150,000"), so figures are parsed
best-effort and `provides_structured_salary` is False — the flag tells the
scorer these numbers are inferred, which is the honest pairing.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from backend.core.config import get_settings
from backend.sources.base import ProviderQuotaExceeded, ProviderUnavailable
from backend.sources.criteria import SearchCriteria
from backend.sources.http import new_async_client
from backend.sources.relevance import mentions_any, query_keywords
from backend.sources.normalize import (
    first_non_empty,
    parse_iso_datetime,
    parse_salary_text,
    seniority_from_text,
    skills_from_text,
    strip_html,
)
from backend.sources.schema import (
    EmploymentType,
    NormalizedJob,
    SalaryPeriod,
    WorkMode,
)

logger = logging.getLogger(__name__)

_API_URL = "https://remotive.com/api/remote-jobs"

# Remotive asks API consumers to identify themselves. A generic client string
# is what gets throttled first.
_USER_AGENT = "RecruitBot/1.0 (+https://github.com/YoussefIbrahim13/Job_Search-_Agent)"

_JOB_TYPE_MAP: dict[str, EmploymentType] = {
    "full_time": EmploymentType.FULL_TIME,
    "part_time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "freelance": EmploymentType.CONTRACT,
    "internship": EmploymentType.INTERNSHIP,
    "temporary": EmploymentType.TEMPORARY,
    "other": EmploymentType.UNKNOWN,
}

_PERIOD_MAP: dict[str, SalaryPeriod] = {
    "HOUR": SalaryPeriod.HOUR,
    "DAY": SalaryPeriod.DAY,
    "WEEK": SalaryPeriod.WEEK,
    "MONTH": SalaryPeriod.MONTH,
    "YEAR": SalaryPeriod.YEAR,
}


class RemotiveSource:
    """Remotive adapter. Conforms to `backend.sources.base.JobSource`."""

    name = "remotive"
    provides_structured_dates = True
    provides_structured_salary = False

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client

    # ── Capability ──────────────────────────────────────────────────────────

    def supports(self, criteria: SearchCriteria) -> bool:
        """
        Every Remotive listing is remote, so the only disqualifying case is a
        search that has explicitly ruled remote out.

        Note what is deliberately *not* checked: the requested location. A
        Cairo-based candidate searching "Cairo" should still see worldwide
        remote roles, because those are jobs they can actually take. Filtering
        on location here would remove the provider's entire value. Whether a
        given remote role's timezone or region requirement fits is a scoring
        question, not a fetch question.
        """
        if not criteria.remote_ok:
            return False
        return bool(criteria.titles or criteria.skills)

    # ── Search ──────────────────────────────────────────────────────────────

    async def search(
        self,
        criteria: SearchCriteria,
        limit: int | None = None,
    ) -> list[NormalizedJob]:
        settings = get_settings()
        effective_limit = limit or criteria.limit

        # These parameters are sent but NOT relied upon. Probing the live
        # endpoint showed Remotive accepts `search`, `category`, and `limit`
        # without error and ignores all three: every combination returned the
        # identical twenty newest jobs. They are kept because they cost nothing
        # and would start working if the provider ever honours them; the
        # keyword gate below is what actually makes the results relevant.
        params: dict[str, Any] = {"limit": effective_limit}
        search_term = criteria.primary_title or " ".join(criteria.skills[:3])
        if search_term:
            params["search"] = search_term

        payload = await self._request(params, settings)

        records = payload.get("jobs")
        if not isinstance(records, list):
            raise ProviderUnavailable(
                f"Remotive returned an unexpected payload shape: jobs was "
                f"{type(records).__name__}, expected list"
            )

        keywords = query_keywords(criteria)

        jobs: list[NormalizedJob] = []
        skipped = 0
        gated_out = 0
        for record in records:
            job = self._map_record(record)
            if job is None:
                skipped += 1
                continue
            # Gate before the limit, not after. Truncating first would fill the
            # quota with whatever happened to be newest and discard the
            # relevant jobs further down the feed.
            if not mentions_any(job, keywords):
                gated_out += 1
                continue
            jobs.append(job)
            if len(jobs) >= effective_limit:
                break

        if skipped or gated_out:
            logger.info(
                "remotive: kept %d, gated out %d, unmappable %d",
                len(jobs), gated_out, skipped,
            )
        return jobs

    async def _request(self, params: dict[str, Any], settings) -> dict[str, Any]:
        headers = {"User-Agent": _USER_AGENT, "Accept": "application/json"}
        try:
            if self._client is not None:
                response = await self._client.get(
                    _API_URL, params=params, headers=headers
                )
            else:
                async with new_async_client(settings.source_http_timeout) as client:
                    response = await client.get(
                        _API_URL, params=params, headers=headers
                    )
        except httpx.TimeoutException as exc:
            raise ProviderUnavailable(f"Remotive timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"Remotive request failed: {exc}") from exc

        if response.status_code == 429:
            # No API key means no quota to exhaust, but the endpoint is still
            # throttled by IP. Same class, so the registry backs off rather
            # than hammering a source that is asking for room.
            raise ProviderQuotaExceeded("Remotive throttled the request")
        if response.status_code >= 500:
            raise ProviderUnavailable(
                f"Remotive server error (HTTP {response.status_code})"
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"Remotive rejected the request (HTTP {response.status_code})"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnavailable(f"Remotive returned non-JSON body: {exc}") from exc

        if not isinstance(payload, dict):
            raise ProviderUnavailable(
                f"Remotive returned {type(payload).__name__}, expected an object"
            )
        return payload

    # ── Record mapping ──────────────────────────────────────────────────────

    def _map_record(self, record: Any) -> Optional[NormalizedJob]:
        if not isinstance(record, dict):
            return None

        try:
            apply_url = first_non_empty(record.get("url"))
            title = first_non_empty(record.get("title"))
            company = first_non_empty(record.get("company_name"))
            if not (apply_url and title and company):
                return None

            description = strip_html(record.get("description"))
            required_location = first_non_empty(
                record.get("candidate_required_location")
            )

            salary_min, salary_max, currency, period = parse_salary_text(
                record.get("salary")
            )

            tags = record.get("tags")
            tag_text = (
                " ".join(str(t) for t in tags) if isinstance(tags, list) else None
            )

            return NormalizedJob(
                provider=self.name,
                source_id=str(record.get("id") or apply_url),
                title=title,
                company=company,
                apply_url=apply_url,
                # The location string describes where the candidate may be
                # ("Worldwide", "Europe"), not where the job is. It is kept as
                # location_raw but deliberately not parsed into city/country:
                # "Worldwide" is not a city, and inventing one would make the
                # scorer's location comparison confidently wrong.
                location_raw=required_location,
                city=None,
                country=None,
                # Every Remotive listing is remote by definition of the board.
                work_mode=WorkMode.REMOTE,
                posted_at=parse_iso_datetime(record.get("publication_date")),
                salary_min=salary_min,
                salary_max=salary_max,
                salary_currency=currency,
                salary_period=_PERIOD_MAP.get(period) if period else None,
                employment_type=self._employment_type(record),
                seniority=seniority_from_text(title),
                description=description,
                required_skills=skills_from_text(tag_text, description),
                raw=record,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate per-record isolation
            logger.debug(
                "remotive: could not map record %r: %s", record.get("id", "<no id>"), exc
            )
            return None

    @staticmethod
    def _employment_type(record: dict) -> EmploymentType:
        raw = record.get("job_type")
        if not raw:
            return EmploymentType.UNKNOWN
        return _JOB_TYPE_MAP.get(str(raw).strip().lower(), EmploymentType.UNKNOWN)
