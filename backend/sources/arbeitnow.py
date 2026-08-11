"""
Arbeitnow source adapter — free, unauthenticated, Europe-weighted.

WHY THIS PROVIDER
-----------------
Like Remotive, it needs no key and no account, so it keeps answering when the
paid quota is gone. Its listings skew German and wider European, including
visa-sponsorship roles — directly relevant to a MENA-based candidate looking to
relocate, which the roadmap's other providers cover poorly.

CAPABILITIES
------------
`created_at` is a Unix timestamp, so recency scoring is trustworthy. There is
no salary field at all, so `provides_structured_salary` is False and salary is
always None — absent rather than guessed.

THE QUERY PROBLEM
-----------------
The public job-board feed is *paginated only*: it has no server-side search
parameter, so every call returns the newest page of the whole board regardless
of what was asked for.

That collides with the rule in base.py that an adapter translates rather than
judges. Passing the raw feed through would technically honour that rule while
flooding the merged result set with a hundred unrelated jobs per search —
diluting the ranking and, in Phase 2.3, spending LLM semantic-pass budget on
records nobody asked about.

The resolution is the coarse keyword gate in `sources.relevance`: a posting is
kept only if it mentions a requested title word or skill somewhere. That is
capability compensation, not relevance judgement — the server-side `search`
parameter this provider does not offer, implemented client-side. It answers "is
this even about the right subject?", never "is this a good match?", which
remains entirely the scorer's job.

That module is shared with Remotive, which turned out to have the same problem
for a less obvious reason: it accepts a `search` parameter and silently ignores
it, which is worse than not offering one.
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
    parse_unix_timestamp,
    seniority_from_text,
    skills_from_text,
    strip_html,
)
from backend.sources.schema import EmploymentType, NormalizedJob, WorkMode

logger = logging.getLogger(__name__)

_API_URL = "https://www.arbeitnow.com/api/job-board-api"

_USER_AGENT = "RecruitBot/1.0 (+https://github.com/YoussefIbrahim13/Job_Search-_Agent)"

# Pages to walk while looking for matches. Each is a free request, but the feed
# is ordered by recency, so walking deep trades latency for older jobs — a poor
# bargain when recency is a scoring component.
_MAX_PAGES = 3

_JOB_TYPE_MAP: dict[str, EmploymentType] = {
    "full-time": EmploymentType.FULL_TIME,
    "full_time": EmploymentType.FULL_TIME,
    "fulltime": EmploymentType.FULL_TIME,
    "part-time": EmploymentType.PART_TIME,
    "part_time": EmploymentType.PART_TIME,
    "contract": EmploymentType.CONTRACT,
    "freelance": EmploymentType.CONTRACT,
    "internship": EmploymentType.INTERNSHIP,
    "intern": EmploymentType.INTERNSHIP,
    "temporary": EmploymentType.TEMPORARY,
}


class ArbeitnowSource:
    """Arbeitnow adapter. Conforms to `backend.sources.base.JobSource`."""

    name = "arbeitnow"
    provides_structured_dates = True
    provides_structured_salary = False

    def __init__(self, client: Optional[httpx.AsyncClient] = None) -> None:
        self._client = client

    # ── Capability ──────────────────────────────────────────────────────────

    def supports(self, criteria: SearchCriteria) -> bool:
        """
        Requires something to gate on.

        With no titles and no skills every keyword is empty, the gate admits
        everything, and the adapter returns the newest page of an unrelated job
        board. Declining is strictly better than that.
        """
        return bool(query_keywords(criteria))

    # ── Search ──────────────────────────────────────────────────────────────

    async def search(
        self,
        criteria: SearchCriteria,
        limit: int | None = None,
    ) -> list[NormalizedJob]:
        settings = get_settings()
        effective_limit = limit or criteria.limit
        keywords = query_keywords(criteria)

        jobs: list[NormalizedJob] = []
        seen_urls: set[str] = set()
        skipped = 0
        gated_out = 0

        for page in range(1, _MAX_PAGES + 1):
            payload = await self._request({"page": page}, settings)

            records = payload.get("data")
            if not isinstance(records, list):
                raise ProviderUnavailable(
                    f"Arbeitnow returned an unexpected payload shape: data was "
                    f"{type(records).__name__}, expected list"
                )
            if not records:
                break

            for record in records:
                job = self._map_record(record)
                if job is None:
                    skipped += 1
                    continue
                if not mentions_any(job, keywords):
                    gated_out += 1
                    continue
                # The feed can repeat a posting across page boundaries as new
                # jobs shift the window.
                if job.apply_url in seen_urls:
                    continue
                seen_urls.add(job.apply_url)
                jobs.append(job)
                if len(jobs) >= effective_limit:
                    break

            if len(jobs) >= effective_limit:
                break

        if skipped or gated_out:
            logger.info(
                "arbeitnow: kept %d, gated out %d, unmappable %d",
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
            raise ProviderUnavailable(f"Arbeitnow timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderUnavailable(f"Arbeitnow request failed: {exc}") from exc

        if response.status_code == 429:
            raise ProviderQuotaExceeded("Arbeitnow throttled the request")
        if response.status_code >= 500:
            raise ProviderUnavailable(
                f"Arbeitnow server error (HTTP {response.status_code})"
            )
        if response.status_code >= 400:
            raise ProviderUnavailable(
                f"Arbeitnow rejected the request (HTTP {response.status_code})"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise ProviderUnavailable(
                f"Arbeitnow returned non-JSON body: {exc}"
            ) from exc

        if not isinstance(payload, dict):
            raise ProviderUnavailable(
                f"Arbeitnow returned {type(payload).__name__}, expected an object"
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
            location = first_non_empty(record.get("location"))

            tags = record.get("tags")
            tag_text = (
                " ".join(str(t) for t in tags) if isinstance(tags, list) else None
            )

            return NormalizedJob(
                provider=self.name,
                source_id=str(record.get("slug") or apply_url),
                title=title,
                company=company,
                apply_url=apply_url,
                location_raw=location,
                # `location` is a single free-text string that may hold a city,
                # a country, or "Remote". Splitting it into city/country would
                # be guesswork; the scorer compares against location_raw until
                # a geocoding step exists.
                city=None,
                country=None,
                work_mode=(
                    WorkMode.REMOTE if record.get("remote") is True else WorkMode.UNKNOWN
                ),
                posted_at=parse_unix_timestamp(record.get("created_at")),
                # Arbeitnow publishes no salary data at all. Absent, not zero.
                salary_min=None,
                salary_max=None,
                salary_currency=None,
                salary_period=None,
                employment_type=self._employment_type(record),
                seniority=seniority_from_text(title),
                description=description,
                required_skills=skills_from_text(tag_text, description),
                raw=record,
            )
        except Exception as exc:  # noqa: BLE001 - deliberate per-record isolation
            logger.debug(
                "arbeitnow: could not map record %r: %s",
                record.get("slug", "<no slug>"),
                exc,
            )
            return None

    @staticmethod
    def _employment_type(record: dict) -> EmploymentType:
        """
        Read the type from `job_types`, which is a list.

        Falls back to a scalar `job_type` in case the feed ever grows one —
        the same defensive pairing used in the JSearch adapter, for the same
        reason: a provider-side rename otherwise turns every job UNKNOWN with
        nothing in the logs.
        """
        values = record.get("job_types")
        if isinstance(values, list) and values:
            raw = values[0]
        else:
            raw = record.get("job_type")

        if not raw:
            return EmploymentType.UNKNOWN
        return _JOB_TYPE_MAP.get(str(raw).strip().lower(), EmploymentType.UNKNOWN)
