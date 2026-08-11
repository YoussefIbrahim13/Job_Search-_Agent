"""
Tavily source adapter — web search as the last-resort provider.

WHY KEEP IT AT ALL
------------------
Every other adapter reads a jobs API. Tavily reads the open web, which makes it
the only coverage for boards no API indexes — and for this app's market that is
not a rounding error. A Wuzzuf or Akhtaboot posting that Google for Jobs has not
picked up is invisible to JSearch and absent from the remote-only boards
entirely.

So it stays, but demoted: lowest priority, both capability flags False, and the
full legacy filter chain applied to everything it returns.

WHY ITS RESULTS ARE STRUCTURALLY WORSE
--------------------------------------
A JSearch record *is* a vacancy. A Tavily result is a web page that might be a
vacancy, might be a category listing, might be a five-year-old archived posting
that still re-crawls daily and therefore looks fresh to any crawl-date filter.

Consequently:

* `posted_at` is always None. Tavily reports when it crawled a page, not when
  the job was posted, and the prototype's habit of presenting one as the other
  is exactly the dishonesty this rewrite removes. No date is better than a
  wrong date: the scorer treats absence as neutral, and a wrong date would let
  an ancient posting win on recency.
* Salary is accepted only when a currency symbol or code is present in the
  snippet. Without that guard, "500+ applicants" or "5+ years" parses as pay.
* Company must be *derived* — from the ATS URL or the page title — and a
  record whose employer cannot be identified is dropped rather than labelled
  "Unknown". The prototype emitted placeholder companies and then needed a
  separate rule to discard them.

The honest summary: this adapter produces the weakest records in the system,
declares that through its capability flags, and lets the scorer weight it
accordingly.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Optional
from urllib.parse import urlparse

from backend.agents.prompts import APPROVED_SEARCH_BOARDS
from backend.core.config import get_settings
from backend.core.resilience import tavily_retry
from backend.ranking.geo import canonical_country, country_for_city, region_countries
from backend.sources.base import (
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.filters import FilterVerdict, survives_filter_chain
from backend.sources.normalize import (
    parse_salary_text,
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

# Board sets by market. Two site: tokens per query keeps this to a single
# Tavily search — it is the fallback provider, and spending several credits per
# call on the weakest records in the system is the wrong trade.
_MENA_BOARDS = ("wuzzuf", "bayt")
_REMOTE_BOARDS = ("weworkremotely", "remoteok")
_GLOBAL_BOARDS = ("linkedin", "indeed")

_PERIOD_MAP: dict[str, SalaryPeriod] = {
    "HOUR": SalaryPeriod.HOUR,
    "DAY": SalaryPeriod.DAY,
    "WEEK": SalaryPeriod.WEEK,
    "MONTH": SalaryPeriod.MONTH,
    "YEAR": SalaryPeriod.YEAR,
}

# Board names that must never be mistaken for an employer. The prototype
# routinely emitted "LinkedIn" as the hiring company.
_BOARD_NAMES: frozenset[str] = frozenset(
    {
        "linkedin", "indeed", "glassdoor", "wuzzuf", "bayt", "akhtaboot",
        "weworkremotely", "we work remotely", "remoteok", "remote ok",
        "himalayas", "wellfound", "dice", "greenhouse", "lever", "jobs",
        "careers", "job search", "naukrigulf", "monster",
    }
)

# "<Role> at <Company>" — the dominant shape across LinkedIn, Wuzzuf and Bayt.
#
# Case-insensitive on the first character on purpose: plenty of real employers
# are lowercase ("apexanalytix", "thoughtbot", "basecamp"), and requiring a
# capital silently dropped them. `_GENERIC_AFTER_AT` is what stops the looser
# pattern from matching ordinary prose.
_AT_COMPANY_RE = re.compile(
    r"\bat\s+([A-Za-z0-9][\w&.,'’\- ]{1,60}?)(?:\s*[|\-–—]|\s+in\s+|$)",
)
# "<Company> is hiring …" and LinkedIn's "<Company> hiring <Role> in <City>".
#
# The "is" is optional because LinkedIn omits it, and LinkedIn is the single
# largest source of individual-posting URLs in a Tavily result set. Requiring
# it made every LinkedIn result unidentifiable, which the live run showed as
# three real Cairo postings dropped for want of a company name.
_IS_HIRING_RE = re.compile(
    r"^([A-Za-z0-9][\w&.,'’\- ]{1,60}?)\s+(?:is\s+)?hiring\b"
)

# Words that follow "at" in ordinary phrasing rather than naming an employer.
_GENERIC_AFTER_AT: frozenset[str] = frozenset(
    {
        "the", "a", "an", "our", "your", "this", "least", "home", "scale",
        "work", "any", "all", "one", "two", "three", "up", "over", "about",
    }
)

# Direct-ATS URLs carry the employer in the path: /<company>/jobs/<id>.
_ATS_HOSTS = ("greenhouse.io", "lever.co", "ashbyhq.com", "workable.com")

_TRAILING_NOISE_RE = re.compile(
    r"\s*(?:[|\-–—]\s*)?(?:jobs?|careers?|hiring|vacanc(?:y|ies))\s*$",
    re.IGNORECASE,
)


def _clean_company(value: Optional[str]) -> Optional[str]:
    """Tidy an extracted employer name, or reject it."""
    if not value:
        return None
    cleaned = _TRAILING_NOISE_RE.sub("", value.strip()).strip(" -–—|,.")
    if len(cleaned) < 2 or len(cleaned) > 60:
        return None
    if cleaned.casefold() in _BOARD_NAMES:
        return None
    # "at the company", "at least 5 years" — prose, not an employer.
    if cleaned.split()[0].casefold() in _GENERIC_AFTER_AT:
        return None
    # A "company" that is entirely digits or punctuation is a parse artefact.
    if not re.search(r"[A-Za-z]", cleaned):
        return None
    return cleaned


def extract_company(title: str, url: str) -> Optional[str]:
    """
    Derive the employer from an ATS URL or a page title.

    Returns None when it cannot be established. The caller drops such records,
    which is deliberate: a listing attributed to "Unknown" — or worse, to
    "LinkedIn" — is misinformation, and the prototype needed a dedicated
    discard rule precisely because it manufactured those values.
    """
    host = (urlparse(url).netloc or "").lower()
    path_parts = [p for p in (urlparse(url).path or "").split("/") if p]

    if any(h in host for h in _ATS_HOSTS) and path_parts:
        # boards.greenhouse.io/<company>/jobs/<id>, jobs.lever.co/<company>/<id>
        candidate = _clean_company(path_parts[0].replace("-", " ").title())
        if candidate:
            return candidate

    hiring = _IS_HIRING_RE.search(title)
    if hiring:
        candidate = _clean_company(hiring.group(1))
        if candidate:
            return candidate

    at_company = _AT_COMPANY_RE.search(title)
    if at_company:
        candidate = _clean_company(at_company.group(1))
        if candidate:
            return candidate

    return None


class TavilySource:
    """Tavily adapter. Conforms to `backend.sources.base.JobSource`."""

    name = "tavily"

    # Both False, and this is the load-bearing part of the class. Tavily
    # reports crawl dates, not posting dates, and its salary figures are
    # regex-scraped from snippet prose. The scorer reads these flags to decide
    # how much to trust the corresponding fields.
    provides_structured_dates = False
    provides_structured_salary = False

    def __init__(self, client: Any = None) -> None:
        self._client = client

    # ── Capability ──────────────────────────────────────────────────────────

    def supports(self, criteria: SearchCriteria) -> bool:
        """
        Requires a key and something to search for.

        No key is not an error — like JSearch, the adapter simply reports
        itself unsupported and the free providers carry the search.
        """
        if not get_settings().tavily_api_key:
            return False
        return bool(criteria.titles or criteria.skills)

    # ── Search ──────────────────────────────────────────────────────────────

    async def search(
        self,
        criteria: SearchCriteria,
        limit: int | None = None,
    ) -> list[NormalizedJob]:
        settings = get_settings()
        if not settings.tavily_api_key:
            raise ProviderConfigError("TAVILY_API_KEY is not set; see .env.example")

        effective_limit = limit or criteria.limit
        query = self.build_query(criteria)

        # TavilyClient is synchronous. Off the event loop it goes, or it blocks
        # the concurrent fan-out it is supposed to be running alongside.
        response = await asyncio.to_thread(self._search_sync, query, settings)

        results = response.get("results")
        if not isinstance(results, list):
            raise ProviderUnavailable(
                f"Tavily returned an unexpected payload shape: results was "
                f"{type(results).__name__}, expected list"
            )

        jobs: list[NormalizedJob] = []
        drops: dict[str, int] = {}

        for result in results:
            if not isinstance(result, dict):
                continue

            url = str(result.get("url") or "")
            title = str(result.get("title") or "")
            snippet = str(result.get("content") or result.get("snippet") or "")

            verdict = survives_filter_chain(url, title, snippet)
            if verdict != FilterVerdict.KEPT:
                drops[verdict] = drops.get(verdict, 0) + 1
                continue

            job = self._map_result(url, title, snippet, result)
            if job is None:
                drops["unidentifiable"] = drops.get("unidentifiable", 0) + 1
                continue

            jobs.append(job)
            if len(jobs) >= effective_limit:
                break

        if drops:
            # Per-bucket counts, not a total. A single bucket swallowing most
            # of a search is the signature of an over-broad pattern, and it is
            # invisible in an aggregate number.
            logger.info(
                "tavily: kept %d/%d — dropped %s",
                len(jobs), len(results),
                ", ".join(f"{reason}={count}" for reason, count in sorted(drops.items())),
            )

        return jobs

    # ── Query construction ──────────────────────────────────────────────────

    def build_query(self, criteria: SearchCriteria) -> str:
        """
        Assemble a single site-scoped query.

        Deterministic, unlike the prototype, where the LLM wrote the query
        string and routinely dropped the location token — the single largest
        cause of empty result sets. Here the location is placed by code and
        cannot go missing.

        Note the absence of negative terms. The prototype appended
        `-"jobs in"` and similar, which collided with ordinary listing-page
        chrome and suppressed real results; category pages are rejected after
        retrieval by the filter chain instead.
        """
        parts: list[str] = []

        # Stack keywords first: a query without one returns generic career
        # advice and aggregator hubs.
        if criteria.skills:
            parts.extend(criteria.skills[:2])
        if criteria.primary_title:
            parts.append(criteria.primary_title)
        if criteria.primary_location:
            parts.append(criteria.primary_location)

        parts.append(
            "internship"
            if EmploymentType.INTERNSHIP in criteria.employment_types
            else "jobs"
        )

        boards = self._boards_for(criteria)
        site_clause = " OR ".join(
            APPROVED_SEARCH_BOARDS[key] for key in boards if key in APPROVED_SEARCH_BOARDS
        )
        if site_clause:
            parts.append(site_clause)

        return " ".join(parts)

    @staticmethod
    def _boards_for(criteria: SearchCriteria) -> tuple[str, ...]:
        """
        Choose which boards to scope the query to, by market.

        Regional boards are where MENA postings actually live; querying
        LinkedIn for a Cairo role returns a fraction of what Wuzzuf carries.
        """
        location = criteria.primary_location
        if not location or not criteria.locations:
            return _REMOTE_BOARDS + _GLOBAL_BOARDS[:1]

        country = canonical_country(location) or country_for_city(location)
        mena = region_countries("mena") or frozenset()
        if country and country in mena:
            return _MENA_BOARDS

        if criteria.work_modes and WorkMode.REMOTE in criteria.work_modes:
            return _REMOTE_BOARDS

        return _GLOBAL_BOARDS

    # ── Transport ───────────────────────────────────────────────────────────

    @tavily_retry
    def _search_sync(self, query: str, settings) -> dict:
        if self._client is not None:
            client = self._client
        else:
            try:
                from tavily import TavilyClient
            except ImportError as exc:
                raise ProviderConfigError(
                    "tavily-python is not installed"
                ) from exc
            client = TavilyClient(api_key=settings.tavily_api_key)

        from backend.sources.filters import build_exclude_domains

        try:
            return client.search(
                query=query,
                search_depth="advanced",
                max_results=settings.tavily_max_results,
                time_range=settings.tavily_time_range,
                exclude_domains=build_exclude_domains(),
                include_answer=False,
                include_raw_content=False,
            )
        except Exception as exc:
            text = str(exc).lower()
            if "rate limit" in text or "429" in text or "quota" in text:
                raise ProviderQuotaExceeded(f"Tavily quota exhausted: {exc}") from exc
            if "unauthor" in text or "invalid api key" in text or "403" in text:
                raise ProviderConfigError(f"Tavily rejected the API key: {exc}") from exc
            raise ProviderUnavailable(f"Tavily search failed: {exc}") from exc

    # ── Mapping ─────────────────────────────────────────────────────────────

    def _map_result(
        self, url: str, title: str, snippet: str, raw: dict
    ) -> Optional[NormalizedJob]:
        try:
            company = extract_company(title, url)
            if not company:
                return None

            clean_title = self._clean_title(title, company)
            if not clean_title:
                return None

            salary_min, salary_max, currency, period = parse_salary_text(snippet)
            if currency is None:
                # No currency marker means the number was probably an applicant
                # count or a years-of-experience figure, not pay.
                salary_min = salary_max = None
                period = None

            return NormalizedJob(
                provider=self.name,
                source_id=url,
                title=clean_title,
                company=company,
                apply_url=url,
                location_raw=None,
                city=None,
                country=None,
                work_mode=WorkMode.UNKNOWN,
                # Always None. Tavily reports crawl date, not posting date;
                # presenting one as the other is the dishonesty this rewrite
                # removes. The staleness filter has already dropped listings
                # whose own text admits to being old.
                posted_at=None,
                salary_min=salary_min,
                salary_max=salary_max,
                salary_currency=currency,
                salary_period=_PERIOD_MAP.get(period) if period else None,
                employment_type=self._employment_type(clean_title, snippet),
                seniority=seniority_from_text(clean_title),
                description=snippet,
                required_skills=skills_from_text(clean_title, snippet),
                raw=raw,
            )
        except Exception as exc:  # noqa: BLE001 - per-record isolation
            logger.debug("tavily: could not map %r: %s", url, exc)
            return None

    @staticmethod
    def _clean_title(title: str, company: str) -> str:
        """
        Strip board branding and the employer from a page title.

        "Senior .NET Developer at Sylndr - Cairo, Egypt | Wuzzuf" should become
        "Senior .NET Developer", or the title component scores a job against
        text that is mostly site furniture.
        """
        cleaned = title
        for separator in ("|", "•"):
            if separator in cleaned:
                cleaned = cleaned.split(separator)[0]

        # "<Role> at <Company> …"
        cleaned = re.sub(
            rf"\bat\s+{re.escape(company)}\b.*$", "", cleaned, flags=re.IGNORECASE
        )
        # "<Company> hiring <Role> in <City>" — LinkedIn's shape. Strip the
        # prefix and the trailing location, or the title component ends up
        # scoring against the company name and a city.
        cleaned = re.sub(
            rf"^{re.escape(company)}\s+(?:is\s+)?hiring\s+(?:an?\s+)?",
            "", cleaned, flags=re.IGNORECASE,
        )
        cleaned = re.sub(r"\s+in\s+[^,]+(?:,.*)?$", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.split(r"\s+[-–—]\s+", cleaned)[0]
        return cleaned.strip(" -–—|,")

    @staticmethod
    def _employment_type(title: str, snippet: str) -> EmploymentType:
        text = f"{title} {snippet}".lower()
        if re.search(r"\b(intern|internship|trainee)\b", text):
            return EmploymentType.INTERNSHIP
        if re.search(r"\bpart[\s-]time\b", text):
            return EmploymentType.PART_TIME
        if re.search(r"\b(contract|freelance|contractor)\b", text):
            return EmploymentType.CONTRACT
        if re.search(r"\bfull[\s-]time\b", text):
            return EmploymentType.FULL_TIME
        return EmploymentType.UNKNOWN
