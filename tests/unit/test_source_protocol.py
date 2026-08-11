"""
Contract tests for the `JobSource` protocol.

Adapters are written in parallel once the schema is frozen, so this file exists
to make "does my adapter conform?" answerable without reading base.py. A
reference implementation lives here as the worked example.
"""
import inspect

import pytest

from backend.sources.base import (
    JobSource,
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
    SourceError,
)
from backend.sources.criteria import SearchCriteria
from backend.sources.schema import NormalizedJob, WorkMode


class ReferenceSource:
    """
    Minimal conforming adapter — the shape every real adapter must match.

    Copy this when starting a new provider.
    """

    name = "reference"
    provides_structured_dates = True
    provides_structured_salary = True

    def supports(self, criteria: SearchCriteria) -> bool:
        return True

    async def search(
        self, criteria: SearchCriteria, limit: int | None = None
    ) -> list[NormalizedJob]:
        return [
            NormalizedJob(
                provider=self.name,
                source_id="1",
                title="Backend Engineer",
                company="Acme",
                apply_url="https://example.com/jobs/1",
                work_mode=WorkMode.REMOTE,
            )
        ]


class MissingSearch:
    name = "broken"
    provides_structured_dates = False
    provides_structured_salary = False

    def supports(self, criteria: SearchCriteria) -> bool:
        return True


class MissingCapabilityFlags:
    name = "broken"

    def supports(self, criteria: SearchCriteria) -> bool:
        return True

    async def search(self, criteria, limit=None):
        return []


def test_reference_implementation_conforms():
    assert isinstance(ReferenceSource(), JobSource)


@pytest.mark.parametrize("incomplete", [MissingSearch, MissingCapabilityFlags])
def test_incomplete_implementations_do_not_conform(incomplete):
    """
    The capability flags are part of the contract, not documentation — the
    scorer trusts `provides_structured_dates` when deciding whether a recency
    score is meaningful, so an adapter that omits it must not pass as a source.
    """
    assert not isinstance(incomplete(), JobSource)


def test_search_is_async():
    """
    The registry fans out with asyncio.gather; a synchronous `search` would
    serialize every provider and blow the request timeout.
    """
    assert inspect.iscoroutinefunction(ReferenceSource.search)


@pytest.mark.asyncio
async def test_reference_search_returns_normalized_jobs():
    jobs = await ReferenceSource().search(SearchCriteria(titles=["Backend Engineer"]))
    assert all(isinstance(job, NormalizedJob) for job in jobs)
    assert jobs[0].provider == "reference"


def test_error_hierarchy_allows_catching_all_source_failures():
    """
    The registry catches SourceError to degrade to the next provider. Anything
    that escapes that base class fails the user's entire search instead of one
    provider's contribution to it.
    """
    for error in (ProviderUnavailable, ProviderQuotaExceeded, ProviderConfigError):
        assert issubclass(error, SourceError)


def test_quota_and_availability_errors_are_distinguishable():
    """
    They demand opposite responses: retry a transient outage, but back off hard
    on a quota refusal — retrying a 429 against a monthly cap burns what is
    left of the budget faster.
    """
    assert not issubclass(ProviderQuotaExceeded, ProviderUnavailable)
    assert not issubclass(ProviderUnavailable, ProviderQuotaExceeded)
