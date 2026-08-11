"""
Source-adapter layer.

Replaces "parse a search-engine snippet and guess" with "read structured fields
from a jobs API". The two schemas exported here — `NormalizedJob` and
`SearchCriteria` — are the frozen contract that the adapters, the scorer, the
persistence layer, and the API all build against; see schema.py for the rules
on changing them.
"""

from backend.sources.base import (
    JobSource,
    ProviderConfigError,
    ProviderQuotaExceeded,
    ProviderUnavailable,
    SourceError,
)
from backend.sources.criteria import (
    DEFAULT_LIMIT,
    DEFAULT_MAX_AGE_DAYS,
    SearchCriteria,
)
from backend.sources.schema import (
    SENIORITY_ORDER,
    EmploymentType,
    NormalizedJob,
    SalaryPeriod,
    Seniority,
    WorkMode,
    normalize_url,
    slugify,
)

__all__ = [
    "DEFAULT_LIMIT",
    "DEFAULT_MAX_AGE_DAYS",
    "EmploymentType",
    "JobSource",
    "NormalizedJob",
    "ProviderConfigError",
    "ProviderQuotaExceeded",
    "ProviderUnavailable",
    "SENIORITY_ORDER",
    "SalaryPeriod",
    "SearchCriteria",
    "Seniority",
    "SourceError",
    "WorkMode",
    "normalize_url",
    "slugify",
]
