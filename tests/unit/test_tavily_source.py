"""
Tests for the Tavily adapter and the shared filter chain.

This adapter produces the weakest records in the system, so most of these tests
are about what it must *refuse* to claim: no posting date, no salary without a
currency marker, no employer it had to invent. The prototype manufactured all
three, and then needed extra rules downstream to discard its own output.

Company extraction gets the most coverage. It is the one field with no source
of truth in a search result — it has to be derived from a URL or a page title —
and getting it wrong means attributing a job to the wrong employer, which is
worse than not showing the job.
"""
import pytest

from backend.core.config import get_settings
from backend.sources.base import JobSource, ProviderConfigError
from backend.sources.criteria import SearchCriteria
from backend.sources.filters import FilterVerdict, survives_filter_chain
from backend.sources.schema import EmploymentType, SalaryPeriod, WorkMode
from backend.sources.tavily import TavilySource, extract_company


@pytest.fixture(autouse=True)
def _configure_key(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-tavily-key")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class FakeTavilyClient:
    """Stands in for TavilyClient, which is synchronous."""

    def __init__(self, results=None, error=None):
        self._results = results if results is not None else []
        self._error = error
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if self._error is not None:
            raise self._error
        return {"results": self._results}


def result(url, title, content=""):
    return {"url": url, "title": title, "content": content, "score": 0.9}


@pytest.fixture
def criteria():
    return SearchCriteria(
        titles=["Backend Engineer"], locations=["Cairo"], skills=["Python", "Django"]
    )


# ===========================================================================
# Protocol and capability
# ===========================================================================


def test_adapter_conforms_to_the_protocol():
    assert isinstance(TavilySource(), JobSource)


def test_capability_flags_are_both_false():
    """
    The load-bearing declaration. Tavily reports crawl dates, not posting
    dates, and its salary figures are scraped from prose — the scorer reads
    these flags to decide how much to trust those fields.
    """
    source = TavilySource()
    assert source.provides_structured_dates is False
    assert source.provides_structured_salary is False


def test_unconfigured_adapter_reports_unsupported(monkeypatch, criteria):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    assert TavilySource().supports(criteria) is False


@pytest.mark.asyncio
async def test_search_without_a_key_raises_config_error(monkeypatch, criteria):
    monkeypatch.setenv("TAVILY_API_KEY", "")
    get_settings.cache_clear()
    with pytest.raises(ProviderConfigError):
        await TavilySource().search(criteria)


# ===========================================================================
# Query construction
# ===========================================================================


def test_query_always_carries_the_location():
    """
    The prototype let the LLM write this string, and it routinely dropped the
    location token — the single largest cause of empty result sets. Placed by
    code, it cannot go missing.
    """
    query = TavilySource().build_query(
        SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"],
                       skills=["Python"])
    )
    assert "Cairo" in query
    assert "Backend Engineer" in query
    assert "Python" in query


def test_query_scopes_to_regional_boards_for_a_mena_search():
    """Wuzzuf carries Cairo roles that LinkedIn simply does not list."""
    query = TavilySource().build_query(
        SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])
    )
    assert "wuzzuf.net" in query


def test_query_scopes_to_remote_boards_when_no_location_given():
    query = TavilySource().build_query(SearchCriteria(titles=["Backend Engineer"]))
    assert "weworkremotely.com" in query or "remoteok.com" in query


def test_query_contains_no_negative_terms():
    """
    The prototype appended -"jobs in" and similar, which collided with ordinary
    listing-page chrome and suppressed real results. Category pages are
    rejected after retrieval instead.
    """
    query = TavilySource().build_query(
        SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"])
    )
    assert '-"' not in query


def test_internship_criteria_change_the_query_modifier():
    query = TavilySource().build_query(
        SearchCriteria(titles=["Backend Engineer"], locations=["Cairo"],
                       employment_types=[EmploymentType.INTERNSHIP])
    )
    assert "internship" in query


# ===========================================================================
# Company extraction
# ===========================================================================


@pytest.mark.parametrize(
    "title,url,expected",
    [
        ("Senior .NET Developer at Sylndr - Cairo, Egypt",
         "https://wuzzuf.net/jobs/p/1-x", "Sylndr"),
        (".NET Application Developer at apexanalytix | LinkedIn",
         "https://www.linkedin.com/jobs/view/4396364201", "apexanalytix"),
        ("Halan is hiring a Backend Engineer",
         "https://wuzzuf.net/jobs/p/2-y", "Halan"),
        ("Backend Engineer, Platform",
         "https://boards.greenhouse.io/anthropic/jobs/4019283", "Anthropic"),
        ("Senior Engineer",
         "https://jobs.lever.co/acme-corp/abc-123", "Acme Corp"),
    ],
)
def test_company_is_extracted_from_title_or_ats_url(title, url, expected):
    assert extract_company(title, url) == expected


@pytest.mark.parametrize(
    "title,url",
    [
        ("Backend Engineer Jobs in Cairo", "https://wuzzuf.net/jobs/p/3-z"),
        ("Software Engineer", "https://www.linkedin.com/jobs/view/123456789"),
        ("", "https://wuzzuf.net/jobs/p/4-a"),
    ],
)
def test_unidentifiable_employer_returns_none(title, url):
    """
    None means the record gets dropped. A listing attributed to "Unknown" — or
    worse, to the job board — is misinformation, and the prototype needed a
    dedicated discard rule precisely because it manufactured those values.
    """
    assert extract_company(title, url) is None


@pytest.mark.parametrize(
    "title,expected",
    [
        ("PentaValue hiring PHP Backend Developer in Cairo, Egypt | LinkedIn",
         "PentaValue"),
        ("Nexus Analytica hiring Senior Backend Developer in Cairo | LinkedIn",
         "Nexus Analytica"),
        ("Halan is hiring a Backend Engineer", "Halan"),
    ],
)
def test_linkedin_hiring_pattern_is_recognised(title, expected):
    """
    Regression from a live run. LinkedIn writes "<Company> hiring <Role>",
    without "is" — requiring it made every LinkedIn result unidentifiable, and
    LinkedIn is the largest source of individual-posting URLs in a Tavily
    result set. Three real Cairo postings were being dropped for want of a
    company name.
    """
    assert extract_company(
        title, "https://www.linkedin.com/jobs/view/4448313818"
    ) == expected


def test_linkedin_title_is_stripped_of_company_and_location():
    """
    "PentaValue hiring PHP Backend Developer in Cairo, Egypt" must reduce to
    the role, or the title component scores against a company name and a city.
    """
    source = TavilySource()
    cleaned = source._clean_title(
        "PentaValue hiring PHP Backend Developer in Cairo, Cairo, Egypt | LinkedIn",
        "PentaValue",
    )
    assert cleaned == "PHP Backend Developer"


@pytest.mark.parametrize(
    "phrase",
    ["Backend Engineer at the company", "Engineer with at least 5 years"],
)
def test_generic_prose_after_at_is_not_an_employer(phrase):
    """
    Relaxing the pattern to accept lowercase names ("apexanalytix") also let it
    match ordinary prose. This is the guard that keeps that fix honest.
    """
    assert extract_company(phrase, "https://wuzzuf.net/jobs/p/1-x") is None


def test_a_board_name_is_never_accepted_as_the_employer():
    """The prototype routinely emitted "LinkedIn" as the hiring company."""
    assert extract_company("Backend Engineer at LinkedIn | LinkedIn",
                           "https://www.linkedin.com/jobs/view/1") is None
    assert extract_company("Developer at Indeed", "https://indeed.com/viewjob?jk=1") is None


def test_trailing_noise_is_stripped_from_the_employer():
    assert extract_company("Backend Engineer at Acme Careers",
                           "https://wuzzuf.net/jobs/p/5-b") == "Acme"


# ===========================================================================
# Mapping and honesty guarantees
# ===========================================================================


async def run_search(criteria, results, **kwargs):
    client = FakeTavilyClient(results)
    jobs = await TavilySource(client=client).search(criteria, **kwargs)
    return jobs, client


@pytest.mark.asyncio
async def test_posted_at_is_always_none(criteria):
    """
    Tavily reports when it crawled a page, not when the job was posted. No date
    is better than a wrong one: the scorer treats absence as neutral, whereas a
    crawl date presented as a posting date would let an ancient listing win on
    recency — exactly the prototype's zombie problem.
    """
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/842119-Senior-Backend-Engineer",
               "Senior Backend Engineer at Sylndr",
               "Python and Django. Posted 3 days ago."),
    ])
    assert jobs[0].posted_at is None


@pytest.mark.asyncio
async def test_records_without_an_identifiable_employer_are_dropped(criteria):
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-good", "Backend Engineer at Sylndr", "Python"),
        result("https://wuzzuf.net/jobs/p/2-anon", "Backend Engineer", "Python"),
    ])
    assert len(jobs) == 1
    assert jobs[0].company == "Sylndr"


@pytest.mark.asyncio
async def test_salary_requires_a_currency_marker(criteria):
    """
    Without this guard "500+ applicants" and "5+ years" parse as pay. A wrong
    salary is worse than none: absence scores neutral, a bad figure mis-ranks.
    """
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-a", "Backend Engineer at Sylndr",
               "Over 500 applicants. 5+ years experience required."),
    ])
    assert jobs[0].salary_min is None
    assert jobs[0].salary_max is None


@pytest.mark.asyncio
async def test_salary_with_a_currency_is_accepted(criteria):
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-a", "Backend Engineer at Sylndr",
               "Salary EGP 40,000 - 60,000 monthly."),
    ])
    job = jobs[0]
    assert (job.salary_min, job.salary_max) == (40000, 60000)
    assert job.salary_currency == "EGP"
    assert job.salary_period is SalaryPeriod.MONTH


@pytest.mark.asyncio
async def test_work_mode_is_unknown_not_guessed(criteria):
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-a", "Backend Engineer at Sylndr", "Python"),
    ])
    assert jobs[0].work_mode is WorkMode.UNKNOWN


@pytest.mark.asyncio
async def test_title_is_stripped_of_board_branding_and_employer(criteria):
    """
    Otherwise the title component scores a job against site furniture rather
    than the role.
    """
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-a",
               "Senior Backend Engineer at Sylndr - Cairo, Egypt | Wuzzuf", "Python"),
    ])
    assert jobs[0].title == "Senior Backend Engineer"


@pytest.mark.asyncio
async def test_skills_are_harvested_through_the_shared_vocabulary(criteria):
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-a", "Backend Engineer at Sylndr",
               "Build services with Python, Django and PostgreSQL."),
    ])
    assert {"Python", "Django", "PostgreSQL"} <= set(jobs[0].required_skills)


@pytest.mark.asyncio
async def test_internship_is_detected_from_the_text(criteria):
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-a", "Backend Intern at Halan",
               "Internship for students. Python."),
    ])
    assert jobs[0].employment_type is EmploymentType.INTERNSHIP


# ===========================================================================
# Filter chain
# ===========================================================================


@pytest.mark.asyncio
async def test_category_pages_and_zombies_are_filtered_out(criteria):
    """
    The chain exists because a Tavily result is a web page that might be a
    vacancy. Every other adapter is handed records that already are.
    """
    jobs, _ = await run_search(criteria, [
        result("https://wuzzuf.net/jobs/p/1-real", "Backend Engineer at Sylndr",
               "Python and Django."),
        result("https://wuzzuf.net/jobs/egypt/software-development",
               "Software Development Jobs in Egypt | Browse 1,240 Vacancies", "Browse"),
        result("https://wuzzuf.net/jobs/p/2-dead", "Backend Engineer at DeadCorp",
               "This position has been filled and applications are closed."),
        result("https://www.reddit.com/r/csharp/comments/1abc/jobs",
               "Any good jobs? at Reddit", "discussion"),
    ])
    assert [j.company for j in jobs] == ["Sylndr"]


def test_tools_reexports_are_the_same_objects_as_filters():
    """
    The chain moved from `agents/tools.py` to `sources/filters.py`, and tools
    re-exports it so existing imports keep working. This asserts identity, not
    equality: two independently-defined copies of a regex would pass an
    equality check and then drift the moment one is edited. Identity makes a
    fork impossible.
    """
    from backend.agents import tools
    from backend.sources import filters

    paired = [
        "_is_blacklisted_domain", "_is_content_pollution_domain",
        "_is_category_page", "_passes_path_gate", "_is_usable_url",
        "_snippet_is_stale", "_snippet_is_zombie", "_normalise_netloc",
        "_build_tavily_exclude_domains", "_arabic_digit_to_int",
        "_BOARD_PATH_PATTERNS", "_BLACKLISTED_DOMAINS",
        "_CONTENT_POLLUTION_DOMAINS", "_CATEGORY_PAGE_TITLE_RE",
        "_CATEGORY_PAGE_URL_RE", "_ZOMBIE_DECLARATION_RE",
        "_BAD_URL_PATTERNS", "_VALID_URL_RE", "STALENESS_MONTHS_THRESHOLD",
    ]
    for name in paired:
        assert getattr(tools, name) is getattr(filters, name), (
            f"{name} has forked between agents.tools and sources.filters"
        )


def test_public_filter_names_alias_the_private_ones():
    """The public surface must not be a second implementation either."""
    from backend.sources import filters

    assert filters.is_blacklisted_domain is filters._is_blacklisted_domain
    assert filters.passes_path_gate is filters._passes_path_gate
    assert filters.snippet_is_stale is filters._snippet_is_stale


def test_filters_module_does_not_import_the_agent_layer():
    """
    The point of the move. `sources` must not depend on `agents`, or deleting
    the agent in Phase 2.5 takes the filter chain with it.
    """
    import ast
    import pathlib

    source = pathlib.Path("backend/sources/filters.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = [
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    ] + [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ]
    assert not any("agents" in module for module in imported), (
        f"sources/filters.py imports from the agent layer: {imported}"
    )


def test_filter_chain_reports_the_first_rejecting_layer():
    """
    Naming the layer rather than returning a bare boolean is what makes
    drop-reason telemetry possible — and that telemetry is what would have
    surfaced the bare-"closed" bug in a day.
    """
    assert survives_filter_chain(
        "https://wuzzuf.net/jobs/p/1-x", "Backend Engineer at Sylndr", "Python"
    ) == FilterVerdict.KEPT

    assert survives_filter_chain(
        "https://wuzzuf.net/jobs/egypt/software-development",
        "Software Jobs in Egypt | Browse 1,240 Vacancies", "",
    ) == FilterVerdict.CATEGORY

    assert survives_filter_chain(
        "https://wuzzuf.net/jobs/p/2-x", "Backend Engineer at Acme",
        "This position has been filled.",
    ) == FilterVerdict.STALE


def test_salary_undisclosed_still_survives_the_chain():
    """
    The Phase 1 regression, re-asserted at the layer that now owns it: a bare
    "closed" substring once matched inside "undisclosed" and killed every
    listing that declined to state a salary.
    """
    assert survives_filter_chain(
        "https://wuzzuf.net/jobs/p/3-x", "Backend Engineer at Sylndr",
        "Salary undisclosed. Python and Django.",
    ) == FilterVerdict.KEPT


@pytest.mark.parametrize(
    "url,should_pass",
    [
        # The form Bayt actually serves today: a country segment before /jobs/,
        # and a trailing numeric id.
        ("https://www.bayt.com/en/egypt/jobs/senior-backend-developer-75011643", True),
        ("https://www.bayt.com/en/uae/jobs/software-engineer-12345", True),
        ("https://www.bayt.com/en/jobs/backend-developer", True),
        ("https://www.bayt.com/job/123456", True),
        # A listing hub, not a posting — no numeric id.
        ("https://www.bayt.com/en/egypt/jobs/junior-backend-jobs", False),
    ],
)
def test_bayt_country_segment_urls_pass_the_path_gate(url, should_pass):
    """
    Regression from a live Cairo search. The gate only knew /en/jobs/<slug>,
    so every real Bayt listing was rejected at the path gate while the category
    pages around it were correctly caught. Bayt is one of the two primary MENA
    boards, so this silently removed a large share of the app's core market —
    for the LangGraph agent as much as for this adapter.
    """
    from backend.sources.filters import passes_path_gate

    assert passes_path_gate(url) is should_pass


def test_social_media_posts_are_filtered_as_pollution():
    """
    Also from the live run: an Instagram post titled "Senior Python Developer
    Nile Bits — Cairo" cleared the entire chain. It reads exactly like a
    listing and is not one.
    """
    assert survives_filter_chain(
        "https://www.instagram.com/p/DawfjSplQiT",
        "Senior Python Developer Nile Bits - Cairo", "",
    ) == FilterVerdict.POLLUTION


def test_linkedin_numeric_url_survives_the_chain():
    """The other Phase 1 regression: the canonical pure-numeric form."""
    assert survives_filter_chain(
        "https://www.linkedin.com/jobs/view/4396364201",
        "Backend Engineer at Sylndr", "Python",
    ) == FilterVerdict.KEPT


# ===========================================================================
# Transport
# ===========================================================================


@pytest.mark.asyncio
async def test_limit_is_honoured(criteria):
    jobs, _ = await run_search(criteria, [
        result(f"https://wuzzuf.net/jobs/p/{i}-x", f"Backend Engineer at Company{i}",
               "Python")
        for i in range(10)
    ], limit=3)
    assert len(jobs) == 3


@pytest.mark.asyncio
async def test_empty_results_are_success(criteria):
    jobs, _ = await run_search(criteria, [])
    assert jobs == []


@pytest.mark.asyncio
async def test_unexpected_payload_shape_is_reported(criteria):
    from backend.sources.base import ProviderUnavailable

    class BadClient:
        def search(self, **kwargs):
            return {"results": {"oops": 1}}

    with pytest.raises(ProviderUnavailable, match="payload shape"):
        await TavilySource(client=BadClient()).search(criteria)


@pytest.mark.asyncio
async def test_quota_errors_are_distinguishable(criteria):
    from backend.sources.base import ProviderQuotaExceeded

    client = FakeTavilyClient(error=RuntimeError("429 rate limit exceeded"))
    with pytest.raises(ProviderQuotaExceeded):
        await TavilySource(client=client).search(criteria)


@pytest.mark.asyncio
async def test_search_runs_off_the_event_loop(criteria):
    """
    TavilyClient is synchronous. Left on the loop it would block the concurrent
    fan-out it is supposed to be running alongside.
    """
    import asyncio
    import threading

    loop_thread = threading.get_ident()
    seen = {}

    class ThreadRecordingClient:
        def search(self, **kwargs):
            seen["thread"] = threading.get_ident()
            return {"results": []}

    await TavilySource(client=ThreadRecordingClient()).search(criteria)
    assert seen["thread"] != loop_thread
