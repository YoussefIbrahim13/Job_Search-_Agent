"""
Tests for deduplication, ranking, and the relevance threshold.

Before this, `_validate_and_fix_output` did none of the three: the two parallel
searches could return the same posting twice, jobs rendered in whatever order
the model emitted them, and nothing was ever excluded for being a poor match.
The product described its output as "ranked" regardless.
"""
import pytest

from backend.agents.recruitment_agent import (
    MIN_MATCH_SCORE,
    _canonical_url_key,
    _validate_and_fix_output,
)


def _job(link, score=80, title="Backend Developer", company="Acme Industrial"):
    return {
        "job_title": title,
        "company_name": company,
        "location": "Cairo, Egypt",
        "application_link": link,
        "match_score": score,
        "match_reason": "Matches on Python and Django.",
        "required_skills": ["Python"],
    }


def _run(jobs):
    return _validate_and_fix_output(
        {"job_title": "Backend Developer", "location": "Cairo", "jobs": jobs}
    )


# --- URL normalisation -------------------------------------------------------

@pytest.mark.parametrize(
    "a,b",
    [
        ("https://wuzzuf.net/jobs/p/abc123", "https://www.wuzzuf.net/jobs/p/abc123"),
        ("https://wuzzuf.net/jobs/p/abc123", "http://wuzzuf.net/jobs/p/abc123"),
        ("https://wuzzuf.net/jobs/p/abc123", "https://wuzzuf.net/jobs/p/abc123/"),
        ("https://wuzzuf.net/jobs/p/abc123", "https://wuzzuf.net/jobs/p/abc123#apply"),
        ("https://wuzzuf.net/jobs/p/abc123", "https://wuzzuf.net/jobs/p/abc123?utm_source=x"),
        ("https://WUZZUF.net/Jobs/P/abc123", "https://wuzzuf.net/jobs/p/abc123"),
    ],
)
def test_urls_that_are_the_same_posting(a, b):
    assert _canonical_url_key(a) == _canonical_url_key(b)


def test_urls_that_are_different_postings():
    assert _canonical_url_key("https://wuzzuf.net/jobs/p/abc123") != _canonical_url_key(
        "https://wuzzuf.net/jobs/p/xyz789"
    )


# --- Deduplication -----------------------------------------------------------

def test_duplicate_postings_collapse_to_one():
    result = _run([
        _job("https://wuzzuf.net/jobs/p/abc123"),
        _job("https://www.wuzzuf.net/jobs/p/abc123/"),   # same posting
        _job("https://wuzzuf.net/jobs/p/xyz789"),
    ])
    assert len(result["jobs"]) == 2
    assert result["total_found"] == 2


def test_first_occurrence_of_a_duplicate_is_kept():
    result = _run([
        _job("https://wuzzuf.net/jobs/p/abc123", company="First Seen"),
        _job("https://wuzzuf.net/jobs/p/abc123", company="Second Seen"),
    ])
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["company_name"] == "First Seen"


# --- Ranking -----------------------------------------------------------------

def test_jobs_are_sorted_by_score_descending():
    result = _run([
        _job("https://wuzzuf.net/jobs/p/low", score=40),
        _job("https://wuzzuf.net/jobs/p/high", score=95),
        _job("https://wuzzuf.net/jobs/p/mid", score=70),
    ])
    scores = [j["match_score"] for j in result["jobs"]]
    assert scores == sorted(scores, reverse=True)
    assert scores == [95, 70, 40]


def test_scores_are_not_capped_below_the_high_match_band():
    """
    Targeted search used to clamp every score to <=75, so no result could reach
    the UI's "high match" band at >=75 — while CV search applied no cap at all.
    """
    result = _run([_job("https://wuzzuf.net/jobs/p/abc123", score=95)])
    assert result["jobs"][0]["match_score"] == 95


# --- Threshold ---------------------------------------------------------------

def test_jobs_below_threshold_are_dropped():
    result = _run([
        _job("https://wuzzuf.net/jobs/p/good", score=80),
        _job("https://wuzzuf.net/jobs/p/bad", score=MIN_MATCH_SCORE - 1),
    ])
    assert len(result["jobs"]) == 1
    assert result["jobs"][0]["match_score"] == 80


def test_job_exactly_at_threshold_is_kept():
    result = _run([_job("https://wuzzuf.net/jobs/p/edge", score=MIN_MATCH_SCORE)])
    assert len(result["jobs"]) == 1


def test_total_found_reflects_returned_jobs():
    """total_found must count what the user actually receives, post-filtering."""
    result = _run([
        _job("https://wuzzuf.net/jobs/p/a", score=80),
        _job("https://wuzzuf.net/jobs/p/a", score=80),        # duplicate
        _job("https://wuzzuf.net/jobs/p/b", score=5),         # below threshold
        _job("https://wuzzuf.net/jobs/p/c", score=60),
    ])
    assert result["total_found"] == len(result["jobs"]) == 2


def test_empty_input_produces_empty_output():
    result = _run([])
    assert result["jobs"] == []
    assert result["total_found"] == 0


# --- Aggregator / category page titles ---------------------------------------

@pytest.mark.parametrize(
    "title",
    [
        "Job Search | Indeed",           # observed live: rendered as a vacancy
        "Jobs Search Results",
        "Job Listings | Bayt.com",
        "Browse Jobs",
        "Search Jobs",
        "All Jobs",
        "Latest Jobs",
        "1,200 Jobs in Cairo",
        "Software Jobs in Egypt",
        "Career Opportunities",
    ],
)
def test_aggregator_titles_are_dropped(title):
    result = _run([_job("https://www.indeed.com/viewjob", title=title)])
    assert result["jobs"] == [], f"aggregator page {title!r} was returned as a vacancy"


@pytest.mark.parametrize(
    "title",
    [
        # Observed live: an Arabic LinkedIn category page rendered as a vacancy.
        # "100+ Java Developer jobs, hiring in Cairo governorate"
        "١٠٠+ Java Developer من الوظائف، التوظيف في محافظة القاهرة",
        "وظائف في القاهرة",              # "jobs in Cairo"
        "التوظيف في مصر",                # "hiring in Egypt"
        "جميع الوظائف",                  # "all jobs"
        "أحدث الوظائف",                  # "latest jobs"
        "100+ وظائف",                    # western digits, Arabic noun
    ],
)
def test_arabic_aggregator_titles_are_dropped(title):
    """
    The approved boards are MENA-heavy (Wuzzuf, Bayt, Akhtaboot) and LinkedIn
    serves Arabic category pages to those users, so an English-only aggregator
    filter misses a large share of real traffic.
    """
    result = _run([_job("https://www.linkedin.com/jobs/view/4449101571", title=title)])
    assert result["jobs"] == [], f"Arabic aggregator page {title!r} was returned"


@pytest.mark.parametrize(
    "title",
    [
        "مطور جافا",                     # "Java Developer" — a real role
        "مهندس برمجيات أول",             # "Senior Software Engineer"
    ],
)
def test_arabic_job_titles_survive(title):
    result = _run([_job("https://wuzzuf.net/jobs/p/abc123", title=title)])
    assert len(result["jobs"]) == 1, f"real Arabic vacancy {title!r} was dropped"


@pytest.mark.parametrize(
    "title",
    [
        "Senior Backend Developer",
        "Python Engineer (Remote)",
        "Full Stack Developer - Fintech",
        "Junior Data Analyst",
    ],
)
def test_real_job_titles_survive(title):
    result = _run([_job("https://wuzzuf.net/jobs/p/abc123", title=title)])
    assert len(result["jobs"]) == 1, f"real vacancy {title!r} was dropped"


# --- Summary / count consistency ---------------------------------------------

def test_summary_reconciles_with_the_displayed_count():
    """
    The model writes agent_summary before validation, so it reports its own raw
    count. Observed live: a summary claiming "I found 8 relevant listings" next
    to a "2 FOUND" header. The summary must acknowledge the difference.
    """
    result = _validate_and_fix_output({
        "job_title": "Backend Developer",
        "location": "Cairo",
        "agent_summary": "I found 8 relevant listings.",
        "jobs": [
            _job("https://wuzzuf.net/jobs/p/a", score=80),
            _job("https://wuzzuf.net/jobs/p/a", score=80),   # duplicate
            _job("https://wuzzuf.net/jobs/p/b", score=5),     # below threshold
        ],
    })
    assert result["total_found"] == 1
    assert "2 of 3" in result["agent_summary"]
    assert "1 shown" in result["agent_summary"]


def test_summary_is_untouched_when_nothing_was_dropped():
    result = _validate_and_fix_output({
        "job_title": "Backend Developer",
        "location": "Cairo",
        "agent_summary": "Found 1 listing.",
        "jobs": [_job("https://wuzzuf.net/jobs/p/a", score=80)],
    })
    assert result["agent_summary"] == "Found 1 listing."
