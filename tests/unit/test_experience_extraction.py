"""
Tests for CV experience extraction.

These replace a hardcoded `experience_level: "Professional"` that the API
returned for every candidate and the UI rendered as a definitive "LEVEL:" badge.

The governing rule, and what most of these tests assert: when the signal isn't
there, return None. A blank field the user can correct is honest; a confident
wrong label is not.
"""
import pytest

from backend.parsers.cv_parser import _extract_years_experience, _infer_seniority


# --- Explicit self-reported claims ------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("5+ years of experience in backend development", 5.0),
        ("Over 8 years experience building web apps", 8.0),
        ("7 yrs experience with Python", 7.0),
        ("approximately 3 years of professional experience", 3.0),
        # Largest plausible claim wins when several appear.
        ("2 years of experience in QA. 6 years of experience in dev.", 6.0),
    ],
)
def test_explicit_years_claims(text, expected):
    assert _extract_years_experience(text, current_year=2026) == expected


# --- Date-range spans --------------------------------------------------------

def test_span_from_date_ranges():
    cv = """
    Software Engineer, Acme Corp
    Jan 2019 - Mar 2022
    Backend Developer, Globex
    2022 - present
    """
    years = _extract_years_experience(cv, current_year=2026)
    assert years is not None
    assert 6.5 <= years <= 8.0, f"expected ~7 years from 2019->2026, got {years}"


def test_span_handles_en_dash_and_present():
    cv = "Senior Developer\n2020 – Present"
    years = _extract_years_experience(cv, current_year=2026)
    assert years is not None and 5.0 <= years <= 7.0


def test_explicit_claim_beats_date_ranges():
    """A self-reported total is more reliable than summing overlapping roles."""
    cv = "10 years of experience.\nDeveloper\nJan 2023 - Mar 2024"
    assert _extract_years_experience(cv, current_year=2026) == 10.0


# --- The None cases (the point of the change) --------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "Jane Doe\nCairo, Egypt\njane@example.com",           # no experience info
        "Skills: Python, Django, PostgreSQL",                  # skills only
        "Graduated 2021 with a BSc in Computer Science",       # education only, no range
    ],
)
def test_returns_none_when_undeterminable(text):
    assert _extract_years_experience(text, current_year=2026) is None


@pytest.mark.parametrize(
    "text",
    [
        "Developer\n2030 - 2035",     # start in the future
        "Developer\n2020 - 2015",     # reversed range
        "Developer\n1850 - 1860",     # implausibly old
        "99 years of experience",     # implausible claim
    ],
)
def test_rejects_implausible_ranges(text):
    assert _extract_years_experience(text, current_year=2026) is None


# --- Seniority ---------------------------------------------------------------

@pytest.mark.parametrize(
    "header,expected",
    [
        ("Jane Doe\nSenior Backend Engineer", "senior"),
        ("John Smith\nJunior Developer", "junior"),
        ("Sam Lee\nSoftware Engineering Intern", "intern"),
        ("Alex Ray\nTech Lead", "lead"),
        ("Pat Kim\nPrincipal Engineer", "principal"),
        ("Chris Fox\nGraduate Trainee", "junior"),
    ],
)
def test_seniority_from_title_keyword(header, expected):
    assert _infer_seniority(header) == expected


@pytest.mark.parametrize(
    "years,expected",
    [(0.5, "junior"), (2.0, "junior"), (4.0, "mid"), (7.0, "senior"), (12.0, "lead")],
)
def test_seniority_falls_back_to_years(years, expected):
    assert _infer_seniority("Jane Doe\nDeveloper", years) == expected


def test_seniority_is_none_without_any_signal():
    assert _infer_seniority("Jane Doe\nCairo, Egypt") is None
    assert _infer_seniority("") is None


def test_seniority_ignores_keywords_outside_the_header():
    """
    "Senior" appearing deep in a past role's description says nothing about the
    candidate's current level, so it must not drive the badge.
    """
    cv = "Jane Doe\nDeveloper\n" + "\n" * 20 + "Reported to the Senior Architect"
    assert _infer_seniority(cv) is None
