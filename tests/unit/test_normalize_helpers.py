"""
Tests for the shared field-normalization helpers.

Pure functions, no I/O — the highest-value tests in the adapter layer, and the
place two real bugs already surfaced during development:

  * "$120,000 - $150,000" parsed as min == max, because the repeated currency
    symbol broke the range pattern and the string fell through to the
    single-amount path. A silently wrong range, not a visible failure.
  * "$50/hr" parsed as nothing, because one plausibility floor tuned for annual
    salaries discards every hourly rate.

Both are the same class of defect: a mis-parse that looks like a legitimate
value downstream. Hence the emphasis below on "returns None" cases — declining
to parse is always safer than guessing, because absence scores neutral while a
wrong figure actively mis-ranks a job.
"""
from datetime import timezone

import pytest

from backend.sources.normalize import (
    parse_iso_datetime,
    parse_salary_text,
    parse_unix_timestamp,
    seniority_from_months,
    seniority_from_text,
    skills_from_text,
    strip_html,
)
from backend.sources.schema import Seniority


# ---------------------------------------------------------------------------
# parse_salary_text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("$120,000 - $150,000 per year", (120000, 150000, "USD", "YEAR")),
        ("USD 100,000 to USD 130,000", (100000, 130000, "USD", None)),
        ("120k - 150k USD", (120000, 150000, "USD", None)),
        ("EGP 40,000 - 60,000 monthly", (40000, 60000, "EGP", "MONTH")),
        ("$60 - $85 per hour", (60, 85, "USD", "HOUR")),
        ("$50/hr", (50, 50, "USD", "HOUR")),
        ("Up to $90,000 annually", (90000, 90000, "USD", "YEAR")),
        ("€70,000 - €90,000", (70000, 90000, "EUR", None)),
    ],
)
def test_recognisable_salary_strings_parse(text, expected):
    assert parse_salary_text(text) == expected


def test_repeated_currency_symbol_does_not_collapse_the_range():
    """
    Regression: the second "$" broke the range pattern, so the string fell
    through to the single-amount path and produced min == max. Both numbers
    looked plausible, so nothing downstream could detect it.
    """
    low, high, _, _ = parse_salary_text("$120,000 - $150,000")
    assert low == 120000
    assert high == 150000
    assert low != high


def test_k_suffix_on_only_the_second_figure_applies_to_both():
    """"$120 - 150k" means 120k-150k in every real posting."""
    assert parse_salary_text("$120 - 150k")[:2] == (120000, 150000)


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "Competitive salary",
        "Competitive salary based on experience",
        "Salary not disclosed",
        "3 years of experience required",
        "Top 10 employer",
    ],
)
def test_unparseable_salary_returns_all_none(text):
    """
    Declining is the safe outcome: absence scores neutral, whereas a figure
    invented from "3 years" actively mis-ranks the job.
    """
    assert parse_salary_text(text) == (None, None, None, None)


def test_european_dot_thousands_are_declined_rather_than_misread():
    """
    "€60.000" means sixty thousand, but parses as 60.0. The plausibility floor
    rejects it, which is intended — reading it as sixty would be far worse than
    returning nothing.
    """
    assert parse_salary_text("€60.000 pro Jahr") == (None, None, None, None)


def test_hourly_rates_survive_the_plausibility_floor():
    """
    Regression: a single floor tuned for annual salaries discarded every
    hourly rate, silently removing all contract listings' pay data.
    """
    assert parse_salary_text("$45 per hour")[:2] == (45, 45)
    assert parse_salary_text("$45")[:2] == (None, None)  # no period: floor applies


# ---------------------------------------------------------------------------
# strip_html
# ---------------------------------------------------------------------------


def test_block_boundaries_become_whitespace_not_concatenation():
    """
    Without this, "<li>Python</li><li>Django</li>" collapses to "PythonDjango"
    and the skill harvester finds neither.
    """
    text = strip_html("<ul><li>Python</li><li>Django</li></ul>")
    assert "Python" in text
    assert "Django" in text
    assert "PythonDjango" not in text


def test_entities_are_unescaped():
    assert "Docker & AWS" in strip_html("<p>Docker &amp; AWS</p>")


def test_tags_are_removed_entirely():
    result = strip_html("<p class='x'>Hello <b>world</b></p>")
    assert "<" not in result and ">" not in result
    assert "Hello" in result and "world" in result


@pytest.mark.parametrize("value", ["", None])
def test_empty_html_is_empty_string(value):
    assert strip_html(value) == ""


def test_stripped_html_feeds_the_skill_harvester():
    """The whole reason stripping happens at the adapter boundary."""
    skills = skills_from_text(strip_html("<ul><li>Python</li><li>Django</li></ul>"))
    assert "Python" in skills
    assert "Django" in skills


# ---------------------------------------------------------------------------
# parse_unix_timestamp
# ---------------------------------------------------------------------------


def test_valid_timestamp_becomes_aware_utc():
    parsed = parse_unix_timestamp(1754476200)
    assert parsed is not None
    assert parsed.tzinfo is not None
    assert parsed.utcoffset().total_seconds() == 0


@pytest.mark.parametrize(
    "value,reason",
    [
        (0, "epoch zero always means 'unset', never 1970"),
        (None, "absent"),
        ("nonsense", "unparseable"),
        (True, "bool is not a timestamp"),
        (1754476200000, "milliseconds passed as seconds land in the year 57000"),
        (-1, "negative"),
    ],
)
def test_implausible_timestamps_are_declined(value, reason):
    assert parse_unix_timestamp(value) is None, reason


# ---------------------------------------------------------------------------
# parse_iso_datetime
# ---------------------------------------------------------------------------


def test_trailing_z_is_handled():
    parsed = parse_iso_datetime("2026-08-05T09:30:00.000Z")
    assert parsed is not None
    assert parsed.astimezone(timezone.utc).hour == 9


def test_naive_iso_is_assumed_utc():
    """
    Remotive publishes naive timestamps. Assuming server-local would make
    recency scoring vary by deployment region.
    """
    parsed = parse_iso_datetime("2026-08-06T10:15:00")
    assert parsed is not None
    assert parsed.tzinfo is not None


@pytest.mark.parametrize("value", ["", None, "not-a-timestamp", "2026-13-45"])
def test_bad_iso_returns_none_rather_than_raising(value):
    """One malformed timestamp must not fail an entire search."""
    assert parse_iso_datetime(value) is None


# ---------------------------------------------------------------------------
# seniority
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "title,expected",
    [
        ("Senior Backend Engineer", Seniority.SENIOR),
        ("Junior Developer", Seniority.JUNIOR),
        ("Backend Development Intern", Seniority.INTERN),
        ("Lead Platform Engineer", Seniority.LEAD),
        ("Principal Engineer", Seniority.PRINCIPAL),
        ("Backend Engineer", None),
    ],
)
def test_seniority_from_title(title, expected):
    assert seniority_from_text(title) is expected


def test_most_senior_keyword_wins():
    """Ordered most-senior-first, so a compound title resolves upward."""
    assert seniority_from_text("Senior Engineering Lead") is Seniority.LEAD


@pytest.mark.parametrize(
    "months,expected",
    [
        (0, Seniority.JUNIOR),
        (24, Seniority.JUNIOR),
        (48, Seniority.MID),
        (84, Seniority.SENIOR),
        (144, Seniority.LEAD),
        (None, None),
        (-5, None),
    ],
)
def test_seniority_from_months(months, expected):
    assert seniority_from_months(months) is expected


def test_seniority_banding_matches_the_cv_parser_boundaries():
    """
    A posting requiring five years and a candidate with five years must land on
    the same band, or the scorer compares two different scales.
    """
    from backend.parsers.cv_parser import _infer_seniority

    # Neutral text carrying no seniority keyword, so both sides fall through to
    # pure years-based banding. `_infer_seniority` short-circuits on empty text
    # and would return None for every year value, making the comparison vacuous.
    neutral = "Backend Engineer"

    for years in (0.5, 2, 4, 7, 12):
        cv_band = _infer_seniority(neutral, years)
        posting_band = seniority_from_months(int(years * 12))
        assert posting_band is not None
        assert posting_band.value == cv_band, (
            f"{years} years: CV parser says {cv_band}, "
            f"posting parser says {posting_band.value}"
        )
