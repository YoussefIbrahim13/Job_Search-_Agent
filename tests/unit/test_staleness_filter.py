"""
Regression tests for the staleness / "zombie listing" filter.

The bug this file exists to prevent: `_ZOMBIE_CONTENT_SNIPPETS_EN` contained the
bare substring "closed", which was `re.escape`d into an alternation with no word
boundary and matched against `f"{title} {snippet}"`. Any listing whose text
contained "undisclosed", "disclosed", or "enclosed" was silently dropped —
and "salary undisclosed" is routine phrasing on Bayt and Wuzzuf, the two boards
that carry most of this app's MENA coverage.

Every FALSE_POSITIVES case below was being dropped before the fix.
"""
import pytest

from backend.agents.tools import STALENESS_MONTHS_THRESHOLD, _snippet_is_stale


# Text that must SURVIVE the filter (real, open listings).
FALSE_POSITIVES = [
    ("salary undisclosed", "Backend Developer", "Compensation: undisclosed"),
    ("undisclosed salary range", "Python Engineer", "Salary undisclosed by employer"),
    ("disclosed", "Data Analyst", "Benefits will be disclosed at interview stage"),
    ("enclosed", "QA Engineer", "Please review the enclosed job description"),
    ("closed-loop domain term", "Control Systems Engineer", "Experience with closed-loop control"),
    ("closed captions domain term", "Frontend Developer", "Build closed captions UI for video"),
    ("non-disclosure", "Security Engineer", "Subject to a non-disclosure agreement"),
]

# Text that must still be DROPPED (genuinely dead listings).
TRUE_POSITIVES = [
    ("explicit no longer accepting", "Developer", "This job is no longer accepting applications"),
    ("position filled", "Developer", "position filled"),
    ("job expired", "Developer", "job expired"),
    ("no longer available", "Developer", "This job is no longer available"),
    ("applications closed", "Developer", "Applications closed as of last week"),
    ("vacancy closed", "Developer", "vacancy closed"),
    ("position is closed", "Developer", "This position is closed"),
    ("role closed", "Developer", "The role closed on 1 March"),
]


@pytest.mark.parametrize(
    "label,title,snippet", FALSE_POSITIVES, ids=[c[0] for c in FALSE_POSITIVES]
)
def test_open_listings_survive(label, title, snippet):
    assert _snippet_is_stale(snippet, title) is False, (
        f"{label!r} was dropped as a zombie listing but describes an open job"
    )


@pytest.mark.parametrize(
    "label,title,snippet", TRUE_POSITIVES, ids=[c[0] for c in TRUE_POSITIVES]
)
def test_dead_listings_are_dropped(label, title, snippet):
    assert _snippet_is_stale(snippet, title) is True, (
        f"{label!r} describes a dead listing but survived the filter"
    )


# --- Posting-age thresholds ------------------------------------------------

@pytest.mark.parametrize(
    "snippet,expected",
    [
        ("Posted 1 month ago", False),
        ("Posted 2 months ago", False),
        ("Posted 3 months ago", True),   # boundary: >= threshold
        ("Posted 4 months ago", True),
        ("Posted 1 year ago", True),
        ("Posted 2 years ago", True),
    ],
)
def test_english_age_thresholds(snippet, expected):
    assert STALENESS_MONTHS_THRESHOLD == 3, "test boundaries assume a 3-month threshold"
    assert _snippet_is_stale(snippet) is expected


@pytest.mark.parametrize(
    "snippet,expected",
    [
        ("منذ شهر", False),        # 1 month
        ("منذ شهرين", False),      # 2 months
        ("منذ 3 أشهر", True),      # 3 months, western digits
        ("منذ ٤ أشهر", True),      # 4 months, Arabic-Indic digits
        ("منذ سنة", True),         # 1 year
        ("منذ سنتين", True),       # 2 years
    ],
)
def test_arabic_age_thresholds(snippet, expected):
    assert _snippet_is_stale(snippet) is expected


def test_arabic_closure_declarations_are_dropped():
    assert _snippet_is_stale("الوظيفة مغلقة") is True
    assert _snippet_is_stale("انتهت فترة التقديم") is True


def test_empty_input_is_not_stale():
    assert _snippet_is_stale("") is False
    assert _snippet_is_stale("", "") is False
