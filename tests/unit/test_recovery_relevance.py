"""
Tests for the relevance gate on the ToolMessage recovery path.

When the agent hits its iteration cap without emitting JSON, the recovery path
scrapes listings straight out of earlier tool results. Those listings were never
scored or judged by the model — nothing vouches for their relevance.

Observed live: a search for "Java Developer" in Cairo returned a "Human
Resources Coordinator" and a "Front End Specialist", each with a hardcoded score
of 50 and company "Unknown".

The gate is deliberately permissive: it removes the obviously unrelated, not the
merely imperfect. Over-filtering here would leave the user with nothing at all,
which is the situation the recovery path exists to avoid.
"""
import pytest

from backend.agents.recruitment_agent import _is_relevant_to_search, _title_tokens


# --- Must be REJECTED (observed live failures) ------------------------------

@pytest.mark.parametrize(
    "job_title,searched",
    [
        ("Zeidex hiring Human Resources Coordinator", "Java Developer"),
        ("Ollie's Bargain Outlet hiring Front End Specialist", "Java Developer"),
        ("Marketing Manager", "Python Backend Developer"),
        ("Sales Representative", "Data Engineer"),
        ("Registered Nurse", "React Frontend Developer"),
    ],
)
def test_unrelated_titles_are_rejected(job_title, searched):
    assert _is_relevant_to_search(job_title, searched) is False, (
        f"{job_title!r} is unrelated to a {searched!r} search but was kept"
    )


# --- Must be KEPT ------------------------------------------------------------

@pytest.mark.parametrize(
    "job_title,searched",
    [
        ("Senior Java Developer", "Java Developer"),
        ("Java Backend Engineer", "Java Developer"),
        ("Bosta hiring Senior Backend Engineer", "Backend Developer"),
        ("Deloitte hiring Senior Python Backend Developer", "Python Backend Developer"),
        ("Frontend Developer (React)", "React Frontend Developer"),
        ("Junior Developer - Java Spring Boot", "Java Developer"),
        (".NET Developer", ".NET Developer"),
        ("C# Software Engineer", "C# Developer"),
    ],
)
def test_related_titles_are_kept(job_title, searched):
    assert _is_relevant_to_search(job_title, searched) is True, (
        f"{job_title!r} is a plausible match for {searched!r} but was dropped"
    )


# --- Fail-open behaviour -----------------------------------------------------

def test_no_search_title_keeps_everything():
    """
    CV-based searches may have no detected title. With nothing to compare
    against, the gate must not filter — returning nothing would be worse.
    """
    assert _is_relevant_to_search("Anything At All", "") is True
    assert _is_relevant_to_search("Human Resources Coordinator", "") is True


def test_stopword_only_search_keeps_everything():
    """A search of pure noise words carries no signal to filter on."""
    assert _is_relevant_to_search("Human Resources Coordinator", "jobs in the") is True


# --- Tokenisation ------------------------------------------------------------

def test_tokens_drop_stopwords_and_punctuation():
    assert _title_tokens("Senior Developer at Acme (Full Time)") == {
        "senior", "developer", "acme"
    }


def test_tokens_preserve_tech_punctuation():
    """'.NET', 'C#', and 'Node.js' lose their meaning if punctuation is stripped."""
    assert "c#" in _title_tokens("C# Engineer")
    assert ".net" in _title_tokens(".NET Developer")
    assert "node.js" in _title_tokens("Node.js Engineer")


def test_matching_is_case_insensitive():
    assert _is_relevant_to_search("SENIOR JAVA DEVELOPER", "java developer") is True
