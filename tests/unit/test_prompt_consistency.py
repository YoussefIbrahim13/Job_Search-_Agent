"""
Consistency checks between the two prompt sources that reach the model.

The model receives instructions from two places simultaneously:
  1. `recruitment_agent._SYSTEM_PROMPT` — the system message.
  2. `tools.tavily_job_search.__doc__` — the tool description LangChain sends
     with every request.

They drifted: the system prompt called a missing location token "the #1 cause of
empty result sets", while the tool docstring's mandatory query format omitted
LOCATION entirely and all three of its required examples were location-free. The
model was being told two different things about the same query in one request.

Both are now generated from shared constants in `backend.agents.prompts` — the
tool description is passed via `@tool(description=...)` rather than written as a
docstring, precisely so it can interpolate them. These tests pin the properties
that generation is supposed to guarantee, so a future edit that reintroduces
hand-written duplication fails here rather than in production.
"""
import re

import pytest

from backend.agents import prompts
from backend.agents.prompts import APPROVED_SEARCH_BOARDS
from backend.agents.recruitment_agent import _SYSTEM_PROMPT
from backend.agents.tools import tavily_job_search

TOOL_DOC = tavily_job_search.description or ""


def test_tool_doc_query_format_includes_location():
    """The mandatory query-format template must name the LOCATION token."""
    match = re.search(r"RULE 3.*?(?=RULE 4)", TOOL_DOC, re.DOTALL)
    assert match, "RULE 3 (query format) not found in the tool description"
    rule3 = match.group(0)
    assert "LOCATION" in rule3, (
        "RULE 3's query format omits the LOCATION token, contradicting the "
        "system prompt's requirement that it appear in every query"
    )


def test_tool_doc_examples_include_a_location():
    """
    Every ✓ example must carry a location token, or the examples teach the
    model to drop it regardless of what the prose says.
    """
    good_examples = re.findall(r"✓\s+\"([^\"]+)\"", TOOL_DOC)
    assert good_examples, "no positive query examples found in the tool description"

    known_locations = ("cairo", "dubai", "remote", "riyadh", "amman", "giza")
    for example in good_examples:
        assert any(loc in example.lower() for loc in known_locations), (
            f"positive example {example!r} contains no location token"
        )


def test_both_sources_require_location():
    """Both prompt sources must state the location requirement."""
    assert re.search(r"location", _SYSTEM_PROMPT, re.IGNORECASE)
    assert re.search(r"location", TOOL_DOC, re.IGNORECASE)


def test_neither_source_claims_an_unapplied_date_filter():
    """
    `start_date` is never passed to Tavily (it is mutually exclusive with
    `time_range`), so no prompt may claim results are filtered by it — that
    tells the model a guarantee holds when it does not.
    """
    assert "start_date" not in TOOL_DOC, (
        "tool description claims a start_date filter that is never applied"
    )
    assert "start_date" not in _SYSTEM_PROMPT


def test_every_approved_board_reaches_both_prompts():
    """
    The bug this guards: board grouping used to be hand-written separately from
    APPROVED_SEARCH_BOARDS, so a board could be added to the registry — and
    honoured by the URL filters — while appearing in neither prompt. The model
    was never told it existed, so it was never searched. Nothing errored; the
    board was simply dead weight.
    """
    for key, token in APPROVED_SEARCH_BOARDS.items():
        assert token in TOOL_DOC, (
            f"approved board {key!r} ({token}) is missing from the tool "
            f"description, so the model will never query it"
        )
        assert token in _SYSTEM_PROMPT, (
            f"approved board {key!r} ({token}) is missing from the system prompt"
        )


def test_ungrouped_board_is_rejected_at_import():
    """
    Adding a board to the registry without grouping it must fail loudly.

    This is the mechanism that makes the test above hold for boards added in
    future, rather than only for the ones that exist today.
    """
    original = prompts.APPROVED_SEARCH_BOARDS.copy()
    try:
        prompts.APPROVED_SEARCH_BOARDS["newboard"] = "site:newboard.com"
        with pytest.raises(ImportError, match="newboard"):
            prompts._assert_every_board_is_grouped()
    finally:
        prompts.APPROVED_SEARCH_BOARDS.clear()
        prompts.APPROVED_SEARCH_BOARDS.update(original)

    # The guard must not be left tripped for the rest of the session.
    prompts._assert_every_board_is_grouped()


def test_bad_examples_constant_matches_rendered_text():
    """
    QUERY_EXAMPLES_BAD is the machine-readable form of the ✗ block, which is
    rendered as hand-aligned literal text. Keep the two in step.
    """
    for example, _reason in prompts.QUERY_EXAMPLES_BAD:
        assert example in TOOL_DOC, (
            f"QUERY_EXAMPLES_BAD entry {example!r} no longer appears in the "
            f"rendered tool description"
        )


def test_no_negative_jobs_in_token_in_query_builder():
    """
    The `-"jobs in"` exclusion collided with ordinary listing-page chrome and
    suppressed real results; category pages are rejected after retrieval
    instead. Guard against it being reintroduced into the query builder.
    """
    import inspect

    from backend.agents import tools

    source = inspect.getsource(tools.tavily_job_search.func)
    # Strip comments — the fix is documented in a comment that mentions the
    # very token being guarded against.
    code_only = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("#")
    )
    assert '-"jobs in"' not in code_only, (
        'the -"jobs in" negative token is back in the query builder; it '
        "suppresses legitimate listings (see _is_category_page instead)"
    )
