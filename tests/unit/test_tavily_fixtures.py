"""
Fixture-driven regression tests for the Tavily pre-filter chain.

Each fixture in tests/fixtures/tavily/ is a full Tavily response — the shape the
provider actually returns, not one hand-picked result — annotated with the
expected outcome per URL. That structure matters for two reasons:

  * It exercises the chain the way production sees it: a mix of good listings,
    hub pages, and zombies in a single response, where a filter that is too
    broad silently eats the good ones alongside the bad.
  * It is the format the recorded fixtures use too, so replacing a synthetic
    fixture with a real capture is a file swap and nothing else.

These fixtures are the regression net for Phase 2, which moves this filter chain
into backend/sources/filters.py and applies it only to the Tavily adapter. The
expectations below must survive that move unchanged.
"""
import json
from pathlib import Path

import pytest

from backend.agents.tools import (
    _is_blacklisted_domain,
    _is_category_page,
    _is_content_pollution_domain,
    _is_usable_url,
    _passes_path_gate,
    _snippet_is_stale,
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "tavily"


def _load_fixtures():
    """Yield (fixture_name, result_dict, expectation) for every annotated URL."""
    cases = []
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        expectations = payload["_meta"]["expectations"]
        for result in payload["response"]["results"]:
            url = result["url"]
            assert url in expectations, (
                f"{path.name}: result {url!r} has no entry in _meta.expectations. "
                f"Every result must state its expected outcome."
            )
            cases.append(
                pytest.param(
                    path.stem, result, expectations[url],
                    id=f"{path.stem}::{url[:70]}",
                )
            )
    return cases


def _classify(result: dict) -> str:
    """
    Run the pre-filter chain and report the layer that rejected the result.

    Mirrors the ordered chain inside `tavily_job_search`. The live shallow probe
    that follows it is deliberately excluded — it makes network calls, and these
    tests must stay offline and deterministic.

    Order is significant: a URL rejected by an earlier layer never reaches a
    later one, so the reported reason is the *first* rejection, matching the
    drop-reason telemetry buckets.
    """
    url = result.get("url", "")
    title = result.get("title", "")
    snippet = result.get("content", result.get("snippet", ""))

    if _is_blacklisted_domain(url):
        return "drop:blacklist"
    if _is_content_pollution_domain(url):
        return "drop:pollution"
    if _is_category_page(title, url):
        return "drop:category"
    if not _passes_path_gate(url):
        return "drop:path_gate"
    if not _is_usable_url(url):
        return "drop:bad_url"
    if _snippet_is_stale(snippet, title):
        return "drop:stale"
    return "survive"


@pytest.mark.parametrize("fixture_name,result,expected", _load_fixtures())
def test_prefilter_outcome_matches_expectation(fixture_name, result, expected):
    actual = _classify(result)

    # A survivor turning into a drop is the expensive direction: it is invisible
    # in production (the user just sees fewer jobs) and it is the exact class of
    # bug — bare "closed", _FAKE_LINK_RE — these fixtures exist to catch.
    assert actual == expected, (
        f"{fixture_name}: {result['url']}\n"
        f"  expected: {expected}\n"
        f"  actual:   {actual}\n"
        f"  title:    {result.get('title')!r}\n"
        f"  snippet:  {result.get('content', '')[:160]!r}"
    )


def test_fixtures_are_present_and_cover_both_outcomes():
    """
    Guard against a fixture set that quietly becomes all-drop or all-survive —
    either would make the suite pass while testing nothing useful.
    """
    outcomes: set[str] = set()
    for path in sorted(FIXTURE_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        outcomes.update(payload["_meta"]["expectations"].values())

    assert "survive" in outcomes, "no fixture expects a listing to survive"
    assert any(o.startswith("drop:") for o in outcomes), "no fixture expects a drop"
