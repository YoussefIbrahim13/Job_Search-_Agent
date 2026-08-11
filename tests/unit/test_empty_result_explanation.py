"""
Tests for the empty-result explanation.

A search returning nothing is a legitimate outcome — a real ".net / cairo" run
examined 10 listings and correctly rejected all of them (5 category pages, 5
verified-closed postings). The user saw "0 FOUND" and "The agent could not
produce a structured response", which was unhelpful and, worse, untrue: the
pipeline ran exactly as designed.
"""
import pytest

from backend.agents.recruitment_agent import _explain_empty_result
from backend.agents.tools import (
    get_filter_stats,
    record_filter_stats,
    reset_filter_stats,
)


def test_explains_the_real_observed_case():
    """10 examined, 5 category pages, 5 closed — the exact reported run."""
    msg = _explain_empty_result(
        {"examined": 10, "kept": 0, "category": 5, "closed": 5,
         "path_gate": 0, "blocked": 0}
    )
    assert "10" in msg
    assert "5 were expired or no longer accepting applications" in msg
    assert "5 were category or search-results pages" in msg
    # Must not claim the agent malfunctioned.
    assert "could not produce" not in msg.lower()


def test_names_every_nonzero_reason():
    msg = _explain_empty_result(
        {"examined": 12, "kept": 0, "category": 3, "closed": 4,
         "path_gate": 2, "blocked": 3}
    )
    for fragment in ("3 were category", "4 were expired", "2 were not individual",
                     "3 were from blocked"):
        assert fragment in msg


def test_omits_zero_reasons():
    msg = _explain_empty_result(
        {"examined": 5, "kept": 0, "category": 5, "closed": 0,
         "path_gate": 0, "blocked": 0}
    )
    assert "category" in msg
    assert "expired" not in msg
    assert "blocked" not in msg


def test_distinguishes_no_results_from_all_filtered():
    """Zero examined is a different problem and needs different advice."""
    msg = _explain_empty_result({"examined": 0, "kept": 0})
    assert "No job listings were returned" in msg
    assert "Checked 0" not in msg


def test_suggests_an_actionable_next_step():
    msg = _explain_empty_result({"examined": 8, "kept": 0, "closed": 8})
    assert "broader job title" in msg


def test_handles_missing_keys():
    assert _explain_empty_result({}) != ""


# --- Request-scoped stats plumbing -------------------------------------------

def test_stats_accumulate_across_tool_calls():
    """Two parallel queries per search, so counts must sum, not overwrite."""
    reset_filter_stats()
    record_filter_stats(examined=5, kept=0, category=5)
    record_filter_stats(examined=5, kept=0, closed=5)

    stats = get_filter_stats()
    assert stats["examined"] == 10
    assert stats["category"] == 5
    assert stats["closed"] == 5
    assert stats["kept"] == 0


def test_reset_clears_previous_request():
    reset_filter_stats()
    record_filter_stats(examined=99, category=99)
    reset_filter_stats()
    assert get_filter_stats()["examined"] == 0


def test_recording_without_a_request_is_a_no_op():
    """A direct tool call outside a tracked request must not raise."""
    import backend.agents.tools as tools

    if hasattr(tools._stats, "data"):
        del tools._stats.data
    record_filter_stats(examined=5)   # must not raise
    assert get_filter_stats() == {}


def test_stats_are_isolated_between_threads():
    """Concurrent searches must not share counters."""
    import threading

    results = {}

    def worker(name, examined):
        reset_filter_stats()
        record_filter_stats(examined=examined)
        results[name] = get_filter_stats()["examined"]

    threads = [
        threading.Thread(target=worker, args=("a", 10)),
        threading.Thread(target=worker, args=("b", 20)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results == {"a": 10, "b": 20}
