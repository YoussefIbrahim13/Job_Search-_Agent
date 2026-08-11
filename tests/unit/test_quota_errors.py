"""
Tests for LLM quota-error classification.

Groq's free tier enforces a daily token budget. Once exhausted every call fails
for hours — observed live at 99,415 of 100,000 tokens used. That surfaced to the
user as a generic 500 "An unexpected error occurred", which invites a retry that
cannot possibly succeed and hides an operational condition from the operator.
"""
import pytest

from backend.api.routes import _is_llm_quota_error, _quota_message

GROQ_429 = (
    "Error code: 429 - {'error': {'message': 'Rate limit reached for model "
    "`llama-3.3-70b-versatile` in organization `org_01k` service tier "
    "`on_demand` on tokens per day (TPD): Limit 100000, Used 99415, Requested "
    "5592. Please try again in 1h12m6.048s.', 'code': 'rate_limit_exceeded'}}"
)


class RateLimitError(Exception):
    """Mimics groq.RateLimitError, which is matched by class name."""


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


def test_detects_by_exception_class_name():
    assert _is_llm_quota_error(RateLimitError("quota gone")) is True


def test_detects_by_status_code():
    assert _is_llm_quota_error(_StatusError(429)) is True


def test_detects_by_message_body():
    assert _is_llm_quota_error(Exception(GROQ_429)) is True


@pytest.mark.parametrize(
    "exc",
    [
        _StatusError(500),
        _StatusError(401),
        ValueError("malformed JSON from model"),
        KeyError("jobs"),
        TimeoutError("agent timed out"),
    ],
)
def test_ignores_non_quota_failures(exc):
    """A genuine bug must keep surfacing as a 500, not be masked as a quota wall."""
    assert _is_llm_quota_error(exc) is False


def test_message_includes_provider_retry_hint():
    msg = _quota_message(Exception(GROQ_429))
    assert "1h12m6.048s" in msg
    assert "usage limit" in msg.lower()


def test_retry_hint_does_not_swallow_the_sentence_period():
    """Observed live: the duration rendered as '55m17.76s..' with a double stop."""
    msg = _quota_message(Exception("Please try again in 55m17.76s. Need more tokens?"))
    assert "55m17.76s." in msg
    assert ".." not in msg


def test_message_falls_back_without_a_hint():
    msg = _quota_message(RateLimitError("rate limit reached"))
    assert "try again later" in msg.lower()
