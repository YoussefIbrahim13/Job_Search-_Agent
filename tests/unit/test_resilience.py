"""
Tests for the transient-failure classifier used by the retry policies.

The important half is the negative cases: retrying a 401 or a 400 burns quota
and latency to produce the same failure, and the agent's iteration budget is
too small to waste on it.
"""
import pytest

from backend.core.resilience import _is_transient, groq_retry


class _HttpError(Exception):
    def __init__(self, status_code):
        super().__init__(f"HTTP {status_code}")
        self.status_code = status_code


@pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
def test_transient_status_codes(status):
    assert _is_transient(_HttpError(status)) is True


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_permanent_status_codes(status):
    assert _is_transient(_HttpError(status)) is False


@pytest.mark.parametrize(
    "message",
    [
        "Rate limit exceeded",
        "429 Too Many Requests",
        "Connection reset by peer",
        "Request timed out",
        "Service Unavailable",
    ],
)
def test_transient_message_text(message):
    assert _is_transient(Exception(message)) is True


@pytest.mark.parametrize(
    "message",
    ["Invalid API key", "Model not found", "malformed request body"],
)
def test_permanent_message_text(message):
    assert _is_transient(Exception(message)) is False


# --- Policy behaviour --------------------------------------------------------

def test_retries_then_succeeds():
    calls = []

    @groq_retry
    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise _HttpError(429)
        return "ok"

    assert flaky() == "ok"
    assert len(calls) == 3


def test_does_not_retry_permanent_errors():
    calls = []

    @groq_retry
    def bad_request():
        calls.append(1)
        raise _HttpError(401)

    with pytest.raises(_HttpError):
        bad_request()
    assert len(calls) == 1, "a 401 must not be retried"


def test_reraises_after_exhausting_attempts():
    calls = []

    @groq_retry
    def always_rate_limited():
        calls.append(1)
        raise _HttpError(429)

    with pytest.raises(_HttpError):
        always_rate_limited()
    assert len(calls) == 3
