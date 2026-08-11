"""
Retry policies for calls to external APIs.

`tenacity` was already a declared dependency but was imported nowhere, so a
single transient 429 from Groq or a blip from Tavily failed the whole search —
and because the agent runs under an iteration cap, that failure also consumed
one of the very few turns available to recover.

Both policies below retry only on transient conditions. A 4xx that is not 429
means the request itself is wrong, and retrying it just burns quota and latency.
"""
from __future__ import annotations

import logging
from typing import Any

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


def _is_transient(exc: BaseException) -> bool:
    """
    True when an exception is worth retrying.

    Provider SDKs raise a wide variety of types, so classify structurally
    (status code where available, else message text) rather than by class.
    """
    status = (
        getattr(exc, "status_code", None)
        or getattr(exc, "status", None)
        or getattr(getattr(exc, "response", None), "status_code", None)
    )
    if isinstance(status, int):
        # 429 = rate limited, 5xx = provider-side fault. Everything else in the
        # 4xx range is a malformed request that will fail identically on retry.
        return status == 429 or 500 <= status < 600

    text = str(exc).lower()
    transient_markers = (
        "rate limit", "429", "too many requests",
        "timeout", "timed out", "connection", "temporarily unavailable",
        "service unavailable", "502", "503", "504",
    )
    return any(marker in text for marker in transient_markers)


# Groq: the agent is latency-sensitive (the user is waiting on an HTTP request
# under a 300s ceiling), so cap total added delay at roughly 10 seconds.
groq_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=1, max=6),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)

# Tavily: already has a basic-depth fallback of its own, so retry the transport
# a couple of times before falling through to it.
tavily_retry = retry(
    retry=retry_if_exception(_is_transient),
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=4),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


__all__: list[Any] = ["groq_retry", "tavily_retry", "_is_transient"]
