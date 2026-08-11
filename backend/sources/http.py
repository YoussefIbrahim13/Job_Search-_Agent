"""
Shared HTTP client construction for source adapters.

WHY THIS EXISTS
---------------
`httpx.AsyncClient()` builds a fresh SSL context on every construction, which
means loading and parsing the CA bundle from disk. Measured on the development
machine that is **~0.6s per client**, and constructing a client with the
default context costs 1.7–2.0s.

That is not a micro-optimization. The registry fans out across every configured
provider on each search, so a per-adapter client would add seconds of pure
setup to every request — on a code path that is already slow — and would do it
again for the next user. The SSL context is the expensive part and is
completely reusable: it is immutable configuration, designed to be shared
across connections, and holds no per-request state.

With the context cached, client construction drops to roughly 1ms.

Adapters should still accept an injected client (see `JSearchSource.__init__`)
so the registry can share one connection pool across a fan-out. This module is
the fallback for standalone use, and it makes that fallback cheap rather than
pathological.
"""

from __future__ import annotations

import logging
import ssl
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_ssl_context: Optional[ssl.SSLContext] = None


def get_ssl_context() -> ssl.SSLContext:
    """
    Return the process-wide SSL context, building it once on first use.

    Safe to share: an `SSLContext` is read-only configuration as far as request
    handling is concerned, and the standard library is explicit that one
    context is intended to serve many connections.

    Built lazily rather than at import so that importing this module — which
    every adapter does — never pays the cost in a process that makes no
    outbound calls, such as a test run that only exercises mapping logic.
    """
    global _ssl_context
    if _ssl_context is None:
        logger.debug("Building shared SSL context for source adapters")
        _ssl_context = httpx.create_ssl_context()
    return _ssl_context


def new_async_client(timeout: float, **kwargs) -> httpx.AsyncClient:
    """
    Construct an `AsyncClient` that reuses the cached SSL context.

    Prefer passing an existing client into an adapter when one is available;
    this is for the standalone path where no shared pool exists.
    """
    kwargs.setdefault("verify", get_ssl_context())
    return httpx.AsyncClient(timeout=timeout, **kwargs)
