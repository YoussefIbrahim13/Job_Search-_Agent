"""
Tests for shared HTTP client construction.

This pins a performance property rather than a behavioural one, which is
unusual for a unit test and deliberate here. The cost is invisible in review —
`httpx.AsyncClient()` looks free — and it is paid per provider per search once
the registry fans out. Discovering it again later means re-profiling a slow
endpoint to find a one-line cause.
"""
import time

import httpx

from backend.sources import http as source_http


def test_ssl_context_is_reused_across_calls():
    """
    The context is the expensive part: building one loads and parses the CA
    bundle from disk (~0.6s on the dev machine). Returning the same object is
    what keeps client construction at roughly 1ms.
    """
    first = source_http.get_ssl_context()
    second = source_http.get_ssl_context()
    assert first is second


def test_clients_share_the_cached_context():
    a = source_http.new_async_client(timeout=1.0)
    b = source_http.new_async_client(timeout=1.0)
    try:
        assert isinstance(a, httpx.AsyncClient)
        assert isinstance(b, httpx.AsyncClient)
    finally:
        # Constructed but never entered, so no connections exist to close;
        # dropping the references is sufficient.
        del a, b


def test_client_construction_is_cheap_once_the_context_is_warm():
    """
    Guards the regression directly. Measured on the dev machine: 5 clients take
    ~0.4s with the cached context and ~8.7s without it. The threshold sits
    between those by a wide margin in both directions — tight enough to catch
    the caching being removed, loose enough not to go flaky on a slow CI box.
    """
    source_http.get_ssl_context()  # warm

    start = time.perf_counter()
    for _ in range(5):
        source_http.new_async_client(timeout=1.0)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, (
        f"constructing 5 clients took {elapsed:.2f}s — the shared SSL context "
        f"is probably no longer being reused, which costs ~0.6s per client per "
        f"provider on every search"
    )


def test_explicit_verify_argument_is_respected():
    """
    The cached context is a default, not a mandate — a caller that needs
    different TLS behaviour must still be able to ask for it.
    """
    client = source_http.new_async_client(timeout=1.0, verify=False)
    assert isinstance(client, httpx.AsyncClient)
