"""
Standalone diagnostic — run this directly in your project root with your
venv activated:

    python diagnose_live_probe.py

This bypasses LangChain, the agent graph, and the ThreadPoolExecutor
entirely. It imports your REAL backend.agents.tools module and calls
_verify_live_url_is_stale() synchronously on a known URL from your logs,
with full print() output (not logger, so nothing can be swallowed by log
config) and a timer so we can see exactly how long the fetch attempt
actually took.

Run this and paste the full output — it will tell us definitively whether
the function is fetching, blocked, timing out, or succeeding silently.
"""

import logging
import sys
import time

# Make sure backend.agents.tools logger output is visible regardless of
# whatever logging config backend.main sets up — attach a basic handler
# directly to root so nothing can filter it out.
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)

print("=" * 70)
print("Importing backend.agents.tools ...")
print("=" * 70)

try:
    from backend.agents.tools import (
        _verify_live_url_is_stale,
        _is_canonical_listing_url,
        _has_explicit_closed_badge,
        _HEAD_PROBE_BYTES,
    )
    import backend.agents.tools as tools_module
    print(f"Imported OK from: {tools_module.__file__}")
except Exception as exc:
    print(f"IMPORT FAILED: {type(exc).__name__}: {exc}")
    sys.exit(1)

print()
print(f"_HEAD_PROBE_BYTES = {_HEAD_PROBE_BYTES}")
print()

# URLs taken directly from the user's most recent log — both confirmed
# CLOSED / stale in earlier screenshots.
test_urls = [
    "https://wuzzuf.net/jobs/p/U1He6Y5s3ykb-Senior-Front-End-Developer---ReactJS---Next-JS-Taqneen-Cairo-Egypt",
    "https://wuzzuf.net/jobs/p/kmiuk743oelq-senior-front-end-developer-vuejs-furat-frat-cairo-egypt",
    "https://www.linkedin.com/jobs/view/4347645998",
]

for url in test_urls:
    print("-" * 70)
    print(f"Testing: {url}")
    print(f"  _is_canonical_listing_url -> {_is_canonical_listing_url(url)}")

    start = time.monotonic()
    try:
        result = _verify_live_url_is_stale(url, timeout=8.0)
        elapsed = time.monotonic() - start
        print(f"  RESULT: is_stale={result}  (took {elapsed:.2f}s)")
    except Exception as exc:
        elapsed = time.monotonic() - start
        print(f"  EXCEPTION ESCAPED THE FUNCTION (took {elapsed:.2f}s): "
              f"{type(exc).__name__}: {exc}")
    print()

print("=" * 70)
print("Done. If any RESULT line shows a multi-second elapsed time, the "
      "fetch is real but slow. If every elapsed time is near-zero, "
      "something is short-circuiting before the network call.")
print("=" * 70)
