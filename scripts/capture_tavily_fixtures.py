"""
Record real Tavily responses as test fixtures.

WHY RUN THIS
------------
The fixtures committed today are synthetic: hand-written to encode the specific
bugs we know about. They are precise regression tests and they are honest about
being constructed, but they cannot surprise us — they only contain failure modes
someone already thought of.

Real captures can. Tavily is also on borrowed time in this codebase: Phase 2
demotes it behind structured providers, so the window to record what it actually
returns for MENA queries closes as the code moves on. Recording a handful now
costs a few API credits and preserves that ground truth.

USAGE
-----
    python scripts/capture_tavily_fixtures.py                 # all queries
    python scripts/capture_tavily_fixtures.py --only cairo    # one, to try it
    python scripts/capture_tavily_fixtures.py --dry-run       # show, spend nothing

Each run costs 2 API credits per query (advanced depth). Output lands in
tests/fixtures/tavily/recorded_<name>.json.

WHAT IS SCRUBBED
----------------
Tavily responses are public web content, but the payload also carries request
metadata. This script keeps only the fields the code actually reads — url,
title, content, score — plus the query, and drops everything else including
`request_id`. Review the output before committing regardless: these files go
into a public repository.

The recorded files carry `"origin": "recorded"` and an empty `expectations`
map. Fill the expectations in by hand — deciding what *should* have happened to
each result is the judgement the fixture exists to capture, and inferring it
from current behaviour would make the test tautological.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Import path setup so the script runs from the repo root without installation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.core.config import get_settings  # noqa: E402

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "tavily"

# Deliberately spread across boards, languages, and job families. Each exists to
# capture a shape the synthetic fixtures can only approximate.
QUERIES: dict[str, str] = {
    "cairo_dotnet":
        "C# ASP.NET Backend Developer Cairo jobs site:wuzzuf.net OR site:bayt.com",
    "cairo_python":
        "Python Django Backend Developer Cairo jobs site:wuzzuf.net",
    "linkedin_egypt":
        "Python Backend Developer Egypt jobs site:linkedin.com/jobs",
    "mena_arabic":
        "مطور برمجيات القاهرة jobs site:bayt.com OR site:akhtaboot.com",
    "gulf_dubai":
        "React Node.js Software Engineer Dubai jobs site:bayt.com OR site:linkedin.com/jobs",
    "remote_global":
        "Python Django Backend Engineer Remote jobs site:weworkremotely.com OR site:himalayas.app",
    "internship_cairo":
        "Python Django Developer Cairo internship site:wuzzuf.net OR site:linkedin.com/jobs",
    "ats_direct":
        "Python Backend Engineer Remote jobs site:greenhouse.io OR site:jobs.lever.co",
}

# Only these keys survive into the fixture — everything else is request metadata
# we neither read nor want in a public repo.
_KEPT_RESULT_KEYS = ("url", "title", "content", "score")


def scrub(response: dict, query: str) -> dict:
    """Reduce a raw Tavily response to the fields the code actually consumes."""
    return {
        "query": query,
        "results": [
            {k: r.get(k) for k in _KEPT_RESULT_KEYS if k in r}
            for r in response.get("results", [])
        ],
    }


def capture(name: str, query: str, client, out_dir: Path) -> Path:
    response = client.search(
        query=query,
        search_depth="advanced",
        max_results=get_settings().tavily_max_results,
        time_range=get_settings().tavily_time_range,
        include_answer=False,
        include_raw_content=False,
    )

    payload = {
        "_meta": {
            "origin": "recorded",
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "why": "TODO: describe what this capture is here to pin.",
            # Left empty on purpose — see the module docstring. Filling this in
            # from current behaviour would assert only that the code does what
            # it currently does.
            "expectations": {},
        },
        "response": scrub(response, query),
    }

    path = out_dir / f"recorded_{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def main() -> int:
    # The Arabic query below is unencodable in the Windows console default
    # (cp1252), which turns a harmless progress line into a crash. Force UTF-8
    # rather than dropping the query that most needs capturing.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, OSError):
        pass

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", help="capture just this query key")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="list what would be captured without calling the API",
    )
    args = parser.parse_args()

    if args.only:
        if args.only not in QUERIES:
            print(f"Unknown query key {args.only!r}. Available: {', '.join(QUERIES)}")
            return 1
        selected = {args.only: QUERIES[args.only]}
    else:
        selected = dict(QUERIES)

    if args.dry_run:
        print(f"Would capture {len(selected)} quer{'y' if len(selected) == 1 else 'ies'} "
              f"(~{len(selected) * 2} API credits):")
        for name, query in selected.items():
            print(f"  {name:20s} {query}")
        return 0

    settings = get_settings()
    if not settings.tavily_api_key:
        print("TAVILY_API_KEY is not set. Populate .env first (see .env.example).")
        return 1

    try:
        from tavily import TavilyClient
    except ImportError:
        print("tavily-python is not installed. Run: pip install tavily-python")
        return 1

    client = TavilyClient(api_key=settings.tavily_api_key)
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for name, query in selected.items():
        try:
            path = capture(name, query, client, FIXTURE_DIR)
        except Exception as exc:
            # One failing query should not discard the captures that succeeded.
            print(f"  {name:20s} FAILED: {exc}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        print(f"  {name:20s} {len(payload['response']['results']):2d} results → {path.name}")

    print(
        "\nNext: fill in `_meta.why` and `_meta.expectations` for each new file, "
        "then review the contents before committing — this repo is public."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
