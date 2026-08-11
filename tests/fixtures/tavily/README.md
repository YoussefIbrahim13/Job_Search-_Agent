# Tavily fixtures

Recorded and constructed Tavily responses, used by
`tests/unit/test_tavily_fixtures.py` to pin the behaviour of the pre-filter
chain in `backend/agents/tools.py`.

## Two kinds of fixture

Every file declares its provenance in `_meta.origin`. The distinction is not
bookkeeping — it changes how much the fixture is worth as evidence.

| `origin` | File prefix | What it is | What it proves |
|---|---|---|---|
| `synthetic` | `synthetic_*.json` | Hand-written by a developer | That a *known* bug stays fixed |
| `recorded` | `recorded_*.json` | A real API response, scrubbed | That the filters work on data nobody curated |

The synthetic fixtures were written to encode the specific defects found in the
Phase 1 audit. They are precise and they cost nothing to run, but they can only
contain failure modes someone already thought of. Recorded fixtures can
surprise you; that is the entire reason to have them.

**No recorded fixtures are committed yet.** Generate them with:

```bash
python scripts/capture_tavily_fixtures.py --dry-run   # see what it would do
python scripts/capture_tavily_fixtures.py             # ~2 API credits per query
```

Do this sooner rather than later: Phase 2 demotes Tavily behind structured
providers, and the window to record what it actually returns for MENA queries
closes as the code moves on.

## Format

```jsonc
{
  "_meta": {
    "origin": "synthetic",          // or "recorded"
    "why": "what this fixture is here to pin",
    "expectations": {
      "<result url>": "survive"     // or "drop:<layer>"
    }
  },
  "response": {                     // verbatim provider payload, scrubbed
    "query": "...",
    "results": [ { "url": ..., "title": ..., "content": ..., "score": ... } ]
  }
}
```

`response` holds only the fields the code reads, so a recorded fixture and a
synthetic one are interchangeable to the test.

### Expectation values

`survive`, or `drop:<layer>` where layer is the **first** chain stage that
rejects the result. Order is significant — a URL rejected early never reaches a
later stage:

```
blacklist → pollution → category → path_gate → bad_url → stale
```

Naming the layer rather than just "dropped" is deliberate: it catches a result
that is still rejected but for a newly-wrong reason, which is usually a filter
quietly growing too broad.

> The live shallow probe that runs after this chain is excluded — it makes
> network calls. These tests stay offline and deterministic.

## Adding a fixture

1. Add the file. Every result in `response.results` must have an entry in
   `_meta.expectations`; the loader fails the suite if one is missing, so a
   half-annotated fixture cannot slip through as a pass.
2. Decide expectations by reading the result, **not** by running the code and
   copying what it does. A fixture derived from current behaviour asserts only
   that the code does what it currently does.
3. Run `pytest tests/unit/test_tavily_fixtures.py`. If an expectation fails,
   work out which is wrong — the fixture or the filter — before changing either.

## Known-bug coverage

These are the regressions the current fixtures exist to prevent:

- **`salary undisclosed` must survive.** A bare `"closed"` substring in the
  zombie list matched inside "undis**closed**", killing every listing that
  declined to state a salary — routine in MENA postings.
- **LinkedIn pure-numeric URLs must survive.** `_FAKE_LINK_RE` rejected
  `/jobs/view/4396364201`, the canonical form, after the path gate had already
  correctly admitted it.
- **Genuinely closed listings must drop**, in English and Arabic — the filters
  above must not be loosened into uselessness while fixing the false positives.
- **Arabic age badges must drop.** `منذ 4 سنوات` pages are re-crawled daily, so
  they pass every crawl-date filter and must be caught by content.
- **Fresh Arabic listings must survive** the same scanner — the failure to avoid
  is rejecting Arabic text wholesale.
- **Hub pages must drop**: Greenhouse company boards, Indeed search-results
  pages, Wuzzuf category pages, Himalayas company profiles. All read like
  listings to a snippet parser; none is a vacancy.
