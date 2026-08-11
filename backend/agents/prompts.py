"""
Single source of truth for prompt text shared between the tool description and
the agent system prompt.

WHY THIS MODULE EXISTS
----------------------
The approved-board list and the query-format rules were previously written out
by hand in two places:

  * the ``tavily_job_search`` docstring in tools.py, which LangChain sends to
    the model as the tool description, and
  * ``_SYSTEM_PROMPT`` in recruitment_agent.py.

Those two texts disagreed. The docstring's query-format string omitted the
location token that the system prompt called "the #1 cause of empty result
sets", so the model was told two different things about the same parameter and
could satisfy one rule by breaking the other.

Worse, the *grouping* of boards ("Global", "MENA/Gulf", ...) was hardcoded
separately from ``APPROVED_SEARCH_BOARDS`` itself. Adding a board to the
registry updated the filter layer but appeared in neither prompt, so the model
was never told the board existed and would never query it. That failure is
silent: nothing errors, the board is simply never used.

This module owns the registry, the grouping, and the shared rule text. Both
consumers render from it, and :func:`_assert_every_board_is_grouped` turns the
silent failure into an ImportError at startup.

RENDERING, NOT WORDING
----------------------
The two call sites still render this data in their own layouts — an aligned
column block for the docstring, a compact one-line-per-group block for the
system prompt. That is deliberate. Unifying the *data* removes the drift; also
unifying the *wording* would change the exact bytes sent to the model, and
there is no eval harness yet (Phase 5) to prove such a change is neutral. The
prompt text below is therefore byte-identical to what shipped before this
refactor; ``tests/unit/test_prompt_consistency.py`` pins that.

Once the Phase 5 eval harness exists, collapsing the two layouts into one is a
worthwhile follow-up — measurable then, guesswork now.
"""

from __future__ import annotations

from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Approved board registry
# ---------------------------------------------------------------------------

APPROVED_SEARCH_BOARDS: dict[str, str] = {
    "linkedin":       "site:linkedin.com/jobs",
    "indeed":         "site:indeed.com",
    "glassdoor":      "site:glassdoor.com",
    "wuzzuf":         "site:wuzzuf.net",
    "bayt":           "site:bayt.com",
    "akhtaboot":      "site:akhtaboot.com",
    "weworkremotely": "site:weworkremotely.com",
    "remoteok":       "site:remoteok.com",
    "himalayas":      "site:himalayas.app",
    "wellfound":      "site:wellfound.com",
    "dice":           "site:dice.com",
    # Direct-ATS boards — the freshest source available: postings are served
    # straight from the hiring company's applicant-tracking system and are
    # taken down the moment a role is filled/closed, so zombie/expired
    # listings are structurally rare here versus scraped aggregators.
    "greenhouse":     "site:greenhouse.io",
    "lever":          "site:jobs.lever.co",
}

APPROVED_SITE_TOKENS: list[str] = list(APPROVED_SEARCH_BOARDS.values())


# Board grouping, defined once. Each entry is:
#     (compact_label, docstring_label, board_keys)
#
# The two labels differ only because the two render targets have different
# width budgets; they describe the same group. Add a board to
# APPROVED_SEARCH_BOARDS *and* to a group here — the assertion below enforces
# it, so a half-finished addition fails loudly at import rather than silently
# hiding the board from the model.
BOARD_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("Global",                "Global generalist",   ("linkedin", "indeed", "glassdoor")),
    ("MENA/Gulf",             "MENA / Egypt / Gulf", ("wuzzuf", "bayt", "akhtaboot")),
    ("Remote",                "Remote-focused",      ("weworkremotely", "remoteok", "himalayas")),
    ("Tech",                  "Tech-specialist",     ("wellfound", "dice")),
    ("Direct ATS (freshest)", "Direct ATS (freshest — prefer when possible)",
                                                     ("greenhouse", "lever")),
)


def _assert_every_board_is_grouped() -> None:
    """
    Fail at import if the registry and the grouping have diverged.

    An ungrouped board is invisible to the model; a grouped-but-unregistered
    key renders as a KeyError-shaped hole. Both are silent in production, so
    they are converted into a startup crash here.
    """
    grouped: list[str] = [key for _, _, keys in BOARD_GROUPS for key in keys]

    ungrouped = set(APPROVED_SEARCH_BOARDS) - set(grouped)
    if ungrouped:
        raise ImportError(
            f"Board(s) {sorted(ungrouped)} are in APPROVED_SEARCH_BOARDS but not in "
            f"any BOARD_GROUPS entry, so the model would never be told they exist. "
            f"Add them to a group in backend/agents/prompts.py."
        )

    unknown = set(grouped) - set(APPROVED_SEARCH_BOARDS)
    if unknown:
        raise ImportError(
            f"BOARD_GROUPS references unknown board key(s) {sorted(unknown)}. "
            f"Add them to APPROVED_SEARCH_BOARDS in backend/agents/prompts.py."
        )

    duplicates = sorted({k for k in grouped if grouped.count(k) > 1})
    if duplicates:
        raise ImportError(
            f"Board key(s) {duplicates} appear in more than one BOARD_GROUPS entry."
        )


_assert_every_board_is_grouped()


# ---------------------------------------------------------------------------
# Board block renderers
# ---------------------------------------------------------------------------

# Column at which site tokens start in the docstring layout. The longest
# inline-capable label ("MENA / Egypt / Gulf") is exactly this wide; a longer
# label drops its tokens to the following line instead of blowing the column.
_DOCSTRING_LABEL_WIDTH = 19
_DOCSTRING_INDENT = " " * 4
# Tokens line up one column past "<label>:" — i.e. label width plus the colon
# plus the separating space.
_DOCSTRING_CONTINUATION = " " * (len(_DOCSTRING_INDENT) + _DOCSTRING_LABEL_WIDTH + 2)


def render_boards_compact() -> str:
    """
    One line per group: ``Global: site:a, site:b``.

    Used by the agent system prompt, where vertical space is at a premium.
    """
    lines: list[str] = []
    for compact_label, _, keys in BOARD_GROUPS:
        tokens = [APPROVED_SEARCH_BOARDS[k] for k in keys if k in APPROVED_SEARCH_BOARDS]
        if tokens:
            lines.append(f"{compact_label}: {', '.join(tokens)}")
    return "\n".join(lines)


def render_boards_docstring() -> str:
    """
    Aligned column layout, one site token per line.

    Used by the tool description, where the model is choosing exactly one token
    per query and the one-per-line form makes that constraint visually obvious.
    """
    lines: list[str] = []
    for _, docstring_label, keys in BOARD_GROUPS:
        tokens = [APPROVED_SEARCH_BOARDS[k] for k in keys if k in APPROVED_SEARCH_BOARDS]
        if not tokens:
            continue

        if len(docstring_label) <= _DOCSTRING_LABEL_WIDTH:
            # Label and first token share a line.
            head = f"{_DOCSTRING_INDENT}{docstring_label.ljust(_DOCSTRING_LABEL_WIDTH)}:"
            lines.append(f"{head} {tokens[0]}")
            rest = tokens[1:]
        else:
            # Label is too wide to share; tokens all drop to the next line.
            lines.append(f"{_DOCSTRING_INDENT}{docstring_label}:")
            rest = tokens

        lines.extend(f"{_DOCSTRING_CONTINUATION}{token}" for token in rest)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Shared query-construction rules
# ---------------------------------------------------------------------------

# The mandatory query shape. Both prompts state this; stating it twice by hand
# is how the location token went missing from one of them.
QUERY_FORMAT = '"<TECH_1> <TECH_2> <role> <LOCATION> <modifier> <site:TOKEN>"'

QUERY_MODIFIERS = "jobs, internship, intern, trainee"

# Why the location token is non-negotiable. Dropping it was the single largest
# source of empty result sets in the prototype.
LOCATION_MANDATE = (
    "The LOCATION token is MANDATORY and must appear verbatim, exactly as the\n"
    "user gave it, with no preposition (\"Cairo\", not \"in Cairo\"). Dropping it\n"
    "is the single most common cause of empty or irrelevant result sets."
)

# Canonical worked examples. Every one carries a location token, so a model
# copying an example verbatim cannot produce the failure mode above.
QUERY_EXAMPLES_GOOD: tuple[str, ...] = (
    '"Python Django Back-End Developer Cairo jobs site:wuzzuf.net OR site:bayt.com"',
    '"React Node.js Software Engineer Dubai internship site:linkedin.com/jobs"',
    '"C# ASP.NET Backend Developer Remote jobs site:glassdoor.com OR site:wellfound.com"',
)

QUERY_EXAMPLES_BAD: tuple[tuple[str, str], ...] = (
    ('"Back-End Developer jobs Cairo"',          "no tech, no site:"),
    ('"Python developer jobs"',                  "no site:, no location"),
    ('"Python Django Developer jobs site:wuzzuf.net"', "location dropped"),
    ('"Go ASP.NET Backend Developer jobs site:linkedin.com/jobs"',
     'leaked operational "Go" corrupts a C#/.NET search.'),
)


def current_year() -> int:
    """Current UTC year, used by the recency rules in both prompts."""
    return datetime.now(timezone.utc).year


def _indent(text: str, spaces: int) -> str:
    """Indent every line of a multi-line block by ``spaces``."""
    pad = " " * spaces
    return "\n".join(f"{pad}{line}" if line else line for line in text.split("\n"))


# ---------------------------------------------------------------------------
# Tool description
# ---------------------------------------------------------------------------
#
# This is the text LangChain sends to the model as the `tavily_job_search` tool
# description. It lives here rather than in the function's docstring because a
# docstring cannot interpolate the shared constants above — which is precisely
# how it drifted out of sync with the system prompt in the first place.
#
# tools.py passes it via `@tool(description=...)`.

_GOOD_EXAMPLES_BLOCK = "\n".join(
    f"    ✓ {example}" for example in QUERY_EXAMPLES_GOOD
)

# Kept as literal text: the ← annotations are hand-aligned to a common column
# and the last entry wraps, so generating them from QUERY_EXAMPLES_BAD would
# cost more in alignment logic than it saves. QUERY_EXAMPLES_BAD remains the
# machine-readable form for tests.
_BAD_EXAMPLES_BLOCK = """\
    ✗ "Back-End Developer jobs Cairo"            ← no tech, no site:
    ✗ "Python developer jobs"                    ← no site:, no location
    ✗ "Python Django Developer jobs site:wuzzuf.net"   ← location dropped
    ✗ "Go ASP.NET Backend Developer jobs site:linkedin.com/jobs"
          ← leaked operational "Go" corrupts a C#/.NET search."""

TAVILY_TOOL_DESCRIPTION = f"""\
Search for REAL, CURRENTLY OPEN job postings on APPROVED premium job boards.

══════════════════════════════════════════════════════════════════════
CRITICAL RULES — read completely before constructing each query call
══════════════════════════════════════════════════════════════════════

RULE 1 — TECHNOLOGY STACK IS MANDATORY IN EVERY QUERY:
  Every query MUST include at least one concrete technology keyword from
  the candidate's stack (Python, React, Django, Node.js, Flutter, C#,
  ASP.NET, etc.). A stack-free query is a CRITICAL FAILURE.

RULE 2 — APPROVED BOARDS ONLY — NO OPEN-WEB QUERIES:
  Every query MUST include a site: clause from the approved list below.

  APPROVED SITE TOKENS (use exactly as written — one per query):
{render_boards_docstring()}

  You MAY combine up to two approved boards using OR in one query.
  You may NOT use any domain not on the list above.
  You may NOT omit the site: clause entirely.

RULE 3 — QUERY FORMAT (mandatory structure — STACK TOKENS ONLY):
  {QUERY_FORMAT}
  where modifier is one of: {QUERY_MODIFIERS}

{_indent(LOCATION_MANDATE, 2)}

  Required examples:
{_GOOD_EXAMPLES_BLOCK}

  Forbidden examples:
{_BAD_EXAMPLES_BLOCK}

RULE 4 — FIRE BOTH REQUIRED QUERIES IN PARALLEL, IN YOUR FIRST TURN.

RULE 5 — NEVER INVENT RESULTS. Copy URLs character-for-character.

RULE 6 — RECENCY: the tool restricts results by CRAWL date via Tavily's
  time_range parameter (default: the past month) and appends the current
  year plus "hiring now" / "actively hiring" tokens to every query, so you
  do not need to add a year yourself — but you MAY append "hiring now" or
  the current year and it will not be penalised. Never add a year OLDER
  than the current one (that reintroduces the archived-listing problem this
  filter exists to solve). Note that crawl date is not the same as posting
  date, so still drop any listing whose own text says it is closed or old.

Returns a numbered list of job postings from approved boards only.
Category/listing pages, templates, blog articles, career-advice pages,
and non-vacancy subpaths are excluded BEFORE results reach you via a
multi-layer filter: (1) URL path gating against each board's canonical
listing structure, (2) snippet/title content scanning for age badges
(e.g. "Posted 5 years ago") and closed declarations, and (3) a live
shallow probe for boards where snippet staleness alone is insufficient.
Every listing shown below is a structurally-validated individual posting
candidate."""
