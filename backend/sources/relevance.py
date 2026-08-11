"""
Client-side subject-matter gating for providers with no server-side search.

WHY THIS EXISTS
---------------
Two of the free providers return an unfiltered feed of the newest postings
regardless of what was asked for:

* **Arbeitnow** documents no search parameter at all — the job-board API is
  paginated only.
* **Remotive** accepts `search`, `category`, and `limit` without error and
  ignores all three. Verified by probing the live endpoint: `search=django`,
  `search=Python Django Developer`, `category=software-dev`, and no parameters
  at all each returned the identical twenty newest jobs. The adapter originally
  trusted the parameter and consequently fed twenty sales and copywriting roles
  into a Django search.

That failure is invisible in mocked tests by construction — a fixture returns
whatever it was written to return, so a parameter the provider silently drops
looks like it works.

WHAT THIS IS AND IS NOT
-----------------------
This is the `search` parameter those providers do not offer, implemented
client-side. It answers "is this posting even about the right subject?" and
nothing more. It does not rank, weight, or judge quality — that is the scorer's
job, and keeping the two separate is what stops relevance logic leaking into
the adapter layer.

Kept deliberately coarse: false positives are harmless (the scorer demotes
them, and the relevance gate drops the truly unconnected), while false
negatives silently hide real jobs the user would have wanted.
"""

from __future__ import annotations

import re

from backend.sources.criteria import SearchCriteria
from backend.sources.schema import NormalizedJob

# Words too common to discriminate. A gate keyed on "developer" or "engineer"
# admits the entire board and stops being a gate.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "and", "the", "of", "for", "in", "at", "to", "with",
        "senior", "junior", "lead", "principal", "staff", "mid", "level",
        "engineer", "developer", "software", "specialist", "expert",
        "manager", "consultant", "analyst", "intern", "internship",
        "remote", "hybrid", "onsite", "contract", "fulltime", "parttime",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9+#.]{2,}")


def query_keywords(criteria: SearchCriteria) -> set[str]:
    """
    Distinctive tokens a posting must mention to clear the gate.

    Skills stay whole ("node.js", "c#") because splitting them destroys exactly
    what makes them distinctive. Title words are split and stripped of
    stopwords, so "Senior Backend Engineer" contributes "backend" — the only
    word in it that narrows anything.
    """
    tokens: set[str] = set()

    for skill in criteria.skills:
        cleaned = skill.strip().lower()
        if len(cleaned) >= 2:
            tokens.add(cleaned)

    for title in criteria.titles:
        for word in _TOKEN_RE.findall(title.lower()):
            if word not in _STOPWORDS and len(word) >= 3:
                tokens.add(word)

    return tokens


def mentions_any(job: NormalizedJob, keywords: set[str]) -> bool:
    """
    Whether a posting mentions any requested keyword.

    Substring matching is intentional: "python" must match "Python3" and
    "PythonDeveloper". Searches title, harvested skills, and description —
    the description matters most, since these providers rarely populate
    structured skill fields.

    An empty keyword set admits everything: there is nothing to gate on, and
    silently returning zero jobs would be worse than returning unfiltered ones.
    """
    if not keywords:
        return True

    haystack = " ".join(
        (job.title, " ".join(job.required_skills), job.description)
    ).lower()
    return any(keyword in haystack for keyword in keywords)
