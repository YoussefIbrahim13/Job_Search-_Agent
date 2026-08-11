"""
Regression tests for the URL gates.

Two separate gates decide whether a listing URL survives:

  1. `tools._passes_path_gate` — positive assertion that the path looks like a
     canonical single-vacancy URL for that board.
  2. The bad-link rejection block in `recruitment_agent._validate_and_fix_output`.

The bug this file exists to prevent: those two gates disagreed. The path gate
admits LinkedIn's canonical `/jobs/view/<id>` form, and then `_FAKE_LINK_RE`
(`/jobs/view/\\d+$`) rejected that exact shape a few lines later — so every
numeric-ID LinkedIn job the pipeline validated was thrown away immediately
afterwards. `test_gates_agree` is the guard against that class of contradiction
returning.
"""
import pytest

from backend.agents.tools import _passes_path_gate

# (label, url, should_pass)
PATH_GATE_CASES = [
    # LinkedIn — both canonical listing forms must pass.
    ("linkedin numeric id", "https://www.linkedin.com/jobs/view/4396364201", True),
    ("linkedin slug + id", "https://www.linkedin.com/jobs/view/backend-developer-at-acme-4396364201", True),
    ("linkedin regional subdomain", "https://eg.linkedin.com/jobs/view/4396364201", True),
    ("linkedin search page", "https://www.linkedin.com/jobs/search/?keywords=python", False),
    ("linkedin jobs hub", "https://www.linkedin.com/jobs/", False),

    # Wuzzuf
    ("wuzzuf listing", "https://wuzzuf.net/jobs/p/kmiuk743oelq-backend-developer", True),
    ("wuzzuf template page", "https://wuzzuf.net/r/template-slug", False),

    # Bayt
    ("bayt english listing", "https://www.bayt.com/en/jobs/senior-developer-5012345/", True),
    ("bayt arabic listing", "https://www.bayt.com/ar/jobs/mudir-tatwir-5012345/", True),
    ("bayt legacy id", "https://www.bayt.com/job/1234567", True),

    # Other approved boards
    ("glassdoor listing", "https://www.glassdoor.com/job-listing/backend-dev-acme-JV123.htm", True),
    ("akhtaboot listing", "https://www.akhtaboot.com/jobs/12345-software-engineer", True),
    ("wwr listing", "https://weworkremotely.com/remote-jobs/engineering/senior-dev", True),
    ("himalayas listing", "https://himalayas.app/jobs/backend-engineer-acme", True),
    ("himalayas company page", "https://himalayas.app/companies/acme", False),
    ("dice listing", "https://www.dice.com/jobs/detail/abc-123", True),
    ("greenhouse listing", "https://boards.greenhouse.io/acme/jobs/4012345", True),
    ("greenhouse board hub", "https://boards.greenhouse.io/acme", False),
    ("lever listing", "https://jobs.lever.co/acme/abc-123-def", True),

    # Boards with no path pattern registered pass through by design.
    ("unregistered board", "https://example-jobs.com/anything/at/all", True),
]


@pytest.mark.parametrize(
    "label,url,should_pass", PATH_GATE_CASES, ids=[c[0] for c in PATH_GATE_CASES]
)
def test_path_gate(label, url, should_pass):
    assert _passes_path_gate(url) is should_pass, (
        f"{label}: _passes_path_gate({url!r}) should be {should_pass}"
    )


def test_malformed_url_passes_through():
    """Malformed URLs are _is_usable_url's job, not the path gate's."""
    assert _passes_path_gate("not a url") is True
    assert _passes_path_gate("") is True


# --- Cross-gate consistency -------------------------------------------------

# URLs that the path gate admits and that must therefore also survive the
# validation layer's bad-link rejection. A URL cannot be simultaneously
# "canonical" and "fake".
GATE_AGREEMENT_URLS = [
    "https://www.linkedin.com/jobs/view/4396364201",
    "https://www.linkedin.com/jobs/view/backend-developer-at-acme-4396364201",
    "https://wuzzuf.net/jobs/p/kmiuk743oelq-backend-developer",
    "https://www.bayt.com/en/jobs/senior-developer-5012345/",
    "https://boards.greenhouse.io/acme/jobs/4012345",
    "https://jobs.lever.co/acme/abc-123-def",
]


@pytest.mark.parametrize("url", GATE_AGREEMENT_URLS)
def test_gates_agree(url):
    """
    Any URL the path gate calls canonical must survive full validation.

    This is the regression guard for the LinkedIn `/jobs/view/<id>` conflict.
    """
    from backend.agents.recruitment_agent import _validate_and_fix_output

    assert _passes_path_gate(url) is True, "test fixture must be a canonical URL"

    result = _validate_and_fix_output({
        "job_title": "Backend Developer",
        "location": "Cairo",
        "jobs": [{
            "job_title": "Backend Developer",
            "company_name": "Acme Industrial",
            "location": "Cairo, Egypt",
            "application_link": url,
            "match_score": 80,
            "match_reason": "Strong match on Python and Django experience.",
            "required_skills": ["Python", "Django"],
        }],
    })

    assert len(result["jobs"]) == 1, (
        f"{url} passes the path gate but was dropped by the validation layer"
    )
    assert result["jobs"][0]["application_link"] == url
