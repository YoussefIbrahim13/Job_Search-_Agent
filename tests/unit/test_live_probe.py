"""
Tests for the live closure probe.

The bug this file exists to prevent, reported with screenshots: two LinkedIn
postings from Akvelon, Inc. — both a year old, both showing "No longer
accepting applications" — were returned to the user at a 90% match score.

Root cause was a byte budget, not a missing rule. The probe looked for exactly
the right phrase, but read only the first 8,000 bytes of the page while
LinkedIn renders its closure badge at roughly byte 31,500. The probe fetched
the page, found nothing, and reported the listing as verified-fresh.

Two properties are asserted here:
  1. The badge check sees deep enough into the page to find the marker.
  2. The prose age check stays shallow, because past the hero section the page
     lists OTHER jobs whose age strings are not about this listing.

The network tests are marked `live` and excluded from CI (`-m "not live"`);
the offline tests below cover the logic without hitting LinkedIn.
"""
import pytest

from backend.agents.tools import (
    _HEAD_PROBE_BYTES,
    _LINKEDIN_HEAD_PROBE_BYTES,
    _has_explicit_closed_badge,
    _is_canonical_listing_url,
)

# Real markup from a closed LinkedIn listing (Akvelon, Inc. — screenshot 1).
LINKEDIN_CLOSED_FRAGMENT = (
    '<figure class="closed-job closed-job__flavor topcard__flavor-row">'
    '<span class="closed-job__icon closed-job__icon--error-pebble lazy-load"></span>'
    '<figcaption class="closed-job__flavor--closed">'
    "No longer accepting applications</figcaption></figure>"
)


def test_badge_is_detected_in_linkedin_markup():
    assert _has_explicit_closed_badge(LINKEDIN_CLOSED_FRAGMENT, "linkedin.com") is True


def test_badge_survives_being_buried_deep_in_the_page():
    """
    The regression itself: the marker sits ~31KB in on a real page. Anything
    that reintroduces an 8KB ceiling for the badge check fails here.
    """
    buried = ("<div>filler</div>" * 3000) + LINKEDIN_CLOSED_FRAGMENT
    assert len(buried) > 30_000, "fixture must exceed the old 8KB budget"
    assert _has_explicit_closed_badge(buried, "linkedin.com") is True


def test_linkedin_budget_covers_the_observed_badge_offset():
    """Measured at 31,505-31,762 bytes across live listings; keep headroom."""
    assert _LINKEDIN_HEAD_PROBE_BYTES >= 40_000
    assert _LINKEDIN_HEAD_PROBE_BYTES > _HEAD_PROBE_BYTES


def test_open_listing_markup_has_no_badge():
    open_page = (
        '<figure class="topcard__flavor-row">'
        '<span class="num-applicants__caption">Be among the first 25 applicants</span>'
        "</figure>"
    )
    assert _has_explicit_closed_badge(open_page, "linkedin.com") is False


def test_structural_class_is_matched_not_only_prose():
    """
    The CSS class is the load-bearing signal: it is locale-independent (so it
    works on the Arabic pages served to MENA users) and it cannot be tripped by
    a job description that merely discusses applications closing.
    """
    class_only = '<figcaption class="closed-job__flavor--closed">لم يعد مفتوحاً</figcaption>'
    assert _has_explicit_closed_badge(class_only, "linkedin.com") is True


def test_job_description_prose_does_not_trip_the_badge():
    """A description mentioning closing dates must not read as a closure badge."""
    prose = (
        "<p>Applications for our graduate scheme close on 31 March. "
        "We are no longer accepting late CVs by post.</p>"
    )
    assert _has_explicit_closed_badge(prose, "linkedin.com") is False


def test_badge_check_is_scoped_per_board():
    """LinkedIn markup must not satisfy the Wuzzuf badge check or vice versa."""
    assert _has_explicit_closed_badge(LINKEDIN_CLOSED_FRAGMENT, "wuzzuf.net") is False
    assert _has_explicit_closed_badge("<span> Closed </span>", "linkedin.com") is False
    assert _has_explicit_closed_badge("<span> Closed </span>", "wuzzuf.net") is True


@pytest.mark.parametrize(
    "url",
    [
        "https://www.linkedin.com/jobs/view/4188849806",
        "https://www.linkedin.com/jobs/view/junior-net-engineer-at-akvelon-inc-4188849806",
        "https://wuzzuf.net/jobs/p/abc123-backend-developer",
    ],
)
def test_canonical_listings_use_the_narrow_probe(url):
    assert _is_canonical_listing_url(url) is True


# --- Live network checks (excluded from CI) ---------------------------------

@pytest.mark.live
@pytest.mark.parametrize(
    "url,expected_stale",
    [
        # The two listings from the bug report — both closed.
        ("https://www.linkedin.com/jobs/view/junior-net-engineer-at-akvelon-inc-4188849806", True),
        ("https://www.linkedin.com/jobs/view/intern-junior-net-developer-at-akvelon-inc-4192743745", True),
        # Open at time of writing; guards against over-filtering.
        ("https://www.linkedin.com/jobs/view/4446186750", False),
        ("https://www.linkedin.com/jobs/view/4448284265", False),
    ],
)
def test_live_linkedin_probe(url, expected_stale):
    from backend.agents.tools import _verify_live_url_is_stale

    assert _verify_live_url_is_stale(url) is expected_stale
