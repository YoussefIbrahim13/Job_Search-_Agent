"""
Tests for location normalization.

Location carries the highest weight in the scorer (25 of 100), and the inputs
are the messiest: the same job arrives as "Cairo"/"EG" from one provider,
"Worldwide" from another, and nothing at all from a third.

The asymmetry worth remembering: a false negative here (failing to match a job
the user could take) is invisible — they simply never see it — while a false
positive is merely a slightly worse ranking. The tests lean accordingly.
"""
import pytest

from backend.ranking.geo import (
    canonical_country,
    country_for_city,
    means_anywhere,
    normalize_place,
    parse_place,
    places_match,
    region_countries,
)


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Cairo", "cairo"),
        ("  CAIRO  ", "cairo"),
        ("München", "munchen"),
        ("Türkiye", "turkiye"),
        ("Cairo,", "cairo"),
        ("", ""),
        (None, ""),
    ],
)
def test_normalize_place(raw, expected):
    assert normalize_place(raw) == expected


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("EG", "egypt"),
        ("eg", "egypt"),
        ("Egypt", "egypt"),
        ("UAE", "united arab emirates"),
        ("AE", "united arab emirates"),
        ("KSA", "saudi arabia"),
        ("US", "united states"),
        ("USA", "united states"),
        ("UK", "united kingdom"),
        ("Deutschland", "germany"),
        ("Atlantis", None),
        (None, None),
    ],
)
def test_canonical_country(raw, expected):
    """
    Providers emit ISO codes ("EG") while users type names ("Egypt"). Without
    this mapping every JSearch record would fail to match a typed location.
    """
    assert canonical_country(raw) == expected


def test_every_lookup_key_is_already_normalized():
    """
    Guards a whole class of silent dead data: a key with an accent ("München")
    can never be hit, because every lookup arrives accent-stripped. Nothing
    errors — the entry simply never matches, and the city quietly stops
    resolving to its country.
    """
    from backend.ranking import geo

    tables = {
        "_COUNTRY_ALIASES": geo._COUNTRY_ALIASES,
        "_CITY_TO_COUNTRY": geo._CITY_TO_COUNTRY,
        "_ANYWHERE_TOKENS": geo._ANYWHERE_TOKENS,
        "_REGIONS": geo._REGIONS,
    }
    unreachable = {
        name: [k for k in table if normalize_place(k) != k]
        for name, table in tables.items()
    }
    unreachable = {name: keys for name, keys in unreachable.items() if keys}

    assert not unreachable, f"unreachable lookup keys: {unreachable}"


def test_known_cities_imply_their_country():
    """
    Lets "Cairo" match a job tagged only country="EG" — a common JSearch shape
    where the city field is null.
    """
    assert country_for_city("Cairo") == "egypt"
    assert country_for_city("Dubai") == "united arab emirates"
    assert country_for_city("München") == "germany"
    assert country_for_city("Nowhereville") is None


# ---------------------------------------------------------------------------
# "Anywhere"
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", ["Worldwide", "Anywhere", "Global", "100% Remote", "remote", "WFH"]
)
def test_unconstrained_locations_are_recognised(value):
    assert means_anywhere(value) is True


@pytest.mark.parametrize(
    "value", ["Remote, Germany", "Cairo", "Remote (US only)", "Berlin", ""]
)
def test_constrained_locations_are_not_treated_as_anywhere(value):
    """
    "Remote, Germany" is a real constraint. Treating it as unconstrained would
    show a Cairo candidate a job they cannot legally take — the substring trap
    that whole-part matching exists to avoid.
    """
    assert means_anywhere(value) is False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def test_parse_place_splits_city_and_country():
    cities, countries = parse_place("Cairo, Egypt")
    assert "cairo" in cities
    assert "egypt" in countries


def test_parse_place_handles_iso_codes():
    cities, countries = parse_place("Cairo, EG")
    assert "cairo" in cities
    assert "egypt" in countries


def test_parse_place_infers_country_from_a_bare_city():
    cities, countries = parse_place("Cairo")
    assert "cairo" in cities
    assert "egypt" in countries


def test_parse_place_keeps_unknown_parts_as_cities():
    """
    An unrecognized token is more likely a city than noise, and treating it as
    one lets an exact string match still succeed.
    """
    cities, _ = parse_place("Smallville, Egypt")
    assert "smallville" in cities


def test_parse_place_ignores_decoration():
    cities, countries = parse_place("Berlin, Germany (Hybrid)")
    assert "berlin" in cities
    assert "germany" in countries


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


def test_exact_city_match():
    city, country = places_match("Cairo", "Cairo, Egypt")
    assert city is True
    assert country is True


def test_city_matches_a_country_only_record():
    """The JSearch shape where job_city is null but job_country is "EG"."""
    city, country = places_match("Cairo", "EG")
    assert city is False
    assert country is True


def test_same_country_different_city():
    city, country = places_match("Cairo", "Alexandria, Egypt")
    assert city is False
    assert country is True


def test_different_country_matches_nothing():
    city, country = places_match("Cairo", "Berlin, Germany")
    assert city is False
    assert country is False


def test_region_covers_member_countries():
    """A job advertised as "Europe" should match a Berlin-based search."""
    _, country = places_match("Berlin", "Europe")
    assert country is True


def test_mena_region_covers_egypt():
    _, country = places_match("Cairo", "MENA")
    assert country is True


def test_region_does_not_cover_outsiders():
    _, country = places_match("Cairo", "Europe")
    assert country is False


def test_empty_candidate_matches_nothing():
    assert places_match("Cairo", None) == (False, False)
    assert places_match("Cairo", "") == (False, False)


def test_region_lookup_is_case_and_accent_insensitive():
    assert region_countries("EMEA") == region_countries("emea")
    assert region_countries("not-a-region") is None
