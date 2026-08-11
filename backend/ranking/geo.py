"""
Location normalization and comparison.

Location is the highest-weighted deterministic component (0-25), and it is the
one where providers disagree most. The same job can arrive as:

    JSearch     city="Cairo",  country="EG"
    Remotive    location_raw="Worldwide"
    Arbeitnow   location_raw="Berlin"
    Tavily      nothing at all

Meanwhile the user typed "Cairo". Comparing those needs a normalization step,
and getting it wrong is expensive in a specific direction: a candidate in Cairo
who is shown only Cairo-tagged jobs loses every remote role they could actually
take, which for many MENA candidates are the ones that escape the local salary
ceiling.

WHAT THIS DELIBERATELY IS NOT
-----------------------------
Not a geocoder. There is no distance calculation, no lat/long, no "within 50km".
Those need a geocoding service and a travel model, and neither is justified for
a comparison that feeds a 25-point component. This is exact and alias matching
over city and country names, which covers the realistic inputs and fails
visibly rather than subtly when it does not.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

# ISO-3166 alpha-2 -> canonical name, plus the aliases people actually type.
# Weighted toward this app's markets (Egypt, Gulf, Levant) and the destinations
# its users search: Europe, US, UK.
_COUNTRY_ALIASES: dict[str, str] = {
    # MENA
    "eg": "egypt", "egypt": "egypt", "arab republic of egypt": "egypt",
    "ae": "united arab emirates", "uae": "united arab emirates",
    "united arab emirates": "united arab emirates", "emirates": "united arab emirates",
    "sa": "saudi arabia", "ksa": "saudi arabia", "saudi arabia": "saudi arabia",
    "saudi": "saudi arabia",
    "qa": "qatar", "qatar": "qatar",
    "kw": "kuwait", "kuwait": "kuwait",
    "bh": "bahrain", "bahrain": "bahrain",
    "om": "oman", "oman": "oman",
    "jo": "jordan", "jordan": "jordan",
    "lb": "lebanon", "lebanon": "lebanon",
    "ma": "morocco", "morocco": "morocco",
    "tn": "tunisia", "tunisia": "tunisia",
    "dz": "algeria", "algeria": "algeria",
    # Europe
    "de": "germany", "germany": "germany", "deutschland": "germany",
    "nl": "netherlands", "netherlands": "netherlands", "holland": "netherlands",
    "fr": "france", "france": "france",
    "es": "spain", "spain": "spain",
    "pt": "portugal", "portugal": "portugal",
    "pl": "poland", "poland": "poland",
    "ie": "ireland", "ireland": "ireland",
    "gb": "united kingdom", "uk": "united kingdom",
    "united kingdom": "united kingdom", "england": "united kingdom",
    "great britain": "united kingdom",
    # Americas / other
    "us": "united states", "usa": "united states",
    "united states": "united states", "united states of america": "united states",
    "america": "united states",
    "ca": "canada", "canada": "canada",
    "in": "india", "india": "india",
    "pk": "pakistan", "pakistan": "pakistan",
    # Keys here must already be in `normalize_place` form — accent-stripped and
    # lowercase. "türkiye" as a key would be unreachable, because every lookup
    # arrives having already had its accents removed.
    "tr": "turkey", "turkey": "turkey", "turkiye": "turkey",
}

# City -> country, for the cities this app's users actually search. Lets
# "Cairo" match a job tagged only with country="EG", which is otherwise a miss.
_CITY_TO_COUNTRY: dict[str, str] = {
    "cairo": "egypt", "giza": "egypt", "alexandria": "egypt",
    "new cairo": "egypt", "6th of october": "egypt", "maadi": "egypt",
    "dubai": "united arab emirates", "abu dhabi": "united arab emirates",
    "sharjah": "united arab emirates",
    "riyadh": "saudi arabia", "jeddah": "saudi arabia", "dammam": "saudi arabia",
    "doha": "qatar", "kuwait city": "kuwait", "manama": "bahrain",
    "muscat": "oman", "amman": "jordan", "beirut": "lebanon",
    "casablanca": "morocco", "rabat": "morocco", "tunis": "tunisia",
    # As above: keys are pre-normalized, so "munchen" not "München".
    "berlin": "germany", "munich": "germany", "munchen": "germany",
    "hamburg": "germany", "frankfurt": "germany", "cologne": "germany",
    "amsterdam": "netherlands", "paris": "france", "madrid": "spain",
    "barcelona": "spain", "lisbon": "portugal", "warsaw": "poland",
    "dublin": "ireland", "london": "united kingdom", "manchester": "united kingdom",
    "new york": "united states", "san francisco": "united states",
    "seattle": "united states", "austin": "united states",
    "toronto": "canada", "vancouver": "canada",
    "bangalore": "india", "bengaluru": "india", "mumbai": "india",
    "istanbul": "turkey",
}

# Strings that mean "location is not a constraint" rather than naming a place.
# Remotive uses these in `candidate_required_location`, where they describe
# where the candidate may live, not where the job is.
_ANYWHERE_TOKENS: frozenset[str] = frozenset(
    {
        "worldwide", "anywhere", "global", "globally", "remote",
        "fully remote", "100% remote", "work from home", "wfh",
        "remote worldwide", "anywhere in the world", "international",
    }
)

# Multi-country regions. Coarse on purpose: "Europe" genuinely is coarse, and
# pretending otherwise would invent precision the source never had.
_REGIONS: dict[str, frozenset[str]] = {
    "europe": frozenset(
        {"germany", "netherlands", "france", "spain", "portugal", "poland",
         "ireland", "united kingdom"}
    ),
    "emea": frozenset(
        {"germany", "netherlands", "france", "spain", "portugal", "poland",
         "ireland", "united kingdom", "egypt", "united arab emirates",
         "saudi arabia", "qatar", "kuwait", "bahrain", "oman", "jordan",
         "lebanon", "morocco", "tunisia", "algeria", "turkey"}
    ),
    "mena": frozenset(
        {"egypt", "united arab emirates", "saudi arabia", "qatar", "kuwait",
         "bahrain", "oman", "jordan", "lebanon", "morocco", "tunisia", "algeria"}
    ),
    "gcc": frozenset(
        {"united arab emirates", "saudi arabia", "qatar", "kuwait", "bahrain",
         "oman"}
    ),
    "middle east": frozenset(
        {"egypt", "united arab emirates", "saudi arabia", "qatar", "kuwait",
         "bahrain", "oman", "jordan", "lebanon", "turkey"}
    ),
}

_SPLIT_RE = re.compile(r"[,/|;()\-–—]+")
_WS_RE = re.compile(r"\s+")


def normalize_place(value: Optional[str]) -> str:
    """
    Casefold, strip accents and punctuation, collapse whitespace.

    Accent stripping is what lets "München" match "Munich" via the city table,
    and "Türkiye" match "turkey".
    """
    if not value:
        return ""
    decomposed = unicodedata.normalize("NFKD", value)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    cleaned = _WS_RE.sub(" ", ascii_only.lower()).strip()
    return cleaned.strip(".,")


def canonical_country(value: Optional[str]) -> Optional[str]:
    """Map a country name or ISO code to a canonical name, or None."""
    normalized = normalize_place(value)
    if not normalized:
        return None
    return _COUNTRY_ALIASES.get(normalized)


def country_for_city(value: Optional[str]) -> Optional[str]:
    """Country a known city belongs to, or None."""
    normalized = normalize_place(value)
    if not normalized:
        return None
    return _CITY_TO_COUNTRY.get(normalized)


def means_anywhere(value: Optional[str]) -> bool:
    """
    True when a location string expresses "no geographic constraint".

    Checked against whole comma-separated parts rather than as a substring, so
    "Remote" matches but "Remote, Germany" does not — the latter is a real
    constraint and treating it as unconstrained would show a Cairo candidate a
    job they cannot legally take.
    """
    normalized = normalize_place(value)
    if not normalized:
        return False
    if normalized in _ANYWHERE_TOKENS:
        return True
    parts = [p.strip() for p in _SPLIT_RE.split(normalized) if p.strip()]
    return len(parts) == 1 and parts[0] in _ANYWHERE_TOKENS


def parse_place(value: Optional[str]) -> tuple[set[str], set[str]]:
    """
    Extract (cities, countries) from a free-text location string.

    Handles the shapes providers actually emit: "Cairo", "Cairo, Egypt",
    "Cairo, EG", "Berlin, Germany (Hybrid)". Any part that is not a recognized
    country is treated as a possible city, and a recognized city implies its
    country — so "Cairo" alone still matches a job tagged only "EG".
    """
    normalized = normalize_place(value)
    if not normalized:
        return (set(), set())

    cities: set[str] = set()
    countries: set[str] = set()

    for part in (p.strip() for p in _SPLIT_RE.split(normalized)):
        if not part:
            continue
        country = _COUNTRY_ALIASES.get(part)
        if country:
            countries.add(country)
            continue
        implied = _CITY_TO_COUNTRY.get(part)
        if implied:
            cities.add(part)
            countries.add(implied)
        else:
            cities.add(part)

    return (cities, countries)


def region_countries(value: Optional[str]) -> Optional[frozenset[str]]:
    """Countries covered by a named region, or None if not a known region."""
    return _REGIONS.get(normalize_place(value))


def places_match(wanted: str, candidate: Optional[str]) -> tuple[bool, bool]:
    """
    Compare a requested location against a job's location string.

    Returns ``(city_match, country_match)``. City match is the strong signal;
    country match is the weaker "same market, wrong city" case, which is still
    worth points because commuting distance and relocation are the user's call,
    not the ranker's.
    """
    if not candidate:
        return (False, False)

    wanted_cities, wanted_countries = parse_place(wanted)
    cand_cities, cand_countries = parse_place(candidate)

    city_match = bool(wanted_cities & cand_cities)

    country_match = bool(wanted_countries & cand_countries)
    if not country_match:
        # "Europe" / "MENA" as the job's stated coverage.
        region = region_countries(candidate)
        if region and wanted_countries & region:
            country_match = True

    return (city_match, country_match)
