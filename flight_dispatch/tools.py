"""The tool surface exposed to an LLM.

WHAT THIS FILE IS FOR
---------------------
CP1-CP3 built a routing engine whose functions take rich Python objects:
`plan_route` wants an `Airport`, a pre-filtered `Sequence[Navaid]`, an
`AircraftProfile`, a `WindSource` and an `AirspaceIndex`. A language model
cannot supply any of those. It can supply strings, numbers and booleans.

So this module is an adapter, not new logic. Each tool here takes simple
JSON-able arguments, does the loading and filtering the CLI used to do,
calls straight into the CP1-CP3 functions, and returns a JSON-able dict.

NOTHING IN CP1-CP3 CHANGED TO MAKE THIS WORK. That is the design
principle the project is built to demonstrate: the routing engine does
not know or care that an LLM exists. Swap the model, or remove it
entirely, and `plan_route` is unaffected.

BACKEND-AGNOSTIC ON PURPOSE
---------------------------
`ToolSpec` describes a tool in neutral terms. Each model backend converts
that description into whatever format it wants -- JSON Schema for the
Claude API, a `Tool` subclass for Apple's Foundation Models SDK. The
descriptions, the dispatch table, and the functions themselves are shared.

DESIGNING FOR A MODEL, NOT A PROGRAMMER
---------------------------------------
Three things matter more here than in ordinary API design:

  1. Descriptions are prompt text. The model decides whether to call a
     tool by reading its description, so each one says WHEN to use it,
     not just what it does.
  2. Arguments must be guessable. A tool taking `radius_nm` and
     `min_neighbors` invites a model to invent plausible-sounding
     numbers. Those stay as defaults; the model only chooses things it
     can reason about.
  3. Errors are answers, not exceptions. A bad ICAO code returns a dict
     explaining the problem so the model can correct itself on the next
     turn. Raising would end the conversation.
"""

import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .aircraft import AIRCRAFT, get_aircraft
from .airspace import (
    AirspaceDataError,
    AirspaceIndex,
    airspace_near_route,
    load_airspace,
)
from .data_loader import (
    MissingDataError,
    load_airports,
    load_runway_area,
    load_navaids,
    navaids_near_route,
)
from .geo import haversine_nm, initial_bearing_deg
from .route import NoRouteFound, plan_route
from .wind import ConstantWindSource

# ---------------------------------------------------------------------------
# Cached reference data
#
# Loading airports and navaids parses ~18 MB of CSV and takes about a
# second. The CLI could afford that once per process; an agent may call
# several tools per turn, so the data is loaded lazily and kept.
# ---------------------------------------------------------------------------

# Routes longer than this return a compact route string instead of a
# per-leg table. Sized for the on-device model's 4,096-token context: a
# 21-waypoint transcontinental route overflowed it, losing a plan that
# had computed correctly.
MAX_DETAILED_WAYPOINTS = 12

# Candidate airports returned by a name search. Was 8; a city lookup then
# cost ~217 tokens against ~36 for an exact code, and seven of the eight
# were discarded the moment the model chose one.
MAX_AIRPORT_MATCHES = 3

DEFAULT_WIND_ALTITUDE_FT = 8000.0

# Altitudes the model may ask for, as strings because that is what a
# schema enum carries.
#
# WHY AN ENUM AT ALL. Apple's schema format cannot express an optional
# parameter, so `backend_apple._is_exposed` withholds free numerics --
# the fix for a model that invented `payload_lb: 1600` for a Cessna 172
# and made both plans refuse. The side effect was that `altitude_ft`
# could not be passed to this tool AT ALL on-device: asked for the wind
# at 35,000 ft it always answered at the 8,000 ft default, and said
# 35,000. An enum survives the filter, exactly as `aircraft` does.
#
# These values are chosen to land on DISTINCT pressure levels. Wind is
# published at pressure levels, not arbitrary altitudes, so two options
# either side of one level would return identical data and imply a
# precision that is not there.
ALTITUDE_CHOICES = (
    "3000",
    "5000",
    "10000",
    "14000",
    "18000",
    "24000",
    "30000",
    "34000",
    "39000",
    "45000",
)

_COMPASS = (
    "north", "north-northeast", "northeast", "east-northeast",
    "east", "east-southeast", "southeast", "south-southeast",
    "south", "south-southwest", "southwest", "west-southwest",
    "west", "west-northwest", "northwest", "north-northwest",
)


def _compass_point(degrees: float) -> str:
    """Name the 16-point compass direction for a bearing.

    Included in wind results because the model got it wrong unprompted:
    it rendered 239 degrees as "from the northeast", which is the
    opposite side of the compass. Converting degrees to a quadrant is
    arithmetic, and arithmetic is not the model's job here.
    """
    return _COMPASS[int((degrees % 360) / 22.5 + 0.5) % 16]


def _coerce_altitude(value: Any, default: float) -> Optional[float]:
    """Read an altitude that may arrive as a number or an enum string.

    The schema carries strings, since that is what enums hold, but the
    CLI and the tests pass floats. Returns None if it is neither, so the
    caller can answer with an error the model can act on.
    """
    if value is None or value == "":
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

_CACHE: Dict[str, Any] = {}


def _airports() -> dict:
    if "airports" not in _CACHE:
        _CACHE["airports"] = load_airports()
    return _CACHE["airports"]


def _navaids() -> list:
    if "navaids" not in _CACHE:
        _CACHE["navaids"] = load_navaids()
    return _CACHE["navaids"]


def _runways() -> dict:
    """Total open runway area per airport, for ranking name searches.

    Loaded lazily and tolerantly: ranking is a nicety, so a missing
    runways.csv degrades the ordering rather than failing the lookup.
    """
    if "runways" not in _CACHE:
        try:
            _CACHE["runways"] = load_runway_area()
        except MissingDataError:
            _CACHE["runways"] = {}
    return _CACHE["runways"]


def _airspace() -> list:
    if "airspace" not in _CACHE:
        _CACHE["airspace"] = load_airspace()
    return _CACHE["airspace"]


@dataclass(frozen=True)
class ToolSpec:
    """A backend-neutral description of one callable tool.

    Attributes:
        name: What the model calls. Snake case, verb-first.
        description: Prompt text. Should say when to reach for this tool,
            since that is what the model reads to decide.
        parameters: Maps argument name -> {type, description, required,
            and optionally enum}. Deliberately a plain dict rather than a
            JSON Schema, so each backend can render it its own way.
        func: The Python callable. Takes keyword arguments matching
            `parameters` and returns a JSON-able dict.
    """

    name: str
    description: str
    parameters: Dict[str, Dict[str, Any]]
    func: Callable[..., Dict[str, Any]]

    def required_names(self) -> List[str]:
        return [
            name
            for name, spec in self.parameters.items()
            if spec.get("required", False)
        ]

    def json_schema(self) -> Dict[str, Any]:
        """Render as JSON Schema, which is what the Claude API expects."""
        properties = {}
        for name, spec in self.parameters.items():
            prop: Dict[str, Any] = {
                "type": spec["type"],
                "description": spec["description"],
            }
            if "enum" in spec:
                prop["enum"] = spec["enum"]
            properties[name] = prop

        return {
            "type": "object",
            "properties": properties,
            "required": self.required_names(),
        }


# ---------------------------------------------------------------------------
# The tools
# ---------------------------------------------------------------------------


# Airport types, most significant first. OurAirports publishes no size
# or traffic figures, but this classification plus scheduled_service is a
# good proxy for the one a pilot means.
_TYPE_RANK = {
    "large_airport": 0,
    "medium_airport": 1,
    "small_airport": 2,
    "seaplane_base": 3,
    "balloonport": 4,
    "heliport": 5,
    "closed": 6,
}


def _match_rank(airport, needle: str = "", runway_area: float = 0.0) -> tuple:
    """Sort key for name-search results: most likely answer first.

    THE BUG THIS REPLACES. Sorting by name length put a Mexican airstrip
    named literally "San Francisco" (MX-1385) ahead of San Francisco
    International, and the agent planned a flight from it. Short is not
    the same as canonical, and neither is a guess at what the name
    contains.

    Four real signals, in priority order:

      TYPE. large_airport (1,172 of 85,825) versus small_airport
        (42,700). The single strongest discriminator in the dataset.
      SCHEDULED SERVICE. 4,365 airports carry commercial airline
        service. If a passenger could book a flight there, it is almost
        certainly what they meant.
      IATA CODE. 9,054 have one. Presence signals commercial relevance;
        an airstrip does not get an IATA code.
      MUNICIPALITY MATCH. Someone typing "San Francisco" usually means
        the airport serving that city, not one that happens to share the
        name -- so a municipality hit outranks a mere name hit.
      RUNWAY AREA. The tiebreaker among airports the first four
        cannot separate. London Heathrow and East London (South Africa)
        are both large airports with scheduled service and IATA codes,
        and "East London Airport" is the shorter name -- so name length
        put a South African regional airport ahead of Heathrow.

    Name length survives only as the final tiebreaker, which is all it
    was ever suited for.

    WHY AREA RATHER THAN THE LONGEST RUNWAY
    ---------------------------------------
    Longest-single-runway came first, and it loses to any airport with
    one long strip and little else. Dubai International -- ~90 million
    passengers a year -- ranked below Al Maktoum, which is nearly empty
    and has a runway 174 ft longer. Tokyo Haneda lost to Narita the same
    way. Total area counts every runway and its width, which tracks how
    much traffic an airport can actually handle. On fifteen
    multi-airport cities: longest runway 12 correct, total area 14.

    The fifteenth is Mexico City, and no runway metric fixes it -- see
    `data_loader.load_runway_area`.
    """
    # `needle` arrives normalised, so the municipality is normalised too
    # -- otherwise "OHare" would never match the city text it came from.
    municipality_match = needle and needle in _normalise(airport.municipality)

    return (
        _TYPE_RANK.get(airport.airport_type, 7),  # large airports first
        not airport.scheduled_service,            # then commercial service
        not airport.iata_code,                    # then an IATA code
        not municipality_match,                   # then serving that city
        -runway_area,                             # then the bigger airport
        len(airport.name),                        # then shortest name
    )


_SEARCH_INDEX: Optional[List] = None


def _normalise(text: str) -> str:
    """Lowercase, strip accents, strip punctuation.

    Punctuation goes because people type identifiers without it far more
    often than with it, and no airport is distinguished from another by
    an apostrophe -- O'Hare and OHare are the same place.

    Accents go for the same reason, and it was a real bug: Guarulhos'
    municipality is "Sao Paulo" with a tilde, so a plain "Sao Paulo"
    search matched nothing there and fell through to a hotel helipad
    whose name happens to spell it without one. Decomposing to NFKD
    splits an accented character into its base letter plus a combining
    mark, and dropping the marks leaves the base letters behind.

    Both the query and the haystack go through this, so the comparison is
    blind to both on both sides.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    unaccented = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", " ", unaccented).strip()


def _search_index() -> List:
    """Airports paired with their searchable text, built once.

    Every name, city, IATA and ICAO code a person might use, normalised.
    85,825 airports is enough that rebuilding this per query would show
    up in a conversation's latency.

    `keywords` is included because it is where OurAirports keeps the
    alternate names: local-language forms ("Londres" for Heathrow,
    "Ciudad de Mexico" for Benito Juarez), former names, and IATA
    metropolitan area codes (LON, NYC, CHI). It is searched but never
    ranked on -- a synonym list says nothing about how big an airport is.
    """
    global _SEARCH_INDEX
    if _SEARCH_INDEX is None:
        _SEARCH_INDEX = [
            (
                airport,
                _normalise(
                    " ".join(
                        (
                            airport.name,
                            airport.municipality,
                            airport.iata_code,
                            airport.icao,
                            airport.keywords,
                        )
                    )
                ),
            )
            for airport in _airports().values()
        ]
    return _SEARCH_INDEX


def find_airport(query: str) -> Dict[str, Any]:
    """Resolve a name or code to an ICAO identifier."""
    query = query.strip()
    airports = _airports()

    def described(airport):
        return {
            "found": True,
            "icao": airport.icao,
            "name": airport.name,
            "latitude": round(airport.lat, 4),
            "longitude": round(airport.lon, 4),
            "elevation_ft": airport.elevation_ft,
        }

    # Exact ICAO match first -- the common case, and unambiguous.
    exact = airports.get(query.upper())
    if exact is not None:
        return described(exact)

    # Then an exact IATA code. "JFK" and "LAX" are what people actually
    # say, and they are unambiguous, but they are not substrings of the
    # airport's name -- so a text search would never find them.
    if len(query) == 3 and query.isalpha():
        by_iata = [a for a in airports.values() if a.iata_code.upper() == query.upper()]
        if len(by_iata) == 1:
            return described(by_iata[0])

    # Otherwise search names AND municipalities. The model usually has a
    # city rather than an airport name -- someone asking for "London"
    # means Heathrow, whose name is "London Heathrow Airport" but whose
    # municipality is simply "London". Searching both catches airports
    # named for their city and airports merely serving it.
    needle = _normalise(query)
    index = _search_index()
    matches = [airport for airport, text in index if needle in text]

    # Widen to airports carrying every word somewhere in their
    # searchable text. "New York JFK" is a substring of nothing: the city
    # lives in one field and the code in another, so the phrase can only
    # match once it is broken apart.
    #
    # This widens rather than falls back, because the phrase search can
    # succeed on the wrong things: "Los Angeles airport" is literally
    # contained in "Hilton Los Angeles Airport Helipad" but not in "Los
    # Angeles International Airport", so stopping at the first non-empty
    # result would hand back a hotel helipad. Both sets go to the ranker,
    # which knows a large airport outranks a helipad.
    tokens = needle.split()
    if len(tokens) > 1:
        seen = {airport.icao for airport in matches}
        matches += [
            airport
            for airport, text in index
            if airport.icao not in seen and all(token in text for token in tokens)
        ]

        # Still nothing: keep the airports matching the most tokens.
        # "Sydney Australia" names a real airport, but no field holds the
        # country, so requiring every word finds nothing at all. Best
        # partial match degrades to "Sydney" on its own, and the ranker
        # picks the significant one from there -- an unhelpful answer is
        # still better than a dead end the model has to apologise for.
        if not matches:
            scored = [
                (sum(1 for token in tokens if token in text), airport)
                for airport, text in index
            ]
            best = max((score for score, _ in scored), default=0)
            if best:
                matches = [airport for score, airport in scored if score == best]

    # Last resort: compare with all spacing removed, so "OHare" reaches
    # "O'Hare". Normalising punctuation to a space handles "O Hare" but
    # not the run-together form, and this pass only runs when everything
    # else has failed, where a loose match beats no match.
    if not matches:
        squashed = needle.replace(" ", "")
        matches = [
            airport for airport, text in index if squashed in text.replace(" ", "")
        ]

    runways = _runways()
    matches.sort(
        key=lambda airport: _match_rank(
            airport, needle, runways.get(airport.icao, 0)
        )
    )

    if not matches:
        return {
            "found": False,
            "error": f"No airport matching {query!r}.",
            "hint": "Try an ICAO code (KORD) or a fuller name.",
        }

    # CANDIDATES ARE FOR CHOOSING, NOT FOR USING.
    #
    # This list used to carry eight matches with latitude and longitude
    # on each. Measured in a real session, one `find_airport("Chicago")`
    # was 1,206 characters -- 41% of the entire transcript, and the
    # single largest thing in it. The model reads the list, picks one
    # airport, and every other field sits in the context for the rest of
    # the conversation doing nothing.
    #
    # Coordinates go because nothing downstream takes them from here:
    # `plan_flight` wants ICAO codes. `get_winds_aloft` does want a
    # position, and gets it by looking the chosen code up again -- an
    # exact-code lookup returns the full record and costs about 36
    # tokens, which is far cheaper than carrying eight positions on the
    # chance one is needed.
    #
    # Three rather than eight because a fourth-ranked match is not a
    # serious candidate. The ranker is good enough that if the answer is
    # not in the top three, more rows will not rescue it -- and
    # `match_count` still reports the true total.
    return {
        "found": True,
        "matches": [
            {"icao": airport.icao, "name": airport.name}
            for airport in matches[:MAX_AIRPORT_MATCHES]
        ],
        "match_count": len(matches),
    }


def list_aircraft(category: Optional[str] = None) -> Dict[str, Any]:
    """List available aircraft.

    ONE LINE PER AIRCRAFT, NOT A DICT PER AIRCRAFT. The structured form
    cost 1,853 tokens unfiltered -- 45% of the on-device model's entire
    4,096-token context, spent on a catalogue when the user asked for a
    flight plan. Conversations overflowed and lost plans that had already
    computed correctly.

    Compact strings carry the same information at a tenth the size, and a
    model reads "c172: Cessna 172S Skyhawk, 120 kt, 8000 ft, 2 seats,
    658 nm" as easily as the equivalent JSON object.
    """
    profiles = list(AIRCRAFT.values())
    if category:
        profiles = [p for p in profiles if p.category == category.lower()]
        if not profiles:
            return {
                "error": f"No aircraft in category {category!r}.",
                "valid_categories": sorted({p.category for p in AIRCRAFT.values()}),
            }

    return {
        "count": len(profiles),
        "format": "key: name, cruise speed, cruise altitude, seats, range",
        "aircraft": [
            f"{p.key}: {p.name}, {p.cruise_tas_kt:.0f} kt, "
            f"{p.cruise_altitude_ft/1000:.0f}k ft, {p.typical_occupancy} seats, "
            f"{p.range_nm():.0f} nm"
            for p in profiles
        ],
    }


# Used when the caller names no aircraft. The choice is deliberately a
# small trainer rather than something that flatters every route: an
# unrealistic plan should look unrealistic. What matters is that picking
# it is REPORTED -- see `aircraft_note` below.
DEFAULT_AIRCRAFT = "c172"


def plan_flight(
    origin: str,
    dest: str,
    aircraft: Optional[str] = None,
    use_wind: bool = True,
    avoid_airspace: bool = True,
    altitude_ft: Optional[Any] = None,
    payload_lb: Optional[float] = None,
    save_map: bool = False,
) -> Dict[str, Any]:
    """Plan a route. The primary tool -- everything else supports it.

    Wraps the whole CP1-CP3 pipeline: load data, filter to the region,
    build the mesh, fetch winds, index airspace, run A*.

    `save_map` writes an interactive HTML map alongside the plan. Note
    what the model gets back is a FILE PATH, not an image -- it can tell
    the user where the map is, but cannot see it. That is the right
    division here: rendering is deterministic work, and describing the
    route is what the model is for.
    """
    airports = _airports()

    origin_airport = airports.get(origin.strip().upper())
    if origin_airport is None:
        return {"error": f"Unknown origin ICAO code {origin!r}.",
                "hint": "Use find_airport to look up the code first."}

    dest_airport = airports.get(dest.strip().upper())
    if dest_airport is None:
        return {"error": f"Unknown destination ICAO code {dest!r}.",
                "hint": "Use find_airport to look up the code first."}

    # A missing aircraft is a real gap, not a detail to fill in quietly.
    # Asked to fly KJFK to EGLL with no type named, this defaulted to a
    # Cessna 172 and returned a straight-faced plan: 22h15m and 195
    # gallons, in an aircraft holding 56. The numbers were arithmetically
    # correct and the answer was nonsense. Tracking the substitution lets
    # it be reported rather than assumed.
    aircraft_defaulted = aircraft is None
    aircraft = aircraft or DEFAULT_AIRCRAFT

    try:
        profile = get_aircraft(aircraft)
    except KeyError:
        return {"error": f"Unknown aircraft {aircraft!r}.",
                "hint": "Use list_aircraft to see valid keys."}

    payload = profile.typical_payload_lb if payload_lb is None else payload_lb
    if profile.range_nm(payload) <= 0:
        return {
            "error": (
                f"{profile.name} cannot carry {payload:,.0f} lb and any usable "
                f"fuel. Its useful load is {profile.useful_load_lb:,.0f} lb."
            )
        }

    # `altitude_ft` may arrive as an enum string from the schema or a
    # float from the CLI.
    altitude = _coerce_altitude(altitude_ft, profile.cruise_altitude_ft)
    if altitude is None:
        return {
            "error": f"Could not read {altitude_ft!r} as an altitude in feet.",
            "hint": f"Use one of: {', '.join(ALTITUDE_CHOICES)}.",
        }
    if not profile.can_fly_at(altitude):
        return {
            "error": (
                f"{profile.name} cannot cruise at {altitude:,.0f} ft; its "
                f"service ceiling is {profile.service_ceiling_ft:,.0f} ft."
            )
        }

    try:
        navaids = navaids_near_route(
            _navaids(),
            origin_airport.lat, origin_airport.lon,
            dest_airport.lat, dest_airport.lon,
            margin_nm=100.0,
        )
    except MissingDataError as exc:
        return {"error": str(exc)}

    wind_source = None
    if use_wind:
        from .wind_openmeteo import OpenMeteoWindSource, WindDataError

        try:
            wind_source = OpenMeteoWindSource()
        except WindDataError as exc:
            # Degrade rather than fail: a route without wind is still
            # useful, and the model can tell the user why.
            wind_source = None
            use_wind = False

    airspace_index = None
    if avoid_airspace:
        try:
            airspace_index = AirspaceIndex(
                airspace_near_route(
                    _airspace(),
                    origin_airport.lat, origin_airport.lon,
                    dest_airport.lat, dest_airport.lon,
                ),
                altitude_ft=altitude,
            )
        except AirspaceDataError:
            airspace_index = None
            avoid_airspace = False

    wind_note = None
    try:
        plan = plan_route(
            origin_airport,
            dest_airport,
            navaids,
            aircraft=profile,
            wind_source=wind_source,
            altitude_ft=altitude,
            airspace=airspace_index,
            use_grid=True,
        )
    except NoRouteFound as exc:
        return {"error": str(exc)}
    except Exception as exc:  # noqa: BLE001
        # The wind service can fail mid-plan -- Open-Meteo rate-limits a
        # burst of batched requests with a 429. Losing the whole flight
        # plan over unavailable weather is the wrong trade: replan in
        # still air and say so, rather than returning nothing.
        if wind_source is None:
            raise
        wind_note = f"Winds unavailable ({_short_error(exc)}); planned in still air."
        wind_source = None
        use_wind = False
        plan = plan_route(
            origin_airport,
            dest_airport,
            navaids,
            aircraft=profile,
            altitude_ft=altitude,
            airspace=airspace_index,
            use_grid=True,
        )

    result: Dict[str, Any] = {
        "origin": {"icao": origin_airport.icao, "name": origin_airport.name},
        "destination": {"icao": dest_airport.icao, "name": dest_airport.name},
        "aircraft": profile.name,
        "cruise_altitude_ft": altitude,
        # The compact form first, and always. This is how a route is
        # actually filed and read aloud, and it costs a handful of tokens
        # where the per-leg table costs hundreds.
        "route": " ".join(w.ident for w in plan.waypoints),
        "waypoint_count": len(plan.waypoints),
        "direct_distance_nm": round(plan.direct_distance_nm, 1),
        "route_distance_nm": round(plan.total_distance_nm, 1),
        "wind_applied": use_wind,
    }

    if wind_note:
        result["wind_note"] = wind_note

    # The per-leg table is only included for routes short enough to be
    # worth reading. A transcontinental route has 21 waypoints, and the
    # full table overflowed the on-device model's 4,096-token context --
    # the plan was computed correctly and then lost because the result
    # describing it would not fit. A compact route string plus totals is
    # what the user actually needs; anyone wanting every leg can run the
    # CLI or open the map.
    if len(plan.waypoints) <= MAX_DETAILED_WAYPOINTS:
        result["waypoints"] = _describe_waypoints(plan)
    else:
        result["waypoints_omitted"] = (
            f"{len(plan.waypoints)} waypoints -- per-leg detail omitted to stay "
            "within context. Report the route string and the totals; offer the "
            "map for detail."
        )

    # A SENTENCE, NOT A COUNT. This previously returned two fields --
    # `airspace_avoidance_applied: true` and
    # `restricted_volumes_considered: 95` -- and the model assembled them
    # into "Route includes prohibited and restricted airspace", which is
    # the exact opposite of what happened. The route crossed none of them.
    #
    # The fault was in the field names, not the model. "Considered" reads
    # equally well as "taken into account" and "included in", and beside
    # a count of 95 the second reading is the more natural one. The
    # numbers were all correct and the safety claim came out inverted --
    # which no test caught, because the router was never wrong.
    #
    # So the tool states the conclusion and the model relays it. The same
    # rule as "the model never does computation", extended to
    # interpretation: a result that can be read two ways should not be
    # handed over as raw fields for the model to interpret.
    if not avoid_airspace:
        result["restricted_airspace"] = (
            "NOT CHECKED -- airspace avoidance was disabled for this plan, so "
            "the route may pass through prohibited or restricted areas."
        )
    elif plan.airspace_avoided:
        result["restricted_airspace"] = (
            f"Routed clear of {plan.airspace_avoided} active "
            "prohibited/restricted/warning areas. The route crosses none of "
            "them."
        )
    else:
        result["restricted_airspace"] = (
            "No active prohibited or restricted areas lie near this route."
        )

    if plan.grid_waypoints_used:
        result["oceanic_waypoints"] = plan.grid_waypoints_used
        result["waypoint_note"] = (
            f"{plan.grid_waypoints_used} waypoints are lat/lon oceanic fixes "
            "(named like 56N020W) rather than ground navaids, because navaids "
            "are ground stations and do not cover open water."
        )

    if save_map:
        # Imported lazily: folium is only needed when a map is asked for,
        # so a plain routing conversation never pays the import.
        from .mapping import save_route_map

        try:
            filename = (
                f"maps/{origin_airport.icao}_{dest_airport.icao}_"
                f"{profile.key}.html"
            ).lower()
            result["map_file"] = save_route_map(
                plan,
                filename,
                airspace=airspace_index.volumes if airspace_index else None,
            )
            result["map_note"] = (
                "Interactive HTML map written. Grey dashed line is the direct "
                "course, blue is the planned route. Restricted airspace is "
                "shaded where avoidance was applied. Tell the user the path; "
                "you cannot see the image yourself."
            )
        except Exception as exc:  # noqa: BLE001 - a failed map must not lose the plan
            result["map_error"] = f"Route planned, but the map failed: {exc}"

    # Time, fuel and range are always reported. With wind applied they
    # come from the search itself, since cost IS time there. Without it,
    # they fall back to still-air arithmetic -- distance over true
    # airspeed. A still-air estimate is less accurate but far more useful
    # than no estimate, and the range check in particular must not depend
    # on whether a weather API happened to answer: a Cessna cannot cross
    # the continent in any wind.
    if plan.ete_hours is not None:
        hours_en_route = plan.ete_hours
        ground_speed = plan.average_ground_speed_kt or profile.cruise_tas_kt
    else:
        hours_en_route = plan.total_distance_nm / profile.cruise_tas_kt
        ground_speed = profile.cruise_tas_kt
        result["estimate_basis"] = "still air (no wind data applied)"

    whole_hours = int(hours_en_route)
    minutes = int(round((hours_en_route - whole_hours) * 60))
    result["ete"] = f"{whole_hours}h{minutes:02d}m"
    result["ete_hours"] = round(hours_en_route, 2)
    result["average_ground_speed_kt"] = round(ground_speed)
    result["fuel_required_gal"] = round(
        plan.fuel_required_gal
        if plan.fuel_required_gal is not None
        else profile.fuel_required_gal(hours_en_route),
        1,
    )

    # One line rather than nine fields. The three phases are worth
    # reporting -- they are why the estimate is no longer distance over
    # cruise speed -- but a model with a 4,096-token context does not
    # need nine numbers to say "twenty minutes up, twenty down".
    if plan.phases is not None:
        phases = plan.phases
        result["flight_profile"] = (
            f"climb {phases.climb_time_hours * 60:.0f} min over "
            f"{phases.climb_distance_nm:.0f} nm, cruise "
            f"{phases.cruise_distance_nm:.0f} nm at "
            f"{phases.cruise_altitude_ft:,.0f} ft, descent "
            f"{phases.descent_time_hours * 60:.0f} min over "
            f"{phases.descent_distance_nm:.0f} nm."
        )
        if not phases.reached_planned_altitude:
            result["altitude_note"] = (
                f"Too short to reach the planned cruise level -- the flight "
                f"tops out around {phases.cruise_altitude_ft:,.0f} ft."
            )
        result["ete_note"] = (
            "ETE is airborne time from takeoff to landing. It excludes taxi, "
            "so it reads lower than a published schedule."
        )

    endurance = profile.endurance_hours(payload)
    result["within_aircraft_range"] = hours_en_route <= endurance

    if hours_en_route > endurance:
        # "A fuel stop is required" is the right advice for an aircraft
        # falling short overland. It is the wrong advice for a Cessna 172
        # over the North Atlantic, where there is nowhere to stop.
        #
        # The discriminator is NOT how far short the aircraft falls. A
        # 172 crossing the United States needs four stops and that is a
        # trip people genuinely make. What makes the Atlantic different
        # is that the legs are over open water, so the advice cannot be
        # followed -- and the oceanic waypoints already tell us that.
        stops = math.ceil(hours_en_route / endurance) - 1
        if plan.grid_waypoints_used:
            result["range_warning"] = (
                f"The {profile.name} cannot fly this route. Flight time is "
                f"{whole_hours}h{minutes:02d}m against {endurance:.1f} h of "
                f"endurance, and the route crosses open water where there is "
                "nowhere to refuel. Say plainly that this is the wrong "
                "aircraft for the trip, and suggest planning again with a "
                "type that has the range."
            )
        else:
            result["range_warning"] = (
                f"Flight time of {whole_hours}h{minutes:02d}m exceeds the "
                f"{profile.name}'s {endurance:.1f} h endurance. About "
                f"{stops} fuel stop{'s' if stops != 1 else ''} would be needed "
                "-- tell the user this plainly."
            )

    if aircraft_defaulted:
        result["aircraft_note"] = (
            f"No aircraft was specified, so this was planned for a "
            f"{profile.name}. Tell the user which aircraft was assumed."
        )

    return result


def _describe_waypoints(plan) -> List[Dict[str, Any]]:
    """Waypoints with per-leg distance and heading, as plain dicts."""
    described = []
    for index, waypoint in enumerate(plan.waypoints):
        entry: Dict[str, Any] = {
            "ident": waypoint.ident,
            "name": waypoint.name,
            "latitude": round(waypoint.lat, 4),
            "longitude": round(waypoint.lon, 4),
        }
        if index > 0:
            previous = plan.waypoints[index - 1]
            entry["leg_distance_nm"] = round(
                haversine_nm(previous.lat, previous.lon, waypoint.lat, waypoint.lon), 1
            )
            entry["leg_course_true"] = round(
                initial_bearing_deg(previous.lat, previous.lon, waypoint.lat, waypoint.lon)
            )
        described.append(entry)
    return described


def get_winds_aloft(
    latitude: float,
    longitude: float,
    altitude_ft: Optional[Any] = None,
) -> Dict[str, Any]:
    """Wind at a single point and altitude."""
    from .wind_openmeteo import OpenMeteoWindSource, WindDataError

    altitude = _coerce_altitude(altitude_ft, DEFAULT_WIND_ALTITUDE_FT)
    if altitude is None:
        return {
            "error": f"Could not read {altitude_ft!r} as an altitude in feet.",
            "hint": f"Use one of: {', '.join(ALTITUDE_CHOICES)}.",
        }

    try:
        wind = OpenMeteoWindSource().wind_at(latitude, longitude, altitude)
    except WindDataError as exc:
        return {"error": f"Could not fetch winds: {exc}"}

    direction = round(wind.direction_deg)
    speed = round(wind.speed_kt)

    # THE RESULT NAMES ITS ALTITUDE IN THE SAME SENTENCE AS THE NUMBERS.
    #
    # Asked for the wind at 35,000 ft, this returned the 8,000 ft wind --
    # 12 kt from 239 degrees at +11.9 C -- and the model reported it as
    # "the wind at 35,000 ft". The real answer was 26 kt from 223 at
    # -41 C. The tool had answered the only question the schema let the
    # model ask, and the model attached the user's question to it.
    #
    # `altitude_ft` was already in the result as its own field, and that
    # was not enough: a bare number beside other bare numbers is easy to
    # skip. Binding the altitude into the same string as the wind is what
    # makes the two impossible to separate.
    #
    # The compass point is here for a related reason. The model rendered
    # 239 degrees as "from the northeast", which is backwards -- 239 is
    # southwest. Naming the quadrant removes the invitation to convert.
    summary = (
        f"At {altitude:,.0f} ft: wind from {direction:03d} degrees true "
        f"({_compass_point(direction)}) at {speed} kt"
    )
    if wind.temperature_c is not None:
        summary += f", temperature {wind.temperature_c:.0f}C"
    summary += "."

    return {
        "summary": summary,
        "latitude": latitude,
        "longitude": longitude,
        "altitude_ft": altitude,
        "wind_from_degrees_true": direction,
        "wind_speed_kt": speed,
        "temperature_c": wind.temperature_c,
        "note": (
            "Report the altitude named in `summary`. Wind comes from fixed "
            "pressure levels, so it may differ from the altitude asked "
            "about. Direction is where the wind blows FROM."
        ),
    }


def check_airspace(
    origin: str, dest: str, altitude_ft: Optional[Any] = None
) -> Dict[str, Any]:
    """Which restricted areas lie along a direct course, before routing."""
    altitude = _coerce_altitude(altitude_ft, DEFAULT_WIND_ALTITUDE_FT)
    if altitude is None:
        return {
            "error": f"Could not read {altitude_ft!r} as an altitude in feet.",
            "hint": f"Use one of: {', '.join(ALTITUDE_CHOICES)}.",
        }
    airports = _airports()
    origin_airport = airports.get(origin.strip().upper())
    dest_airport = airports.get(dest.strip().upper())

    if origin_airport is None or dest_airport is None:
        return {"error": "Unknown ICAO code. Use find_airport first."}

    try:
        volumes = _airspace()
    except AirspaceDataError as exc:
        return {"error": str(exc)}

    index = AirspaceIndex(
        airspace_near_route(
            volumes,
            origin_airport.lat, origin_airport.lon,
            dest_airport.lat, dest_airport.lon,
        ),
        altitude_ft=altitude,
    )

    crossings = index.crossings(
        origin_airport.lat, origin_airport.lon,
        dest_airport.lat, dest_airport.lon,
    )

    return {
        "route": f"{origin_airport.icao} to {dest_airport.icao}",
        "altitude_ft": altitude,
        "active_volumes_in_region": len(index),
        "direct_course_crossings": [
            {
                "name": volume.name,
                "type": volume.type_name,
                "floor_ft": volume.lower_ft,
                "ceiling_ft": volume.upper_ft,
                "state": volume.state,
                "blocking": volume.is_blocking,
            }
            for volume in crossings
        ],
        "note": (
            "These are crossings of the DIRECT course. Call plan_flight with "
            "avoid_airspace=true to route around them."
        ),
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TOOLS: List[ToolSpec] = [
    ToolSpec(
        name="find_airport",
        description=(
            "Look up an airport by ICAO code, name, or city. Call this FIRST "
            "whenever the user names an airport in plain language rather than "
            "giving a four-letter ICAO code, so you route from the right "
            "place. A name or city returns the best few matches, best first. "
            "An exact ICAO or IATA code returns that one airport with its "
            "position -- so if you need coordinates for get_winds_aloft, look "
            "up the code you chose."
        ),
        parameters={
            "query": {
                "type": "string",
                "description": "ICAO code, airport name, or city. e.g. 'KORD', 'Heathrow', 'Chicago Executive'.",
                "required": True,
            },
        },
        func=find_airport,
    ),
    ToolSpec(
        name="list_aircraft",
        description=(
            "List the aircraft this system can plan for, with cruise speed, "
            "altitude, seats and range. Call this ONLY when the user asks what "
            "aircraft are available or wants their specifications. You do NOT "
            "need it to plan a flight -- plan_flight's aircraft parameter "
            "already lists every valid key, so calling this first only wastes "
            "a turn. Filter by category when you can, to keep the reply short."
        ),
        parameters={
            "category": {
                "type": "string",
                "description": "Optional filter.",
                "enum": ["ga", "business", "regional", "narrowbody", "widebody"],
                "required": False,
            },
        },
        func=list_aircraft,
    ),
    ToolSpec(
        name="plan_flight",
        description=(
            "Plan a complete flight between two airports: route, waypoints, "
            "estimated time en route, and fuel. This is the main tool -- call "
            "this whenever the user asks to plan, fly, or route between two "
            "places. Both airports must be given as ICAO codes, so use "
            "find_airport first if you only have a name. Winds aloft and "
            "restricted-airspace avoidance are applied internally by default, "
            "so you do not need to call get_winds_aloft or check_airspace "
            "beforehand."
        ),
        parameters={
            "origin": {
                "type": "string",
                "description": "Origin ICAO code, e.g. KPWK.",
                "required": True,
            },
            "dest": {
                "type": "string",
                "description": "Destination ICAO code, e.g. KMSP.",
                "required": True,
            },
            # An enum rather than a free string, for two reasons. It stops
            # a model inventing "cessna172" or "B737-800" and burning a
            # turn on the error. And it is the only way an optional
            # parameter survives on a backend whose schema cannot express
            # optionality -- see backend_apple._is_exposed.
            "aircraft": {
                "type": "string",
                # The enum supplies keys but not names, so a model cannot
                # map "a Cirrus" to sr22 from the key list alone -- it
                # planned a Cessna for exactly that request until these
                # examples were added. Naming the common types here is far
                # cheaper than making the model call list_aircraft first.
                "description": (
                    "Which aircraft to plan for. Common keys: c172 (Cessna 172), "
                    "sr22 (Cirrus SR22), b350 (King Air), cj2 (Citation), "
                    "e175 (Embraer E175), a320n (Airbus A320neo), "
                    "b738 (Boeing 737-800), b789 (Boeing 787-9), "
                    "b77w (Boeing 777-300ER), a388 (Airbus A380). "
                    # This used to read "if the user did not name an
                    # aircraft, use c172", and the model dutifully passed
                    # c172 for a transatlantic crossing. Omitting the
                    # parameter lets the tool record that nobody chose,
                    # and say so in the result.
                    "OMIT this parameter entirely if the user did not name an "
                    "aircraft -- do not guess one."
                ),
                "enum": sorted(AIRCRAFT),
                "required": False,
            },
            "use_wind": {
                "type": "boolean",
                "description": "Apply live winds aloft. Default true. Set false for a still-air estimate.",
                "required": False,
            },
            "avoid_airspace": {
                "type": "boolean",
                "description": "Route around prohibited and restricted airspace. Default true.",
                "required": False,
            },
            "altitude_ft": {
                "type": "string",
                "description": (
                    "Cruise altitude in feet. Omit to use the aircraft's own "
                    "normal cruise altitude, which is usually right."
                ),
                "enum": list(ALTITUDE_CHOICES),
                "required": False,
            },
            "payload_lb": {
                "type": "number",
                "description": "Payload in pounds. Defaults to typical occupancy for the aircraft.",
                "required": False,
            },
            "save_map": {
                "type": "boolean",
                "description": (
                    "Write an interactive HTML map of the route. Set true when "
                    "the user asks to see, view, visualise or map the flight. "
                    "Returns a file path for them to open -- you will not be "
                    "able to see the image yourself. Default false."
                ),
                "required": False,
            },
        },
        func=plan_flight,
    ),
    ToolSpec(
        name="get_winds_aloft",
        description=(
            "Get the forecast wind at one specific point and altitude. Use "
            "this when the user asks about wind conditions somewhere, not "
            "when planning a route -- plan_flight already applies winds "
            "along the whole route by itself."
        ),
        parameters={
            "latitude": {"type": "number", "description": "Latitude in degrees.", "required": True},
            "longitude": {"type": "number", "description": "Longitude in degrees.", "required": True},
            "altitude_ft": {
                "type": "string",
                "description": (
                    "Altitude in feet. ALWAYS set this when the user names an "
                    "altitude or flight level -- wind changes completely with "
                    "height, and the default of 8000 is a low-level answer. "
                    "Pick the closest listed value."
                ),
                "enum": list(ALTITUDE_CHOICES),
                "required": False,
            },
        },
        func=get_winds_aloft,
    ),
    ToolSpec(
        name="check_airspace",
        description=(
            "Report which restricted, prohibited or warning areas lie on the "
            "DIRECT course between two airports. Use this when the user asks "
            "what airspace is in the way, or why a route detours. To actually "
            "route around it, call plan_flight instead."
        ),
        parameters={
            "origin": {"type": "string", "description": "Origin ICAO code.", "required": True},
            "dest": {"type": "string", "description": "Destination ICAO code.", "required": True},
            "altitude_ft": {
                "type": "string",
                "description": (
                    "Cruise altitude in feet. Airspace is altitude-banded, so "
                    "this changes which areas are active. Default 8000."
                ),
                "enum": list(ALTITUDE_CHOICES),
                "required": False,
            },
        },
        func=check_airspace,
    ),
]

TOOLS_BY_NAME: Dict[str, ToolSpec] = {tool.name: tool for tool in TOOLS}


def dispatch(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a tool by name.

    Every failure comes back as a dict rather than an exception. The model
    is mid-conversation: an error it can read is something it can recover
    from, whereas a traceback ends the turn. That includes unexpected
    exceptions from the engine itself -- an agent that reports "that
    lookup failed, let me try another way" is more useful than one that
    crashes the process.
    """
    tool = TOOLS_BY_NAME.get(name)
    if tool is None:
        return {
            "error": f"No tool named {name!r}.",
            "available_tools": sorted(TOOLS_BY_NAME),
        }

    unexpected = set(arguments) - set(tool.parameters)
    if unexpected:
        return {
            "error": f"Unexpected arguments for {name}: {sorted(unexpected)}.",
            "accepted_arguments": sorted(tool.parameters),
        }

    missing = [n for n in tool.required_names() if n not in arguments]
    if missing:
        return {"error": f"{name} requires: {missing}."}

    try:
        return tool.func(**arguments)
    except Exception as exc:  # noqa: BLE001 - deliberately broad; see docstring
        return {"error": f"{name} failed: {_short_error(exc)}"}


# Error text goes straight into the model's context, so its length is a
# cost. An HTTP failure from a batched request carries the full URL --
# 100 comma-separated coordinate pairs, roughly 2,000 characters -- and
# putting that in a tool result burned ~500 tokens of a 4,096-token
# window on a string the model can do nothing with. It overflowed the
# conversation and lost the turn.
MAX_ERROR_CHARS = 180


def _short_error(exc: Exception) -> str:
    """One readable line describing a failure, with any URL stripped."""
    text = str(exc)

    # Cut anything from the first URL onward -- the useful part of an
    # HTTP error is the status code, not the query string.
    for marker in (" for url:", " url:", "https://", "http://"):
        index = text.find(marker)
        if index != -1:
            text = text[:index].rstrip(" :")
            break

    if len(text) > MAX_ERROR_CHARS:
        text = text[:MAX_ERROR_CHARS] + "..."

    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__
