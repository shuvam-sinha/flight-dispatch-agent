"""The core data types this project passes around.

Everything else in the package either produces these (data_loader) or
consumes them (route). Start here when reading the codebase -- once you
know what an Airport and a Navaid are, the rest follows.

WHY DATACLASSES INSTEAD OF DICTS
--------------------------------
The CSV loaders could just hand back dicts: {"icao": "KORD", "lat": ...}.
Dataclasses are better here for three reasons:

  1. Typos fail loudly. `airport.latitude` is an AttributeError;
     `airport["latitude"]` on a dict full of "lat" keys is a KeyError at
     best and silently wrong at worst.
  2. Editors autocomplete the fields and type-checkers can verify them.
  3. The class definition IS the documentation of what a row contains.

WHY frozen=True
---------------
It makes instances immutable -- `airport.lat = 5` raises an error. Once
loaded, reference data should never be modified; if some routing code
mutated a shared Airport, every later route would silently inherit the
change. Freezing makes that class of bug impossible rather than merely
unlikely.

It also makes instances hashable, so they can go in sets and be used as
dict keys. CP2 needs exactly that: A* keeps sets of visited nodes and
dicts mapping each node to its cost so far.
"""

from dataclasses import dataclass
from typing import Optional, Union


@dataclass(frozen=True)
class Airport:
    """An airport, parsed from a row of OurAirports `airports.csv`.

    Fields:
        icao: The four-letter ICAO identifier pilots use -- KORD, KPWK,
            EGLL. The lookup key for the whole program.
        name: Human-readable name, for display only.
        lat: Latitude in decimal degrees. Positive north, negative south.
        lon: Longitude in decimal degrees. Positive east, negative west --
            so everywhere in the continental US is negative.
        elevation_ft: Field elevation in feet above sea level. Optional
            because the dataset does not always have it. Unused in CP1;
            CP3 needs it for climb/descent and fuel calculations.

    Note `elevation_ft` has a default, so it must be declared last --
    Python does not allow a field without a default to follow one with it.
    """

    icao: str
    name: str
    lat: float
    lon: float
    elevation_ft: Optional[float] = None

    # Significance signals, used to rank name searches. OurAirports
    # carries no size or traffic figures, but these four together
    # separate a major airport from an airstrip that happens to share
    # its name -- see tools._match_rank.
    airport_type: str = ""        # large_airport / medium_airport / ...
    scheduled_service: bool = False  # has commercial airline service
    iata_code: str = ""           # SFO, LAX -- only commercial fields have one
    municipality: str = ""        # the city, which is what users usually type

    @property
    def is_major(self) -> bool:
        """A rough "would a passenger have heard of it" test."""
        return self.airport_type == "large_airport" or self.scheduled_service

    @property
    def ident(self) -> str:
        """Uniform identifier across waypoint types.

        Airport calls its identifier `icao`, Navaid calls its own `ident`.
        Routing code should not have to care which kind of waypoint it is
        holding, so both types expose `.ident`. This property is the
        adapter that makes an Airport satisfy that shared shape.

        Accessed as `airport.ident`, no parentheses -- that is what
        @property does: it makes a method look like a plain attribute.
        """
        return self.icao


@dataclass(frozen=True)
class Navaid:
    """A ground-based navigation aid, from OurAirports `navaids.csv`.

    A navaid is a radio transmitter on the ground that aircraft tune to
    for position information. They are the fixed, named points that
    routes are traditionally built from -- which is why this project
    routes over them rather than over arbitrary lat/lon coordinates.

    Fields:
        ident: Short identifier, typically 2-3 letters -- OBK, DPA, MSN.
            NOT globally unique: several countries have their own "OB",
            which is why navaids are held in a list rather than a dict
            keyed by ident.
        name: Human-readable name, e.g. "Northbrook".
        navaid_type: VOR, VORTAC, NDB and so on. See
            ROUTABLE_NAVAID_TYPES in data_loader.py for which types are
            usable as waypoints and why.
        lat: Latitude in decimal degrees.
        lon: Longitude in decimal degrees.
    """

    ident: str
    name: str
    navaid_type: str
    lat: float
    lon: float


# Anything that can appear as a point on a route.
#
# `Union[Airport, Navaid]` tells a type-checker "either of these two". It
# is a type-level annotation only -- it creates no class and changes no
# runtime behaviour. Its purpose is to let `RoutePlan.waypoints` be
# declared as List[Waypoint], documenting that a route mixes both kinds:
# airports at the ends, navaids in the middle.
#
# The code that walks a route only ever touches .ident, .name, .lat and
# .lon -- the four attributes both types share -- so it never needs to
# check which kind it is holding.
Waypoint = Union[Airport, Navaid]
