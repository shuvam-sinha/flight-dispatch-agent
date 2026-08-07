"""Aircraft performance profiles.

Cruise performance for a fixed catalogue of aircraft, selectable by key.
There is deliberately no "recommend the best aircraft for this mission"
advisory engine -- choosing between aircraft is the user's job; this
module only says how each one performs.

WHY THIS EXISTS AT ALL
----------------------
CP2 measured routes in nautical miles, which is fine when cost is
distance. CP3 makes cost TIME, and time depends on how fast the aircraft
moves over the ground -- which depends on its airspeed and on the wind.
So the router now needs to know something about the aircraft.

WHY SO MANY PROFILES
--------------------
Wind matters in proportion to how slow you are. A 30 kt headwind costs a
Cessna 172 at 120 kt a quarter of its ground speed; the same wind costs
an A320 at 450 kt under 7%. Running one route across different aircraft
makes that visible, and it is what lets CP4's agent answer a follow-up
like "what if I flew a 787 instead".

The airliners also cruise in the flight levels where winds are far
stronger -- jet stream cores routinely exceed 100 kt. So although they
are less sensitive to a given wind, they meet much bigger ones, and they
are where wind-optimal routing actually pays off in practice.

ACCURACY -- READ THIS BEFORE QUOTING ANY NUMBER
-----------------------------------------------
These are PLANNING APPROXIMATIONS, not certified performance data. They
are typical published cruise values, rounded. Real figures vary
substantially with:

  - engine option (a 777-300ER's burn depends on its GE90 variant)
  - operator configuration (seating, galley, winglets, aux tanks)
  - weight, altitude, temperature and stage length

Confidence is highest for common in-production types (737-800, A320neo,
787-9) and lowest for older or rarer variants (A318, A340-200,
767-200ER). Nothing here substitutes for the aircraft's AFM/FCOM.

None of this affects the ROUTING logic, which only reads cruise TAS and
altitude; the fuel figures feed the estimate printed at the end.

THE MODEL IS DELIBERATELY SIMPLE
--------------------------------
One cruise true airspeed and one fuel burn per aircraft, not altitude-,
temperature- or weight-dependent performance curves. Real dispatch
software interpolates manufacturer tables across all three. That detail
would not change any routing decision here, and it is where the effort
would balloon.

A NOTE ON FUEL UNITS
--------------------
Everything is US gallons per hour, including the jets. Real turbine
operations plan in pounds or kilograms, because fuel density varies with
temperature and weight is what matters for performance. Gallons are used
here for consistency across piston and turbine aircraft; turbine figures
are converted from published pounds-per-hour at roughly 6.7 lb per US
gallon of Jet A.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AircraftProfile:
    """Cruise performance for one aircraft type.

    Attributes:
        key: Short lookup name used by the CLI, e.g. "b738".
        name: Display name, e.g. "Boeing 737-800".
        category: "ga", "business", "regional", "narrowbody" or
            "widebody". Used only for grouping in listings.
        cruise_tas_kt: True airspeed in cruise, knots. TAS is speed
            through the AIR, not over the ground -- the distinction that
            makes wind matter at all. See `wind.ground_speed_kt`.
        cruise_altitude_ft: Default planned cruise altitude, feet MSL.
            Decides which wind level to sample and which airspace applies.
        service_ceiling_ft: Highest altitude the aircraft can maintain.
        fuel_burn_gph: Cruise fuel consumption, US gallons per hour.
        usable_fuel_gal: Total usable fuel on board.
        reserve_minutes: Fuel that must still be aboard on landing. The
            FAA minimum is 30 minutes for day VFR and 45 for IFR; 45 is
            used throughout as the conservative default.
    """

    key: str
    name: str
    category: str
    cruise_tas_kt: float
    cruise_altitude_ft: float
    service_ceiling_ft: float
    fuel_burn_gph: float
    usable_fuel_gal: float
    mtow_lb: float
    empty_weight_lb: float
    seats: int
    typical_occupancy: int
    reserve_minutes: float = 45.0

    @property
    def fuel_density_lb_gal(self) -> float:
        """Jet A is about 6.7 lb per US gallon, avgas about 6.0.

        Both vary with temperature, which is one reason real turbine
        operations plan fuel in weight rather than volume.
        """
        return 6.0 if self.category == "ga" else 6.7

    @property
    def typical_payload_lb(self) -> float:
        """Default payload: `typical_occupancy` people at 220 lb each.

        220 lb is roughly the FAA standard average passenger weight plus a
        checked bag. Belly cargo would add more; this is passenger-only.

        WHY OCCUPANCY IS SEPARATE FROM SEATS
        ------------------------------------
        `seats` is a specification -- how many the aircraft holds.
        `typical_occupancy` is an assumption -- how many usually fly. For
        airliners the two are equal, because airlines fly full and plan
        for it.

        For light aircraft they are not. A Cessna 172 has four seats and a
        useful load of 870 lb; four adults at 220 lb is 880 lb, which
        leaves negative room for fuel. Real 172 flights carry two people,
        occasionally three with partial tanks. Defaulting GA aircraft to a
        full cabin would report zero range for the most common trainer in
        the world, which is arithmetically correct and practically absurd.

        The full-cabin case is still reachable -- pass the payload
        explicitly, and the model will correctly tell you it cannot be
        flown. That answer is worth being able to get.
        """
        return self.typical_occupancy * 220.0

    @property
    def max_payload_lb(self) -> float:
        """Every seat filled. May exceed what the aircraft can lift with
        any usable fuel aboard -- see `typical_payload_lb`."""
        return self.seats * 220.0

    @property
    def useful_load_lb(self) -> float:
        """Everything that can be loaded: fuel plus payload combined."""
        return self.mtow_lb - self.empty_weight_lb

    @property
    def reserve_gal(self) -> float:
        """Fuel that must remain untouched on landing."""
        return (self.reserve_minutes / 60.0) * self.fuel_burn_gph

    def fuel_required_gal(self, hours: float) -> float:
        """Fuel to fly for `hours`, including the reserve."""
        return hours * self.fuel_burn_gph + self.reserve_gal

    def max_fuel_gal(self, payload_lb: Optional[float] = None) -> float:
        """Fuel actually loadable with a given payload aboard.

        THE CONSTRAINT THIS MODELS
        --------------------------
        You usually cannot fill the tanks AND fill the seats. Maximum
        takeoff weight caps the total, so fuel and payload trade against
        each other:

            available for fuel = MTOW - empty weight - payload

        That trade is what a payload-range diagram shows. Below a certain
        payload you are limited by tank capacity; above it, by weight.

        A Cessna 172 makes this vivid: with four adults aboard it cannot
        carry full fuel -- a real and frequently fatal planning trap in
        light aircraft.

        Args:
            payload_lb: Weight carried. Defaults to a full cabin.
        """
        payload = self.typical_payload_lb if payload_lb is None else payload_lb
        weight_available = self.useful_load_lb - payload
        if weight_available <= 0:
            return 0.0  # payload alone exceeds the useful load
        return min(self.usable_fuel_gal, weight_available / self.fuel_density_lb_gal)

    def is_weight_limited(self, payload_lb: Optional[float] = None) -> bool:
        """True when MTOW, not tank size, is what caps the fuel load."""
        return self.max_fuel_gal(payload_lb) < self.usable_fuel_gal

    def endurance_hours(self, payload_lb: Optional[float] = None) -> float:
        """Flying time available before eating into reserves."""
        usable = self.max_fuel_gal(payload_lb) - self.reserve_gal
        return max(0.0, usable / self.fuel_burn_gph)

    def range_nm(self, payload_lb: Optional[float] = None) -> float:
        """Still-air range carrying `payload_lb` (default: a full cabin).

        Still-air means no wind. Wind makes the real figure better or
        worse, which is exactly what CP3's routing accounts for.

        This will not exactly match a manufacturer's published range,
        which also accounts for climb and descent profiles, step climbs,
        alternate and holding fuel, and a burn rate that falls as the
        aircraft gets lighter. Those need a weight-dependent performance
        model -- see the module docstring.
        """
        return self.endurance_hours(payload_lb) * self.cruise_tas_kt

    def ferry_range_nm(self) -> float:
        """Range with full tanks and an empty cabin.

        The most optimistic figure the model can produce. An earlier
        version of this class reported this as plain "range", which was
        misleading -- no revenue flight looks like this.
        """
        return self.range_nm(payload_lb=0.0)

    def can_fly_at(self, altitude_ft: float) -> bool:
        return altitude_ft <= self.service_ceiling_ft


# key, name, category, TAS kt, cruise ft, ceiling ft, burn gph, fuel gal,
# MTOW lb, empty lb, seats, typical occupancy
#
# Held as a compact table rather than fifty constructor calls so the whole
# catalogue can be scanned and compared at a glance.
#
# Burn figures are cruise fuel flow at typical operating weight, converted
# from published kg/hr. Seat counts are typical single-class or common
# two-class layouts and vary widely by operator -- a 737-800 ranges from
# 160 to 189 seats depending on the airline.
_SPECS: List[
    Tuple[str, str, str, float, float, float, float, float, float, float, int, int]
] = [
    # ---- General aviation -------------------------------------------
    ("c172",  "Cessna 172S Skyhawk",     "ga",         120,  8000, 14000,    8.5,    53,    2550,   1680,   4,   2),
    ("sr22",  "Cirrus SR22",             "ga",         180, 10000, 17500,   17.0,    92,    3600,   2300,   4,   3),
    ("b350",  "Beechcraft King Air 350", "business",   310, 27000, 35000,  100.0,   539,   15000,   9955,   9,   6),
    ("cj2",   "Cessna Citation CJ2+",    "business",   413, 41000, 45000,  137.0,   500,   12500,   7500,   7,   4),

    # ---- Regional ----------------------------------------------------
    ("e170",  "Embraer E170",            "regional",   447, 35000, 41000,  380.0,  2400,   82673,  46605,  70,  70),
    ("e175",  "Embraer E175",            "regional",   450, 35000, 41000,  400.0,  3100,   89000,  47836,  76,  76),
    ("e190",  "Embraer E190",            "regional",   450, 36000, 41000,  460.0,  3500,  105359,  62935, 100, 100),
    ("e195",  "Embraer E195",            "regional",   450, 36000, 41000,  490.0,  3500,  107916,  63946, 116, 116),
    ("e175l", "Embraer E175-E2",         "regional",   450, 35000, 41000,  345.0,  3200,   98767,  48500,  80,  80),
    ("e190l", "Embraer E190-E2",         "regional",   450, 36000, 41000,  400.0,  3600,  124340,  66500, 106, 106),
    ("e195l", "Embraer E195-E2",         "regional",   450, 36000, 41000,  430.0,  4000,  133821,  77000, 132, 132),

    # ---- Airbus narrowbody -------------------------------------------
    ("a220a", "Airbus A220-100",         "narrowbody", 447, 35000, 41000,  560.0,  5790,  138000,  77650, 110, 110),
    ("a220c", "Airbus A220-300",         "narrowbody", 450, 35000, 41000,  590.0,  5790,  149000,  81700, 137, 137),
    ("a319",  "Airbus A319",             "narrowbody", 450, 36000, 39800,  750.0,  6400,  166400,  88000, 140, 140),
    ("a320",  "Airbus A320ceo",          "narrowbody", 450, 36000, 39800,  820.0,  6300,  172000,  93900, 165, 165),
    ("a320n", "Airbus A320neo",          "narrowbody", 450, 36000, 39800,  700.0,  7060,  174200,  94800, 180, 180),
    ("a321",  "Airbus A321ceo",          "narrowbody", 450, 36000, 39800,  890.0,  7600,  196200, 105300, 200, 200),
    ("a321n", "Airbus A321neo",          "narrowbody", 450, 36000, 39800,  790.0,  8700,  213000, 110000, 220, 220),
    ("a321x", "Airbus A321XLR",          "narrowbody", 450, 36000, 39800,  810.0, 11400,  222000, 112000, 220, 220),

    # ---- Airbus widebody ---------------------------------------------
    ("a332",  "Airbus A330-200",         "widebody",   475, 37000, 41100, 1810.0, 36744,  533500, 264600, 247, 247),
    ("a333",  "Airbus A330-300",         "widebody",   475, 37000, 41100, 1880.0, 25765,  533500, 286600, 277, 277),
    ("a338",  "Airbus A330-800neo",      "widebody",   475, 37000, 41100, 1550.0, 36744,  553400, 291000, 257, 257),
    ("a339",  "Airbus A330-900neo",      "widebody",   475, 37000, 41100, 1580.0, 36744,  553400, 300000, 287, 287),
    ("a343",  "Airbus A340-300",         "widebody",   475, 35000, 41000, 2140.0, 38295,  610700, 291000, 295, 295),
    ("a346",  "Airbus A340-600",         "widebody",   480, 35000, 41500, 2630.0, 51750,  811300, 384200, 326, 326),
    ("a359",  "Airbus A350-900",         "widebody",   488, 39000, 43100, 1910.0, 36600,  617300, 313000, 315, 315),
    ("a35k",  "Airbus A350-1000",        "widebody",   488, 39000, 43100, 2200.0, 41000,  705000, 342000, 369, 369),
    ("a388",  "Airbus A380-800",         "widebody",   490, 39000, 43000, 3780.0, 84535, 1268000, 610700, 555, 555),

    # ---- Boeing narrowbody -------------------------------------------
    ("b737",  "Boeing 737-700",          "narrowbody", 450, 35000, 41000,  760.0,  6875,  154500,  83000, 143, 143),
    ("b738",  "Boeing 737-800",          "narrowbody", 453, 35000, 41000,  830.0,  6875,  174200,  91300, 189, 189),
    ("b739",  "Boeing 737-900ER",        "narrowbody", 453, 35000, 41000,  860.0,  7837,  187700,  98495, 215, 215),
    ("b37m",  "Boeing 737 MAX 7",        "narrowbody", 450, 35000, 41000,  675.0,  6820,  177000,  99000, 153, 153),
    ("b38m",  "Boeing 737 MAX 8",        "narrowbody", 453, 35000, 41000,  710.0,  6820,  182200,  99360, 189, 189),
    ("b39m",  "Boeing 737 MAX 9",        "narrowbody", 453, 35000, 41000,  740.0,  6820,  194700, 101000, 193, 193),
    ("b3xm",  "Boeing 737 MAX 10",       "narrowbody", 453, 35000, 41000,  775.0,  6820,  197900, 104000, 204, 204),
    ("b752",  "Boeing 757-200",          "narrowbody", 460, 37000, 42000, 1050.0, 11489,  255000, 128730, 200, 200),
    ("b753",  "Boeing 757-300",          "narrowbody", 460, 37000, 42000, 1150.0, 11489,  273000, 141690, 243, 243),

    # ---- Boeing widebody ---------------------------------------------
    ("b763",  "Boeing 767-300ER",        "widebody",   460, 37000, 43100, 1510.0, 24140,  412000, 198440, 261, 261),
    ("b764",  "Boeing 767-400ER",        "widebody",   460, 37000, 43100, 1610.0, 24140,  450000, 229000, 296, 296),
    ("b772",  "Boeing 777-200ER",        "widebody",   490, 37000, 43100, 2240.0, 31000,  656000, 307500, 313, 313),
    ("b77l",  "Boeing 777-200LR",        "widebody",   490, 37000, 43100, 2270.0, 47890,  766000, 320000, 317, 317),
    ("b77w",  "Boeing 777-300ER",        "widebody",   490, 37000, 43100, 2470.0, 47890,  775000, 370000, 396, 396),
    ("b788",  "Boeing 787-8",            "widebody",   488, 39000, 43000, 1780.0, 33340,  502500, 264500, 248, 248),
    ("b789",  "Boeing 787-9",            "widebody",   490, 39000, 43000, 1840.0, 33384,  561500, 284000, 296, 296),
    ("b78x",  "Boeing 787-10",           "widebody",   490, 39000, 43000, 1940.0, 33384,  560000, 298700, 336, 336),
    ("b744",  "Boeing 747-400",          "widebody",   490, 35000, 45100, 3450.0, 57285,  875000, 394100, 416, 416),
    ("b748",  "Boeing 747-8",            "widebody",   495, 35000, 43100, 3290.0, 63034,  987000, 485300, 467, 467),
]

AIRCRAFT: Dict[str, AircraftProfile] = {
    spec[0]: AircraftProfile(
        key=spec[0],
        name=spec[1],
        category=spec[2],
        cruise_tas_kt=float(spec[3]),
        cruise_altitude_ft=float(spec[4]),
        service_ceiling_ft=float(spec[5]),
        fuel_burn_gph=float(spec[6]),
        usable_fuel_gal=float(spec[7]),
        mtow_lb=float(spec[8]),
        empty_weight_lb=float(spec[9]),
        seats=int(spec[10]),
        typical_occupancy=int(spec[11]),
    )
    for spec in _SPECS
}

DEFAULT_AIRCRAFT = AIRCRAFT["c172"]


def get_aircraft(key: str) -> AircraftProfile:
    """Look up a profile by key, with a helpful error listing the options."""
    profile = AIRCRAFT.get(key.strip().lower())
    if profile is None:
        raise KeyError(
            f"Unknown aircraft {key!r}. "
            f"Run with --list-aircraft to see all {len(AIRCRAFT)} options."
        )
    return profile


def aircraft_by_category() -> Dict[str, List[AircraftProfile]]:
    """Profiles grouped by category, preserving catalogue order."""
    grouped: Dict[str, List[AircraftProfile]] = {}
    for profile in AIRCRAFT.values():
        grouped.setdefault(profile.category, []).append(profile)
    return grouped
