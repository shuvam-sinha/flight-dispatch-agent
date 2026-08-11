"""Winds aloft, and the vector maths that turns them into ground speed.

WHY WIND IS THE POINT OF CP3
----------------------------
CP2 measured routes in nautical miles and every result came out at ~100%
of the direct distance, because with distance as the cost the straight
line always wins. That made A* look like an elaborate way to draw a
straight line.

Wind breaks that. An aircraft flies through the AIR, not over the ground.
Its true airspeed (TAS) is fixed by the throttle; the ground speed --
what actually determines how long the flight takes -- is TAS combined
with whatever the air itself is doing. So a longer route through
favourable wind can be genuinely faster than a shorter one into a
headwind, and the cheapest path stops being the straight line.

At 120 kt TAS a 300 nm leg into a 30 kt headwind takes 3h20m. A 340 nm
leg with a 25 kt tailwind takes 2h21m. The longer route wins by an hour.

WHERE THE DATA COMES FROM
-------------------------
Open-Meteo (open-meteo.com): free, no API key, JSON, and global. It
serves numerical weather prediction output -- the same GFS and ECMWF
model runs that national weather services publish -- at any coordinate.

The obvious aviation alternative is NOAA's FD winds-aloft product on
aviationweather.gov, which is the bulletin pilots actually read at
preflight. It was evaluated and rejected: it is a fixed-width legacy text
format covering ~218 US stations, with no data over any ocean and none
outside the United States. This project routes globally, and the whole
point of CP3's routing grid is oceanic and long-haul flights, where FD
has nothing to say.

`WindSource` exists so that judgement can be revisited without touching
the router: an FDWindSource implementing the same interface would drop
straight in.

A NOTE ON WHAT THIS IS NOT
--------------------------
A real dispatcher reads the FD bulletin; a real flight planning SYSTEM
consumes model grids directly, which is what this does. Do not describe
this as "the same product a pilot reads at preflight".
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

# Pressure levels Open-Meteo publishes, paired with their approximate
# altitude in a standard atmosphere. Real geopotential height varies with
# temperature -- a cold air mass sits lower -- but for choosing which
# level to sample, the standard figures are close enough.
#
# The aviation shorthand in the comments is the flight level: FL350 means
# 35,000 ft on a standard altimeter setting.
PRESSURE_LEVELS: Tuple[Tuple[int, float], ...] = (
    (1000, 360.0),     # near surface
    (925, 2500.0),
    (850, 4800.0),
    (700, 9900.0),     # ~FL100, GA cruise
    (600, 14100.0),
    (500, 18300.0),    # ~FL180
    (400, 23600.0),
    (300, 30100.0),    # ~FL300, jet cruise, jet stream core
    (250, 34000.0),    # ~FL340
    (200, 38700.0),    # ~FL390
    (150, 44600.0),
    (100, 52700.0),
)

KMH_PER_KNOT = 1.852


@dataclass(frozen=True)
class Wind:
    """A wind vector at one point.

    Attributes:
        direction_deg: The direction the wind is blowing FROM, in degrees
            true. This is the meteorological convention and it trips
            everyone up at least once: a "270 wind" blows from the west,
            towards the east. Aviation uses the same convention.
        speed_kt: Wind speed in knots.
        altitude_ft: Altitude this sample applies to.
        temperature_c: Air temperature, if known. Unused by the routing
            maths; carried because it feeds a future TAS correction.
    """

    direction_deg: float
    speed_kt: float
    altitude_ft: float
    temperature_c: Optional[float] = None

    @property
    def blowing_towards_deg(self) -> float:
        """The direction the wind is blowing TOWARDS.

        The reciprocal of `direction_deg`. Useful when reasoning about the
        vector rather than the report.
        """
        return (self.direction_deg + 180.0) % 360.0

    def components(self, course_deg: float) -> Tuple[float, float]:
        """Split the wind into headwind and crosswind for a given course.

        Returns (headwind_kt, crosswind_kt), where headwind is positive
        when it opposes you and negative when it helps (a tailwind), and
        crosswind is positive from the right.

        The angle between the course and the wind's origin decides the
        split: wind straight down the nose is all headwind, wind straight
        off the wingtip is all crosswind.
        """
        angle = math.radians(self.direction_deg - course_deg)
        return self.speed_kt * math.cos(angle), self.speed_kt * math.sin(angle)


CALM = Wind(direction_deg=0.0, speed_kt=0.0, altitude_ft=0.0)


def ground_speed_kt(
    true_airspeed_kt: float, course_deg: float, wind: Wind
) -> float:
    """Ground speed achieved flying `course_deg` at `true_airspeed_kt`.

    THE WIND TRIANGLE
    -----------------
    Three vectors: where the aircraft points and how fast it moves through
    the air (heading + TAS), what the air is doing (the wind), and the
    resulting motion over the ground (track + ground speed). The first two
    add to give the third.

    The subtlety is that we want to fly a specific COURSE over the ground,
    not a specific heading. A crosswind pushes the aircraft sideways, so
    to track straight the pilot must angle into the wind -- the wind
    correction angle -- and that angling "wastes" some airspeed holding
    position against the drift. The remaining airspeed, plus whatever
    tailwind or headwind component exists, is the ground speed:

        drift correction:  sin(wca) = crosswind / TAS
        ground speed    :  TAS * cos(wca) - headwind

    Note a pure crosswind still SLOWS you down, because part of the
    airspeed is spent countering drift rather than making progress. That
    surprises people who expect a 90-degree wind to be free.

    Returns:
        Ground speed in knots, floored at a small positive value. A wind
        stronger than the aircraft's airspeed would mathematically give a
        negative or impossible result -- physically the aircraft cannot
        hold that course at all. Rather than propagate a NaN into the
        router, the leg is made prohibitively slow, so A* routes around it.
    """
    if wind.speed_kt == 0:
        return true_airspeed_kt

    headwind_kt, crosswind_kt = wind.components(course_deg)

    # The aircraft cannot hold this course if the crosswind exceeds its
    # airspeed -- asin would be undefined. Treat it as effectively
    # unflyable rather than crashing.
    if abs(crosswind_kt) >= true_airspeed_kt:
        return 0.1

    wind_correction_angle = math.asin(crosswind_kt / true_airspeed_kt)
    speed = true_airspeed_kt * math.cos(wind_correction_angle) - headwind_kt

    # Headwind at or above airspeed: no forward progress over the ground.
    return max(speed, 0.1)


def nearest_pressure_level(altitude_ft: float) -> int:
    """The Open-Meteo pressure level closest to a given altitude.

    Sampling the nearest published level rather than interpolating between
    two is a deliberate simplification. The levels are close enough
    together through the cruise band that the difference does not change
    routing decisions, and it halves the number of values requested.
    """
    return min(PRESSURE_LEVELS, key=lambda pair: abs(pair[1] - altitude_ft))[0]


def altitude_for_level(level_hpa: int) -> float:
    """Approximate standard-atmosphere altitude of a pressure level."""
    return dict(PRESSURE_LEVELS)[level_hpa]


class WindSource(Protocol):
    """Anything that can report wind at a point.

    Deliberately narrow. The router only ever needs "what is the wind
    here, at this altitude", so any backend satisfying that -- a live API,
    a cached file, a fixed value in a test -- can be substituted without
    the routing code knowing or caring.

    This is the same swappable-backend pattern CP4 will use for the LLM.
    """

    def wind_at(self, lat: float, lon: float, altitude_ft: float) -> Wind:
        """Wind at a single point."""
        ...

    def wind_at_many(
        self, points: Sequence[Tuple[float, float]], altitude_ft: float
    ) -> List[Wind]:
        """Wind at several points. Implementations should batch."""
        ...


class ConstantWindSource:
    """A uniform wind everywhere. For tests and for `--wind` on the CLI.

    Useful beyond testing: a constant wind makes the routing effect easy
    to reason about, because any bend in the route must come from the
    geometry rather than from a complicated wind field.
    """

    def __init__(self, direction_deg: float = 0.0, speed_kt: float = 0.0):
        self.direction_deg = direction_deg
        self.speed_kt = speed_kt

    def wind_at(self, lat: float, lon: float, altitude_ft: float) -> Wind:
        return Wind(
            direction_deg=self.direction_deg,
            speed_kt=self.speed_kt,
            altitude_ft=altitude_ft,
        )

    def wind_at_many(
        self, points: Sequence[Tuple[float, float]], altitude_ft: float
    ) -> List[Wind]:
        return [self.wind_at(lat, lon, altitude_ft) for lat, lon in points]


class NoWind(ConstantWindSource):
    """Calm air. Makes CP3 reproduce CP2's distance-optimal routes exactly,
    which is a useful regression check that nothing else changed."""

    def __init__(self):
        super().__init__(direction_deg=0.0, speed_kt=0.0)
