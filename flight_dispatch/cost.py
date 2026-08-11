"""Edge cost functions for A*.

WHAT CHANGES IN CP3, AND WHAT DOES NOT
--------------------------------------
Nothing in `search.py` or `graph.py` changes. A* already accepts an
optional `cost_function` and minimises whatever number it returns. CP2
returned distance; CP3 returns TIME. That single substitution is what
makes routes bend.

    CP2:  cost = distance_nm
    CP3:  cost = distance_nm / ground_speed_kt   (hours)

Ground speed is not airspeed. An aircraft flies through the air at a
fixed true airspeed, but the air itself is moving, so progress over the
ground depends on the wind along that specific leg. A leg flown into a
headwind takes longer than an identical leg flown with a tailwind, and A*
will now prefer the second even if it is further.

WHY THE HEURISTIC HAS TO CHANGE TOO
-----------------------------------
A* is only guaranteed to find the cheapest path if its heuristic never
OVERESTIMATES the remaining cost. In CP2 the heuristic was great-circle
distance, and cost was distance, so the units matched and the bound held
trivially.

Now cost is time. Great-circle distance in nautical miles is not a lower
bound on hours -- it is not even the same unit. The correct admissible
heuristic is:

    remaining time >= straight-line distance / best possible ground speed

and the best possible ground speed is TAS plus the strongest tailwind
that could exist anywhere en route. Using the maximum wind speed found in
the region guarantees the estimate is never optimistic in the wrong
direction. See `time_heuristic`.

Getting this wrong is a subtle and common bug: the search still returns a
route, it just is not necessarily the best one.
"""

import math
from typing import Callable, Optional, Sequence, Tuple

from .aircraft import AircraftProfile
from .geo import great_circle_point, haversine_nm, initial_bearing_deg
from .graph import WaypointGraph
from .wind import Wind, WindSource, ground_speed_kt

# How many points along an edge to sample the wind at. Edges can be 150 nm
# long, over which wind can shift appreciably, so a single midpoint sample
# would misprice them. Three points -- start, middle, end -- captures the
# variation cheaply, and every sample is a cache hit after prefetching.
EDGE_WIND_SAMPLES = 3


def sample_points_along_edge(
    lat1: float, lon1: float, lat2: float, lon2: float, samples: int = EDGE_WIND_SAMPLES
) -> list:
    """Evenly spaced points along the great circle between two waypoints."""
    if samples <= 1:
        return [great_circle_point(0.5, lat1, lon1, lat2, lon2)]
    return [
        great_circle_point(i / (samples - 1), lat1, lon1, lat2, lon2)
        for i in range(samples)
    ]


def leg_time_hours(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    aircraft: AircraftProfile,
    wind_source: WindSource,
    altitude_ft: Optional[float] = None,
) -> float:
    """Time to fly one leg, accounting for wind.

    The leg is divided into equal-distance segments, each flown at the
    ground speed implied by the wind sampled at its midpoint. Summing the
    segment times is a better approximation than applying one wind to the
    whole leg, because a 150 nm edge can start in a headwind and end in a
    crosswind.

    Course is recomputed per segment. On a great circle the heading
    changes continuously, and on a long leg at high latitude that shift is
    large enough to change which component of the wind is a headwind.

    Returns:
        Hours. Effectively infinite if the wind makes the leg unflyable.
    """
    altitude = aircraft.cruise_altitude_ft if altitude_ft is None else altitude_ft
    total_nm = haversine_nm(lat1, lon1, lat2, lon2)
    if total_nm == 0:
        return 0.0

    points = sample_points_along_edge(lat1, lon1, lat2, lon2)
    winds = wind_source.wind_at_many(points, altitude)

    segment_nm = total_nm / (len(points) - 1)
    hours = 0.0

    for index in range(len(points) - 1):
        start, end = points[index], points[index + 1]
        course = initial_bearing_deg(start[0], start[1], end[0], end[1])

        # Average the winds at the two ends of the segment. Averaging the
        # vectors would be more correct than averaging the resulting
        # speeds, but at segment scale the difference is negligible.
        speed = 0.5 * (
            ground_speed_kt(aircraft.cruise_tas_kt, course, winds[index])
            + ground_speed_kt(aircraft.cruise_tas_kt, course, winds[index + 1])
        )
        hours += segment_nm / speed

    return hours


def make_wind_cost(
    aircraft: AircraftProfile,
    wind_source: WindSource,
    altitude_ft: Optional[float] = None,
) -> Callable[[WaypointGraph, int, int, float], float]:
    """Build a cost function returning flight time in hours.

    The returned callable matches the signature A* expects:
    (graph, from_index, to_index, base_distance_nm) -> cost.

    Args:
        aircraft: Supplies cruise TAS and default altitude.
        wind_source: Where wind comes from. Prefetch it first.
        altitude_ft: Override the aircraft's default cruise altitude.
    """
    altitude = aircraft.cruise_altitude_ft if altitude_ft is None else altitude_ft

    def cost(
        graph: WaypointGraph, from_index: int, to_index: int, base_nm: float
    ) -> float:
        a, b = graph.nodes[from_index], graph.nodes[to_index]
        return leg_time_hours(
            a.lat, a.lon, b.lat, b.lon, aircraft, wind_source, altitude
        )

    return cost


def max_wind_speed_kt(
    wind_source: WindSource,
    points: Sequence[Tuple[float, float]],
    altitude_ft: float,
) -> float:
    """Strongest wind anywhere in a set of points.

    Used to bound the heuristic. Sampling the graph's own nodes is the
    right set: A* can only fly between those, so no leg can encounter a
    wind stronger than the strongest one found among them.
    """
    winds = wind_source.wind_at_many(points, altitude_ft)
    return max((wind.speed_kt for wind in winds), default=0.0)


def make_time_heuristic(
    aircraft: AircraftProfile, max_tailwind_kt: float
) -> Callable[[float], float]:
    """Build an admissible heuristic for time-based costs.

    Returns a function mapping remaining distance (nm) to a lower bound on
    remaining time (hours):

        hours >= distance / (TAS + strongest possible tailwind)

    Dividing by the BEST case ground speed is what makes this a lower
    bound. Any real leg will be flown at some ground speed no greater than
    that, so it can only take longer than this estimate -- never less.
    Which is exactly the admissibility condition A* needs.

    The tighter this bound, the fewer nodes A* expands. With no wind it is
    exact. In a 100 kt jet stream it is loose, and the search does more
    work -- correctly, because the wind genuinely opens up more routes
    worth considering.
    """
    best_ground_speed = aircraft.cruise_tas_kt + max(0.0, max_tailwind_kt)

    def heuristic(distance_nm: float) -> float:
        return distance_nm / best_ground_speed

    return heuristic


def distance_cost(
    graph: WaypointGraph, from_index: int, to_index: int, base_nm: float
) -> float:
    """CP2's cost function: plain great-circle distance.

    Kept so the two regimes can be compared directly, which is the
    clearest way to show what wind routing actually buys.
    """
    return base_nm
