"""CP1 route construction.

WHAT THIS IS, AND WHAT IT DELIBERATELY IS NOT
----------------------------------------------
This is NOT a path search. There is no graph, no A*, no notion of "best"
route. It draws a straight line from origin to destination and collects
real navaids that happen to lie near that line.

That is intentional for CP1, whose only job is to prove the data layer
works end to end: real CSVs load, real coordinates parse, real navaid
identifiers come out in a sensible order.

The visible consequence is that on SHORT legs this produces zigzags. If
the corridor is 15 nm wide and the flight is only 8 nm long, navaids
scattered around the destination all qualify, and the route wanders
between them. CP2 fixes this properly by building a waypoint mesh graph
and running A* over it, where a zigzag simply costs more and loses.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

from .aircraft import AircraftProfile
from .airspace import AirspaceIndex, make_airspace_cost
from .cost import leg_time_hours, make_time_heuristic, make_wind_cost, max_wind_speed_kt
from .geo import cross_and_along_track_nm, haversine_nm
from .graph import DEFAULT_MIN_NEIGHBORS, DEFAULT_RADIUS_NM, WaypointGraph, build_mesh
from .grid import count_grid_points, waypoints_for_route
from .models import Airport, Navaid, Waypoint
from .phases import FlightPhases, flight_phases
from .search import a_star
from .wind import WindSource


@dataclass(frozen=True)
class RoutePlan:
    """An ordered set of waypoints from origin to destination.

    `frozen=True` makes instances immutable: once a plan is built, nothing
    downstream can quietly mutate it. That matters more later, when the
    CP4 agent is passing plans around between tool calls.

    The two distances are computed on demand (@property) rather than
    stored, so they can never drift out of sync with `waypoints`.
    """

    origin: Airport
    dest: Airport
    waypoints: List[Waypoint]  # includes origin at [0] and dest at [-1]

    # Search diagnostics, populated by plan_route and left as None by the
    # CP1 naive_route (which does no searching). Kept on the plan so the
    # CLI can report them without re-running anything.
    graph_nodes: Optional[int] = None
    graph_edges: Optional[int] = None
    nodes_expanded: Optional[int] = None

    # CP3. Present only when the route was planned against a wind source;
    # a distance-only route has no meaningful time or fuel figure.
    aircraft: Optional[AircraftProfile] = None
    ete_hours: Optional[float] = None

    # Volumes that were active at cruise altitude near this route and had
    # to be routed around. Empty when airspace was checked and none
    # applied; None when airspace was not checked at all.
    airspace_avoided: Optional[int] = None

    # How many waypoints on this route are generated lat/lon fixes rather
    # than charted navaids. None when the grid was not used.
    grid_waypoints_used: Optional[int] = None

    # The climb/cruise/descent split. Present whenever an aircraft was
    # given; `ete_hours` is this profile's total when it is.
    phases: Optional[FlightPhases] = None

    @property
    def fuel_required_gal(self) -> Optional[float]:
        """Fuel for the planned flight, including reserve.

        Per phase when the profile is available: a jet burns about 1.6x
        cruise flow climbing and a third of it descending, so billing the
        whole flight at cruise flow overstates short sectors and
        understates the climb.
        """
        if self.aircraft is None or self.ete_hours is None:
            return None
        if self.phases is not None:
            return self.phases.total_fuel_gal + self.aircraft.reserve_gal
        return self.aircraft.fuel_required_gal(self.ete_hours)

    @property
    def average_ground_speed_kt(self) -> Optional[float]:
        """Distance flown divided by time taken.

        Comparing this against the aircraft's cruise TAS shows at a glance
        whether the flight was helped or hindered overall: above TAS means
        a net tailwind, below means a net headwind.
        """
        if self.ete_hours is None or self.ete_hours == 0:
            return None
        return self.total_distance_nm / self.ete_hours

    def is_within_range(self, payload_lb: Optional[float] = None) -> Optional[bool]:
        """Whether the aircraft can actually make this flight.

        A route can be geometrically fine and still unflyable -- that is
        what an alternate or a fuel stop is for.
        """
        if self.aircraft is None or self.ete_hours is None:
            return None
        return self.ete_hours <= self.aircraft.endurance_hours(payload_lb)

    @property
    def direct_distance_nm(self) -> float:
        """Straight-line distance origin to destination, ignoring waypoints.

        This is the theoretical best case -- the distance if you could fly
        directly with no detours at all.
        """
        return haversine_nm(
            self.origin.lat, self.origin.lon, self.dest.lat, self.dest.lon
        )

    @property
    def total_distance_nm(self) -> float:
        """Distance actually flown, summed leg by leg.

        Always >= direct_distance_nm; the gap is how much detour the
        waypoints cost you. Comparing the two is the quickest sanity check
        that a route is reasonable.

        `zip(waypoints, waypoints[1:])` is the standard Python idiom for
        walking consecutive pairs: given [A, B, C] it yields (A,B), (B,C).
        """
        return sum(
            haversine_nm(a.lat, a.lon, b.lat, b.lon)
            for a, b in zip(self.waypoints, self.waypoints[1:])
        )


def naive_route(
    origin: Airport,
    dest: Airport,
    navaids: Sequence[Navaid],
    corridor_width_nm: float = 15.0,
    max_waypoints: int = 5,
) -> RoutePlan:
    """Build a route by sampling navaids inside a corridor around the
    origin->destination great circle.

    Picture a rectangle laid over the direct course line: as long as the
    flight, and `corridor_width_nm` wide on EACH side. Any navaid inside
    that rectangle is a candidate.

        origin +--------------------------------+ dest
               |        <- corridor ->          |   each side is
               +--------------------------------+   corridor_width_nm

    Args:
        corridor_width_nm: HALF-width. A navaid qualifies when its
            perpendicular distance from the course is within this many nm
            on either side. Bigger = more candidates but sloppier routes.
        max_waypoints: Cap on INTERMEDIATE waypoints (origin and dest are
            always included on top of this).

    Returns:
        A RoutePlan whose `waypoints` runs origin -> picks -> dest.
    """
    total_nm = haversine_nm(origin.lat, origin.lon, dest.lat, dest.lon)

    # Degenerate case: origin and destination are the same point. There is
    # no course line to measure against, and the along-track math below
    # would divide by a zero-length course. Bail out early.
    if total_nm == 0:
        return RoutePlan(origin=origin, dest=dest, waypoints=[origin, dest])

    # --- Step 1: find every navaid inside the corridor -------------------
    candidates = []  # list of (along_track_nm, navaid), unsorted for now
    for navaid in navaids:
        cross_track, along_track = cross_and_along_track_nm(
            navaid.lat, navaid.lon, origin.lat, origin.lon, dest.lat, dest.lon
        )

        # Two independent tests, both of which must pass:
        #   abs(cross_track) <= width  -> close enough to the course line
        #   0 < along_track < total_nm -> actually BETWEEN the endpoints
        #
        # That second test is why along-track had to be signed. A negative
        # value means the navaid is behind the departure airport; a value
        # above total_nm means it is past the destination. Both are on the
        # course line, both are wrong to fly to.
        if abs(cross_track) <= corridor_width_nm and 0 < along_track < total_nm:
            candidates.append((along_track, navaid))

    # --- Step 2: put them in the order you'd actually fly them -----------
    # Sorting by along-track distance = sorting by progress toward the
    # destination, which is exactly flight order.
    candidates.sort(key=lambda item: item[0])

    # --- Step 3: thin them down to at most max_waypoints -----------------
    selected = _spread_evenly(candidates, total_nm, max_waypoints)

    return RoutePlan(origin=origin, dest=dest, waypoints=[origin, *selected, dest])


class NoRouteFound(Exception):
    """Raised when no path exists through the mesh between two airports."""


def plan_route(
    origin: Airport,
    dest: Airport,
    navaids: Sequence[Navaid],
    radius_nm: float = DEFAULT_RADIUS_NM,
    min_neighbors: int = DEFAULT_MIN_NEIGHBORS,
    aircraft: Optional[AircraftProfile] = None,
    wind_source: Optional[WindSource] = None,
    altitude_ft: Optional[float] = None,
    airspace: Optional["AirspaceIndex"] = None,
    use_grid: bool = False,
) -> RoutePlan:
    """CP2 routing: shortest path over a waypoint mesh graph.

    This is what `naive_route` should have been. Instead of sampling
    points near the direct course and connecting them in order, it builds
    a graph of every plausible leg in the region and asks A* for the
    cheapest way through it. A detour now has to justify itself against
    every alternative, so the zigzags CP1 produced on short legs simply
    lose the search.

    Cost is plain great-circle distance at this checkpoint. CP3 swaps in
    time-given-wind and blocks edges crossing restricted airspace by
    passing a cost_function to `a_star` -- neither the graph nor the
    search needs to change for that.

    Args:
        navaids: Candidate waypoints, pre-filtered to the region. Use
            `data_loader.navaids_near_route` to build this.
        radius_nm: Edge connection radius for the mesh.
        min_neighbors: Nearest-neighbour floor, so sparse regions cannot
            fragment the graph.

    Raises:
        NoRouteFound: if the mesh has no path between origin and dest.
    """
    # Top up with virtual waypoints where navaids do not reach. Over land
    # this adds almost nothing -- a real navaid is a better waypoint than
    # a generated one, so the grid defers wherever coverage exists. Over
    # water it supplies the only waypoints there are. See grid.py.
    waypoints = list(navaids)
    if use_grid:
        waypoints = waypoints_for_route(
            origin.lat, origin.lon, dest.lat, dest.lon, navaids
        )

    # Origin and destination join the mesh as nodes so the path has
    # endpoints; no other airports participate, since real routes are
    # defined navaid-to-navaid.
    graph = build_mesh(
        [origin, *waypoints, dest],
        radius_nm=radius_nm,
        min_neighbors=min_neighbors,
    )

    cost_function = None
    distance_to_cost = None
    total_hours: Optional[float] = None

    if wind_source is not None:
        if aircraft is None:
            raise ValueError("wind_source requires an aircraft profile")

        altitude = aircraft.cruise_altitude_ft if altitude_ft is None else altitude_ft

        # Warm the wind cache for every node before A* starts, so the cost
        # function below never issues a network request from inside the
        # search loop. See wind_openmeteo for how this collapses tens of
        # thousands of edge lookups into a handful of HTTP calls.
        node_points = [(node.lat, node.lon) for node in graph.nodes]
        if hasattr(wind_source, "prefetch"):
            wind_source.prefetch(node_points, altitude)

        cost_function = make_wind_cost(aircraft, wind_source, altitude)

        # Cost is now hours, not miles, so the heuristic must convert. The
        # bound uses the strongest wind anywhere in the graph as a
        # best-case tailwind, which keeps it admissible.
        strongest_kt = max_wind_speed_kt(wind_source, node_points, altitude)
        distance_to_cost = make_time_heuristic(aircraft, strongest_kt)

    airspace_avoided = None if airspace is None else len(airspace)

    if airspace is not None:
        # Layered on top of the wind cost rather than replacing it, so the
        # result is the fastest route that is also legal. An edge crossing
        # prohibited or restricted airspace costs infinity, and A* routes
        # around it for the same reason it avoids anything expensive.
        cost_function = make_airspace_cost(airspace, cost_function)

    # Origin is at index 0 and destination last, by construction above.
    result = a_star(
        graph,
        0,
        graph.node_count - 1,
        cost_function=cost_function,
        distance_to_cost=distance_to_cost,
    )

    if not result.found:
        raise NoRouteFound(
            f"No path from {origin.icao} to {dest.icao} through "
            f"{graph.node_count} waypoints. Try a larger --radius-nm."
        )

    if wind_source is not None and airspace is None:
        # With a time-based cost, the search's own total IS the flight
        # time -- no need to recompute it leg by leg.
        total_hours = result.cost
    elif wind_source is not None:
        # The airspace wrapper may have inflated costs, so recompute the
        # time honestly from the chosen waypoints.
        total_hours = sum(
            leg_time_hours(a.lat, a.lon, b.lat, b.lon, aircraft, wind_source,
                           aircraft.cruise_altitude_ft if altitude_ft is None else altitude_ft)
            for a, b in zip(
                [graph.nodes[i] for i in result.path],
                [graph.nodes[i] for i in result.path][1:],
            )
        )

    path_nodes = [graph.nodes[i] for i in result.path]
    route_nm = sum(
        haversine_nm(a.lat, a.lon, b.lat, b.lon)
        for a, b in zip(path_nodes, path_nodes[1:])
    )

    # Split the flight into climb, cruise and descent. Everything above
    # produced CRUISE time: A* costs an edge at cruise speed, because
    # that is the speed the route is flown at between top of climb and
    # top of descent. The ends are slower, and until now they were billed
    # at cruise speed too.
    #
    # This is applied after the search rather than inside the cost
    # function on purpose. Climb and descent depend on the route's total
    # length and on the two field elevations -- not on which waypoints
    # are chosen -- so they are the same for every candidate path and
    # cannot change which route A* prefers. Folding them into the edge
    # cost would slow the search down to compute a constant.
    phases = None
    if aircraft is not None:
        cruise_ground_speed = (
            route_nm / total_hours
            if total_hours
            else aircraft.cruise_tas_kt
        )
        phases = flight_phases(
            aircraft,
            route_distance_nm=route_nm,
            cruise_ground_speed_kt=cruise_ground_speed,
            origin_elevation_ft=origin.elevation_ft or 0.0,
            dest_elevation_ft=dest.elevation_ft or 0.0,
            cruise_altitude_ft=altitude_ft,
        )
        total_hours = phases.total_time_hours

    return RoutePlan(
        origin=origin,
        dest=dest,
        waypoints=path_nodes,
        graph_nodes=graph.node_count,
        graph_edges=graph.edge_count,
        nodes_expanded=result.nodes_expanded,
        aircraft=aircraft,
        ete_hours=total_hours,
        phases=phases,
        airspace_avoided=airspace_avoided,
        grid_waypoints_used=(
            count_grid_points([graph.nodes[i] for i in result.path])
            if use_grid
            else None
        ),
    )


def _spread_evenly(
    candidates: Sequence[tuple], total_nm: float, max_waypoints: int
) -> List[Navaid]:
    """Thin a distance-sorted candidate list down to evenly spaced picks.

    THE PROBLEM THIS SOLVES
    -----------------------
    Navaid density is wildly uneven -- there are far more around a major
    metro area than over farmland. The obvious thinning approach, "take
    every Nth item from the list", follows that density: if 9 of 10
    candidates are bunched near the departure city, most of your picks
    land in that cluster and the rest of the route gets nothing.

    THE APPROACH
    ------------
    Ignore the list and think in distances instead. Lay down evenly spaced
    target points along the course, then for each one grab the nearest
    candidate that hasn't already been taken.

    For max_waypoints=3 on a 300 nm route, spacing = 300/4 = 75 nm, so the
    targets sit at 75, 150 and 225 nm:

        origin ----X--------X--------X---- dest
                  75nm    150nm    225nm

    The picks land as close to evenly spread as the available navaids
    allow, regardless of how they clump.

    Args:
        candidates: (along_track_nm, navaid) pairs, already sorted by
            along-track distance.

    Returns:
        The chosen navaids, in course order.
    """
    # Nothing to do, or nothing to thin.
    if max_waypoints <= 0 or not candidates:
        return []
    if len(candidates) <= max_waypoints:
        return [navaid for _, navaid in candidates]

    # Divide by (max_waypoints + 1) so the targets sit at interior points
    # and none lands on top of the origin or destination. 3 waypoints means
    # 4 gaps between 5 points total.
    spacing = total_nm / (max_waypoints + 1)

    chosen: set = set()  # indices into `candidates`, so we never pick twice

    for step in range(1, max_waypoints + 1):
        target_nm = spacing * step

        # Of the candidates not yet taken, find the one whose along-track
        # distance is closest to this target. min() with a key= is just
        # "argmin": it returns the index minimising the distance to target.
        best_index = min(
            (i for i in range(len(candidates)) if i not in chosen),
            key=lambda i: abs(candidates[i][0] - target_nm),
        )
        chosen.add(best_index)

    # `chosen` is a set, so it has no order, and we filled it in target
    # order anyway. But `candidates` is sorted by along-track distance, so
    # sorting the INDICES puts the picks back into course order.
    return [candidates[i][1] for i in sorted(chosen)]
