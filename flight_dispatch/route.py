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

from .geo import cross_and_along_track_nm, haversine_nm
from .graph import DEFAULT_MIN_NEIGHBORS, DEFAULT_RADIUS_NM, WaypointGraph, build_mesh
from .models import Airport, Navaid, Waypoint
from .search import a_star


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
    # Origin and destination join the mesh as nodes so the path has
    # endpoints; no other airports participate, since real routes are
    # defined navaid-to-navaid.
    graph = build_mesh(
        [origin, *navaids, dest],
        radius_nm=radius_nm,
        min_neighbors=min_neighbors,
    )

    # Origin is at index 0 and destination last, by construction above.
    result = a_star(graph, 0, graph.node_count - 1)

    if not result.found:
        raise NoRouteFound(
            f"No path from {origin.icao} to {dest.icao} through "
            f"{graph.node_count} waypoints. Try a larger --radius-nm."
        )

    return RoutePlan(
        origin=origin,
        dest=dest,
        waypoints=[graph.nodes[i] for i in result.path],
        graph_nodes=graph.node_count,
        graph_edges=graph.edge_count,
        nodes_expanded=result.nodes_expanded,
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
