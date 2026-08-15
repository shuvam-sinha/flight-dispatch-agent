"""Virtual waypoints, for routing where navaids do not reach.

THE PROBLEM
-----------
Navaids are radio transmitters on the ground, so coverage ends where the
ground does. Measured across the whole dataset, overland routes come out
at 100-105% of the direct distance on every continent -- but KJFK to EGLL
runs 133%, crawling up through Greenland and Iceland because those are
the only waypoints that exist. San Francisco to London produced 34
waypoints threading through Canada.

Real oceanic flights do not use ground navaids. They use named lat/lon
points: `56N020W` means 56 degrees north, 20 degrees west. Whole
organised track systems are built from them, republished daily to follow
the jet stream.

THE APPROACH
------------
Generate those points. A lattice is laid over the great circle -- columns
spaced along the course, lanes offset either side of it -- giving A* room
to manoeuvre where nothing else exists:

              +400nm  ·    ·    ·    ·    ·    ·
                      ·    ·    ·    ·    ·    ·
    origin ───────────·────·────·────·────·──────── destination
                      ·    ·    ·    ·    ·    ·
              -400nm  ·    ·    ·    ·    ·    ·

WHY THIS IS NOT JUST AN OCEAN FIX
---------------------------------
Lateral freedom is what makes wind routing meaningful. To bend 200 nm
north for a jet stream, there must be a waypoint 200 nm north. Over the
US Midwest, navaid density happens to supply that; over an ocean it does
not, and even overland the navaid positions constrain where a route can
go. The grid gives A* a smooth surface to optimise over, which is what
turns CP3's wind cost from a number into a visible route change.

HOW REAL THIS IS
----------------
Over water: genuinely real. Oceanic waypoints ARE lat/lon points, and
`56N020W` is the actual naming convention, so the grid points are
generated on whole degrees of latitude and longitude to land where real
ones do.

Over land: not real. A generated point over Nebraska corresponds to
nothing on any chart, and no controller would recognise it. That is why
grid points are only emitted where navaids are genuinely absent -- see
`fill_navaid_gaps`. Overland routes keep naming real VORs; oceanic legs
get lat/lon fixes. Which is, as it happens, exactly how a real flight
plan reads.
"""

import math
from typing import List, Sequence, Tuple

from .geo import (
    destination_point,
    great_circle_point,
    haversine_nm,
    initial_bearing_deg,
)
from .models import Navaid

# Spacing along the course. Matched to the mesh connection radius so
# consecutive grid points are always joinable without relying on the
# k-nearest fallback.
DEFAULT_SPACING_NM = 150.0

# How far either side of the course to offer lanes, and how many. Four
# lanes at 200 nm gives +/-400 nm of lateral freedom -- enough to reach a
# jet stream core, without exploding the node count.
DEFAULT_LANE_SPACING_NM = 200.0
DEFAULT_LANES = 2

# A grid point closer than this to a real navaid is redundant: the navaid
# is a better waypoint, being an actual charted fix. 120 nm is slightly
# under the mesh radius, so anywhere a navaid can serve, it does.
DEFAULT_NAVAID_CLEARANCE_NM = 120.0


def oceanic_name(lat: float, lon: float) -> str:
    """Name a lat/lon fix the way oceanic waypoints are actually named.

    `56N020W` is 56 degrees north, 20 degrees west -- the convention used
    across organised track systems. Latitude takes two digits, longitude
    three, which is why the two are padded differently.
    """
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(round(lat)):02d}{ns}{abs(round(lon)):03d}{ew}"


def make_grid_point(lat: float, lon: float) -> Navaid:
    """Wrap a coordinate as a Navaid so the mesh can treat it uniformly.

    Reusing `Navaid` rather than adding a type means `build_mesh`,
    `a_star`, the cost functions and the map renderer all work unchanged
    -- a grid point is just another thing with an ident and a position.
    The `navaid_type` marks its provenance so output can distinguish a
    charted VOR from a generated fix.
    """
    # Snap to whole degrees. Real oceanic waypoints sit on whole or half
    # degrees, and snapping also means nearby routes reuse identical
    # points rather than generating near-duplicates.
    lat, lon = round(lat), round(lon)
    return Navaid(
        ident=oceanic_name(lat, lon),
        name=f"{abs(lat)}{'N' if lat >= 0 else 'S'} "
        f"{abs(lon)}{'E' if lon >= 0 else 'W'}",
        navaid_type="GRID",
        lat=float(lat),
        lon=float(lon),
    )


def routing_grid(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    spacing_nm: float = DEFAULT_SPACING_NM,
    lane_spacing_nm: float = DEFAULT_LANE_SPACING_NM,
    lanes: int = DEFAULT_LANES,
) -> List[Navaid]:
    """Build a lattice of virtual waypoints around a great circle.

    Columns are spaced `spacing_nm` along the course; each column carries
    `2 * lanes + 1` points, offset perpendicular in `lane_spacing_nm`
    steps either side.

    The perpendicular bearing is recomputed per column rather than taken
    once at the origin. On a great circle the course changes continuously
    -- a transatlantic flight departs pointing well north of east and
    arrives pointing south of east -- so a single fixed offset direction
    would skew the lattice badly at high latitudes.
    """
    total_nm = haversine_nm(lat1, lon1, lat2, lon2)
    if total_nm < spacing_nm:
        return []

    columns = int(total_nm // spacing_nm)
    points: List[Navaid] = []
    seen: set = set()

    for column in range(1, columns + 1):
        fraction = (column * spacing_nm) / total_nm
        centre_lat, centre_lon = great_circle_point(
            fraction, lat1, lon1, lat2, lon2
        )

        # Local course at this column, so the offset is genuinely
        # perpendicular here rather than perpendicular at the origin.
        ahead_lat, ahead_lon = great_circle_point(
            min(fraction + 0.01, 1.0), lat1, lon1, lat2, lon2
        )
        course = initial_bearing_deg(centre_lat, centre_lon, ahead_lat, ahead_lon)

        for lane in range(-lanes, lanes + 1):
            if lane == 0:
                lat, lon = centre_lat, centre_lon
            else:
                bearing = (course + (90 if lane > 0 else 270)) % 360
                lat, lon = destination_point(
                    centre_lat, centre_lon, bearing, abs(lane) * lane_spacing_nm
                )

            point = make_grid_point(lat, lon)
            if point.ident not in seen:
                seen.add(point.ident)
                points.append(point)

    return points


def fill_navaid_gaps(
    grid_points: Sequence[Navaid],
    navaids: Sequence[Navaid],
    clearance_nm: float = DEFAULT_NAVAID_CLEARANCE_NM,
) -> List[Navaid]:
    """Keep only grid points that no navaid already covers.

    This is what keeps overland routes honest. A generated fix over
    Nebraska corresponds to nothing on any chart, so where real navaids
    exist they should be used and the grid should stay out of the way.
    Over open water there is nothing to defer to, and the grid supplies
    the only waypoints available.

    The result is a route that names real VORs overland and lat/lon fixes
    oceanically -- which is how an actual flight plan reads.

    Complexity is O(grid x navaids), but the grid is tens of points and
    the navaid list is already filtered to the region, so this is cheap
    next to mesh construction.
    """
    if not navaids:
        return list(grid_points)

    kept: List[Navaid] = []
    for point in grid_points:
        nearest = min(
            haversine_nm(point.lat, point.lon, navaid.lat, navaid.lon)
            for navaid in navaids
        )
        if nearest > clearance_nm:
            kept.append(point)
    return kept


def waypoints_for_route(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    navaids: Sequence[Navaid],
    use_grid: bool = True,
    **grid_kwargs,
) -> List[Navaid]:
    """Navaids for a route, topped up with grid points where they run out.

    The single entry point callers need: hand it the region's navaids and
    get back everything the mesh should be built from.
    """
    if not use_grid:
        return list(navaids)

    grid = routing_grid(lat1, lon1, lat2, lon2, **grid_kwargs)
    return list(navaids) + fill_navaid_gaps(grid, navaids)


def count_grid_points(waypoints: Sequence) -> int:
    """How many waypoints in a list are generated rather than charted."""
    return sum(1 for w in waypoints if getattr(w, "navaid_type", "") == "GRID")
