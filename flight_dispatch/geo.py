"""Great-circle geometry helpers.

WHY THIS FILE EXISTS
--------------------
The Earth is a sphere, so you cannot use flat-plane geometry (Pythagoras,
y = mx + b) for anything at flight distances. Every distance and heading
here is computed on the surface of a sphere.

This module is kept free of project-specific types (no Airport, no Navaid)
so that CP2's A* heuristic and CP3's wind-vector math can reuse it
directly. It takes plain floats and returns plain floats.

UNITS, USED CONSISTENTLY THROUGHOUT THE PROJECT
-----------------------------------------------
- Distance: nautical miles (nm). Aviation's unit. 1 nm = one minute of
  arc on the Earth's surface, which is exactly why a degree of latitude
  is 60 nm -- a fact the tests lean on.
- Angles in the public API: degrees.
- Angles inside the math: radians, because Python's math module needs
  them. Conversions happen at the boundary of each function.
"""

import math
from typing import List, Sequence, Tuple

# Mean radius of the Earth in nautical miles. "Mean" because the Earth is
# slightly squashed at the poles; a sphere is a simplification that is
# accurate to a few tenths of a percent, which is far better than this
# project needs.
EARTH_RADIUS_NM = 3440.065


def haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points, in nautical miles.

    A "great circle" is the shortest path between two points on a sphere
    -- the path you'd get by stretching a string over a globe. This is the
    haversine formula, the standard way to compute its length.

    The idea in two steps:
      1. Find the ANGLE between the two points as seen from the centre of
         the Earth (call it c, in radians).
      2. Multiply that angle by the Earth's radius. Arc length = r * theta.

    Step 1 is the fiddly part. The `a` term below is a trigonometric
    identity that is numerically stable even for very short distances --
    the naive spherical law of cosines loses precision there because
    cos(x) is nearly flat near x = 0.
    """
    # Convert everything to radians up front.
    phi1, phi2 = math.radians(lat1), math.radians(lat2)  # phi = latitude
    dphi = math.radians(lat2 - lat1)  # change in latitude
    dlambda = math.radians(lon2 - lon1)  # change in longitude

    # `a` is the square of half the straight-line (chord) distance between
    # the points, expressed on a unit sphere.
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )

    # atan2(sqrt(a), sqrt(1-a)) converts that chord back into the central
    # angle. Multiply by radius to get an actual distance.
    return EARTH_RADIUS_NM * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def initial_bearing_rad(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing from point 1 to point 2, in radians.

    "Bearing" is the compass direction you point the aircraft. "Initial"
    matters: on a great circle your compass heading changes continuously
    as you fly. Flying from Chicago to London you depart pointing well
    north of east, yet arrive pointing south of east, without ever turning
    -- the path is straight, but the meridians you cross are not parallel.
    So this is the heading at the START of the leg only.

    "True" means relative to true north (the geographic pole), not
    magnetic north. Real charts use magnetic; converting between them
    requires local magnetic variation, which this project does not model.
    """
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)

    # Standard forward-azimuth formula. y and x are the components of the
    # direction vector to point 2, projected into the local tangent plane
    # at point 1; atan2 turns them back into an angle.
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(
        dlambda
    )
    return math.atan2(y, x)


def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial true bearing from point 1 to point 2, in degrees (0-360).

    atan2 returns -pi..+pi, so a westbound course comes out negative. The
    `% 360` wraps that into the 0-360 range a compass actually shows:
    -90 degrees becomes 270 degrees.
    """
    return math.degrees(initial_bearing_rad(lat1, lon1, lat2, lon2)) % 360.0


def cross_track_nm(
    point_lat: float,
    point_lon: float,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> float:
    """Signed perpendicular distance from a point to the start->end course.

    "Cross-track error" is a real instrument reading in aircraft: how far
    off the intended course line you have drifted. Zero means dead on
    course.

    The sign tells you WHICH SIDE: positive means the point lies to the
    right of the course, negative to the left. That matters for CP3 --
    knowing a restricted zone is to your left tells you which way to bend
    the route.
    """
    # d13 = angular distance from start to the point (in radians, i.e. on
    # a unit sphere -- hence the division by the radius).
    d13 = haversine_nm(start_lat, start_lon, point_lat, point_lon) / EARTH_RADIUS_NM
    theta13 = initial_bearing_rad(start_lat, start_lon, point_lat, point_lon)  # to point
    theta12 = initial_bearing_rad(start_lat, start_lon, end_lat, end_lon)  # to dest

    # (theta13 - theta12) is the angle between "direction to the point" and
    # "direction to the destination". sin() of that angle picks out the
    # sideways component -- exactly the perpendicular offset we want.
    return math.asin(_clamp(math.sin(d13) * math.sin(theta13 - theta12))) * EARTH_RADIUS_NM


def cross_and_along_track_nm(
    point_lat: float,
    point_lon: float,
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
) -> Tuple[float, float]:
    """Decompose a point's position relative to the start->end course.

    Returns (cross_track_nm, along_track_nm). Think of it as rotating the
    map so the course line becomes the x-axis:

        along-track  = how far ALONG the course the point sits (x)
        cross-track  = how far OFF the course it sits (y)

                        . point
                        |
                        | cross-track (perpendicular offset)
        start ----------+--------------------> dest
              along-track

    Both are signed:
      - cross-track: positive right of course, negative left.
      - along-track: negative means the point is BEHIND the start;
        greater than the course length means it is PAST the destination.

    THE SIGN BUG THIS FUNCTION EXISTS TO AVOID
    ------------------------------------------
    The textbook along-track formula is:

        along = acos(cos(d13) / cos(cross_track))

    but acos() only ever returns values in 0..pi -- it can never be
    negative. So a navaid sitting 60 nm BEHIND your departure airport
    reports the exact same "+60 nm of progress" as one 60 nm ahead of it.
    Route selection would then happily pick up waypoints pointing the
    opposite direction of travel.

    The fix: recover the sign separately. If the angle between "bearing to
    the point" and "bearing to the destination" is obtuse (i.e. its cosine
    is negative), the point is behind us, so negate the result.
    """
    d13 = haversine_nm(start_lat, start_lon, point_lat, point_lon) / EARTH_RADIUS_NM
    theta13 = initial_bearing_rad(start_lat, start_lon, point_lat, point_lon)
    theta12 = initial_bearing_rad(start_lat, start_lon, end_lat, end_lon)

    # Perpendicular offset, in radians for now (converted at the return).
    cross_track = math.asin(_clamp(math.sin(d13) * math.sin(theta13 - theta12)))

    # Right-triangle relation on a sphere: knowing the hypotenuse (d13) and
    # one leg (cross_track), solve for the other leg (along_track).
    along_track = math.acos(_clamp(math.cos(d13) / math.cos(cross_track)))

    # Restore the sign that acos() threw away. See the docstring above.
    if math.cos(theta13 - theta12) < 0:
        along_track = -along_track

    # Convert both from radians on a unit sphere back to nautical miles.
    return cross_track * EARTH_RADIUS_NM, along_track * EARTH_RADIUS_NM


def unwrap_longitudes(lons: Sequence[float]) -> List[float]:
    """Make a sequence of longitudes continuous across the antimeridian.

    Longitude is conventionally reported in [-180, +180], which means a
    path crossing the antimeridian jumps discontinuously: 179, 180, then
    suddenly -179. Any min/max over those values spans the globe the
    WRONG WAY -- 358 degrees the long way round instead of 2 degrees the
    short way.

    This shifts each value by whole multiples of 360 so consecutive
    entries never differ by more than 180. The result may fall outside
    [-180, 180] -- e.g. [179, 181, 183] -- which is exactly the point:
    the numbers are now continuous and safe to take min/max over.

    Callers comparing a real longitude against such a range must use
    `longitude_within`, which tries the +/-360 shifts.
    """
    if not lons:
        return []

    unwrapped = [lons[0]]
    for lon in lons[1:]:
        previous = unwrapped[-1]
        # Move `lon` into the same "lap" as its predecessor.
        lon -= 360.0 * round((lon - previous) / 360.0)
        unwrapped.append(lon)
    return unwrapped


def longitude_within(lon: float, min_lon: float, max_lon: float) -> bool:
    """Is `lon` inside a possibly-unwrapped longitude range?

    A range produced by `unwrap_longitudes` can extend past +/-180, so a
    navaid reported at -179 may need testing as +181 to fall inside it.
    Checking the value shifted by 0, +360 and -360 covers every case
    without the caller needing to know whether a wrap happened.
    """
    return any(
        min_lon <= lon + shift <= max_lon for shift in (0.0, 360.0, -360.0)
    )


def bounding_box(
    lat1: float, lon1: float, lat2: float, lon2: float, margin_nm: float
) -> Tuple[float, float, float, float]:
    """Lat/lon rectangle enclosing both points, padded by margin_nm.

    Returns (min_lat, min_lon, max_lat, max_lon). The longitude bounds
    may lie outside [-180, 180] when the pair straddles the antimeridian;
    test membership with `longitude_within` rather than a bare comparison.

    This is a cheap prefilter, not precise geometry. The navaid dataset is
    global (~11,000 rows); testing every one against the course line is
    wasteful when four float comparisons can throw out 99% of them first.

    WHY LATITUDE AND LONGITUDE PAD DIFFERENTLY
    -------------------------------------------
    Lines of latitude are evenly spaced everywhere: 1 degree = 60 nm,
    always. So lat_pad is a simple division.

    Lines of longitude converge at the poles. At the equator 1 degree of
    longitude is 60 nm; at 60 degrees north it is only 30 nm; at the pole
    itself, zero. So covering a fixed distance in nm requires MORE degrees
    of longitude the further from the equator you are -- hence dividing by
    cos(latitude), which shrinks toward zero as latitude grows.
    """
    lat_pad = margin_nm / 60.0

    # Use whichever endpoint is nearer a pole -- that's where a degree of
    # longitude is narrowest, so it needs the most padding. Padding both
    # sides by that amount is a slight over-estimate, which is fine: this
    # filter must never discard a navaid it should have kept.
    widest_lat = max(abs(lat1), abs(lat2))

    # Two guards against division blowup: clamp the latitude to 89 degrees
    # (cos(90) is 0), and floor the whole divisor at a tiny positive value.
    lon_pad = margin_nm / max(60.0 * math.cos(math.radians(min(widest_lat, 89.0))), 1e-6)

    # Unwrap before min/max so an antimeridian crossing measures the short
    # way round rather than the long way.
    west, east = unwrap_longitudes([lon1, lon2])

    return (
        min(lat1, lat2) - lat_pad,
        min(west, east) - lon_pad,
        max(lat1, lat2) + lat_pad,
        max(west, east) + lon_pad,
    )


def great_circle_point(
    fraction: float,
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> Tuple[float, float]:
    """Interpolate a point along the great circle from 1 to 2.

    `fraction` is 0.0 at the start and 1.0 at the end. This is spherical
    linear interpolation ("slerp"): treat both endpoints as vectors from
    the Earth's centre, blend them with weights that keep the result on
    the sphere, then convert back to lat/lon.

    You cannot get this by averaging latitudes and longitudes -- that
    gives a point on the flat map, which is not on the actual flown path.
    """
    phi1, lam1, phi2, lam2 = map(math.radians, (lat1, lon1, lat2, lon2))

    # Angular distance between the endpoints.
    delta = 2 * math.asin(_clamp(math.sqrt(
        math.sin((phi2 - phi1) / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin((lam2 - lam1) / 2) ** 2
    )))

    # Coincident endpoints: no path to interpolate along.
    if delta == 0:
        return lat1, lon1

    # Blend weights. At fraction=0 these give (1, 0), at fraction=1 (0, 1).
    a = math.sin((1 - fraction) * delta) / math.sin(delta)
    b = math.sin(fraction * delta) / math.sin(delta)

    # Combine as 3D vectors, then back to spherical coordinates.
    x = a * math.cos(phi1) * math.cos(lam1) + b * math.cos(phi2) * math.cos(lam2)
    y = a * math.cos(phi1) * math.sin(lam1) + b * math.cos(phi2) * math.sin(lam2)
    z = a * math.sin(phi1) + b * math.sin(phi2)

    return (
        math.degrees(math.atan2(z, math.sqrt(x * x + y * y))),
        math.degrees(math.atan2(y, x)),
    )


def route_bounding_box(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    margin_nm: float,
    samples: int = 32,
) -> Tuple[float, float, float, float]:
    """Bounding box around the actual great-circle path, not just its ends.

    Returns (min_lat, min_lon, max_lat, max_lon).

    WHY THIS EXISTS RATHER THAN JUST bounding_box()
    -----------------------------------------------
    A great circle between two points arcs POLEWARD of both of them. On
    New York to London the endpoints top out at 51.5N, but the flown path
    peaks near 53.7N -- over a degree outside a box built from the
    endpoints alone. Any waypoint along that bulge would be filtered out
    before the search ever saw it.

    Sampling points along the real path and boxing those instead fixes it.
    32 samples is plenty: the bulge is a smooth arc, so the sampled peak
    lands within a few nm of the true one at any realistic route length.
    """
    lats, lons = [], []
    for step in range(samples + 1):
        lat, lon = great_circle_point(step / samples, lat1, lon1, lat2, lon2)
        lats.append(lat)
        lons.append(lon)

    # Samples along a Pacific crossing run ...178, 179, -180, -179... The
    # jump is an artefact of the [-180, 180] convention, not of the path.
    # Unwrapping removes it so min/max measure the real span.
    lons = unwrap_longitudes(lons)

    lat_pad = margin_nm / 60.0
    widest_lat = max(abs(lat) for lat in lats)
    lon_pad = margin_nm / max(60.0 * math.cos(math.radians(min(widest_lat, 89.0))), 1e-6)

    return (
        min(lats) - lat_pad,
        min(lons) - lon_pad,
        max(lats) + lat_pad,
        max(lons) + lon_pad,
    )


def _clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    """Pin a value into [low, high], default [-1, 1].

    asin() and acos() are only defined on [-1, 1]. Floating-point rounding
    can produce something like 1.0000000000000002 for a point that should
    give exactly 1.0 -- and Python raises ValueError on that, crashing on
    a perfectly valid input. Clamping absorbs the rounding noise.
    """
    return min(high, max(low, value))
