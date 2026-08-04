"""Interactive map rendering for a RoutePlan.

An early slice of CP6, pulled forward because seeing the route on a map
is by far the fastest way to sanity-check that waypoint selection is
behaving. Numbers tell you a route is 289 nm; a map tells you instantly
whether it zigzags.

`folium` wraps Leaflet.js: you build the map in Python and it writes a
self-contained HTML file you open in a browser. It needs an internet
connection to fetch the basemap tiles, but the route data itself is
baked into the file.

CP6 will extend this with restricted-airspace polygons (CP3) and embed
it in the full dispatch report. Nothing here is imported by the routing
code -- it is a pure consumer of RoutePlan, so folium stays an optional
dependency of the CLI rather than of the engine.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from .geo import haversine_nm, initial_bearing_deg
from .models import Airport

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    from .route import RoutePlan

# Colours kept as named constants so the CP6 report can reuse them and
# stay visually consistent with whatever this produces.
DIRECT_COLOR = "#888888"  # the straight-line reference course
ROUTE_COLOR = "#0066cc"  # the actual flown path
AIRPORT_COLOR = "#d62728"  # origin and destination
NAVAID_COLOR = "#0066cc"  # intermediate beacons


def route_map(plan: "RoutePlan", zoom_padding: float = 0.15):
    """Build a folium Map showing a route plan.

    Two lines are drawn deliberately:
      - a grey dashed line for the DIRECT great-circle course, and
      - a solid blue line for the route actually flown.

    The gap between them is the visual version of the "direct distance vs
    route distance" numbers the CLI prints. A route that hugs the dashed
    line is efficient; one that wanders away from it is not. On short
    legs with CP1's naive selection you will see obvious zigzags, which
    is the limitation A* removes in CP2.

    Args:
        plan: The route to draw.
        zoom_padding: Fraction of the route's extent to add as margin
            around the initial view, so endpoints are not flush against
            the window edge.

    Returns:
        A folium.Map. Call `.save("path.html")` to write it out.
    """
    import folium  # imported lazily so the core engine never needs folium

    lats = [wp.lat for wp in plan.waypoints]
    lons = [wp.lon for wp in plan.waypoints]

    route_map_obj = folium.Map(tiles="CartoDB positron", control_scale=True)

    # Great-circle reference course, origin straight to destination.
    folium.PolyLine(
        [(plan.origin.lat, plan.origin.lon), (plan.dest.lat, plan.dest.lon)],
        color=DIRECT_COLOR,
        weight=2,
        opacity=0.8,
        dash_array="8,8",
        tooltip=f"Direct course: {plan.direct_distance_nm:.1f} nm",
    ).add_to(route_map_obj)

    # The route actually flown, waypoint to waypoint.
    folium.PolyLine(
        list(zip(lats, lons)),
        color=ROUTE_COLOR,
        weight=3.5,
        opacity=0.9,
        tooltip=f"Route: {plan.total_distance_nm:.1f} nm",
    ).add_to(route_map_obj)

    _add_waypoint_markers(folium, route_map_obj, plan)

    # Frame the view on the route's extent rather than guessing a zoom
    # level -- this works equally well for an 8 nm hop and a 3000 nm haul.
    lat_pad = (max(lats) - min(lats)) * zoom_padding
    lon_pad = (max(lons) - min(lons)) * zoom_padding
    route_map_obj.fit_bounds(
        [
            [min(lats) - lat_pad, min(lons) - lon_pad],
            [max(lats) + lat_pad, max(lons) + lon_pad],
        ]
    )

    return route_map_obj


def _add_waypoint_markers(folium, route_map_obj, plan: "RoutePlan") -> None:
    """Drop a labelled marker on every waypoint.

    Airports get large red circles, navaids smaller blue ones, so the
    endpoints of the route read at a glance. Each popup carries the leg
    that led into that waypoint -- distance and true course -- which is
    the same information the CLI table prints, just spatially placed.
    """
    cumulative_nm = 0.0

    for index, waypoint in enumerate(plan.waypoints):
        is_airport = isinstance(waypoint, Airport)

        detail_lines = [
            f"<b>{waypoint.ident}</b>",
            waypoint.name,
            f"{waypoint.lat:.4f}, {waypoint.lon:.4f}",
        ]

        if index == 0:
            detail_lines.append("<i>Departure</i>")
        else:
            previous = plan.waypoints[index - 1]
            leg_nm = haversine_nm(
                previous.lat, previous.lon, waypoint.lat, waypoint.lon
            )
            cumulative_nm += leg_nm
            course = initial_bearing_deg(
                previous.lat, previous.lon, waypoint.lat, waypoint.lon
            )
            detail_lines.append(
                f"Leg from {previous.ident}: {leg_nm:.1f} nm on {course:03.0f}T"
            )
            detail_lines.append(f"Cumulative: {cumulative_nm:.1f} nm")

        folium.CircleMarker(
            location=(waypoint.lat, waypoint.lon),
            radius=7 if is_airport else 5,
            color=AIRPORT_COLOR if is_airport else NAVAID_COLOR,
            fill=True,
            fill_opacity=0.9,
            popup=folium.Popup("<br>".join(detail_lines), max_width=260),
            tooltip=waypoint.ident,
        ).add_to(route_map_obj)

        # A permanent text label next to each marker, so identifiers are
        # readable without clicking every point.
        folium.Marker(
            location=(waypoint.lat, waypoint.lon),
            icon=folium.DivIcon(
                html=(
                    '<div style="font:600 11px sans-serif;color:#222;'
                    'text-shadow:0 0 3px #fff,0 0 3px #fff,0 0 3px #fff;'
                    'white-space:nowrap;transform:translate(10px,-8px)">'
                    f"{waypoint.ident}</div>"
                )
            ),
        ).add_to(route_map_obj)


def save_route_map(plan: "RoutePlan", path: str) -> str:
    """Render a route to an HTML file and return the path written.

    Creates the parent directory if needed, so `--map maps/foo.html`
    works without the user having to mkdir first.
    """
    destination = Path(path)
    if destination.parent != Path(""):
        destination.parent.mkdir(parents=True, exist_ok=True)

    route_map(plan).save(str(destination))
    return str(destination)
