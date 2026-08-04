"""CP1 CLI: print a naive waypoint route between two airports.

    python plan_route.py --origin KPWK --dest KMSP

This file is the "thin shell" of the program. It parses arguments, calls
into flight_dispatch/, and formats the result for a terminal. It contains
no routing logic of its own -- that all lives in the package, so CP4's
agent can call the same functions without dragging along any CLI code.
"""

import argparse
import sys

from flight_dispatch.data_loader import (
    MissingDataError,
    load_airports,
    load_navaids,
    navaids_in_bounds,
)
from flight_dispatch.geo import haversine_nm, initial_bearing_deg
from flight_dispatch.models import Airport
from flight_dispatch.route import RoutePlan, naive_route


def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse the command-line interface.

    Taking `argv` as a parameter (rather than always reading sys.argv)
    means tests can call parse_args(["--origin", "KPWK", ...]) directly.
    Passing None makes argparse fall back to the real command line.
    """
    parser = argparse.ArgumentParser(
        description="Plan a naive waypoint route between two airports (CP1)."
    )
    parser.add_argument("--origin", required=True, help="Origin ICAO code, e.g. KPWK")
    parser.add_argument("--dest", required=True, help="Destination ICAO code, e.g. KORD")
    parser.add_argument(
        "--corridor-nm",
        type=float,
        default=15.0,
        help="Half-width of the navaid search corridor, in nm (default: 15)",
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=5,
        help="Maximum intermediate waypoints (default: 5)",
    )
    parser.add_argument(
        "--map",
        metavar="PATH",
        nargs="?",
        const="route_map.html",
        help="Write an interactive HTML map (default filename: route_map.html)",
    )
    return parser.parse_args(argv)


def lookup_airport(airports: dict, code: str, label: str) -> Airport:
    """Resolve an ICAO code to an Airport, or exit with a clear message.

    `.strip().upper()` so that " kord " and "kord" both work -- users type
    lowercase constantly.

    SystemExit with a string prints it to stderr and exits with status 1,
    which is the conventional Unix way for a CLI to fail. `label` just
    makes the message say "origin" or "destination" appropriately.
    """
    airport = airports.get(code.strip().upper())
    if airport is None:
        raise SystemExit(f"Unknown {label} ICAO code: {code!r}")
    return airport


def format_plan(plan: RoutePlan) -> str:
    """Render a route plan as an aligned, human-readable block.

    Kept as a pure string-returning function rather than printing
    directly, so it can be unit-tested and later reused by CP6's report
    generator.

    Each row shows the waypoint, its coordinates, and the leg that got you
    there: distance, initial true course, and running total.
    """
    lines = [
        f"{plan.origin.icao} ({plan.origin.name})"
        f"  ->  {plan.dest.icao} ({plan.dest.name})",
        "",
    ]

    # Width of the widest identifier, so the columns line up regardless of
    # whether idents are 2 characters ("OH") or 4 ("KMSP").
    width = max(len(wp.ident) for wp in plan.waypoints)

    cumulative_nm = 0.0

    for index, waypoint in enumerate(plan.waypoints):
        if index > 0:
            # Every waypoint after the first has a leg leading into it.
            previous = plan.waypoints[index - 1]
            leg_nm = haversine_nm(
                previous.lat, previous.lon, waypoint.lat, waypoint.lon
            )
            cumulative_nm += leg_nm
            course = initial_bearing_deg(
                previous.lat, previous.lon, waypoint.lat, waypoint.lon
            )
            # {course:03.0f} zero-pads to three digits, which is how
            # headings are always written in aviation: 090T, not 90T.
            # The T marks it as degrees TRUE rather than magnetic.
            leg = f"{leg_nm:6.1f} nm  {course:03.0f}T  {cumulative_nm:6.1f} nm total"
        else:
            # The origin has no preceding leg.
            leg = "departure"

        # {value:<width} left-aligns in a field of that width; the width
        # itself is substituted from the variable computed above.
        lines.append(
            f"  {waypoint.ident:<{width}}  "
            f"{waypoint.lat:9.4f} {waypoint.lon:10.4f}  "
            f"{leg}"
        )

    # Comparing route distance against direct distance is the quickest way
    # to eyeball whether the waypoint picks were sensible: a big gap means
    # the route is wandering.
    lines += [
        "",
        f"Waypoints:       {len(plan.waypoints)} "
        f"({len(plan.waypoints) - 2} intermediate)",
        f"Direct distance: {plan.direct_distance_nm:.1f} nm",
        f"Route distance:  {plan.total_distance_nm:.1f} nm",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    """Wire the pieces together: load -> look up -> filter -> route -> print."""
    args = parse_args(argv)

    # Load reference data. If the CSVs were never downloaded, turn the
    # exception into a plain one-line message rather than a traceback --
    # a stack trace is noise when the fix is "run the download script".
    try:
        airports = load_airports()
        navaids = load_navaids()
    except MissingDataError as exc:
        raise SystemExit(str(exc))

    origin = lookup_airport(airports, args.origin, "origin")
    dest = lookup_airport(airports, args.dest, "destination")

    # Cheap geographic prefilter before the expensive corridor test.
    nearby = navaids_in_bounds(navaids, origin.lat, origin.lon, dest.lat, dest.lon)

    plan = naive_route(
        origin,
        dest,
        nearby,
        corridor_width_nm=args.corridor_nm,
        max_waypoints=args.max_waypoints,
    )

    print(format_plan(plan))

    # Map rendering is optional, so folium is imported only when asked
    # for. That keeps the routing engine usable without it installed.
    if args.map:
        from flight_dispatch.mapping import save_route_map

        written = save_route_map(plan, args.map)
        print(f"\nMap written to {written}")

    return 0  # exit status 0 = success


# Only run main() when this file is executed directly, not when it is
# imported. sys.exit() propagates the return value as the process's exit
# status, so shell scripts and CI can check whether it succeeded.
if __name__ == "__main__":
    sys.exit(main())
