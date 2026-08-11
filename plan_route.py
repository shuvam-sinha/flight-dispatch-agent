"""CP1 CLI: print a naive waypoint route between two airports.

    python plan_route.py --origin KPWK --dest KMSP

This file is the "thin shell" of the program. It parses arguments, calls
into flight_dispatch/, and formats the result for a terminal. It contains
no routing logic of its own -- that all lives in the package, so CP4's
agent can call the same functions without dragging along any CLI code.
"""

import argparse
import sys
from typing import Optional

from flight_dispatch.data_loader import (
    MissingDataError,
    load_airports,
    load_navaids,
    navaids_near_route,
)
from flight_dispatch.aircraft import (
    AIRCRAFT as AIRCRAFT_KEYS,
    DEFAULT_AIRCRAFT,
    AircraftProfile,
    aircraft_by_category,
    get_aircraft,
)
from flight_dispatch.geo import haversine_nm, initial_bearing_deg
from flight_dispatch.graph import DEFAULT_RADIUS_NM
from flight_dispatch.models import Airport
from flight_dispatch.route import NoRouteFound, RoutePlan, naive_route, plan_route
from flight_dispatch.wind import ConstantWindSource


def parse_args(argv=None) -> argparse.Namespace:
    """Define and parse the command-line interface.

    Taking `argv` as a parameter (rather than always reading sys.argv)
    means tests can call parse_args(["--origin", "KPWK", ...]) directly.
    Passing None makes argparse fall back to the real command line.
    """
    parser = argparse.ArgumentParser(
        description="Plan a naive waypoint route between two airports (CP1)."
    )
    # --list-aircraft prints a catalogue and exits, so origin/dest cannot
    # be unconditionally required. They are checked in main() instead.
    parser.add_argument("--origin", help="Origin ICAO code, e.g. KPWK")
    parser.add_argument("--dest", help="Destination ICAO code, e.g. KORD")
    parser.add_argument(
        "--aircraft",
        default=DEFAULT_AIRCRAFT.key,
        help=f"Aircraft key (default: {DEFAULT_AIRCRAFT.key}). See --list-aircraft",
    )
    parser.add_argument(
        "--payload",
        type=float,
        help="Payload in pounds. Defaults to the aircraft's typical occupancy",
    )
    parser.add_argument(
        "--list-aircraft",
        action="store_true",
        help="Print the aircraft catalogue and exit",
    )
    parser.add_argument(
        "--radius-nm",
        type=float,
        default=DEFAULT_RADIUS_NM,
        help=f"Mesh edge connection radius, in nm (default: {DEFAULT_RADIUS_NM:.0f})",
    )
    parser.add_argument(
        "--margin-nm",
        type=float,
        default=100.0,
        help="How far either side of the course to gather navaids (default: 100)",
    )
    parser.add_argument(
        "--naive",
        action="store_true",
        help="Use the CP1 corridor-sampling router instead of A* (for comparison)",
    )
    parser.add_argument(
        "--corridor-nm",
        type=float,
        default=15.0,
        help="CP1 only: half-width of the navaid corridor, in nm (default: 15)",
    )
    parser.add_argument(
        "--max-waypoints",
        type=int,
        default=5,
        help="CP1 only: maximum intermediate waypoints (default: 5)",
    )
    parser.add_argument(
        "--wind",
        nargs="?",
        const="live",
        metavar="SPEC",
        help='Wind-aware routing: "live" for Open-Meteo, or "DDD/SS" for a '
             'constant wind (e.g. 270/40 = 40 kt from the west)',
    )
    parser.add_argument(
        "--altitude",
        type=float,
        help="Cruise altitude in feet (default: the aircraft's own)",
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
    efficiency = plan.total_distance_nm / plan.direct_distance_nm * 100 if plan.direct_distance_nm else 100.0

    lines += [
        "",
        f"Waypoints:       {len(plan.waypoints)} "
        f"({len(plan.waypoints) - 2} intermediate)",
        f"Direct distance: {plan.direct_distance_nm:.1f} nm",
        f"Route distance:  {plan.total_distance_nm:.1f} nm "
        f"({efficiency:.1f}% of direct)",
    ]

    # CP3 figures, present only when the route was planned against wind.
    if plan.ete_hours is not None and plan.aircraft is not None:
        hours = int(plan.ete_hours)
        minutes = int(round((plan.ete_hours - hours) * 60))
        ground_speed = plan.average_ground_speed_kt or 0.0

        # Average ground speed against cruise TAS says at a glance whether
        # the flight was helped or hindered overall.
        delta = ground_speed - plan.aircraft.cruise_tas_kt
        wind_note = (
            f"net tailwind {delta:+.0f} kt"
            if delta >= 0
            else f"net headwind {-delta:.0f} kt"
        )

        lines += [
            f"ETE:             {hours}h{minutes:02d}m "
            f"(avg {ground_speed:.0f} kt GS, {wind_note})",
            f"Fuel required:   {plan.fuel_required_gal:.1f} gal "
            f"(incl. {plan.aircraft.reserve_minutes:.0f} min reserve)",
        ]

        if plan.is_within_range() is False:
            lines.append(
                f"WARNING:         exceeds endurance of "
                f"{plan.aircraft.endurance_hours():.1f} h -- a fuel stop is required"
            )

    # Search diagnostics, present only for A* routes. Worth surfacing:
    # nodes_expanded vs graph size is the visible payoff of the heuristic.
    if plan.graph_nodes is not None:
        lines.append(
            f"Mesh graph:      {plan.graph_nodes} nodes, {plan.graph_edges} edges; "
            f"A* expanded {plan.nodes_expanded}"
        )

    return "\n".join(lines)


CATEGORY_LABELS = {
    "ga": "General aviation",
    "business": "Business",
    "regional": "Regional jets",
    "narrowbody": "Narrowbody airliners",
    "widebody": "Widebody airliners",
}


def format_aircraft_catalogue() -> str:
    """Render the aircraft catalogue as a grouped table.

    `range` assumes the aircraft's typical occupancy, not a full cabin --
    see AircraftProfile.typical_payload_lb for why those differ. The
    `limit` column shows whether tank capacity or maximum takeoff weight
    is what caps the fuel load.
    """
    lines = [f"{len(list(AIRCRAFT_KEYS))} aircraft. Use --aircraft KEY.", ""]

    for category, profiles in aircraft_by_category().items():
        lines.append(f"  {CATEGORY_LABELS.get(category, category)}")
        lines.append(
            f"    {'key':<6} {'aircraft':<26} {'TAS':>5} {'cruise':>7} "
            f"{'seats':>6} {'range':>8}  limit"
        )
        for profile in profiles:
            limit = "weight" if profile.is_weight_limited() else "tank"
            lines.append(
                f"    {profile.key:<6} {profile.name:<26} "
                f"{profile.cruise_tas_kt:>4.0f}k {profile.cruise_altitude_ft/1000:>6.0f}k "
                f"{profile.typical_occupancy:>3}/{profile.seats:<3} "
                f"{profile.range_nm():>7.0f}n  {limit}"
            )
        lines.append("")

    return "\n".join(lines)


def format_aircraft_summary(
    aircraft: AircraftProfile, payload_lb: float
) -> str:
    """One-line description of the aircraft and load being planned."""
    fuel_gal = aircraft.max_fuel_gal(payload_lb)
    limit = "MTOW-limited" if aircraft.is_weight_limited(payload_lb) else "full tanks"
    return (
        f"{aircraft.name}  |  {aircraft.cruise_tas_kt:.0f} kt TAS at "
        f"{aircraft.cruise_altitude_ft:,.0f} ft  |  payload {payload_lb:,.0f} lb  |  "
        f"fuel {fuel_gal:,.0f} gal ({limit})  |  range {aircraft.range_nm(payload_lb):,.0f} nm"
    )


def build_wind_source(spec: Optional[str]):
    """Turn a --wind argument into a WindSource.

    Accepts "live" for real Open-Meteo forecasts, or "DDD/SS" for a
    uniform wind -- 270/40 meaning 40 kt blowing from the west. The
    constant form is useful for demonstrating the routing effect in
    isolation, since any bend in the route must then come from geometry
    rather than from a complicated wind field.

    Returns None when no wind was requested, which keeps the router on
    CP2's distance-only cost.
    """
    if spec is None:
        return None

    if spec == "live":
        # Imported lazily so the router works without `requests` when no
        # live data is asked for.
        from flight_dispatch.wind_openmeteo import OpenMeteoWindSource

        return OpenMeteoWindSource()

    try:
        direction, speed = spec.split("/")
        return ConstantWindSource(float(direction), float(speed))
    except ValueError:
        raise SystemExit(
            f"Bad --wind value {spec!r}. Use 'live' or 'DDD/SS', e.g. 270/40."
        )


def main(argv=None) -> int:
    """Wire the pieces together: load -> look up -> filter -> route -> print."""
    args = parse_args(argv)

    if args.list_aircraft:
        print(format_aircraft_catalogue())
        return 0

    if not args.origin or not args.dest:
        raise SystemExit("--origin and --dest are required (or use --list-aircraft)")

    try:
        aircraft = get_aircraft(args.aircraft)
    except KeyError as exc:
        raise SystemExit(str(exc).strip('"'))

    payload_lb = (
        aircraft.typical_payload_lb if args.payload is None else args.payload
    )

    # A payload heavy enough to leave no room for fuel is a real answer,
    # not an error -- but the route below would be meaningless, so stop
    # here and say why.
    if aircraft.range_nm(payload_lb) <= 0:
        raise SystemExit(
            f"{aircraft.name} cannot carry {payload_lb:,.0f} lb and any usable fuel.\n"
            f"Useful load is {aircraft.useful_load_lb:,.0f} lb "
            f"(MTOW {aircraft.mtow_lb:,.0f} - empty {aircraft.empty_weight_lb:,.0f})."
        )

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
    wind_source = build_wind_source(args.wind)

    nearby = navaids_near_route(
        navaids,
        origin.lat,
        origin.lon,
        dest.lat,
        dest.lon,
        margin_nm=args.margin_nm,
    )

    if args.naive:
        plan = naive_route(
            origin,
            dest,
            nearby,
            corridor_width_nm=args.corridor_nm,
            max_waypoints=args.max_waypoints,
        )
    else:
        try:
            plan = plan_route(
                origin,
                dest,
                nearby,
                radius_nm=args.radius_nm,
                aircraft=aircraft,
                wind_source=wind_source,
                altitude_ft=args.altitude,
            )
        except NoRouteFound as exc:
            raise SystemExit(str(exc))

    print(format_aircraft_summary(aircraft, payload_lb))
    print()
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
