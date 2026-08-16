import unittest

from flight_dispatch.models import Airport, Navaid
from flight_dispatch.geo import haversine_nm
from flight_dispatch.route import naive_route, plan_route

# A long westbound leg gives room for intermediate waypoints.
ORIGIN = Airport(icao="KPWK", name="Chicago Executive", lat=42.1142, lon=-87.9014)
DEST = Airport(icao="KMSP", name="Minneapolis St Paul", lat=44.8820, lon=-93.2218)


def navaid(ident: str, lat: float, lon: float) -> Navaid:
    return Navaid(ident=ident, name=f"{ident} VOR", navaid_type="VOR", lat=lat, lon=lon)


def idents(plan) -> list:
    return [waypoint.ident for waypoint in plan.waypoints]


class TestNaiveRoute(unittest.TestCase):
    def test_route_starts_at_origin_and_ends_at_dest(self):
        plan = naive_route(ORIGIN, DEST, [])
        self.assertEqual(idents(plan), ["KPWK", "KMSP"])
        self.assertIs(plan.waypoints[0], ORIGIN)
        self.assertIs(plan.waypoints[-1], DEST)

    def test_includes_navaid_near_the_course_line(self):
        # Roughly the midpoint of the KPWK->KMSP great circle.
        on_course = navaid("MID", 43.55, -90.62)
        plan = naive_route(ORIGIN, DEST, [on_course])
        self.assertIn("MID", idents(plan))

    def test_excludes_navaid_far_off_course(self):
        plan = naive_route(ORIGIN, DEST, [navaid("FAR", 35.0, -100.0)])
        self.assertEqual(idents(plan), ["KPWK", "KMSP"])

    def test_excludes_navaid_behind_the_origin(self):
        # Due east of KPWK: on the extended course line, but the wrong way.
        # An unsigned along-track calculation would wrongly include this.
        behind = navaid("BEHIND", 41.7, -85.0)
        plan = naive_route(ORIGIN, DEST, [behind], corridor_width_nm=100.0)
        self.assertNotIn("BEHIND", idents(plan))

    def test_excludes_navaid_past_the_destination(self):
        beyond = navaid("BEYOND", 46.0, -96.0)
        plan = naive_route(ORIGIN, DEST, [beyond], corridor_width_nm=100.0)
        self.assertNotIn("BEYOND", idents(plan))

    def test_corridor_width_controls_inclusion(self):
        # ~30 nm north of the midpoint of the course.
        offset = navaid("OFFSET", 44.05, -90.62)
        narrow = naive_route(ORIGIN, DEST, [offset], corridor_width_nm=5.0)
        wide = naive_route(ORIGIN, DEST, [offset], corridor_width_nm=60.0)
        self.assertNotIn("OFFSET", idents(narrow))
        self.assertIn("OFFSET", idents(wide))

    def test_respects_max_waypoints(self):
        # Ten navaids strung along the course line.
        candidates = [
            navaid(f"N{i}", 42.1142 + i * 0.25, -87.9014 - i * 0.48) for i in range(1, 11)
        ]
        plan = naive_route(ORIGIN, DEST, candidates, corridor_width_nm=60.0, max_waypoints=3)
        self.assertEqual(len(plan.waypoints), 5)  # origin + 3 + dest

    def test_waypoints_are_ordered_along_the_course(self):
        candidates = [
            navaid("FAR_ALONG", 44.3, -92.2),
            navaid("NEAR_START", 42.6, -88.9),
            navaid("MIDWAY", 43.55, -90.62),
        ]
        plan = naive_route(ORIGIN, DEST, candidates, corridor_width_nm=60.0)
        self.assertEqual(
            idents(plan), ["KPWK", "NEAR_START", "MIDWAY", "FAR_ALONG", "KMSP"]
        )

    def test_thinning_spreads_picks_along_the_course(self):
        # Nine navaids bunched in the first third of the route, one late.
        clustered = [navaid(f"C{i}", 42.2 + i * 0.05, -88.0 - i * 0.1) for i in range(9)]
        late = navaid("LATE", 44.3, -92.2)
        plan = naive_route(
            ORIGIN, DEST, clustered + [late], corridor_width_nm=60.0, max_waypoints=2
        )
        # Even-spacing selection must reach the far end rather than taking
        # two neighbours out of the cluster.
        self.assertIn("LATE", idents(plan))

    def test_same_origin_and_dest_yields_trivial_route(self):
        plan = naive_route(ORIGIN, ORIGIN, [navaid("ANY", 43.0, -89.0)])
        self.assertEqual(len(plan.waypoints), 2)
        self.assertEqual(plan.direct_distance_nm, 0.0)


class TestRoutePlanDistances(unittest.TestCase):
    def test_direct_distance_matches_known_leg(self):
        plan = naive_route(ORIGIN, DEST, [])
        # KPWK -> KMSP is roughly 290 nm.
        self.assertAlmostEqual(plan.direct_distance_nm, 290.0, delta=15.0)

    def test_route_distance_equals_direct_with_no_waypoints(self):
        plan = naive_route(ORIGIN, DEST, [])
        self.assertAlmostEqual(
            plan.total_distance_nm, plan.direct_distance_nm, places=6
        )

    def test_detour_is_never_shorter_than_direct(self):
        offset = navaid("OFFSET", 44.05, -90.62)
        plan = naive_route(ORIGIN, DEST, [offset], corridor_width_nm=60.0)
        self.assertGreater(plan.total_distance_nm, plan.direct_distance_nm)


class TestPlanRouteAStar(unittest.TestCase):
    """CP2 routing. The contrast with naive_route is the point."""

    def line_of_navaids(self, count: int = 6) -> list:
        """Navaids strung roughly along the KPWK->KMSP course."""
        return [
            navaid(f"W{i}", 42.1142 + i * 0.46, -87.9014 - i * 0.89)
            for i in range(1, count + 1)
        ]

    def test_route_starts_and_ends_at_the_airports(self):
        plan = plan_route(ORIGIN, DEST, self.line_of_navaids())
        self.assertIs(plan.waypoints[0], ORIGIN)
        self.assertIs(plan.waypoints[-1], DEST)

    def test_beats_naive_on_the_short_leg_dogleg(self):
        # The CP1 failure case: a corridor wider than the flight is long.
        near_ord = Airport(icao="KORD", name="O'Hare", lat=41.9786, lon=-87.9048)
        scattered = [
            navaid("E1", 41.98, -87.60),
            navaid("E2", 42.06, -88.00),
            navaid("E3", 42.05, -88.01),
        ]
        astar = plan_route(ORIGIN, near_ord, scattered)
        naive = naive_route(ORIGIN, near_ord, scattered, corridor_width_nm=15)

        self.assertLess(astar.total_distance_nm, naive.total_distance_nm)
        # With nothing to gain from a detour, A* should fly it direct.
        self.assertAlmostEqual(
            astar.total_distance_nm, astar.direct_distance_nm, delta=0.01
        )

    def test_never_longer_than_the_direct_course_by_much(self):
        plan = plan_route(ORIGIN, DEST, self.line_of_navaids())
        self.assertLess(plan.total_distance_nm, plan.direct_distance_nm * 1.05)

    def test_no_navaids_still_routes_directly(self):
        plan = plan_route(ORIGIN, DEST, [])
        self.assertEqual([wp.ident for wp in plan.waypoints], ["KPWK", "KMSP"])
        self.assertAlmostEqual(
            plan.total_distance_nm, plan.direct_distance_nm, places=6
        )

    def test_reports_search_diagnostics(self):
        plan = plan_route(ORIGIN, DEST, self.line_of_navaids())
        self.assertEqual(plan.graph_nodes, 8)  # 6 navaids + origin + dest
        self.assertGreater(plan.graph_edges, 0)
        self.assertGreaterEqual(plan.nodes_expanded, 1)

    def test_naive_route_leaves_diagnostics_unset(self):
        plan = naive_route(ORIGIN, DEST, self.line_of_navaids())
        self.assertIsNone(plan.graph_nodes)
        self.assertIsNone(plan.nodes_expanded)

    def test_larger_radius_never_produces_a_longer_route(self):
        navaids = self.line_of_navaids(10)
        tight = plan_route(ORIGIN, DEST, navaids, radius_nm=60)
        loose = plan_route(ORIGIN, DEST, navaids, radius_nm=300)
        self.assertLessEqual(
            loose.total_distance_nm, tight.total_distance_nm + 1e-6
        )

    def test_waypoints_progress_toward_the_destination(self):
        plan = plan_route(ORIGIN, DEST, self.line_of_navaids())
        remaining = [
            haversine_nm(wp.lat, wp.lon, DEST.lat, DEST.lon)
            for wp in plan.waypoints
        ]
        # A shortest path over a distance-weighted mesh should never
        # double back, so each hop must get closer to the destination.
        for earlier, later in zip(remaining, remaining[1:]):
            self.assertLess(later, earlier)


if __name__ == "__main__":
    unittest.main()


class TestRangeWarning(unittest.TestCase):
    """ONE SOURCE OF TRUTH, BECAUSE THERE WERE TWO AND THEY DISAGREED.

    `plan_flight` distinguished a shortfall a fuel stop solves from one
    it does not: a Cessna crossing the United States needs four stops and
    that is a trip people make, while over the Atlantic there is nowhere
    to stop. The report reimplemented the check and got it wrong, telling
    a reader that KJFK to EGLL in a 172 needed "a fuel stop" over an
    ocean with no airfields.
    """

    @classmethod
    def setUpClass(cls):
        from flight_dispatch.data_loader import (
            load_airports,
            load_navaids,
            navaids_near_route,
        )

        cls.airports = load_airports()
        cls.navaids = load_navaids()
        cls._navaids_near_route = staticmethod(navaids_near_route)

    def plan(self, origin_icao, dest_icao, aircraft_key):
        from flight_dispatch.aircraft import get_aircraft
        from flight_dispatch.data_loader import navaids_near_route
        from flight_dispatch.route import plan_route

        origin = self.airports[origin_icao]
        dest = self.airports[dest_icao]
        near = navaids_near_route(
            self.navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        return plan_route(
            origin, dest, near, aircraft=get_aircraft(aircraft_key), use_grid=True
        )

    def test_a_flight_within_range_has_no_warning(self):
        self.assertIsNone(self.plan("KPWK", "KMSP", "sr22").range_warning())

    def test_an_overland_shortfall_suggests_fuel_stops(self):
        warning = self.plan("KJFK", "KLAX", "c172").range_warning()
        self.assertIn("fuel stop", warning)
        self.assertNotIn("cannot fly", warning)

    def test_an_oceanic_shortfall_says_it_cannot_be_flown(self):
        warning = self.plan("KJFK", "EGLL", "c172").range_warning()
        self.assertIn("cannot fly this route", warning)
        self.assertIn("nowhere to refuel", warning)
        self.assertNotIn("fuel stop", warning)

    def test_a_plan_without_an_aircraft_has_no_warning(self):
        from flight_dispatch.data_loader import navaids_near_route
        from flight_dispatch.route import plan_route

        origin = self.airports["KPWK"]
        dest = self.airports["KMSP"]
        near = navaids_near_route(
            self.navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        self.assertIsNone(plan_route(origin, dest, near).range_warning())

    def test_the_tool_and_the_report_say_the_same_thing(self):
        # The bug this closes: two implementations of one judgement.
        from flight_dispatch.report import build_report
        from flight_dispatch.tools import dispatch

        plan = self.plan("KJFK", "EGLL", "c172")
        report = build_report(plan).to_dict()
        result = dispatch(
            "plan_flight",
            {"origin": "KJFK", "dest": "EGLL", "aircraft": "c172", "use_wind": False},
        )
        self.assertEqual(report["range_warning"], result["range_warning"])

    def test_the_report_renders_the_plans_own_words(self):
        from flight_dispatch.report import build_report

        page = build_report(self.plan("KJFK", "EGLL", "c172")).to_html()
        self.assertIn("nowhere to refuel", page)
        self.assertNotIn("A fuel stop is required", page)
