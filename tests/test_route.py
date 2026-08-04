import unittest

from flight_dispatch.models import Airport, Navaid
from flight_dispatch.route import naive_route

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


if __name__ == "__main__":
    unittest.main()
