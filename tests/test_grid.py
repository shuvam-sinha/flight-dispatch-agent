"""Tests for the virtual routing grid.

Navaids are ground stations, so oceanic routes had nothing to route over
and ran 133-147% of the direct distance. The grid supplies lat/lon fixes
where navaids do not reach -- and, importantly, stays out of the way
where they do.
"""

import unittest

from flight_dispatch.geo import destination_point, haversine_nm, initial_bearing_deg
from flight_dispatch.grid import (
    count_grid_points,
    fill_navaid_gaps,
    make_grid_point,
    oceanic_name,
    routing_grid,
    waypoints_for_route,
)
from flight_dispatch.models import Navaid

# A transatlantic pair, the case the grid exists for.
KJFK = (40.6413, -73.7781)
EGLL = (51.4706, -0.4619)


class TestDestinationPoint(unittest.TestCase):
    """The new geometry the grid needed: project a point, don't measure one."""

    def test_north_increases_latitude_by_a_degree_per_60nm(self):
        lat, lon = destination_point(0.0, 0.0, 0.0, 60.0)
        self.assertAlmostEqual(lat, 1.0, delta=0.01)
        self.assertAlmostEqual(lon, 0.0, delta=0.01)

    def test_east_along_the_equator(self):
        lat, lon = destination_point(0.0, 0.0, 90.0, 60.0)
        self.assertAlmostEqual(lat, 0.0, delta=0.01)
        self.assertAlmostEqual(lon, 1.0, delta=0.01)

    def test_round_trips_with_haversine(self):
        lat, lon = destination_point(42.0, -88.0, 037.0, 250.0)
        self.assertAlmostEqual(haversine_nm(42.0, -88.0, lat, lon), 250.0, delta=0.5)

    def test_round_trips_with_bearing(self):
        lat, lon = destination_point(42.0, -88.0, 137.0, 200.0)
        self.assertAlmostEqual(
            initial_bearing_deg(42.0, -88.0, lat, lon), 137.0, delta=0.5
        )

    def test_longitude_stays_in_range_across_the_antimeridian(self):
        _, lon = destination_point(60.0, 179.0, 90.0, 300.0)
        self.assertTrue(-180 <= lon <= 180)


class TestOceanicNaming(unittest.TestCase):
    """Real oceanic waypoints are named 56N020W. The grid uses the same
    convention so its points look like what they represent."""

    def test_north_west(self):
        self.assertEqual(oceanic_name(56.0, -20.0), "56N020W")

    def test_south_east(self):
        self.assertEqual(oceanic_name(-33.0, 151.0), "33S151E")

    def test_latitude_is_two_digits_longitude_three(self):
        self.assertEqual(oceanic_name(5.0, 5.0), "05N005E")

    def test_rounds_to_whole_degrees(self):
        self.assertEqual(oceanic_name(56.4, -19.6), "56N020W")


class TestGridGeneration(unittest.TestCase):
    def test_generates_points_along_a_long_route(self):
        grid = routing_grid(*KJFK, *EGLL)
        self.assertGreater(len(grid), 50)

    def test_short_routes_get_no_grid(self):
        # Shorter than one column spacing -- nothing to interpolate.
        self.assertEqual(routing_grid(42.0, -88.0, 42.5, -88.0), [])

    def test_points_are_marked_as_generated(self):
        for point in routing_grid(*KJFK, *EGLL):
            self.assertEqual(point.navaid_type, "GRID")

    def test_idents_are_unique(self):
        grid = routing_grid(*KJFK, *EGLL)
        self.assertEqual(len(grid), len({p.ident for p in grid}))

    def test_points_lie_near_the_course(self):
        # Every point should be within the configured lateral reach.
        from flight_dispatch.geo import cross_and_along_track_nm

        for point in routing_grid(*KJFK, *EGLL, lanes=2, lane_spacing_nm=200):
            cross, _ = cross_and_along_track_nm(point.lat, point.lon, *KJFK, *EGLL)
            self.assertLess(abs(cross), 500)  # 2 lanes x 200 nm, plus rounding

    def test_lanes_control_lateral_spread(self):
        narrow = routing_grid(*KJFK, *EGLL, lanes=1)
        wide = routing_grid(*KJFK, *EGLL, lanes=3)
        self.assertLess(len(narrow), len(wide))

    def test_spacing_controls_column_count(self):
        coarse = routing_grid(*KJFK, *EGLL, spacing_nm=400)
        fine = routing_grid(*KJFK, *EGLL, spacing_nm=150)
        self.assertLess(len(coarse), len(fine))


class TestGapFilling(unittest.TestCase):
    """The rule that keeps overland routes honest: a generated fix over
    Nebraska corresponds to nothing on any chart, so where real navaids
    exist the grid should defer to them."""

    def test_grid_points_near_a_navaid_are_dropped(self):
        point = make_grid_point(40.0, -90.0)
        nearby = Navaid("REAL", "Real VOR", "VOR", 40.1, -90.0)  # ~6 nm away
        self.assertEqual(fill_navaid_gaps([point], [nearby]), [])

    def test_grid_points_far_from_any_navaid_are_kept(self):
        point = make_grid_point(40.0, -30.0)  # mid-Atlantic
        far = Navaid("REAL", "Real VOR", "VOR", 40.0, -90.0)
        self.assertEqual(len(fill_navaid_gaps([point], [far])), 1)

    def test_everything_is_kept_when_there_are_no_navaids(self):
        points = [make_grid_point(40.0, -30.0), make_grid_point(41.0, -35.0)]
        self.assertEqual(len(fill_navaid_gaps(points, [])), 2)

    def test_clearance_distance_is_configurable(self):
        point = make_grid_point(40.0, -90.0)
        navaid = Navaid("REAL", "Real", "VOR", 41.0, -90.0)  # 60 nm
        self.assertEqual(len(fill_navaid_gaps([point], [navaid], clearance_nm=30)), 1)
        self.assertEqual(len(fill_navaid_gaps([point], [navaid], clearance_nm=120)), 0)


class TestWaypointsForRoute(unittest.TestCase):
    def test_returns_navaids_unchanged_when_disabled(self):
        navaids = [Navaid("A", "A", "VOR", 40.0, -90.0)]
        self.assertEqual(
            waypoints_for_route(*KJFK, *EGLL, navaids, use_grid=False), navaids
        )

    def test_adds_grid_points_when_enabled(self):
        navaids = [Navaid("A", "A", "VOR", 40.0, -90.0)]
        result = waypoints_for_route(*KJFK, *EGLL, navaids, use_grid=True)
        self.assertGreater(len(result), len(navaids))

    def test_original_navaids_are_preserved(self):
        navaids = [Navaid("A", "A", "VOR", 40.0, -90.0)]
        result = waypoints_for_route(*KJFK, *EGLL, navaids, use_grid=True)
        self.assertIn(navaids[0], result)

    def test_count_grid_points(self):
        mixed = [
            Navaid("A", "A", "VOR", 40.0, -90.0),
            make_grid_point(40.0, -30.0),
            make_grid_point(41.0, -35.0),
        ]
        self.assertEqual(count_grid_points(mixed), 2)


class TestRoutingImprovement(unittest.TestCase):
    """The measurable point of the whole module. Slow -- these build real
    meshes from the full dataset."""

    @classmethod
    def setUpClass(cls):
        from flight_dispatch.data_loader import load_airports, load_navaids

        cls.airports = load_airports()
        cls.navaids = load_navaids()

    def plan_both(self, origin_icao, dest_icao):
        from flight_dispatch.data_loader import navaids_near_route
        from flight_dispatch.route import plan_route

        origin = self.airports[origin_icao]
        dest = self.airports[dest_icao]
        near = navaids_near_route(
            self.navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        return (
            plan_route(origin, dest, near),
            plan_route(origin, dest, near, use_grid=True),
        )

    def efficiency(self, plan):
        return plan.total_distance_nm / plan.direct_distance_nm

    def test_transatlantic_improves_dramatically(self):
        without, with_grid = self.plan_both("KJFK", "EGLL")
        self.assertGreater(self.efficiency(without), 1.25)   # was 132.7%
        self.assertLess(self.efficiency(with_grid), 1.05)    # now ~101%

    def test_mid_atlantic_improves_dramatically(self):
        without, with_grid = self.plan_both("LPPT", "TNCM")
        self.assertGreater(self.efficiency(without), 1.30)   # was 146.9%
        self.assertLess(self.efficiency(with_grid), 1.10)

    def test_dense_overland_route_is_unchanged(self):
        # The grid must defer where real navaids exist.
        without, with_grid = self.plan_both("KPWK", "KMSP")
        self.assertAlmostEqual(
            without.total_distance_nm, with_grid.total_distance_nm, delta=1.0
        )

    def test_transcontinental_overland_is_unchanged(self):
        without, with_grid = self.plan_both("KJFK", "KLAX")
        self.assertAlmostEqual(
            without.total_distance_nm, with_grid.total_distance_nm, delta=5.0
        )

    def test_oceanic_route_reports_its_generated_waypoints(self):
        _, with_grid = self.plan_both("KJFK", "EGLL")
        self.assertGreater(with_grid.grid_waypoints_used, 0)

    def test_overland_route_uses_no_generated_waypoints(self):
        _, with_grid = self.plan_both("KPWK", "KMSP")
        self.assertEqual(with_grid.grid_waypoints_used, 0)

    def test_grid_is_off_by_default(self):
        without, _ = self.plan_both("KPWK", "KMSP")
        self.assertIsNone(without.grid_waypoints_used)


if __name__ == "__main__":
    unittest.main()
