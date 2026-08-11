import math
import unittest

from flight_dispatch.airspace import (
    BLOCKED_TYPES,
    UNLIMITED_FT,
    AirspaceIndex,
    SpecialUseAirspace,
    _altitude_to_ft,
    airspace_near_route,
    make_airspace_cost,
)
from flight_dispatch.graph import build_mesh
from flight_dispatch.models import Navaid
from flight_dispatch.search import a_star


def box(west, south, east, north):
    """A rectangular polygon, in shapely's (lon, lat) order."""
    from shapely.geometry import Polygon

    return Polygon(
        [(west, south), (east, south), (east, north), (west, north)]
    )


def volume(name, type_code="R", lower=0.0, upper=UNLIMITED_FT, geometry=None):
    return SpecialUseAirspace(
        name=name,
        type_code=type_code,
        lower_ft=lower,
        upper_ft=upper,
        state="XX",
        times_of_use="CONTINUOUS",
        geometry=geometry if geometry is not None else box(-90.5, 39.5, -89.5, 40.5),
    )


class TestAltitudeParsing(unittest.TestCase):
    """The FAA mixes feet, flight levels and sentinels in string fields."""

    def test_plain_feet(self):
        self.assertEqual(_altitude_to_ft("4500", "FT", "MSL"), 4500.0)

    def test_flight_levels_are_hundreds_of_feet(self):
        self.assertEqual(_altitude_to_ft("180", "FL", "STD"), 18000.0)

    def test_std_without_uom_is_also_a_flight_level(self):
        self.assertEqual(_altitude_to_ft("310", None, "STD"), 31000.0)

    def test_unlimited_code(self):
        self.assertEqual(_altitude_to_ft("-9998", None, "UNLTD"), UNLIMITED_FT)

    def test_unlimited_sentinel_without_the_code(self):
        self.assertEqual(_altitude_to_ft("-9998", "FT", "MSL"), UNLIMITED_FT)

    def test_surface_is_zero(self):
        self.assertEqual(_altitude_to_ft("0", "FT", "SFC"), 0.0)

    def test_unparseable_values_fail_open_to_unlimited(self):
        # Conservative: an unreadable ceiling should block, not permit.
        self.assertEqual(_altitude_to_ft("garbage", "FT", "MSL"), UNLIMITED_FT)
        self.assertEqual(_altitude_to_ft(None, None, None), UNLIMITED_FT)


class TestSpecialUseAirspace(unittest.TestCase):
    def test_blocking_types(self):
        for code in ("P", "R", "W"):
            self.assertTrue(volume("x", code).is_blocking)

    def test_advisory_types_do_not_block(self):
        for code in ("MOA", "A", "D"):
            self.assertFalse(volume("x", code).is_blocking)

    def test_active_within_its_band(self):
        area = volume("x", lower=5000, upper=18000)
        self.assertTrue(area.active_at(10000))
        self.assertTrue(area.active_at(5000))
        self.assertTrue(area.active_at(18000))

    def test_inactive_outside_its_band(self):
        area = volume("x", lower=5000, upper=18000)
        self.assertFalse(area.active_at(4999))
        self.assertFalse(area.active_at(35000))

    def test_describe_mentions_surface_and_unlimited(self):
        text = volume("R-1", lower=0, upper=UNLIMITED_FT).describe()
        self.assertIn("surface", text)
        self.assertIn("unlimited", text)


class TestAirspaceIndex(unittest.TestCase):
    def setUp(self):
        # A 1-degree box centred on 40N, 90W.
        self.low = volume("LOW", lower=0, upper=8000)
        self.high = volume("HIGH", lower=20000, upper=UNLIMITED_FT)

    def test_altitude_filter_selects_the_relevant_volumes(self):
        self.assertEqual(len(AirspaceIndex([self.low, self.high], 5000)), 1)
        self.assertEqual(len(AirspaceIndex([self.low, self.high], 30000)), 1)
        self.assertEqual(len(AirspaceIndex([self.low, self.high], 15000)), 0)

    def test_no_altitude_filter_keeps_everything(self):
        self.assertEqual(len(AirspaceIndex([self.low, self.high])), 2)

    def test_leg_through_a_volume_is_detected(self):
        index = AirspaceIndex([self.low], 5000)
        # West to east straight through the box at 40N.
        self.assertTrue(index.blocks(40.0, -91.0, 40.0, -89.0))

    def test_leg_clear_of_a_volume_is_not_blocked(self):
        index = AirspaceIndex([self.low], 5000)
        # Well north of the box.
        self.assertFalse(index.blocks(45.0, -91.0, 45.0, -89.0))

    def test_leg_blocked_at_one_altitude_is_clear_at_another(self):
        low_index = AirspaceIndex([self.low], 5000)
        high_index = AirspaceIndex([self.low], 35000)
        self.assertTrue(low_index.blocks(40.0, -91.0, 40.0, -89.0))
        self.assertFalse(high_index.blocks(40.0, -91.0, 40.0, -89.0))

    def test_crossings_names_the_volumes(self):
        index = AirspaceIndex([self.low], 5000)
        hits = index.crossings(40.0, -91.0, 40.0, -89.0)
        self.assertEqual([h.name for h in hits], ["LOW"])

    def test_advisory_airspace_is_crossed_but_not_blocking(self):
        moa = volume("MOA-1", "MOA", 0, 20000)
        index = AirspaceIndex([moa], 10000)
        self.assertEqual(len(index.crossings(40.0, -91.0, 40.0, -89.0)), 1)
        self.assertFalse(index.blocks(40.0, -91.0, 40.0, -89.0))

    def test_containing_finds_a_point_inside(self):
        index = AirspaceIndex([self.low], 5000)
        self.assertEqual([v.name for v in index.containing(40.0, -90.0)], ["LOW"])
        self.assertEqual(index.containing(50.0, -90.0), [])

    def test_empty_index_blocks_nothing(self):
        index = AirspaceIndex([], 5000)
        self.assertFalse(index.blocks(40.0, -91.0, 40.0, -89.0))
        self.assertEqual(index.crossings(40.0, -91.0, 40.0, -89.0), [])


class TestAirspaceCostFunction(unittest.TestCase):
    def setUp(self):
        # Three navaids in a line west to east, with a restricted box
        # sitting on the middle one, plus a clear detour to the north.
        self.nodes = [
            Navaid("W", "West", "VOR", 40.0, -91.0),
            Navaid("M", "Middle", "VOR", 40.0, -90.0),
            Navaid("N", "North", "VOR", 41.5, -90.0),
            Navaid("E", "East", "VOR", 40.0, -89.0),
        ]
        self.graph = build_mesh(self.nodes, radius_nm=200, min_neighbors=2)
        self.index = AirspaceIndex(
            [volume("R-TEST", geometry=box(-90.3, 39.7, -89.7, 40.3))], 8000
        )

    def test_blocked_edge_costs_infinity(self):
        cost = make_airspace_cost(self.index)
        # W -> M runs into the box.
        self.assertEqual(cost(self.graph, 0, 1, 60.0), math.inf)

    def test_clear_edge_keeps_its_base_cost(self):
        cost = make_airspace_cost(self.index)
        self.assertEqual(cost(self.graph, 0, 2, 120.0), 120.0)

    def test_finite_penalty_scales_instead_of_blocking(self):
        cost = make_airspace_cost(self.index, penalty_factor=10.0)
        self.assertEqual(cost(self.graph, 0, 1, 60.0), 600.0)

    def test_wraps_an_existing_cost_function(self):
        def double(graph, i, j, base):
            return base * 2

        cost = make_airspace_cost(self.index, double)
        self.assertEqual(cost(self.graph, 0, 2, 100.0), 200.0)  # clear edge
        self.assertEqual(cost(self.graph, 0, 1, 100.0), math.inf)  # blocked

    def test_astar_routes_around_blocked_airspace(self):
        unrestricted = a_star(self.graph, 0, 3)
        avoiding = a_star(self.graph, 0, 3, cost_function=make_airspace_cost(self.index))

        self.assertTrue(avoiding.found)
        # The direct path through the middle is no longer available.
        self.assertNotIn(1, avoiding.path)
        self.assertIn(2, avoiding.path)  # detoured north
        self.assertGreater(avoiding.cost, unrestricted.cost)

    def test_fully_blocked_graph_reports_no_route(self):
        everywhere = AirspaceIndex([volume("BIG", geometry=box(-180, -90, 180, 90))], 8000)
        result = a_star(self.graph, 0, 3, cost_function=make_airspace_cost(everywhere))
        self.assertFalse(result.found)


class TestRegionPrefilter(unittest.TestCase):
    def test_keeps_nearby_volumes(self):
        near = volume("NEAR", geometry=box(-90.5, 39.5, -89.5, 40.5))
        result = airspace_near_route([near], 40.0, -91.0, 40.0, -89.0)
        self.assertEqual([v.name for v in result], ["NEAR"])

    def test_discards_distant_volumes(self):
        far = volume("FAR", geometry=box(-120.5, 34.5, -119.5, 35.5))
        result = airspace_near_route([far], 40.0, -91.0, 40.0, -89.0)
        self.assertEqual(result, [])

    def test_margin_widens_the_catchment(self):
        edge = volume("EDGE", geometry=box(-90.5, 44.0, -89.5, 45.0))
        tight = airspace_near_route([edge], 40.0, -91.0, 40.0, -89.0, margin_deg=1.0)
        loose = airspace_near_route([edge], 40.0, -91.0, 40.0, -89.0, margin_deg=6.0)
        self.assertEqual(tight, [])
        self.assertEqual([v.name for v in loose], ["EDGE"])


class TestBlockedTypes(unittest.TestCase):
    def test_prohibited_and_restricted_are_blocked(self):
        self.assertIn("P", BLOCKED_TYPES)
        self.assertIn("R", BLOCKED_TYPES)

    def test_moas_are_not_blocked(self):
        # Legal to transit VFR, so advisory rather than prohibited.
        self.assertNotIn("MOA", BLOCKED_TYPES)


if __name__ == "__main__":
    unittest.main()
