"""Tests for the climb/cruise/descent split.

Flight time used to be route distance over cruise speed, which bills the
first and last twenty minutes of every flight at a speed the aircraft has
not reached yet. KORD to KMIA in a 777 came out at 2h07m against an
airline block time near three hours.
"""

import unittest

from flight_dispatch.aircraft import get_aircraft
from flight_dispatch.phases import PHASE_MODELS, flight_phases, phase_model

C172 = get_aircraft("c172")
B738 = get_aircraft("b738")
B77W = get_aircraft("b77w")


class TestPhaseModels(unittest.TestCase):
    def test_every_category_has_a_model(self):
        from flight_dispatch.aircraft import AIRCRAFT

        for profile in AIRCRAFT.values():
            self.assertIn(profile.category, PHASE_MODELS, profile.key)

    def test_a_piston_single_climbs_slower_than_a_jet(self):
        self.assertLess(
            phase_model(C172).climb_rate_fpm, phase_model(B738).climb_rate_fpm
        )

    def test_jets_burn_more_climbing_and_less_descending(self):
        model = phase_model(B738)
        self.assertGreater(model.climb_burn_factor, 1.0)
        self.assertLess(model.descent_burn_factor, 1.0)


class TestSegmentsAddUp(unittest.TestCase):
    """Whatever else changes, the three phases must tile the route."""

    def test_distances_sum_to_the_route(self):
        result = flight_phases(B738, 1000.0, 450.0)
        total = (
            result.climb_distance_nm
            + result.cruise_distance_nm
            + result.descent_distance_nm
        )
        self.assertAlmostEqual(total, 1000.0, delta=0.01)

    def test_short_flight_distances_also_sum_to_the_route(self):
        result = flight_phases(B738, 120.0, 450.0)
        total = (
            result.climb_distance_nm
            + result.cruise_distance_nm
            + result.descent_distance_nm
        )
        self.assertAlmostEqual(total, 120.0, delta=0.01)

    def test_times_sum_to_the_total(self):
        result = flight_phases(B77W, 3000.0, 490.0)
        self.assertAlmostEqual(
            result.total_time_hours,
            result.climb_time_hours
            + result.cruise_time_hours
            + result.descent_time_hours,
        )


class TestAgainstCruiseOnlyArithmetic(unittest.TestCase):
    def test_flight_takes_longer_than_distance_over_cruise_speed(self):
        # The whole point: the ends are slower than the middle.
        result = flight_phases(B77W, 1042.0, 495.0)
        self.assertGreater(result.total_time_hours, 1042.0 / 495.0)

    def test_the_difference_is_minutes_not_hours(self):
        # A correction this size should be ~10 min on a 2 h flight. If it
        # ever balloons, the rates are wrong.
        result = flight_phases(B77W, 1042.0, 495.0)
        added_minutes = (result.total_time_hours - 1042.0 / 495.0) * 60
        self.assertTrue(3 < added_minutes < 25, added_minutes)

    def test_long_flights_are_barely_affected(self):
        # Climb and descent are a fixed cost, so they matter less the
        # longer the flight.
        short = flight_phases(B738, 300.0, 450.0)
        long = flight_phases(B738, 3000.0, 450.0)
        short_penalty = short.total_time_hours / (300.0 / 450.0)
        long_penalty = long.total_time_hours / (3000.0 / 450.0)
        self.assertGreater(short_penalty, long_penalty)


class TestShortFlights(unittest.TestCase):
    """A 777 cannot reach FL370 in 150 nm and come back down."""

    def test_short_hop_does_not_reach_planned_altitude(self):
        result = flight_phases(B77W, 150.0, 490.0)
        self.assertFalse(result.reached_planned_altitude)
        self.assertLess(result.cruise_altitude_ft, B77W.cruise_altitude_ft)

    def test_short_hop_has_no_negative_cruise(self):
        # The bug this guards: subtracting climb and descent from a route
        # too short for them gives a negative cruise leg, and a flight
        # that arrives before it departs.
        for distance in (5.0, 20.0, 80.0, 200.0):
            result = flight_phases(B77W, distance, 490.0)
            self.assertGreaterEqual(result.cruise_distance_nm, 0.0, distance)
            self.assertGreater(result.total_time_hours, 0.0, distance)

    def test_long_enough_flight_reaches_altitude(self):
        result = flight_phases(B77W, 2000.0, 490.0)
        self.assertTrue(result.reached_planned_altitude)
        self.assertEqual(result.cruise_altitude_ft, B77W.cruise_altitude_ft)

    def test_altitude_scales_with_available_distance(self):
        shorter = flight_phases(B738, 100.0, 450.0)
        longer = flight_phases(B738, 250.0, 450.0)
        self.assertLess(shorter.cruise_altitude_ft, longer.cruise_altitude_ft)


class TestFieldElevation(unittest.TestCase):
    def test_departing_high_shortens_the_climb(self):
        sea_level = flight_phases(B738, 800.0, 450.0, origin_elevation_ft=0)
        denver = flight_phases(B738, 800.0, 450.0, origin_elevation_ft=5400)
        self.assertLess(denver.climb_time_hours, sea_level.climb_time_hours)

    def test_a_lower_cruise_level_means_less_climbing(self):
        high = flight_phases(B738, 800.0, 450.0, cruise_altitude_ft=39000)
        low = flight_phases(B738, 800.0, 450.0, cruise_altitude_ft=24000)
        self.assertLess(low.climb_distance_nm, high.climb_distance_nm)


class TestFuel(unittest.TestCase):
    def test_climb_burns_harder_than_cruise(self):
        result = flight_phases(B738, 1000.0, 450.0)
        climb_gph = result.climb_fuel_gal / result.climb_time_hours
        cruise_gph = result.cruise_fuel_gal / result.cruise_time_hours
        self.assertGreater(climb_gph, cruise_gph)

    def test_descent_burns_less_than_cruise(self):
        result = flight_phases(B738, 1000.0, 450.0)
        descent_gph = result.descent_fuel_gal / result.descent_time_hours
        cruise_gph = result.cruise_fuel_gal / result.cruise_time_hours
        self.assertLess(descent_gph, cruise_gph)

    def test_total_fuel_excludes_reserve(self):
        # The reserve is the caller's to add -- it was already added
        # there, and adding it twice is a silent 45 minutes of fuel.
        result = flight_phases(B738, 1000.0, 450.0)
        self.assertAlmostEqual(
            result.total_fuel_gal,
            result.climb_fuel_gal + result.cruise_fuel_gal + result.descent_fuel_gal,
        )


class TestWind(unittest.TestCase):
    def test_a_tailwind_shortens_the_cruise_but_not_the_climb(self):
        still = flight_phases(B738, 1500.0, B738.cruise_tas_kt)
        tailwind = flight_phases(B738, 1500.0, B738.cruise_tas_kt + 80)
        self.assertLess(tailwind.cruise_time_hours, still.cruise_time_hours)
        # Climb and descent are computed in still air -- documented, and
        # this is the assertion that says so out loud.
        self.assertEqual(tailwind.climb_time_hours, still.climb_time_hours)

    def test_zero_ground_speed_falls_back_to_true_airspeed(self):
        result = flight_phases(B738, 1000.0, 0.0)
        self.assertGreater(result.cruise_time_hours, 0.0)


class TestIntegrationWithRoutePlan(unittest.TestCase):
    """Slow -- builds real meshes."""

    @classmethod
    def setUpClass(cls):
        from flight_dispatch.data_loader import load_airports, load_navaids

        cls.airports = load_airports()
        cls.navaids = load_navaids()

    def plan(self, origin_icao, dest_icao, aircraft_key):
        from flight_dispatch.data_loader import navaids_near_route
        from flight_dispatch.route import plan_route

        origin = self.airports[origin_icao]
        dest = self.airports[dest_icao]
        near = navaids_near_route(
            self.navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        return plan_route(origin, dest, near, aircraft=get_aircraft(aircraft_key))

    def test_plan_carries_a_profile(self):
        plan = self.plan("KORD", "KMIA", "b77w")
        self.assertIsNotNone(plan.phases)
        self.assertAlmostEqual(
            plan.phases.total_time_hours, plan.ete_hours, delta=1e-9
        )

    def test_ete_is_longer_than_the_old_cruise_only_figure(self):
        plan = self.plan("KORD", "KMIA", "b77w")
        cruise_only = plan.total_distance_nm / plan.aircraft.cruise_tas_kt
        self.assertGreater(plan.ete_hours, cruise_only)

    def test_phase_distances_match_the_route(self):
        plan = self.plan("KORD", "KMIA", "b77w")
        total = (
            plan.phases.climb_distance_nm
            + plan.phases.cruise_distance_nm
            + plan.phases.descent_distance_nm
        )
        self.assertAlmostEqual(total, plan.total_distance_nm, delta=0.1)

    def test_fuel_uses_the_profile_not_a_flat_cruise_burn(self):
        plan = self.plan("KORD", "KMIA", "b77w")
        flat = plan.aircraft.fuel_required_gal(plan.ete_hours)
        self.assertNotAlmostEqual(plan.fuel_required_gal, flat, delta=1.0)

    def test_field_elevation_is_taken_from_the_airports(self):
        # KDEN sits at 5,400 ft, so there is less climbing to do.
        denver = self.plan("KDEN", "KMCI", "b738")
        self.assertGreater(denver.origin.elevation_ft, 5000)
        self.assertLess(
            denver.phases.climb_time_hours,
            (denver.aircraft.cruise_altitude_ft / 2000.0) / 60.0,
        )

    def test_a_plan_without_an_aircraft_has_no_profile(self):
        from flight_dispatch.data_loader import navaids_near_route
        from flight_dispatch.route import plan_route

        origin = self.airports["KPWK"]
        dest = self.airports["KMSP"]
        near = navaids_near_route(
            self.navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        self.assertIsNone(plan_route(origin, dest, near).phases)


if __name__ == "__main__":
    unittest.main()
