import math
import unittest

from flight_dispatch.wind import (
    CALM,
    ConstantWindSource,
    NoWind,
    PRESSURE_LEVELS,
    Wind,
    altitude_for_level,
    ground_speed_kt,
    nearest_pressure_level,
)

TAS = 120.0
EAST = 90.0  # flying east


class TestWindConvention(unittest.TestCase):
    """Wind direction is where it blows FROM. This trips everyone up."""

    def test_blowing_towards_is_the_reciprocal(self):
        self.assertAlmostEqual(Wind(270, 30, 8000).blowing_towards_deg, 90.0)
        self.assertAlmostEqual(Wind(90, 30, 8000).blowing_towards_deg, 270.0)

    def test_reciprocal_wraps_at_360(self):
        self.assertAlmostEqual(Wind(200, 10, 0).blowing_towards_deg, 20.0)


class TestWindComponents(unittest.TestCase):
    def test_wind_from_dead_ahead_is_all_headwind(self):
        head, cross = Wind(EAST, 30, 8000).components(EAST)
        self.assertAlmostEqual(head, 30.0, places=6)
        self.assertAlmostEqual(cross, 0.0, places=6)

    def test_wind_from_behind_is_negative_headwind(self):
        head, cross = Wind(270, 30, 8000).components(EAST)
        self.assertAlmostEqual(head, -30.0, places=6)
        self.assertAlmostEqual(cross, 0.0, places=6)

    def test_wind_from_the_right_is_positive_crosswind(self):
        head, cross = Wind(180, 30, 8000).components(EAST)
        self.assertAlmostEqual(head, 0.0, places=6)
        self.assertAlmostEqual(cross, 30.0, places=6)

    def test_wind_from_the_left_is_negative_crosswind(self):
        _, cross = Wind(360, 30, 8000).components(EAST)
        self.assertAlmostEqual(cross, -30.0, places=6)

    def test_quartering_wind_splits_evenly_at_45_degrees(self):
        head, cross = Wind(45, 30, 8000).components(EAST)
        expected = 30.0 / math.sqrt(2)
        self.assertAlmostEqual(head, expected, places=4)
        self.assertAlmostEqual(abs(cross), expected, places=4)

    def test_components_form_the_original_speed(self):
        head, cross = Wind(123, 42, 8000).components(EAST)
        self.assertAlmostEqual(math.hypot(head, cross), 42.0, places=6)


class TestGroundSpeed(unittest.TestCase):
    def test_calm_air_gives_true_airspeed(self):
        self.assertAlmostEqual(ground_speed_kt(TAS, EAST, CALM), TAS)

    def test_tailwind_adds(self):
        self.assertAlmostEqual(ground_speed_kt(TAS, EAST, Wind(270, 30, 0)), 150.0)

    def test_headwind_subtracts(self):
        self.assertAlmostEqual(ground_speed_kt(TAS, EAST, Wind(90, 30, 0)), 90.0)

    def test_pure_crosswind_still_costs_speed(self):
        # The counter-intuitive one: holding a track against drift spends
        # part of the airspeed, so a 90-degree wind is not free.
        speed = ground_speed_kt(TAS, EAST, Wind(180, 30, 0))
        self.assertLess(speed, TAS)
        # 120 * cos(asin(30/120)) = 116.19
        self.assertAlmostEqual(speed, 116.19, places=1)

    def test_crosswind_from_either_side_costs_the_same(self):
        left = ground_speed_kt(TAS, EAST, Wind(360, 30, 0))
        right = ground_speed_kt(TAS, EAST, Wind(180, 30, 0))
        self.assertAlmostEqual(left, right, places=9)

    def test_headwind_equal_to_airspeed_stops_progress(self):
        self.assertAlmostEqual(ground_speed_kt(TAS, EAST, Wind(90, 120, 0)), 0.1)

    def test_headwind_exceeding_airspeed_never_goes_negative(self):
        # Physically the aircraft moves backwards over the ground; for
        # routing we only need "effectively impassable".
        self.assertGreater(ground_speed_kt(TAS, EAST, Wind(90, 200, 0)), 0)

    def test_crosswind_exceeding_airspeed_is_survivable(self):
        # asin() would be undefined here; must not raise.
        speed = ground_speed_kt(TAS, EAST, Wind(180, 300, 0))
        self.assertGreater(speed, 0)
        self.assertLess(speed, 1)

    def test_faster_aircraft_are_less_affected(self):
        wind = Wind(90, 30, 0)  # 30 kt headwind
        slow_loss = 1 - ground_speed_kt(120, EAST, wind) / 120
        fast_loss = 1 - ground_speed_kt(450, EAST, wind) / 450
        self.assertGreater(slow_loss, fast_loss)
        self.assertAlmostEqual(slow_loss, 0.25, places=2)

    def test_tailwind_on_one_course_is_a_headwind_on_the_reciprocal(self):
        wind = Wind(270, 40, 0)
        eastbound = ground_speed_kt(TAS, 90, wind)
        westbound = ground_speed_kt(TAS, 270, wind)
        self.assertAlmostEqual(eastbound, TAS + 40)
        self.assertAlmostEqual(westbound, TAS - 40)


class TestTheCP3Thesis(unittest.TestCase):
    """A longer route can be faster. This is the whole point of CP3."""

    def test_longer_route_with_tailwind_beats_shorter_into_headwind(self):
        direct_hours = 300 / ground_speed_kt(TAS, EAST, Wind(90, 30, 0))
        detour_hours = 340 / ground_speed_kt(TAS, EAST, Wind(270, 25, 0))
        self.assertLess(detour_hours, direct_hours)
        self.assertGreater(direct_hours - detour_hours, 0.9)  # about an hour


class TestPressureLevels(unittest.TestCase):
    def test_ga_cruise_maps_to_700hpa(self):
        self.assertEqual(nearest_pressure_level(8000), 700)

    def test_jet_cruise_maps_to_the_jet_stream_levels(self):
        self.assertEqual(nearest_pressure_level(35000), 250)
        self.assertEqual(nearest_pressure_level(41000), 200)

    def test_sea_level_maps_to_the_lowest_level(self):
        self.assertEqual(nearest_pressure_level(0), 1000)

    def test_every_level_round_trips(self):
        for level, altitude in PRESSURE_LEVELS:
            with self.subTest(level):
                self.assertEqual(altitude_for_level(level), altitude)
                self.assertEqual(nearest_pressure_level(altitude), level)

    def test_absurd_altitude_clamps_to_the_highest_level(self):
        self.assertEqual(nearest_pressure_level(200000), 100)


class TestWindSources(unittest.TestCase):
    def test_constant_source_reports_the_same_wind_everywhere(self):
        source = ConstantWindSource(direction_deg=270, speed_kt=50)
        here = source.wind_at(0, 0, 8000)
        far = source.wind_at(60, 120, 8000)
        self.assertEqual(here.direction_deg, far.direction_deg)
        self.assertEqual(here.speed_kt, 50)

    def test_constant_source_batches(self):
        source = ConstantWindSource(270, 50)
        winds = source.wind_at_many([(0, 0), (1, 1), (2, 2)], 8000)
        self.assertEqual(len(winds), 3)

    def test_no_wind_is_calm(self):
        self.assertEqual(NoWind().wind_at(42, -88, 35000).speed_kt, 0.0)

    def test_no_wind_gives_true_airspeed(self):
        wind = NoWind().wind_at(42, -88, 35000)
        self.assertAlmostEqual(ground_speed_kt(TAS, EAST, wind), TAS)


if __name__ == "__main__":
    unittest.main()


class TestRateLimitBackoff(unittest.TestCase):
    """THE BUG THESE COVER.

    Open-Meteo meters work, not requests: a call costs roughly
    locations x variables x forecast days. Batching 50 coordinates made
    each request fifty times as expensive, and a KJFK-KLAX mesh needs 20
    of them -- about 1,000 units against a 600-per-minute allowance. The
    server answered "Minutely API request limit exceeded", the backoff
    waited 2s then 4s then gave up, and every long route silently planned
    in still air.
    """

    class FakeResponse:
        def __init__(self, headers=None, payload=None, text=""):
            self.headers = headers or {}
            self._payload = payload
            self.text = text

        def json(self):
            if self._payload is None:
                raise ValueError("no json")
            return self._payload

    def source(self):
        from flight_dispatch.wind_openmeteo import OpenMeteoWindSource

        return OpenMeteoWindSource()

    def test_a_minutely_limit_waits_out_the_minute(self):
        from flight_dispatch.wind_openmeteo import MINUTELY_RESET_S

        response = self.FakeResponse(
            payload={"reason": "Minutely API request limit exceeded."}
        )
        self.assertEqual(self.source()._retry_wait(response, 0), MINUTELY_RESET_S)

    def test_the_old_backoff_could_not_have_worked(self):
        # Two seconds against a sixty-second window.
        response = self.FakeResponse(
            payload={"reason": "Minutely API request limit exceeded."}
        )
        self.assertGreater(self.source()._retry_wait(response, 0), 30.0)

    def test_retry_after_header_wins_when_present(self):
        response = self.FakeResponse(headers={"Retry-After": "5"})
        self.assertEqual(self.source()._retry_wait(response, 0), 5.0)

    def test_unparseable_retry_after_falls_through(self):
        response = self.FakeResponse(headers={"Retry-After": "soon"})
        self.assertGreater(self.source()._retry_wait(response, 0), 0)

    def test_other_limits_back_off_exponentially(self):
        response = self.FakeResponse(payload={"reason": "Hourly limit exceeded."})
        wait_first = self.source()._retry_wait(response, 0)
        wait_later = self.source()._retry_wait(response, 2)
        self.assertLess(wait_first, wait_later)

    def test_never_sleeps_longer_than_the_cap(self):
        from flight_dispatch.wind_openmeteo import MAX_RETRY_WAIT_S

        response = self.FakeResponse(headers={"Retry-After": "99999"})
        self.assertLessEqual(self.source()._retry_wait(response, 0), MAX_RETRY_WAIT_S)

    def test_a_body_that_is_not_json_does_not_crash(self):
        response = self.FakeResponse(text="rate limited")
        self.assertGreater(self.source()._retry_wait(response, 0), 0)


class TestRequestCost(unittest.TestCase):
    """Every variable and every forecast day multiplies the quota cost."""

    def captured_params(self, **kwargs):
        from unittest.mock import patch

        from flight_dispatch.wind_openmeteo import OpenMeteoWindSource

        source = OpenMeteoWindSource(**kwargs)
        with patch.object(source, "_get_with_retry", return_value=None) as get:
            source._fetch_batch([(40.0, -90.0)], 250)
            return get.call_args[0][0]

    def test_only_one_forecast_day_is_requested(self):
        # The second day was fetched and thrown away -- only one hour is
        # ever read, and it is today's.
        self.assertEqual(self.captured_params()["forecast_days"], 1)

    def test_routing_does_not_request_temperature(self):
        params = self.captured_params(want_temperature=False)
        self.assertNotIn("temperature", params["hourly"])
        self.assertIn("wind_speed", params["hourly"])
        self.assertIn("wind_direction", params["hourly"])

    def test_single_point_lookups_still_get_temperature(self):
        # get_winds_aloft reports it, so it is part of that answer.
        self.assertIn("temperature", self.captured_params()["hourly"])

    def test_routing_asks_for_two_variables_not_three(self):
        routing = self.captured_params(want_temperature=False)["hourly"].split(",")
        display = self.captured_params()["hourly"].split(",")
        self.assertEqual(len(routing), 2)
        self.assertEqual(len(display), 3)
