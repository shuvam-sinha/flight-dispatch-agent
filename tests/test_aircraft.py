import unittest

from flight_dispatch.aircraft import (
    AIRCRAFT,
    DEFAULT_AIRCRAFT,
    AircraftProfile,
    aircraft_by_category,
    get_aircraft,
)

C172 = AIRCRAFT["c172"]
B738 = AIRCRAFT["b738"]
B788 = AIRCRAFT["b788"]


class TestCatalogue(unittest.TestCase):
    def test_lookup_is_case_and_space_insensitive(self):
        self.assertIs(get_aircraft("  B738 "), B738)

    def test_unknown_key_raises_with_a_hint(self):
        with self.assertRaises(KeyError) as ctx:
            get_aircraft("concorde")
        self.assertIn("--list-aircraft", str(ctx.exception))

    def test_default_is_the_c172(self):
        self.assertIs(DEFAULT_AIRCRAFT, C172)

    def test_keys_match_their_dict_entries(self):
        for key, profile in AIRCRAFT.items():
            self.assertEqual(key, profile.key)

    def test_grouping_covers_every_profile(self):
        grouped = aircraft_by_category()
        self.assertEqual(sum(len(v) for v in grouped.values()), len(AIRCRAFT))


class TestCatalogueSanity(unittest.TestCase):
    """Guards against typos in a hand-entered table of 47 aircraft."""

    def test_all_values_are_positive(self):
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertGreater(profile.cruise_tas_kt, 0)
                self.assertGreater(profile.fuel_burn_gph, 0)
                self.assertGreater(profile.usable_fuel_gal, 0)
                self.assertGreater(profile.seats, 0)

    def test_empty_weight_is_below_mtow(self):
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertLess(profile.empty_weight_lb, profile.mtow_lb)

    def test_cruise_altitude_is_within_the_service_ceiling(self):
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertLessEqual(
                    profile.cruise_altitude_ft, profile.service_ceiling_ft
                )

    def test_occupancy_never_exceeds_seats(self):
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertLessEqual(profile.typical_occupancy, profile.seats)

    def test_airliners_plan_for_a_full_cabin(self):
        for profile in AIRCRAFT.values():
            if profile.category in ("regional", "narrowbody", "widebody"):
                with self.subTest(profile.key):
                    self.assertEqual(profile.typical_occupancy, profile.seats)

    def test_every_aircraft_can_actually_fly_its_default_load(self):
        # The bug that motivated splitting occupancy from seats: a default
        # payload heavy enough to leave no room for fuel.
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertGreater(profile.range_nm(), 0)

    def test_ranges_are_physically_plausible(self):
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertGreater(profile.range_nm(), 100)
                self.assertLess(profile.range_nm(), 15000)


class TestFuelAndReserves(unittest.TestCase):
    def test_reserve_is_45_minutes_of_burn(self):
        self.assertAlmostEqual(C172.reserve_gal, 8.5 * 0.75)

    def test_fuel_required_includes_the_reserve(self):
        required = C172.fuel_required_gal(2.0)
        self.assertAlmostEqual(required, 2.0 * 8.5 + C172.reserve_gal)

    def test_endurance_excludes_the_reserve(self):
        # Full tanks minus reserve, divided by burn.
        expected = (C172.max_fuel_gal() - C172.reserve_gal) / C172.fuel_burn_gph
        self.assertAlmostEqual(C172.endurance_hours(), expected)

    def test_piston_and_turbine_use_different_fuel_densities(self):
        self.assertAlmostEqual(C172.fuel_density_lb_gal, 6.0)
        self.assertAlmostEqual(B738.fuel_density_lb_gal, 6.7)


class TestPayloadRange(unittest.TestCase):
    """The payload/fuel trade against maximum takeoff weight."""

    def test_light_payload_is_tank_limited(self):
        # One person in a 172: tanks fill, weight is not the constraint.
        self.assertAlmostEqual(C172.max_fuel_gal(220), C172.usable_fuel_gal)
        self.assertFalse(C172.is_weight_limited(220))

    def test_heavy_payload_becomes_weight_limited(self):
        # Three adults: 660 lb of the 870 lb useful load, leaving 210 lb
        # for fuel -- 35 gal of the 53 the tanks hold.
        self.assertAlmostEqual(C172.max_fuel_gal(660), 35.0)
        self.assertTrue(C172.is_weight_limited(660))

    def test_payload_exceeding_useful_load_leaves_no_fuel(self):
        # Four adults in a 172 is 880 lb against an 870 lb useful load.
        # The aircraft cannot carry them and any fuel at all.
        self.assertEqual(C172.max_fuel_gal(880), 0.0)
        self.assertEqual(C172.range_nm(880), 0.0)
        self.assertEqual(C172.endurance_hours(880), 0.0)

    def test_range_falls_as_payload_rises(self):
        ranges = [C172.range_nm(pax * 220) for pax in range(1, 5)]
        for heavier, lighter in zip(ranges[1:], ranges):
            self.assertLessEqual(heavier, lighter)

    def test_ferry_range_beats_or_matches_loaded_range(self):
        for profile in AIRCRAFT.values():
            with self.subTest(profile.key):
                self.assertGreaterEqual(profile.ferry_range_nm(), profile.range_nm())

    def test_ferry_range_is_the_zero_payload_case(self):
        self.assertAlmostEqual(B788.ferry_range_nm(), B788.range_nm(0.0))

    def test_widebodies_are_weight_limited_with_a_full_cabin(self):
        # A 787-8 cannot fill its tanks with 248 passengers aboard -- the
        # constraint that makes published range shorter than tank range.
        self.assertTrue(B788.is_weight_limited())
        self.assertLess(B788.max_fuel_gal(), B788.usable_fuel_gal)

    def test_useful_load_is_mtow_minus_empty(self):
        self.assertAlmostEqual(C172.useful_load_lb, 2550 - 1680)

    def test_typical_payload_uses_occupancy_not_seats(self):
        self.assertEqual(C172.seats, 4)
        self.assertEqual(C172.typical_occupancy, 2)
        self.assertAlmostEqual(C172.typical_payload_lb, 440.0)
        self.assertAlmostEqual(C172.max_payload_lb, 880.0)


class TestAltitude(unittest.TestCase):
    def test_can_fly_at_or_below_ceiling(self):
        self.assertTrue(C172.can_fly_at(14000))
        self.assertTrue(C172.can_fly_at(8000))

    def test_cannot_fly_above_ceiling(self):
        self.assertFalse(C172.can_fly_at(14001))
        self.assertFalse(C172.can_fly_at(35000))


class TestKnownFigures(unittest.TestCase):
    """Spot checks against published data, loose enough to tolerate the
    model's simplifications but tight enough to catch a bad table entry."""

    def test_787_8_range_is_near_published(self):
        # Published is roughly 7,300 nm.
        self.assertAlmostEqual(B788.range_nm(), 7300, delta=1200)

    def test_737_800_range_is_near_published(self):
        # Published is roughly 2,900 nm.
        self.assertAlmostEqual(B738.range_nm(), 2900, delta=600)

    def test_c172_endurance_is_about_five_hours(self):
        self.assertAlmostEqual(C172.endurance_hours(), 5.5, delta=0.5)

    def test_airliners_cruise_near_450_knots(self):
        for profile in AIRCRAFT.values():
            if profile.category in ("narrowbody", "widebody"):
                with self.subTest(profile.key):
                    self.assertGreater(profile.cruise_tas_kt, 400)
                    self.assertLess(profile.cruise_tas_kt, 520)


class TestProfileIsImmutable(unittest.TestCase):
    def test_frozen(self):
        with self.assertRaises(Exception):
            C172.cruise_tas_kt = 999  # type: ignore[misc]

    def test_hashable_for_use_as_a_dict_key(self):
        self.assertIsInstance({C172: "ok"}[C172], str)

    def test_constructing_a_custom_profile(self):
        custom = AircraftProfile(
            key="test", name="Test", category="ga",
            cruise_tas_kt=100, cruise_altitude_ft=5000, service_ceiling_ft=10000,
            fuel_burn_gph=10, usable_fuel_gal=40,
            mtow_lb=2000, empty_weight_lb=1200, seats=2, typical_occupancy=2,
        )
        self.assertAlmostEqual(custom.useful_load_lb, 800)


if __name__ == "__main__":
    unittest.main()
