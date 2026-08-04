import unittest

from flight_dispatch.geo import (
    bounding_box,
    cross_and_along_track_nm,
    cross_track_nm,
    haversine_nm,
    initial_bearing_deg,
)

# Chicago Executive and O'Hare: a short, real, roughly north-south leg.
KPWK = (42.1142, -87.9014)
KORD = (41.9786, -87.9048)


class TestHaversine(unittest.TestCase):
    def test_zero_distance(self):
        self.assertAlmostEqual(haversine_nm(41.88, -87.63, 41.88, -87.63), 0.0, places=6)

    def test_one_degree_of_latitude_is_sixty_nm(self):
        # The nautical mile is defined as one minute of arc, so a degree of
        # latitude is 60 nm anywhere on the globe.
        self.assertAlmostEqual(haversine_nm(0.0, 0.0, 1.0, 0.0), 60.0, delta=0.1)
        self.assertAlmostEqual(haversine_nm(45.0, -87.0, 46.0, -87.0), 60.0, delta=0.1)

    def test_longitude_degrees_shrink_with_latitude(self):
        at_equator = haversine_nm(0.0, 0.0, 0.0, 1.0)
        at_sixty_north = haversine_nm(60.0, 0.0, 60.0, 1.0)
        # cos(60 deg) == 0.5, so the higher-latitude degree is half as wide.
        self.assertAlmostEqual(at_sixty_north / at_equator, 0.5, delta=0.01)

    def test_is_symmetric(self):
        self.assertAlmostEqual(
            haversine_nm(*KPWK, *KORD), haversine_nm(*KORD, *KPWK), places=9
        )


class TestBearing(unittest.TestCase):
    def test_due_north(self):
        self.assertAlmostEqual(initial_bearing_deg(0.0, 0.0, 1.0, 0.0), 0.0, delta=0.01)

    def test_due_east(self):
        self.assertAlmostEqual(initial_bearing_deg(0.0, 0.0, 0.0, 1.0), 90.0, delta=0.01)

    def test_due_south(self):
        self.assertAlmostEqual(
            initial_bearing_deg(1.0, 0.0, 0.0, 0.0), 180.0, delta=0.01
        )

    def test_always_in_zero_to_360(self):
        # Westbound would be negative before normalisation.
        self.assertAlmostEqual(initial_bearing_deg(0.0, 0.0, 0.0, -1.0), 270.0, delta=0.01)


class TestCrossTrack(unittest.TestCase):
    def test_point_on_path_has_zero_cross_track(self):
        cross, along = cross_and_along_track_nm(0.0, 5.0, 0.0, 0.0, 0.0, 10.0)
        self.assertAlmostEqual(cross, 0.0, delta=0.01)
        self.assertAlmostEqual(along, 300.0, delta=1.0)

    def test_sign_indicates_which_side_of_course(self):
        # Flying east along the equator: north of course is to the left.
        north_of_course = cross_track_nm(1.0, 5.0, 0.0, 0.0, 0.0, 10.0)
        south_of_course = cross_track_nm(-1.0, 5.0, 0.0, 0.0, 0.0, 10.0)
        self.assertLess(north_of_course, 0)
        self.assertGreater(south_of_course, 0)
        self.assertAlmostEqual(abs(north_of_course), 60.0, delta=0.5)

    def test_offset_point_cross_track_magnitude(self):
        cross, _ = cross_and_along_track_nm(0.5, 5.0, 0.0, 0.0, 0.0, 10.0)
        self.assertAlmostEqual(abs(cross), 30.0, delta=0.5)

    def test_point_behind_origin_has_negative_along_track(self):
        # The textbook acos() form returns 0..pi and would report this
        # point as +60 nm of progress, which would let route selection
        # pick up waypoints in the opposite direction of travel.
        _, along = cross_and_along_track_nm(0.0, -1.0, 0.0, 0.0, 0.0, 10.0)
        self.assertAlmostEqual(along, -60.0, delta=0.5)

    def test_point_past_destination_exceeds_course_length(self):
        course_nm = haversine_nm(0.0, 0.0, 0.0, 10.0)
        _, along = cross_and_along_track_nm(0.0, 11.0, 0.0, 0.0, 0.0, 10.0)
        self.assertGreater(along, course_nm)

    def test_point_at_origin_has_zero_along_track(self):
        _, along = cross_and_along_track_nm(0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
        self.assertAlmostEqual(along, 0.0, delta=0.01)


class TestBoundingBox(unittest.TestCase):
    def test_encloses_both_points(self):
        min_lat, min_lon, max_lat, max_lon = bounding_box(*KPWK, *KORD, margin_nm=0.0)
        for lat, lon in (KPWK, KORD):
            self.assertTrue(min_lat <= lat <= max_lat)
            self.assertTrue(min_lon <= lon <= max_lon)

    def test_margin_widens_the_box(self):
        tight = bounding_box(*KPWK, *KORD, margin_nm=0.0)
        padded = bounding_box(*KPWK, *KORD, margin_nm=60.0)
        self.assertAlmostEqual(tight[0] - padded[0], 1.0, delta=0.01)
        self.assertGreater(padded[2], tight[2])

    def test_longitude_padding_widens_with_latitude(self):
        equator = bounding_box(0.0, 0.0, 0.0, 1.0, margin_nm=60.0)
        far_north = bounding_box(60.0, 0.0, 60.0, 1.0, margin_nm=60.0)
        equator_lon_pad = equator[1]
        far_north_lon_pad = far_north[1]
        # A degree of longitude is narrower up north, so the box needs more
        # degrees to cover the same distance.
        self.assertLess(far_north_lon_pad, equator_lon_pad)


if __name__ == "__main__":
    unittest.main()
