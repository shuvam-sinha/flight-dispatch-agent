"""The tool layer is an adapter, not new routing logic.

THE CLAIM THIS PINS
-------------------
CP4's definition of done says a natural-language request should produce
"the same underlying route as a direct CP3 call, but produced via the
agent's tool orchestration". That is the project's central architectural
claim in one sentence: `tools.py` translates arguments and formats
results, and does not decide anything about flying.

It is easy to say and easy to break. Someone adding a "small improvement"
inside `plan_flight` -- a nudged radius, a filtered waypoint, a rounded
distance -- would create two routers that disagree, and nothing else in
the suite would notice. These tests compare the two paths directly.

NO MODEL IS INVOLVED. `dispatch()` is what the agent loop calls once the
model has chosen a tool, so exercising it covers the whole path from tool
call to route without needing a language model, a network, or an API key.
Whether the model picks the right tool is a separate question, tested in
test_agent.py against a scripted backend.

Wind is off throughout: two calls a second apart could otherwise fetch
different forecasts and disagree for reasons that have nothing to do with
the code.
"""

import unittest

from flight_dispatch.aircraft import get_aircraft
from flight_dispatch.data_loader import (
    load_airports,
    load_navaids,
    navaids_near_route,
)
from flight_dispatch.route import plan_route
from flight_dispatch.tools import dispatch

# Chosen to cover the cases that route differently: a short domestic hop,
# a transcontinental route, and an ocean crossing that needs the grid.
ROUTES = [
    ("KPWK", "KMSP", "sr22"),
    ("KJFK", "KLAX", "b738"),
    ("KJFK", "EGLL", "b789"),
]


class TestToolMatchesDirectCall(unittest.TestCase):
    """Slow -- each case builds a real mesh twice."""

    @classmethod
    def setUpClass(cls):
        cls.airports = load_airports()
        cls.navaids = load_navaids()

    def direct(self, origin_icao, dest_icao, aircraft_key):
        """Plan the CP3 way: load, filter, call plan_route."""
        origin = self.airports[origin_icao]
        dest = self.airports[dest_icao]
        near = navaids_near_route(
            self.navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        return plan_route(
            origin,
            dest,
            near,
            aircraft=get_aircraft(aircraft_key),
            use_grid=True,
        )

    def via_tool(self, origin_icao, dest_icao, aircraft_key):
        """Plan the CP4 way: the call the agent loop makes."""
        return dispatch(
            "plan_flight",
            {
                "origin": origin_icao,
                "dest": dest_icao,
                "aircraft": aircraft_key,
                "use_wind": False,
                "avoid_airspace": False,
            },
        )

    def test_same_waypoints(self):
        # The strongest form of the claim: not merely a similar distance,
        # but the identical ordered list of fixes.
        for origin, dest, aircraft in ROUTES:
            with self.subTest(f"{origin}->{dest}"):
                expected = " ".join(
                    w.ident for w in self.direct(origin, dest, aircraft).waypoints
                )
                self.assertEqual(self.via_tool(origin, dest, aircraft)["route"], expected)

    def test_same_distance(self):
        for origin, dest, aircraft in ROUTES:
            with self.subTest(f"{origin}->{dest}"):
                plan = self.direct(origin, dest, aircraft)
                result = self.via_tool(origin, dest, aircraft)
                self.assertAlmostEqual(
                    result["route_distance_nm"], plan.total_distance_nm, places=1
                )
                self.assertAlmostEqual(
                    result["direct_distance_nm"], plan.direct_distance_nm, places=1
                )

    def test_same_flight_time(self):
        for origin, dest, aircraft in ROUTES:
            with self.subTest(f"{origin}->{dest}"):
                plan = self.direct(origin, dest, aircraft)
                result = self.via_tool(origin, dest, aircraft)
                self.assertAlmostEqual(result["ete_hours"], plan.ete_hours, places=2)

    def test_same_fuel(self):
        for origin, dest, aircraft in ROUTES:
            with self.subTest(f"{origin}->{dest}"):
                plan = self.direct(origin, dest, aircraft)
                result = self.via_tool(origin, dest, aircraft)
                self.assertAlmostEqual(
                    result["fuel_required_gal"], plan.fuel_required_gal, places=1
                )

    def test_same_phase_profile(self):
        # Climb and descent are computed in route.py, so the tool must
        # not be recomputing them differently on its way to a string.
        for origin, dest, aircraft in ROUTES:
            with self.subTest(f"{origin}->{dest}"):
                phases = self.direct(origin, dest, aircraft).phases
                result = self.via_tool(origin, dest, aircraft)
                self.assertIn(
                    f"{phases.cruise_distance_nm:.0f} nm", result["flight_profile"]
                )
                self.assertIn(
                    f"{phases.cruise_altitude_ft:,.0f} ft", result["flight_profile"]
                )

    def test_grid_use_agrees(self):
        # The oceanic route should use generated waypoints and the
        # domestic ones should not -- through either path.
        oceanic = self.via_tool("KJFK", "EGLL", "b789")
        domestic = self.via_tool("KPWK", "KMSP", "sr22")
        self.assertGreater(oceanic["oceanic_waypoints"], 0)
        self.assertNotIn("oceanic_waypoints", domestic)


class TestToolAddsPresentationOnly(unittest.TestCase):
    """What the tool layer IS allowed to add: phrasing, never numbers."""

    def result(self):
        return dispatch(
            "plan_flight",
            {
                "origin": "KPWK",
                "dest": "KMSP",
                "aircraft": "sr22",
                "use_wind": False,
                "avoid_airspace": False,
            },
        )

    def test_adds_prose_the_engine_does_not_produce(self):
        # These exist because the model misread bare fields -- see the
        # comments in tools.py. They are presentation, and they belong
        # here rather than in route.py, which has no model to talk to.
        result = self.result()
        for field in ("wind", "restricted_airspace", "ete_spoken", "flight_profile"):
            self.assertIn(field, result)

    def test_every_number_traces_to_the_engine(self):
        # The spoken time must describe the computed time, not a
        # separately derived one.
        result = self.result()
        hours = int(result["ete_hours"])
        self.assertIn(str(hours), result["ete_spoken"])


if __name__ == "__main__":
    unittest.main()
