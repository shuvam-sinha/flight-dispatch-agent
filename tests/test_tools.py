"""Tests for the LLM tool surface.

These exercise the tools directly, with no model involved -- which is the
point of the layer. If these pass, any failure in a real conversation is
the model's tool selection, not the tools themselves.

Tests that would hit the network (live winds) pass `use_wind=False`.
"""

import unittest

from flight_dispatch.tools import (
    TOOLS,
    TOOLS_BY_NAME,
    ToolSpec,
    dispatch,
)


class TestRegistry(unittest.TestCase):
    def test_every_tool_has_a_description(self):
        for tool in TOOLS:
            with self.subTest(tool.name):
                self.assertGreater(len(tool.description), 40)

    def test_descriptions_say_when_to_use_the_tool(self):
        # The model picks tools by reading these, so each should give a
        # usage cue, not just describe the return value.
        cues = ("call this", "use this", "when the user", "when they", "set true when")
        for tool in TOOLS:
            with self.subTest(tool.name):
                lowered = tool.description.lower()
                self.assertTrue(
                    any(cue in lowered for cue in cues),
                    f"{tool.name} description gives no usage cue",
                )

    def test_every_parameter_is_documented(self):
        for tool in TOOLS:
            for name, spec in tool.parameters.items():
                with self.subTest(f"{tool.name}.{name}"):
                    self.assertIn("type", spec)
                    self.assertGreater(len(spec.get("description", "")), 10)

    def test_names_match_the_lookup_table(self):
        self.assertEqual(len(TOOLS), len(TOOLS_BY_NAME))
        for tool in TOOLS:
            self.assertIs(TOOLS_BY_NAME[tool.name], tool)

    def test_json_schema_shape(self):
        schema = TOOLS_BY_NAME["plan_flight"].json_schema()
        self.assertEqual(schema["type"], "object")
        self.assertIn("origin", schema["properties"])
        self.assertEqual(sorted(schema["required"]), ["dest", "origin"])

    def test_json_schema_carries_enums(self):
        schema = TOOLS_BY_NAME["list_aircraft"].json_schema()
        self.assertIn("ga", schema["properties"]["category"]["enum"])

    def test_required_names(self):
        self.assertEqual(
            sorted(TOOLS_BY_NAME["plan_flight"].required_names()),
            ["dest", "origin"],
        )


class TestDispatch(unittest.TestCase):
    def test_unknown_tool(self):
        result = dispatch("teleport", {})
        self.assertIn("error", result)
        self.assertIn("available_tools", result)

    def test_missing_required_argument(self):
        result = dispatch("find_airport", {})
        self.assertIn("query", result["error"])

    def test_unexpected_argument_is_rejected(self):
        # Catches a model hallucinating a parameter.
        result = dispatch("find_airport", {"query": "KORD", "colour": "blue"})
        self.assertIn("colour", result["error"])
        self.assertIn("accepted_arguments", result)

    def test_never_raises(self):
        # The contract the agent loop depends on.
        for name, args in [
            ("plan_flight", {"origin": "!!!", "dest": "???"}),
            ("get_winds_aloft", {"latitude": 999, "longitude": 999}),
            ("check_airspace", {"origin": "ZZZZ", "dest": "ZZZZ"}),
            ("find_airport", {"query": ""}),
        ]:
            with self.subTest(name):
                self.assertIsInstance(dispatch(name, args), dict)


class TestFindAirport(unittest.TestCase):
    def test_exact_icao(self):
        result = dispatch("find_airport", {"query": "KORD"})
        self.assertTrue(result["found"])
        self.assertEqual(result["icao"], "KORD")

    def test_icao_is_case_insensitive(self):
        self.assertEqual(dispatch("find_airport", {"query": "kord"})["icao"], "KORD")

    def test_name_search_returns_matches(self):
        result = dispatch("find_airport", {"query": "Chicago Executive"})
        self.assertTrue(result["found"])
        self.assertEqual(result["matches"][0]["icao"], "KPWK")

    def test_ambiguous_search_reports_the_count(self):
        # "Minneapolis" matches several airports; the model needs to know
        # so it can choose or ask.
        result = dispatch("find_airport", {"query": "Minneapolis"})
        self.assertGreater(result["match_count"], 1)

    def test_matches_are_capped(self):
        result = dispatch("find_airport", {"query": "International"})
        self.assertLessEqual(len(result["matches"]), 8)

    def test_no_match_gives_a_hint(self):
        result = dispatch("find_airport", {"query": "zzzznotanairport"})
        self.assertFalse(result["found"])
        self.assertIn("hint", result)


class TestListAircraft(unittest.TestCase):
    def test_lists_everything_by_default(self):
        result = dispatch("list_aircraft", {})
        self.assertEqual(result["count"], len(result["aircraft"]))
        self.assertGreater(result["count"], 40)

    def test_category_filter(self):
        from flight_dispatch.aircraft import AIRCRAFT

        result = dispatch("list_aircraft", {"category": "widebody"})
        widebody_keys = {k for k, p in AIRCRAFT.items() if p.category == "widebody"}
        listed = {line.split(":")[0] for line in result["aircraft"]}
        self.assertEqual(listed, widebody_keys)

    def test_bad_category_lists_the_valid_ones(self):
        result = dispatch("list_aircraft", {"category": "spaceship"})
        self.assertIn("valid_categories", result)

    def test_entries_are_compact_lines_not_dicts(self):
        # Structured entries cost 1,853 tokens unfiltered -- 45% of the
        # on-device model's whole context. One line each carries the same
        # information at a fraction of the size.
        entry = dispatch("list_aircraft", {"category": "ga"})["aircraft"][0]
        self.assertIsInstance(entry, str)
        self.assertTrue(entry.startswith("c172:"))
        for fragment in ("Cessna", "kt", "seats", "nm"):
            self.assertIn(fragment, entry)

    def test_result_stays_within_a_sane_token_budget(self):
        import json

        size = len(json.dumps(dispatch("list_aircraft", {})))
        self.assertLess(size // 4, 900, "list_aircraft result grew past its budget")

    def test_format_line_explains_the_layout(self):
        self.assertIn("key", dispatch("list_aircraft", {})["format"])


class TestPlanFlight(unittest.TestCase):
    def plan(self, **kwargs):
        args = {"origin": "KPWK", "dest": "KMSP", "use_wind": False}
        args.update(kwargs)
        return dispatch("plan_flight", args)

    def test_returns_a_route(self):
        result = self.plan()
        self.assertNotIn("error", result)
        idents = [w["ident"] for w in result["waypoints"]]
        self.assertEqual(idents[0], "KPWK")
        self.assertEqual(idents[-1], "KMSP")

    def test_legs_carry_distance_and_course(self):
        second = self.plan()["waypoints"][1]
        self.assertIn("leg_distance_nm", second)
        self.assertIn("leg_course_true", second)

    def test_origin_has_no_leg(self):
        self.assertNotIn("leg_distance_nm", self.plan()["waypoints"][0])

    def test_unknown_origin_points_at_find_airport(self):
        result = self.plan(origin="Chicago")
        self.assertIn("error", result)
        self.assertIn("find_airport", result["hint"])

    def test_unknown_aircraft_points_at_list_aircraft(self):
        result = self.plan(aircraft="concorde")
        self.assertIn("list_aircraft", result["hint"])

    def test_overloaded_aircraft_is_refused(self):
        result = self.plan(aircraft="c172", payload_lb=880)
        self.assertIn("error", result)
        self.assertIn("useful load", result["error"])

    def test_altitude_above_ceiling_is_refused(self):
        result = self.plan(aircraft="c172", altitude_ft=45000)
        self.assertIn("service ceiling", result["error"])

    def test_reports_whether_wind_was_applied(self):
        self.assertFalse(self.plan()["wind_applied"])

    def test_airspace_flag_is_reported(self):
        self.assertIn("airspace_avoidance_applied", self.plan())

    def test_no_map_unless_asked(self):
        self.assertNotIn("map_file", self.plan())

    def test_route_is_never_shorter_than_direct(self):
        result = self.plan()
        self.assertGreaterEqual(
            result["route_distance_nm"], result["direct_distance_nm"] - 0.01
        )

    def test_out_of_range_flight_warns(self):
        # A 172 cannot cross the continent.
        result = dispatch(
            "plan_flight",
            {"origin": "KJFK", "dest": "KLAX", "aircraft": "c172", "use_wind": False},
        )
        self.assertFalse(result["within_aircraft_range"])
        self.assertIn("range_warning", result)


class TestSaveMap(unittest.TestCase):
    def test_map_is_written_and_path_returned(self):
        import os

        result = dispatch(
            "plan_flight",
            {"origin": "KPWK", "dest": "KMSP", "use_wind": False, "save_map": True},
        )
        self.assertIn("map_file", result)
        self.assertTrue(os.path.exists(result["map_file"]))

    def test_model_is_told_it_cannot_see_the_map(self):
        # Guards against the model claiming to have looked at the image.
        result = dispatch(
            "plan_flight",
            {"origin": "KPWK", "dest": "KMSP", "use_wind": False, "save_map": True},
        )
        self.assertIn("cannot see", result["map_note"])

    def test_filename_identifies_the_flight(self):
        result = dispatch(
            "plan_flight",
            {
                "origin": "KPWK", "dest": "KMSP", "aircraft": "sr22",
                "use_wind": False, "save_map": True,
            },
        )
        self.assertIn("kpwk_kmsp_sr22", result["map_file"])


class TestCheckAirspace(unittest.TestCase):
    def test_reports_crossings_on_a_busy_corridor(self):
        result = dispatch(
            "check_airspace",
            {"origin": "KLAX", "dest": "KSLC", "altitude_ft": 10000},
        )
        self.assertGreater(result["active_volumes_in_region"], 0)
        self.assertIn("direct_course_crossings", result)

    def test_altitude_changes_what_is_active(self):
        low = dispatch("check_airspace", {"origin": "KLAX", "dest": "KSLC", "altitude_ft": 5000})
        high = dispatch("check_airspace", {"origin": "KLAX", "dest": "KSLC", "altitude_ft": 41000})
        self.assertGreater(
            low["active_volumes_in_region"], high["active_volumes_in_region"]
        )

    def test_unknown_airport(self):
        self.assertIn("error", dispatch("check_airspace", {"origin": "ZZZZ", "dest": "KORD"}))

    def test_points_the_model_at_plan_flight_for_routing(self):
        result = dispatch("check_airspace", {"origin": "KLAX", "dest": "KSLC"})
        self.assertIn("plan_flight", result["note"])


if __name__ == "__main__":
    unittest.main()


class TestErrorShortening(unittest.TestCase):
    """Error text lands in the model's context, so its length is a cost.
    A 429 from a batched wind request carried a 2,000-character URL --
    ~500 tokens of a 4,096-token window, spent on a string the model can
    do nothing with. It overflowed the conversation."""

    def test_url_is_stripped(self):
        from flight_dispatch.tools import _short_error

        long_url = "https://api.open-meteo.com/v1/forecast?latitude=" + "41.0%2C" * 200
        exc = RuntimeError(f"429 Client Error: Too Many Requests for url: {long_url}")
        short = _short_error(exc)

        self.assertNotIn("http", short)
        self.assertIn("429", short)
        self.assertLess(len(short), 100)

    def test_long_messages_are_truncated(self):
        from flight_dispatch.tools import MAX_ERROR_CHARS, _short_error

        short = _short_error(ValueError("x" * 1000))
        self.assertLessEqual(len(short), MAX_ERROR_CHARS + 40)

    def test_short_messages_pass_through(self):
        from flight_dispatch.tools import _short_error

        self.assertIn("no such thing", _short_error(KeyError("no such thing")))

    def test_type_name_is_kept(self):
        from flight_dispatch.tools import _short_error

        self.assertTrue(_short_error(TimeoutError("slow")).startswith("TimeoutError"))


class TestMatchRanking(unittest.TestCase):
    """Sorting purely by name length put a Mexican airstrip literally
    named "San Francisco" ahead of San Francisco International, and the
    agent planned a flight from it."""

    def first_match(self, query):
        return dispatch("find_airport", {"query": query})["matches"][0]["icao"]

    def test_major_airports_rank_first(self):
        self.assertEqual(self.first_match("San Francisco"), "KSFO")
        self.assertEqual(self.first_match("Los Angeles"), "KLAX")
        self.assertEqual(self.first_match("Minneapolis"), "KMSP")

    def test_placeholder_identifiers_are_deprioritised(self):
        # OurAirports uses MX-1385 / US-3912 style ids for fields with no
        # real ICAO code. Those are almost never what a pilot means.
        for query in ("San Francisco", "Los Angeles", "Minneapolis"):
            with self.subTest(query):
                self.assertNotIn("-", self.first_match(query))

    def test_ranking_prefers_four_letter_icao_codes(self):
        from flight_dispatch.tools import _match_rank
        from flight_dispatch.models import Airport

        real = Airport(icao="KSFO", name="San Francisco International Airport",
                       lat=37.6, lon=-122.4)
        placeholder = Airport(icao="MX-1385", name="San Francisco", lat=20.0, lon=-100.0)
        self.assertLess(_match_rank(real), _match_rank(placeholder))
