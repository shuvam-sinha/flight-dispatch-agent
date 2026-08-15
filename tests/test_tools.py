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
        self.assertLessEqual(len(result["matches"]), 3)

    def test_candidates_carry_no_coordinates(self):
        # A candidate list is for choosing from, not for using. Eight
        # matches with lat/lon each made one "Chicago" lookup 41% of the
        # whole transcript, and the model discards all but one of them.
        # Coordinates come from looking the chosen code up.
        result = dispatch("find_airport", {"query": "Chicago"})
        self.assertNotIn("latitude", result["matches"][0])

    def test_an_exact_code_still_returns_a_position(self):
        # get_winds_aloft needs coordinates, and this is where they
        # come from now.
        result = dispatch("find_airport", {"query": "KORD"})
        self.assertIn("latitude", result)
        self.assertIn("longitude", result)

    def test_true_match_count_survives_the_cap(self):
        result = dispatch("find_airport", {"query": "Chicago"})
        self.assertEqual(len(result["matches"]), 3)
        self.assertGreater(result["match_count"], 3)

    def test_no_match_gives_a_hint(self):
        result = dispatch("find_airport", {"query": "zzzznotanairport"})
        self.assertFalse(result["found"])
        self.assertIn("hint", result)


class TestFindAirportQueryForms(unittest.TestCase):
    """The ways people and models actually name airports.

    Every case here comes from a real transcript or from testing one.
    Substring search alone handles almost none of them.
    """

    def first(self, query: str) -> str:
        result = dispatch("find_airport", {"query": query})
        self.assertTrue(result["found"], f"no match for {query!r}")
        return result.get("icao") or result["matches"][0]["icao"]

    def test_iata_code_alone(self):
        # Not a substring of "John F. Kennedy International Airport".
        self.assertEqual(self.first("JFK"), "KJFK")
        self.assertEqual(self.first("LAX"), "KLAX")

    def test_city_plus_code(self):
        # The failure that prompted all of this: the city is in one field
        # and the code in another, so the phrase matches nothing.
        self.assertEqual(self.first("New York JFK"), "KJFK")

    def test_city_plus_generic_word(self):
        # "Los Angeles airport" IS a substring of "Hilton Los Angeles
        # Airport Helipad" and is NOT one of "Los Angeles International
        # Airport", so a phrase-only search returns a hotel helipad.
        self.assertEqual(self.first("Los Angeles airport"), "KLAX")

    def test_punctuation_is_ignored(self):
        for query in ("O'Hare", "O Hare", "OHare", "Chicago OHare"):
            self.assertEqual(self.first(query), "KORD", query)

    def test_unmatchable_word_does_not_kill_the_query(self):
        # No field holds the country, so requiring every word finds
        # nothing. Best partial match still gets there.
        self.assertEqual(self.first("Sydney Australia"), "YSSY")

    def test_common_international_forms(self):
        self.assertEqual(self.first("London Heathrow"), "EGLL")
        self.assertEqual(self.first("Tokyo Haneda"), "RJTT")
        self.assertEqual(self.first("Paris Charles de Gaulle"), "LFPG")

    def test_still_rejects_genuine_nonsense(self):
        # The relaxations must not make everything match something.
        self.assertFalse(dispatch("find_airport", {"query": "asdfghjkl"})["found"])

    def test_accents_are_ignored(self):
        # Guarulhos' municipality is "Sao Paulo" with a tilde, so an
        # unaccented search matched nothing there and fell through to a
        # hotel helipad that spells it without one.
        self.assertEqual(self.first("Sao Paulo"), "SBGR")
        self.assertEqual(self.first("Sao Paulo"), self.first("São Paulo"))
        self.assertEqual(self.first("Zurich"), "LSZH")
        self.assertEqual(self.first("Malaga"), "LEMG")

    def test_alternate_names_from_keywords(self):
        # OurAirports keeps local-language and former names in
        # `keywords`, which nothing used to read.
        self.assertEqual(self.first("Londres"), "EGLL")
        self.assertEqual(self.first("Ciudad de Mexico"), "MMMX")


class TestCityNamePicksTheMainAirport(unittest.TestCase):
    """A city name should resolve to the airport people mean by it.

    These are the cases the ranker exists for, and several of them are
    ones it previously got wrong.
    """

    EXPECTED = {
        "Chicago": "KORD",
        "New York": "KJFK",
        "Los Angeles": "KLAX",
        "Houston": "KIAH",
        "San Francisco": "KSFO",  # was a Mexican airstrip
        "London": "EGLL",         # was East London, South Africa
        "Paris": "LFPG",
        "Tokyo": "RJTT",          # was Narita, on longest-runway
        "Dubai": "OMDB",          # was Al Maktoum, on longest-runway
        "Sao Paulo": "SBGR",      # was a hotel helipad, on the accent
        "Rome": "LIRF",
        "Seoul": "RKSI",
        "Istanbul": "LTFM",
        "Milan": "LIMC",
        "Osaka": "RJBB",
        "Toronto": "CYYZ",
        "Sydney": "YSSY",
        "Madrid": "LEMD",
    }

    def test_each_city_resolves_to_its_main_airport(self):
        for city, icao in self.EXPECTED.items():
            result = dispatch("find_airport", {"query": city})
            first = result.get("icao") or result["matches"][0]["icao"]
            self.assertEqual(first, icao, f"{city} -> {first}")

    def test_mexico_city_is_a_known_miss(self):
        # Documented rather than hidden. Felipe Angeles is a converted
        # air force base with more pavement than Benito Juarez and almost
        # no traffic, and this dataset carries no traffic figures. If a
        # future change fixes it, this test should be the one that fails.
        result = dispatch("find_airport", {"query": "Mexico City"})
        self.assertEqual(result["matches"][0]["icao"], "MMSM")

    def test_naming_the_airport_reaches_it_anyway(self):
        for query in ("Benito Juarez", "MEX", "AICM", "Ciudad de Mexico"):
            result = dispatch("find_airport", {"query": query})
            first = result.get("icao") or result["matches"][0]["icao"]
            self.assertEqual(first, "MMMX", query)


class TestCompassPoint(unittest.TestCase):
    """The model rendered 239 degrees as "from the northeast". It is not."""

    def test_cardinals(self):
        from flight_dispatch.tools import _compass_point

        self.assertEqual(_compass_point(0), "north")
        self.assertEqual(_compass_point(90), "east")
        self.assertEqual(_compass_point(180), "south")
        self.assertEqual(_compass_point(270), "west")

    def test_the_bearing_that_was_reported_backwards(self):
        from flight_dispatch.tools import _compass_point

        self.assertEqual(_compass_point(239), "west-southwest")
        self.assertNotIn("northeast", _compass_point(239))

    def test_wraps_past_360(self):
        from flight_dispatch.tools import _compass_point

        self.assertEqual(_compass_point(359), "north")
        self.assertEqual(_compass_point(361), "north")


class TestAltitudeCoercion(unittest.TestCase):
    """The schema carries enum strings; the CLI passes floats."""

    def test_a_string_from_the_enum(self):
        from flight_dispatch.tools import _coerce_altitude

        self.assertEqual(_coerce_altitude("34000", 8000.0), 34000.0)

    def test_a_float_from_the_cli(self):
        from flight_dispatch.tools import _coerce_altitude

        self.assertEqual(_coerce_altitude(34000.0, 8000.0), 34000.0)

    def test_missing_falls_back_to_the_default(self):
        from flight_dispatch.tools import _coerce_altitude

        self.assertEqual(_coerce_altitude(None, 8000.0), 8000.0)
        self.assertEqual(_coerce_altitude("", 8000.0), 8000.0)

    def test_nonsense_is_reported_not_guessed(self):
        from flight_dispatch.tools import _coerce_altitude

        self.assertIsNone(_coerce_altitude("high", 8000.0))

    def test_every_choice_parses(self):
        from flight_dispatch.tools import ALTITUDE_CHOICES, _coerce_altitude

        for choice in ALTITUDE_CHOICES:
            self.assertIsInstance(_coerce_altitude(choice, 0.0), float)

    def test_choices_land_on_distinct_pressure_levels(self):
        # Two options either side of one level would return identical
        # wind and imply precision that is not there.
        from flight_dispatch.wind import nearest_pressure_level
        from flight_dispatch.tools import ALTITUDE_CHOICES

        levels = [nearest_pressure_level(float(c)) for c in ALTITUDE_CHOICES]
        self.assertEqual(len(levels), len(set(levels)))


class TestWindResultNamesItsAltitude(unittest.TestCase):
    """THE BUG THIS COVERS.

    Asked for the wind at 35,000 ft, this answered at its 8,000 ft
    default -- 12 kt from 239 degrees at +11.9 C -- and the model
    reported it as the 35,000 ft wind. The true value was 26 kt from
    223 at -41 C. The altitude was already a field in the result; that
    was not enough. It now sits in the same sentence as the numbers.

    Uses a stubbed wind source so the assertions do not depend on
    today's weather.
    """

    def call(self, **kwargs):
        from unittest.mock import patch

        from flight_dispatch.wind import Wind

        stub = Wind(direction_deg=239.0, speed_kt=12.0, altitude_ft=0.0,
                    temperature_c=11.9)
        with patch("flight_dispatch.wind_openmeteo.OpenMeteoWindSource") as source:
            source.return_value.wind_at.return_value = stub
            return dispatch(
                "get_winds_aloft",
                {"latitude": 39.86, "longitude": -104.67, **kwargs},
            )

    def test_summary_states_the_altitude_used(self):
        result = self.call(altitude_ft="34000")
        self.assertIn("34,000 ft", result["summary"])

    def test_summary_states_the_default_when_none_was_asked_for(self):
        result = self.call()
        self.assertIn("8,000 ft", result["summary"])

    def test_summary_carries_the_wind_in_the_same_sentence(self):
        result = self.call(altitude_ft="34000")
        self.assertIn("239", result["summary"])
        self.assertIn("12 kt", result["summary"])

    def test_summary_names_the_compass_quadrant(self):
        result = self.call(altitude_ft="34000")
        self.assertIn("west-southwest", result["summary"])

    def test_altitude_actually_reaches_the_wind_source(self):
        # The whole failure was that it did not.
        from unittest.mock import patch

        from flight_dispatch.wind import Wind

        stub = Wind(direction_deg=223.0, speed_kt=26.0, altitude_ft=0.0,
                    temperature_c=-41.0)
        with patch("flight_dispatch.wind_openmeteo.OpenMeteoWindSource") as source:
            source.return_value.wind_at.return_value = stub
            dispatch(
                "get_winds_aloft",
                {"latitude": 39.86, "longitude": -104.67, "altitude_ft": "34000"},
            )
            _, _, altitude = source.return_value.wind_at.call_args[0]
            self.assertEqual(altitude, 34000.0)

    def test_unreadable_altitude_is_an_error_not_a_guess(self):
        result = dispatch(
            "get_winds_aloft",
            {"latitude": 39.86, "longitude": -104.67, "altitude_ft": "cruise"},
        )
        self.assertIn("error", result)


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

    def test_wind_is_stated_as_a_sentence(self):
        # THE BUG THIS REPLACES. `wind_applied: true` was the only
        # statement that live winds had been used, and the model skipped
        # it -- described a whole plan without mentioning wind. The user
        # asked "with wind considerations?" and the agent re-ran the
        # identical call, because wind had been on all along. Airspace,
        # already a sentence by then, WAS reported in that same reply.
        text = self.plan()["wind"]
        self.assertIn("still air", text)
        self.assertIn("not requested", text)

    def test_airspace_result_is_an_unambiguous_sentence(self):
        # THE BUG THIS REPLACES. The result used to carry
        # `airspace_avoidance_applied: true` alongside
        # `restricted_volumes_considered: 95`, and the model combined
        # them into "Route includes prohibited and restricted airspace"
        # -- the precise opposite of the truth. Every number was right;
        # the English was inverted. The tool now states the conclusion.
        text = self.plan()["restricted_airspace"]
        self.assertIn("crosses none", text)
        self.assertNotIn("considered", text)

    def test_airspace_says_so_when_avoidance_is_off(self):
        text = self.plan(avoid_airspace=False)["restricted_airspace"]
        self.assertIn("NOT CHECKED", text)
        self.assertNotIn("crosses none", text)

    def test_no_map_unless_asked(self):
        self.assertNotIn("map_file", self.plan())

    def test_unspecified_aircraft_is_reported_not_assumed(self):
        # THE BUG THIS REPLACES. Asked for KJFK to EGLL with no aircraft
        # named, the tool silently defaulted to a Cessna 172 and returned
        # a straight-faced plan. Defaulting is fine; doing it quietly is
        # not.
        self.assertIn("aircraft_note", self.plan())

    def test_named_aircraft_gets_no_note(self):
        self.assertNotIn("aircraft_note", self.plan(aircraft="sr22"))

    def test_oceanic_route_beyond_range_is_called_impossible(self):
        # 22h15m in an aircraft holding 56 gal, over the Atlantic. "A
        # fuel stop is required" is advice that cannot be taken.
        result = dispatch(
            "plan_flight",
            {"origin": "KJFK", "dest": "EGLL", "aircraft": "c172", "use_wind": False},
        )
        self.assertFalse(result["within_aircraft_range"])
        self.assertIn("cannot fly this route", result["range_warning"])
        self.assertIn("nowhere to refuel", result["range_warning"])

    def test_overland_route_beyond_range_suggests_fuel_stops(self):
        # A 172 crossing the US needs several stops, which is a trip
        # people actually make. The distinction is water, not distance --
        # this route is longer in hours than some oceanic ones.
        result = dispatch(
            "plan_flight",
            {"origin": "KJFK", "dest": "KLAX", "aircraft": "c172", "use_wind": False},
        )
        self.assertFalse(result["within_aircraft_range"])
        self.assertIn("fuel stop", result["range_warning"])
        self.assertNotIn("cannot fly", result["range_warning"])

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

    def test_summary_names_the_altitude_it_used(self):
        # Airspace is altitude-banded, so a count means nothing without
        # the altitude it was computed at -- and a wrong altitude here
        # would be silently wrong rather than an error.
        result = dispatch(
            "check_airspace",
            {"origin": "KLAX", "dest": "KSLC", "altitude_ft": "10000"},
        )
        self.assertIn("10,000 ft", result["summary"])

    def test_altitude_changes_the_summary(self):
        low = dispatch(
            "check_airspace",
            {"origin": "KLAX", "dest": "KSLC", "altitude_ft": "10000"},
        )
        high = dispatch(
            "check_airspace",
            {"origin": "KLAX", "dest": "KSLC", "altitude_ft": "39000"},
        )
        self.assertNotEqual(low["summary"], high["summary"])

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

    def test_major_city_queries_resolve_to_the_primary_airport(self):
        # Each of these previously returned the wrong airport, because
        # ties among equally-major airports fell through to name length.
        for query, expected in [
            ("London", "EGLL"),       # was FAEL, East London, South Africa
            ("Chicago", "KORD"),      # was KMDW, Midway
            ("Paris", "LFPG"),        # was LFPO, Orly
            ("New York", "KJFK"),     # was KLGA, LaGuardia
            ("Tokyo", "RJTT"),
        ]:
            with self.subTest(query):
                self.assertEqual(self.first_match(query), expected)

    def test_ranking_prefers_a_large_airport_over_an_airstrip(self):
        from flight_dispatch.models import Airport
        from flight_dispatch.tools import _match_rank

        real = Airport(
            icao="KSFO", name="San Francisco International Airport",
            lat=37.6, lon=-122.4, airport_type="large_airport",
            scheduled_service=True, iata_code="SFO", municipality="San Francisco",
        )
        airstrip = Airport(
            icao="MX-1385", name="Pista San Francisco", lat=24.8, lon=-107.4,
            airport_type="small_airport", municipality="Culiacan",
        )
        self.assertLess(_match_rank(real, "san francisco"),
                        _match_rank(airstrip, "san francisco"))

    def test_runway_area_breaks_ties_between_major_airports(self):
        from flight_dispatch.models import Airport
        from flight_dispatch.tools import _match_rank

        # Both large, scheduled, IATA-coded, both match "london".
        # Only runway area separates them -- and "East London Airport"
        # is the shorter name, so name length alone got this backwards.
        heathrow = Airport(icao="EGLL", name="London Heathrow Airport",
                           lat=51.5, lon=-0.5, airport_type="large_airport",
                           scheduled_service=True, iata_code="LHR",
                           municipality="London")
        east_london = Airport(icao="FAEL", name="East London Airport",
                              lat=-33.0, lon=27.8, airport_type="large_airport",
                              scheduled_service=True, iata_code="ELS",
                              municipality="East London")
        self.assertLess(
            _match_rank(heathrow, "london", runway_area=4_067_200),
            _match_rank(east_london, "london", runway_area=1_170_000),
        )

    def test_municipality_is_searched_not_just_the_name(self):
        # Heathrow's name contains "London", but many airports are named
        # for something else entirely and only their municipality says
        # which city they serve.
        result = dispatch("find_airport", {"query": "Minneapolis"})
        self.assertGreater(result["match_count"], 1)


class TestWindIsDescribed(unittest.TestCase):
    """The wind sentence, with a stubbed plan so it does not depend on
    today's weather or on Open-Meteo being up."""

    def describe(self, ground_speed_kt, tas_kt=180.0):
        from types import SimpleNamespace

        from flight_dispatch.tools import _describe_wind

        phases = SimpleNamespace(
            cruise_distance_nm=ground_speed_kt, cruise_time_hours=1.0
        )
        plan = SimpleNamespace(phases=phases)
        profile = SimpleNamespace(cruise_tas_kt=tas_kt)
        return _describe_wind(plan, profile)

    def test_names_a_tailwind(self):
        text = self.describe(230.0)
        self.assertIn("tailwind", text)
        self.assertIn("50 kt", text)

    def test_names_a_headwind(self):
        text = self.describe(140.0)
        self.assertIn("headwind", text)
        self.assertIn("40 kt", text)

    def test_a_trivial_difference_is_not_called_a_wind(self):
        # Reporting "a net tailwind of 1 kt" is noise dressed as insight.
        self.assertIn("still air", self.describe(182.0))

    def test_says_the_wind_was_live(self):
        self.assertIn("Live winds aloft applied", self.describe(230.0))

    def test_compares_cruise_not_the_whole_flight(self):
        # Climb and descent are flown below cruise speed by design, so an
        # average over the whole flight makes every flight look like a
        # headwind -- the trap the CLI fell into when phases landed.
        self.assertIn("still air", self.describe(180.0))

    def test_falls_back_when_there_is_no_phase_profile(self):
        from types import SimpleNamespace

        from flight_dispatch.tools import _describe_wind

        plan = SimpleNamespace(phases=None)
        profile = SimpleNamespace(cruise_tas_kt=180.0)
        self.assertIn("Live winds aloft applied", _describe_wind(plan, profile))


class TestSpokenDuration(unittest.TestCase):
    """THE BUG THIS COVERS.

    Given `ete: "1h01m"` on a KSFO-KLAS plan, the reply said "1 hour 4
    minutes". Every other figure in that reply was exact -- distance,
    fuel, altitude, wind, airspace count -- and the wind and airspace
    sentences were quoted verbatim. The compact token was the one thing
    the model had to rewrite, and rewriting is where the error entered.

    So the phrasing is supplied rather than left to be derived, the same
    move as `_compass_point`.
    """

    def spoken(self, hours, minutes):
        from flight_dispatch.tools import _spoken_duration

        return _spoken_duration(hours, minutes)

    def test_the_case_that_was_reported_wrong(self):
        self.assertEqual(self.spoken(1, 1), "1 hour 1 minute")

    def test_singular_and_plural(self):
        self.assertEqual(self.spoken(1, 0), "1 hour")
        self.assertEqual(self.spoken(2, 0), "2 hours")
        self.assertEqual(self.spoken(0, 1), "1 minute")
        self.assertEqual(self.spoken(0, 45), "45 minutes")

    def test_hours_and_minutes_together(self):
        self.assertEqual(self.spoken(18, 2), "18 hours 2 minutes")

    def test_zero_is_still_a_duration(self):
        self.assertEqual(self.spoken(0, 0), "0 minutes")

    def test_plan_carries_both_forms(self):
        result = dispatch(
            "plan_flight",
            {"origin": "KPWK", "dest": "KMSP", "aircraft": "sr22", "use_wind": False},
        )
        self.assertIn("h", result["ete"])
        self.assertIn("minute", result["ete_spoken"])

    def test_range_warning_uses_the_spoken_form(self):
        # The warning is prose the model copies, so the duration inside
        # it should be prose too rather than a token to reformat.
        result = dispatch(
            "plan_flight",
            {"origin": "KJFK", "dest": "KLAX", "aircraft": "c172", "use_wind": False},
        )
        self.assertIn("hours", result["range_warning"])
        self.assertNotIn("h02m", result["range_warning"])
