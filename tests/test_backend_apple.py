"""Tests for the Apple Foundation Models backend.

Split in two. The schema-shaping logic is pure and always runs. The tests
that need the on-device model are skipped when the SDK or Apple
Intelligence is unavailable, so the suite still passes on any machine.
"""

import unittest

from flight_dispatch.tools import TOOLS_BY_NAME

try:
    from flight_dispatch.backend_apple import (
        SDK_AVAILABLE,
        AppleBackend,
        AppleBackendUnavailable,
        _is_exposed,
    )
except ImportError:  # pragma: no cover
    SDK_AVAILABLE = False


def _model_ready() -> bool:
    if not SDK_AVAILABLE:
        return False
    try:
        AppleBackend()
        return True
    except Exception:  # noqa: BLE001
        return False


MODEL_READY = _model_ready()
requires_model = unittest.skipUnless(
    MODEL_READY, "on-device model unavailable (needs macOS 26+, Xcode, Apple Intelligence)"
)


@unittest.skipUnless(SDK_AVAILABLE, "apple-fm-sdk not installed")
class TestParameterExposure(unittest.TestCase):
    """Apple's schema cannot express optionality, so an exposed parameter
    is one the model WILL fill. These guard the rule that keeps it from
    inventing harmful values."""

    def test_required_parameters_are_exposed(self):
        self.assertTrue(_is_exposed({"type": "string", "required": True}))

    def test_booleans_are_exposed(self):
        # Two possible values, both survivable.
        self.assertTrue(_is_exposed({"type": "boolean", "required": False}))

    def test_enums_are_exposed(self):
        # anyOf constrains the model to real values.
        self.assertTrue(
            _is_exposed({"type": "string", "required": False, "enum": ["a", "b"]})
        )

    def test_free_numbers_are_withheld(self):
        # The regression this rule exists for: the model supplied
        # payload_lb=1600 unprompted, exceeding a Cessna 172's entire
        # 870 lb useful load, and the plan was refused.
        self.assertFalse(_is_exposed({"type": "number", "required": False}))

    def test_free_strings_are_withheld(self):
        self.assertFalse(_is_exposed({"type": "string", "required": False}))

    def test_plan_flight_withholds_free_numerics(self):
        # payload_lb has no sensible enum -- any weight is arguable -- so
        # it stays withheld, which is what stopped the model inventing
        # 1,600 lb for a Cessna 172.
        params = TOOLS_BY_NAME["plan_flight"].parameters
        self.assertFalse(_is_exposed(params["payload_lb"]))

    def test_altitude_reaches_the_model_where_it_is_the_question(self):
        # Withholding altitude_ft everywhere was wrong. Asked for the
        # wind at 35,000 ft, get_winds_aloft could not be given one and
        # answered at its 8,000 ft default -- 12 kt from 239 at +11.9 C,
        # reported as the 35,000 ft wind. The real value was 26 kt from
        # 223 at -41 C. Wind without an altitude is meaningless, so an
        # enum lets it through while still stopping invented values.
        for name in ("get_winds_aloft", "check_airspace"):
            spec = TOOLS_BY_NAME[name].parameters["altitude_ft"]
            self.assertTrue(_is_exposed(spec), name)
            self.assertIn("enum", spec, name)

    def test_plan_flight_offers_no_altitude_to_any_backend(self):
        # And exposing it everywhere was also wrong. Given "KJFK to KLAX
        # in a 737" the model volunteered altitude_ft=30000 unasked, then
        # carried it into "what about in a 172?" -- an aircraft with a
        # 14,000 ft ceiling -- and the plan was refused.
        #
        # That was first fixed by removing an enum so _is_exposed would
        # withhold it, which put a decision about the TOOL inside ONE
        # BACKEND. Ollama's JSON Schema can express optionality, so it
        # rebuilt the parameter and volunteered 30,000 ft for a Cirrus.
        # The parameter is now simply not declared, so no backend can
        # offer it.
        self.assertNotIn("altitude_ft", TOOLS_BY_NAME["plan_flight"].parameters)

    def test_plan_flight_still_exposes_what_the_model_needs(self):
        params = TOOLS_BY_NAME["plan_flight"].parameters
        for name in ("origin", "dest", "aircraft", "use_wind", "avoid_airspace"):
            with self.subTest(name):
                self.assertTrue(_is_exposed(params[name]))

    def test_aircraft_is_an_enum_of_real_keys(self):
        # Without this it would be a free string and get withheld, so
        # "in a Cirrus" could never reach the tool.
        from flight_dispatch.aircraft import AIRCRAFT

        spec = TOOLS_BY_NAME["plan_flight"].parameters["aircraft"]
        self.assertIn("enum", spec)
        self.assertEqual(set(spec["enum"]), set(AIRCRAFT))


@requires_model
class TestBackendWiring(unittest.TestCase):
    def setUp(self):
        self.backend = AppleBackend()

    def test_reports_context_size(self):
        self.assertGreater(self.backend.context_size, 0)

    def test_start_builds_one_bridge_tool_per_spec(self):
        tools = [TOOLS_BY_NAME["find_airport"], TOOLS_BY_NAME["list_aircraft"]]
        self.backend.start("You are a test.", tools)
        self.assertEqual(
            [t.name for t in self.backend.bridge_tools],
            ["find_airport", "list_aircraft"],
        )

    def test_send_tool_results_is_unreachable(self):
        # Apple runs the loop internally, so the hand-written loop never
        # calls this. Asserting it stays unreachable documents the
        # architectural difference.
        self.backend.start("test", [TOOLS_BY_NAME["find_airport"]])
        with self.assertRaises(AssertionError):
            self.backend.send_tool_results([])

    def test_reset_clears_recorded_calls(self):
        self.backend.start("test", [TOOLS_BY_NAME["find_airport"]])
        self.backend.record_call("id1", "find_airport", {"query": "KORD"})
        self.assertEqual(len(self.backend.calls_this_turn), 1)
        self.backend.reset()
        self.assertEqual(self.backend.calls_this_turn, [])


@requires_model
class TestLiveModel(unittest.TestCase):
    """Actually generates. Slow (tens of seconds) and non-deterministic,
    so assertions are on behaviour, never on exact wording."""

    def test_model_calls_a_tool_rather_than_answering_from_memory(self):
        backend = AppleBackend()
        backend.start(
            "You are a flight dispatch assistant. Use tools for every fact.",
            [TOOLS_BY_NAME["find_airport"]],
        )
        backend.send_user_message("What airport is KORD?")

        self.assertEqual(
            [c.name for c in backend.calls_this_turn], ["find_airport"]
        )
        self.assertEqual(backend.calls_this_turn[0].arguments["query"].upper(), "KORD")

    def test_tool_result_reaches_the_engine(self):
        backend = AppleBackend()
        backend.start("Use tools for every fact.", [TOOLS_BY_NAME["find_airport"]])
        backend.send_user_message("What airport is KORD?")

        result = backend.results_this_turn[0]
        self.assertEqual(result.content["icao"], "KORD")



@unittest.skipUnless(SDK_AVAILABLE, "apple-fm-sdk not installed")
class TestCallResultPairing(unittest.TestCase):
    """Apple runs tools concurrently, so calls and results can be recorded
    in different orders. Pairing by position mispaired them -- a
    "Minneapolis" lookup was displayed showing Chicago Executive's
    result. These pin the id-based pairing."""

    def setUp(self):
        if not MODEL_READY:
            self.skipTest("model unavailable")
        self.backend = AppleBackend()

    def test_ids_are_unique_per_call(self):
        ids = {self.backend.next_call_id() for _ in range(5)}
        self.assertEqual(len(ids), 5)

    def test_pairing_survives_out_of_order_results(self):
        first = self.backend.next_call_id()
        second = self.backend.next_call_id()

        self.backend.record_call(first, "find_airport", {"query": "Minneapolis"})
        self.backend.record_call(second, "find_airport", {"query": "Chicago Executive"})
        # Results arrive in the opposite order, as concurrent tools can.
        self.backend.record_result(second, "find_airport", {"icao": "KPWK"})
        self.backend.record_result(first, "find_airport", {"match_count": 3})

        pairs = dict(
            (call.arguments["query"], result.content)
            for call, result in self.backend.paired_calls()
        )
        self.assertEqual(pairs["Chicago Executive"]["icao"], "KPWK")
        self.assertEqual(pairs["Minneapolis"]["match_count"], 3)

    def test_unfinished_calls_are_omitted(self):
        call_id = self.backend.next_call_id()
        self.backend.record_call(call_id, "find_airport", {"query": "KORD"})
        self.assertEqual(self.backend.paired_calls(), [])


if __name__ == "__main__":
    unittest.main()


class TestContextDisplay(unittest.TestCase):
    """The context meter shown after every turn.

    The on-device window is 4,096 tokens and nothing else reveals it
    filling. Measuring it is also how the oversized find_airport result
    was found -- one lookup was 41% of a whole conversation.
    """

    def usage(self, total=1000, limit=4096, **roles):
        return {
            "total": total,
            "limit": limit,
            "percent": 100.0 * total / limit,
            "by_role": roles or {"user": total},
        }

    def test_no_usage_renders_nothing(self):
        import sys, pathlib

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
        from dispatch import format_context

        self.assertEqual(format_context(None, None), "")

    def test_shows_total_and_limit(self):
        from dispatch import format_context

        line = format_context(self.usage(2781, 4096), None)
        self.assertIn("2,781/4,096", line)

    def test_bar_grows_with_usage(self):
        from dispatch import format_context

        empty = format_context(self.usage(0, 4096), None)
        full = format_context(self.usage(4096, 4096), None)
        self.assertLess(empty.count("█"), full.count("█"))

    def test_bar_does_not_overflow_past_the_limit(self):
        from dispatch import format_context

        line = format_context(self.usage(9999, 4096), None)
        self.assertLessEqual(line.count("█"), 20)

    def test_delta_appears_once_there_is_a_previous_total(self):
        from dispatch import format_context

        self.assertNotIn("this turn", format_context(self.usage(2000), None))
        self.assertIn("+500", format_context(self.usage(2000), 1500))

    def test_breakdown_is_ordered_largest_first(self):
        from dispatch import format_context

        line = format_context(
            self.usage(1000, instructions=700, tool=200, user=100), None
        )
        self.assertLess(line.index("instructions"), line.index("tool"))
        self.assertLess(line.index("tool"), line.index("user"))
