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

    def test_plan_flight_withholds_the_dangerous_numerics(self):
        params = TOOLS_BY_NAME["plan_flight"].parameters
        self.assertFalse(_is_exposed(params["payload_lb"]))
        self.assertFalse(_is_exposed(params["altitude_ft"]))

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
        self.backend.record_call("find_airport", {"query": "KORD"})
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


if __name__ == "__main__":
    unittest.main()
