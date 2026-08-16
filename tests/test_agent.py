import unittest
from typing import Any, Dict, List, Sequence

from flight_dispatch.agent import (
    DEFAULT_SYSTEM_PROMPT,
    DispatcherAgent,
    ModelResponse,
    ToolCall,
    ToolResult,
)
from flight_dispatch.tools import TOOLS_BY_NAME, ToolSpec


class ScriptedBackend:
    """A model backend that replays a fixed script.

    The loop's logic is independent of which model drives it, so a
    scripted backend tests the orchestration exactly -- deterministically,
    offline, and with no API cost. This is the same trick
    `ConstantWindSource` plays for the wind maths.

    Args:
        script: One ModelResponse per call the backend receives, in
            order. The loop is expected to consume them all.
    """

    name = "scripted"

    def __init__(self, script: Sequence[ModelResponse]):
        self.script = list(script)
        self.calls = 0
        self.system_prompt = ""
        self.tools: List[ToolSpec] = []
        self.user_messages: List[str] = []
        self.results_received: List[List[ToolResult]] = []

    def start(self, system_prompt: str, tools: Sequence[ToolSpec]) -> None:
        self.system_prompt = system_prompt
        self.tools = list(tools)

    def _next(self) -> ModelResponse:
        if self.calls >= len(self.script):
            raise AssertionError(
                f"Backend called {self.calls + 1} times but script has "
                f"{len(self.script)} entries"
            )
        response = self.script[self.calls]
        self.calls += 1
        return response

    def send_user_message(self, text: str) -> ModelResponse:
        self.user_messages.append(text)
        return self._next()

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelResponse:
        self.results_received.append(list(results))
        return self._next()


def text(message: str) -> ModelResponse:
    return ModelResponse(text=message)


def calls(*specs) -> ModelResponse:
    """Build a tool-call response from (name, args) pairs."""
    return ModelResponse(
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(specs)
        ]
    )


class TestLoopBasics(unittest.TestCase):
    def test_reply_without_tools(self):
        backend = ScriptedBackend([text("Hello, I plan flights.")])
        agent = DispatcherAgent(backend)

        turn = agent.ask("hi")

        self.assertEqual(turn.reply, "Hello, I plan flights.")
        self.assertEqual(turn.tool_calls, [])
        self.assertEqual(turn.rounds, 0)

    def test_backend_receives_system_prompt_and_tools(self):
        backend = ScriptedBackend([text("ok")])
        DispatcherAgent(backend)

        self.assertEqual(backend.system_prompt, DEFAULT_SYSTEM_PROMPT)
        self.assertGreater(len(backend.tools), 0)

    def test_custom_system_prompt(self):
        backend = ScriptedBackend([text("ok")])
        DispatcherAgent(backend, system_prompt="Be terse.")
        self.assertEqual(backend.system_prompt, "Be terse.")

    def test_user_message_reaches_the_backend(self):
        backend = ScriptedBackend([text("ok")])
        DispatcherAgent(backend).ask("plan KPWK to KMSP")
        self.assertEqual(backend.user_messages, ["plan KPWK to KMSP"])


class TestToolExecution(unittest.TestCase):
    def test_single_tool_call_then_answer(self):
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("That's O'Hare."),
        ])
        agent = DispatcherAgent(backend)

        turn = agent.ask("what is KORD")

        self.assertEqual(turn.reply, "That's O'Hare.")
        self.assertEqual(turn.rounds, 1)
        self.assertEqual([c.name for c in turn.tool_calls], ["find_airport"])

    def test_tool_actually_ran(self):
        # The result must be real output from the real function, not a
        # placeholder -- that is the whole point of the tool layer.
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("done"),
        ])
        turn = DispatcherAgent(backend).ask("?")

        result = turn.tool_results[0]
        self.assertTrue(result.content["found"])
        self.assertEqual(result.content["icao"], "KORD")

    def test_results_are_returned_to_the_backend(self):
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("done"),
        ])
        DispatcherAgent(backend).ask("?")

        self.assertEqual(len(backend.results_received), 1)
        self.assertEqual(backend.results_received[0][0].name, "find_airport")

    def test_parallel_tool_calls_run_together(self):
        # A model resolving two airports at once should get both results
        # back in a single batch, which is what both APIs expect.
        backend = ScriptedBackend([
            calls(
                ("find_airport", {"query": "KORD"}),
                ("find_airport", {"query": "KMSP"}),
            ),
            text("Both found."),
        ])
        turn = DispatcherAgent(backend).ask("?")

        self.assertEqual(len(turn.tool_calls), 2)
        self.assertEqual(turn.rounds, 1)
        self.assertEqual(len(backend.results_received[0]), 2)

    def test_multiple_rounds(self):
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "Chicago Executive"})),
            calls(("find_airport", {"query": "Minneapolis"})),
            text("Planned."),
        ])
        turn = DispatcherAgent(backend).ask("?")

        self.assertEqual(turn.rounds, 2)
        self.assertEqual(len(turn.tool_calls), 2)

    def test_call_ids_are_preserved(self):
        # Results must be matchable back to their requests.
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("done"),
        ])
        turn = DispatcherAgent(backend).ask("?")
        self.assertEqual(turn.tool_calls[0].id, turn.tool_results[0].call_id)


class TestErrorHandling(unittest.TestCase):
    def test_tool_error_is_returned_not_raised(self):
        backend = ScriptedBackend([
            calls(("plan_flight", {"origin": "ZZZZ", "dest": "KORD"})),
            text("That origin isn't valid."),
        ])
        turn = DispatcherAgent(backend).ask("?")

        self.assertTrue(turn.tool_results[0].is_error)
        self.assertEqual(turn.reply, "That origin isn't valid.")

    def test_unknown_tool_does_not_crash(self):
        backend = ScriptedBackend([
            calls(("teleport", {})),
            text("No such tool."),
        ])
        turn = DispatcherAgent(backend).ask("?")

        self.assertTrue(turn.tool_results[0].is_error)
        self.assertIn("available_tools", turn.tool_results[0].content)

    def test_model_can_recover_after_an_error(self):
        # The realistic recovery path: bad call, error, corrected call.
        backend = ScriptedBackend([
            calls(("plan_flight", {"origin": "Chicago", "dest": "KMSP"})),
            calls(("find_airport", {"query": "Chicago"})),
            text("Recovered."),
        ])
        turn = DispatcherAgent(backend).ask("?")

        self.assertTrue(turn.tool_results[0].is_error)
        self.assertFalse(turn.tool_results[1].is_error)
        self.assertEqual(turn.reply, "Recovered.")


class TestRunawayProtection(unittest.TestCase):
    def test_max_rounds_stops_an_endless_loop(self):
        # A model that never stops calling tools.
        backend = ScriptedBackend(
            [calls(("find_airport", {"query": "KORD"}))] * 20
        )
        agent = DispatcherAgent(backend, max_rounds=3)

        turn = agent.ask("?")

        self.assertIn("Stopped after 3 rounds", turn.reply)
        self.assertEqual(turn.rounds, 4)  # the round that tripped the limit

    def test_max_rounds_does_not_fire_on_normal_conversations(self):
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("done"),
        ])
        turn = DispatcherAgent(backend, max_rounds=3).ask("?")
        self.assertNotIn("Stopped after", turn.reply)


class TestConversationHistory(unittest.TestCase):
    def test_turns_accumulate(self):
        backend = ScriptedBackend([text("one"), text("two"), text("three")])
        agent = DispatcherAgent(backend)

        agent.ask("first")
        agent.ask("second")
        agent.ask("third")

        self.assertEqual(len(agent.turns), 3)
        self.assertEqual([t.user_message for t in agent.turns],
                         ["first", "second", "third"])
        self.assertEqual([t.reply for t in agent.turns], ["one", "two", "three"])

    def test_tool_calls_tallied_across_turns(self):
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("one"),
            calls(("find_airport", {"query": "KMSP"})),
            text("two"),
        ])
        agent = DispatcherAgent(backend)
        agent.ask("first")
        agent.ask("second")

        self.assertEqual(agent.total_tool_calls, 2)
        self.assertEqual(agent.tool_call_names(), ["find_airport", "find_airport"])


class TestObserverCallbacks(unittest.TestCase):
    def test_callbacks_fire(self):
        seen_calls, seen_results = [], []
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("done"),
        ])
        agent = DispatcherAgent(
            backend,
            on_tool_call=seen_calls.append,
            on_tool_result=seen_results.append,
        )
        agent.ask("?")

        self.assertEqual([c.name for c in seen_calls], ["find_airport"])
        self.assertEqual(len(seen_results), 1)

    def test_loop_works_without_callbacks(self):
        backend = ScriptedBackend([
            calls(("find_airport", {"query": "KORD"})),
            text("done"),
        ])
        self.assertEqual(DispatcherAgent(backend).ask("?").reply, "done")


class TestToolSubset(unittest.TestCase):
    def test_agent_can_be_given_a_restricted_tool_set(self):
        from flight_dispatch.tools import TOOLS_BY_NAME

        backend = ScriptedBackend([text("ok")])
        DispatcherAgent(backend, tools=[TOOLS_BY_NAME["find_airport"]])

        self.assertEqual([t.name for t in backend.tools], ["find_airport"])


if __name__ == "__main__":
    unittest.main()


class TestRepeatedFailureEscalation(unittest.TestCase):
    """THE LOOP THIS BREAKS.

    Observed in a real session: list_aircraft(category='wide-body')
    failed, the error listed the five valid categories, and the very next
    call was list_aircraft(category='wide-body'). The system prompt
    already said not to repeat a failing call.

    The same error text, returned twice, reads to a model as the same
    situation -- so it tries the same thing again. Changing the message
    changes the situation. By the time that session recovered it had
    forgotten half the request and never planned the flight.
    """

    def agent(self, replies, tools=("list_aircraft",)):
        backend = ScriptedBackend(replies)
        return backend, DispatcherAgent(
            backend, tools=[TOOLS_BY_NAME[name] for name in tools]
        )

    def failing(self, call_id):
        return ModelResponse(
            tool_calls=[
                ToolCall(call_id, "list_aircraft", {"category": "wide-body"})
            ]
        )

    def test_the_first_failure_is_reported_normally(self):
        _, agent = self.agent([self.failing("1"), ModelResponse(text="done")])
        turn = agent.ask("find a widebody")
        self.assertIn("No aircraft in category", turn.tool_results[0].content["error"])

    def test_an_identical_second_failure_is_escalated(self):
        _, agent = self.agent(
            [self.failing("1"), self.failing("2"), ModelResponse(text="done")]
        )
        turn = agent.ask("find a widebody")
        second = turn.tool_results[1].content["error"]
        self.assertIn("already called", second)
        self.assertIn("third time", second)

    def test_the_original_error_is_preserved(self):
        # The model still needs to know WHAT failed, not only that it
        # repeated itself.
        _, agent = self.agent(
            [self.failing("1"), self.failing("2"), ModelResponse(text="done")]
        )
        turn = agent.ask("find a widebody")
        self.assertIn(
            "No aircraft in category", turn.tool_results[1].content["previous_error"]
        )

    def test_helpful_fields_survive_the_escalation(self):
        # valid_categories is the thing that would let it recover.
        _, agent = self.agent(
            [self.failing("1"), self.failing("2"), ModelResponse(text="done")]
        )
        turn = agent.ask("find a widebody")
        self.assertIn("valid_categories", turn.tool_results[1].content)

    def test_different_arguments_are_not_escalated(self):
        # A model narrowing in on the right arguments is making progress
        # and should be left alone.
        backend = ScriptedBackend(
            [
                ModelResponse(
                    tool_calls=[ToolCall("1", "list_aircraft", {"category": "wide"})]
                ),
                ModelResponse(
                    tool_calls=[ToolCall("2", "list_aircraft", {"category": "big"})]
                ),
                ModelResponse(text="done"),
            ]
        )
        agent = DispatcherAgent(backend, tools=[TOOLS_BY_NAME["list_aircraft"]])
        turn = agent.ask("find a widebody")
        self.assertNotIn("already called", turn.tool_results[1].content["error"])

    def test_argument_order_does_not_hide_a_repeat(self):
        backend = ScriptedBackend(
            [
                ModelResponse(
                    tool_calls=[
                        ToolCall("1", "plan_flight", {"origin": "ZZZZ", "dest": "YYYY"})
                    ]
                ),
                ModelResponse(
                    tool_calls=[
                        ToolCall("2", "plan_flight", {"dest": "YYYY", "origin": "ZZZZ"})
                    ]
                ),
                ModelResponse(text="done"),
            ]
        )
        agent = DispatcherAgent(backend, tools=[TOOLS_BY_NAME["plan_flight"]])
        turn = agent.ask("plan it")
        self.assertIn("already called", turn.tool_results[1].content["error"])

    def test_a_successful_call_is_never_escalated(self):
        backend = ScriptedBackend(
            [
                ModelResponse(
                    tool_calls=[ToolCall("1", "list_aircraft", {"category": "ga"})]
                ),
                ModelResponse(
                    tool_calls=[ToolCall("2", "list_aircraft", {"category": "ga"})]
                ),
                ModelResponse(text="done"),
            ]
        )
        agent = DispatcherAgent(backend, tools=[TOOLS_BY_NAME["list_aircraft"]])
        turn = agent.ask("list them")
        self.assertNotIn("error", turn.tool_results[1].content)


class TestWholeRequestRule(unittest.TestCase):
    """A checklist is not a plan, and a plan is not a checklist.

    The checklist rule was written to stop the model writing one from
    memory. It worked, and then it overcorrected: asked to "plan a flight
    from KSFO to KEWR on a 777 and give me a checklist", the model called
    find_procedures and never called plan_flight at all.
    """

    def test_the_prompt_requires_both_calls(self):
        from flight_dispatch.agent import DEFAULT_SYSTEM_PROMPT

        prompt = DEFAULT_SYSTEM_PROMPT.lower()
        self.assertIn("a checklist is not a plan", prompt)
        self.assertIn("you need both", prompt)

    def test_the_prompt_says_failures_do_not_excuse_dropping_half(self):
        from flight_dispatch.agent import DEFAULT_SYSTEM_PROMPT

        self.assertIn("do not excuse dropping half", DEFAULT_SYSTEM_PROMPT)
