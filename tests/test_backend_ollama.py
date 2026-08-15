"""Tests for the Ollama backend.

Structured like test_backend_apple.py: the message-shaping logic is pure
and always runs against a fake HTTP session, and the tests that need a
real server are skipped when one is not listening. The suite passes on a
machine with no Ollama installed.

The point of this backend is not the larger context window. It is that
Ollama returns tool calls for the CALLER to run, so `agent.py`'s
hand-written loop actually drives the conversation -- against the Apple
backend that loop makes one pass and exits, because Apple's SDK does the
orchestrating itself.
"""

import copy
import json
import unittest
from typing import Any, Dict, List

from flight_dispatch.agent import DispatcherAgent, ToolResult
from flight_dispatch.tools import TOOLS_BY_NAME

from flight_dispatch.backend_ollama import (
    DEFAULT_NUM_CTX,
    OllamaBackend,
    OllamaUnavailable,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Stands in for `requests.Session`, recording what was sent."""

    def __init__(self, replies=None, models=("llama3.1",)):
        self.replies = list(replies or [])
        self.models = models
        self.posts: List[Dict[str, Any]] = []

    def get(self, url, timeout=None):
        return FakeResponse(
            {"models": [{"name": f"{m}:latest"} for m in self.models]}
        )

    def post(self, url, json=None, timeout=None):
        # Deep-copy: the payload holds a reference to the backend's live
        # message list, so recording it directly would show every post
        # with the final state of the conversation.
        self.posts.append(copy.deepcopy(json))
        reply = self.replies.pop(0) if self.replies else {"content": "done."}
        return FakeResponse({"message": reply})


def backend(**kwargs) -> OllamaBackend:
    kwargs.setdefault("session", FakeSession())
    return OllamaBackend(**kwargs)


class TestAvailability(unittest.TestCase):
    """A connection error mid-conversation is a stack trace; the same
    condition caught at construction is one actionable line."""

    def test_unreachable_server_says_how_to_start_it(self):
        class Dead:
            def get(self, *a, **k):
                raise OSError("connection refused")

        with self.assertRaises(OllamaUnavailable) as caught:
            OllamaBackend(session=Dead())
        self.assertIn("ollama serve", str(caught.exception))

    def test_unreachable_server_offers_the_other_backend(self):
        class Dead:
            def get(self, *a, **k):
                raise OSError("connection refused")

        with self.assertRaises(OllamaUnavailable) as caught:
            OllamaBackend(session=Dead())
        self.assertIn("--backend apple", str(caught.exception))

    def test_missing_model_says_how_to_pull_it(self):
        session = FakeSession(models=("qwen2.5",))
        with self.assertRaises(OllamaUnavailable) as caught:
            OllamaBackend(model="llama3.1", session=session)
        self.assertIn("ollama pull llama3.1", str(caught.exception))

    def test_installed_model_is_accepted(self):
        self.assertIsNotNone(OllamaBackend(session=FakeSession(models=("llama3.1",))))

    def test_a_tagged_model_name_still_matches(self):
        # `ollama list` reports "llama3.1:latest"; users type "llama3.1".
        OllamaBackend(model="llama3.1:8b", session=FakeSession(models=("llama3.1",)))


class TestContextWindow(unittest.TestCase):
    """Ollama does not give a model its advertised window by default:
    0.32 sizes it from VRAM and reported 4,096 on an M3, identical to the
    on-device model. It truncates silently, so this must be explicit."""

    def test_context_is_set_explicitly_and_is_large(self):
        self.assertGreaterEqual(DEFAULT_NUM_CTX, 32768)

    def test_num_ctx_is_sent_on_every_request(self):
        source = backend()
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("hello")
        self.assertEqual(
            source.session.posts[0]["options"]["num_ctx"], DEFAULT_NUM_CTX
        )

    def test_context_beats_the_on_device_model(self):
        from flight_dispatch.backend_apple import CONTEXT_LIMIT_TOKENS

        self.assertGreater(backend().context_size, CONTEXT_LIMIT_TOKENS)

    def test_temperature_is_zero_by_default(self):
        # Tool selection is a decision, not a creative act.
        source = backend()
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("hello")
        self.assertEqual(source.session.posts[0]["options"]["temperature"], 0.0)


class TestToolSchemas(unittest.TestCase):
    """ToolSpec.json_schema() already emits what Ollama wants, because
    that format is OpenAI's and so is Claude's."""

    def schemas(self):
        source = backend()
        source.start("test", [TOOLS_BY_NAME["plan_flight"]])
        return source.tool_schemas

    def test_shape(self):
        schema = self.schemas()[0]
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "plan_flight")

    def test_description_is_carried_through(self):
        # It is prompt text, and it is the only thing the model reads to
        # decide whether to call the tool.
        self.assertIn("plan", self.schemas()[0]["function"]["description"].lower())

    def test_required_arguments_are_declared(self):
        parameters = self.schemas()[0]["function"]["parameters"]
        self.assertEqual(sorted(parameters["required"]), ["dest", "origin"])

    def test_optional_numerics_survive_here(self):
        # Apple withholds these because its schema cannot express
        # optionality. JSON Schema can, so payload_lb is offered without
        # the model being forced to invent a value.
        parameters = self.schemas()[0]["function"]["parameters"]
        self.assertIn("payload_lb", parameters["properties"])
        self.assertNotIn("payload_lb", parameters["required"])


class TestToolCallParsing(unittest.TestCase):
    def response_for(self, message):
        source = backend()
        source.session.replies = [message]
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        return source.send_user_message("hi")

    def test_prose_reply_wants_no_tools(self):
        reply = self.response_for({"content": "KORD is Chicago O'Hare."})
        self.assertFalse(reply.wants_tools)
        self.assertIn("O'Hare", reply.text)

    def test_a_tool_call_is_read(self):
        reply = self.response_for(
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "KORD"}}}
                ]
            }
        )
        self.assertTrue(reply.wants_tools)
        self.assertEqual(reply.tool_calls[0].name, "find_airport")
        self.assertEqual(reply.tool_calls[0].arguments["query"], "KORD")

    def test_arguments_as_a_json_string_are_accepted(self):
        # Well-behaved models send a dict; some send a JSON string.
        reply = self.response_for(
            {
                "tool_calls": [
                    {
                        "function": {
                            "name": "find_airport",
                            "arguments": '{"query": "KORD"}',
                        }
                    }
                ]
            }
        )
        self.assertEqual(reply.tool_calls[0].arguments["query"], "KORD")

    def test_malformed_arguments_do_not_raise(self):
        # dispatch() will answer with a readable error about the missing
        # argument, which the model can recover from. Raising would end
        # the conversation.
        reply = self.response_for(
            {"tool_calls": [{"function": {"name": "find_airport", "arguments": "{{{"}}]}
        )
        self.assertEqual(reply.tool_calls[0].arguments, {})

    def test_parallel_calls_get_distinct_ids(self):
        reply = self.response_for(
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "A"}}},
                    {"function": {"name": "find_airport", "arguments": {"query": "B"}}},
                ]
            }
        )
        ids = [call.id for call in reply.tool_calls]
        self.assertEqual(len(set(ids)), 2)


class TestHistory(unittest.TestCase):
    """Ollama is stateless per request, like the Claude API, so the whole
    conversation is resent every time and lives here."""

    def test_system_prompt_leads(self):
        source = backend()
        source.start("You are a dispatcher.", [TOOLS_BY_NAME["find_airport"]])
        self.assertEqual(source.messages[0]["role"], "system")

    def test_turns_accumulate(self):
        source = backend()
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("first")
        source.send_user_message("second")
        roles = [m["role"] for m in source.messages]
        self.assertEqual(roles, ["system", "user", "assistant", "user", "assistant"])

    def test_every_message_carries_a_role(self):
        # The whole history is resent on the next request, and a message
        # without a role would be rejected -- poisoning the conversation
        # from that point rather than failing where the mistake was.
        source = backend()
        source.session.replies = [{"content": "no role field here"}]
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("hello")
        for message in source.messages:
            self.assertIn("role", message)

    def test_tool_results_are_appended_as_tool_messages(self):
        source = backend()
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        source.send_tool_results(
            [ToolResult(call_id="call_1", name="find_airport", content={"icao": "KORD"})]
        )
        tool_message = source.messages[1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertEqual(json.loads(tool_message["content"])["icao"], "KORD")

    def test_reset_keeps_the_system_prompt(self):
        source = backend()
        source.start("You are a dispatcher.", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("hello")
        source.reset()
        self.assertEqual(len(source.messages), 1)
        self.assertEqual(source.messages[0]["role"], "system")

    def test_full_history_is_resent_each_time(self):
        source = backend()
        source.start("test", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("first")
        source.send_user_message("second")
        self.assertGreater(
            len(source.session.posts[1]["messages"]),
            len(source.session.posts[0]["messages"]),
        )


class TestTheLoopActuallyRuns(unittest.TestCase):
    """THE REASON THIS BACKEND EXISTS.

    Against the Apple backend, `send_tool_results` is unreachable --
    asserted in its own tests -- because Apple's SDK runs tools itself
    and `agent.py`'s loop makes a single pass. Here the loop drives.
    """

    def test_loop_executes_a_tool_and_comes_back(self):
        source = backend()
        source.session.replies = [
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "KORD"}}}
                ]
            },
            {"content": "KORD is Chicago O'Hare International Airport."},
        ]

        agent = DispatcherAgent(source, tools=[TOOLS_BY_NAME["find_airport"]])
        turn = agent.ask("What is KORD?")

        self.assertEqual(turn.rounds, 1)
        self.assertEqual(len(turn.tool_calls), 1)
        self.assertIn("O'Hare", turn.reply)

    def test_the_tool_really_ran(self):
        source = backend()
        source.session.replies = [
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "KORD"}}}
                ]
            },
            {"content": "done"},
        ]
        agent = DispatcherAgent(source, tools=[TOOLS_BY_NAME["find_airport"]])
        turn = agent.ask("What is KORD?")

        # The result came from the real engine, not a stub.
        self.assertEqual(turn.tool_results[0].content["icao"], "KORD")

    def test_several_rounds(self):
        source = backend()
        source.session.replies = [
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "KORD"}}}
                ]
            },
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "KMIA"}}}
                ]
            },
            {"content": "Both found."},
        ]
        agent = DispatcherAgent(source, tools=[TOOLS_BY_NAME["find_airport"]])
        turn = agent.ask("Look up two airports")
        self.assertEqual(turn.rounds, 2)

    def test_runaway_protection_still_applies(self):
        source = backend()
        # Always asks for another tool, never concludes.
        source.session.replies = [
            {
                "tool_calls": [
                    {"function": {"name": "find_airport", "arguments": {"query": "KORD"}}}
                ]
            }
        ] * 50
        agent = DispatcherAgent(
            source, tools=[TOOLS_BY_NAME["find_airport"]], max_rounds=3
        )
        turn = agent.ask("loop forever")
        self.assertIn("Stopped after 3 rounds", turn.reply)


class TestContextMeter(unittest.TestCase):
    """The CLI shows the same meter for either backend."""

    def usage(self):
        source = backend()
        source.start("You are a dispatcher.", [TOOLS_BY_NAME["find_airport"]])
        source.send_user_message("plan a flight")
        return source.context_usage()

    def test_reports_totals_against_the_real_limit(self):
        usage = self.usage()
        self.assertGreater(usage["total"], 0)
        self.assertEqual(usage["limit"], DEFAULT_NUM_CTX)

    def test_counts_the_tool_schemas(self):
        # They are resent on every request and are a large fixed cost --
        # a third of the window on the on-device model, which was only
        # visible once counted.
        self.assertGreater(self.usage()["by_role"]["tools"], 0)

    def test_no_usage_before_a_conversation_starts(self):
        self.assertIsNone(backend().context_usage())


if __name__ == "__main__":
    unittest.main()
