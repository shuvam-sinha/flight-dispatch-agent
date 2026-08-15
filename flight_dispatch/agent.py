"""The dispatcher agent: conversation history and the tool-use loop.

WHAT AN AGENT LOOP ACTUALLY IS
------------------------------
Less than it sounds like. The whole thing:

    1. Send the conversation, plus a description of every tool, to a model.
    2. The model replies either with prose (it is done) or with a request
       to call one or more tools.
    3. If it asked for tools, run them, append the results to the
       conversation, and go back to step 1.

That is the entire mechanism behind "AI agents". Everything else -- the
routing, the wind, the airspace -- is ordinary Python that was already
written and tested before any model existed. The model never computes
anything; it decides which function to call and narrates what came back.

WHY THIS IS HAND-WRITTEN
------------------------
Both Anthropic's SDK and Apple's provide a helper that runs this loop for
you (`client.beta.messages.tool_runner` and `LanguageModelSession`
respectively). Using one would be the right call in production, and would
be about ten lines.

It is written by hand here for the same reason A* was written by hand
instead of imported from networkx: the orchestration is the thing the
project exists to demonstrate. The helpers are noted so it is clear the
choice was deliberate rather than uninformed.

WHY THE HISTORY IS RESENT EVERY TIME
------------------------------------
Chat APIs are stateless. The model retains nothing between calls, so
"memory" is just the client resending the entire conversation on every
request. That is what makes a follow-up like "what if I flew the 172
instead" work -- the model can see the previous turn because it is in the
list being sent, not because anything remembered it.

BACKEND-NEUTRAL
---------------
`ModelBackend` is a narrow protocol: given messages and tools, return
either text or tool calls. Anything satisfying it can drive the loop --
the Claude API, Apple's on-device Foundation Models, or a scripted fake
in a test. The loop, the tools, and the history handling are shared; only
schema formatting and response parsing differ per backend.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence

from .tools import TOOLS, ToolSpec, dispatch

DEFAULT_SYSTEM_PROMPT = """\
You are a flight dispatch assistant for pilots and dispatchers planning \
general-aviation and commercial flights.

You have tools that perform real routing, weather and airspace \
computation. Use them. Never estimate a distance, flight time, fuel \
figure or heading yourself -- every number you report must come from a \
tool result. If you do not have a tool for something, say so plainly \
rather than guessing.

Working style:
- When the user names an airport in plain language, call find_airport to \
resolve it to an ICAO code before planning.
- When they name an aircraft loosely, call list_aircraft to find the key.
- Prefer one plan_flight call over several: it already applies winds \
aloft and airspace avoidance internally.
- If a tool returns an error, read it and correct your approach rather \
than repeating the same call.
- Call tools through the tool mechanism. Never write a tool call as JSON \
in your reply -- a call written as text does not run.

When reporting a plan, lead with the outcome -- the route, time and fuel \
-- then add detail. Mention the waypoint identifiers, since those are \
what a pilot files. If a result carries a warning (insufficient range, \
airspace detour), say so prominently.

Be concise and factual. You are talking to someone who knows aviation.\
"""


@dataclass
class ToolCall:
    """A model's request to run one tool.

    `id` matters: a model may request several tools at once, and each
    result must be matched back to its request. Both backends supply
    some identifier for this; the loop just carries it through.
    """

    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class ModelResponse:
    """One reply from a model backend.

    Exactly one of these is meaningful per turn: either the model
    produced prose and is finished, or it wants tools run first.
    """

    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw: Any = None

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


class ModelBackend(Protocol):
    """Anything that can drive the loop.

    Deliberately narrow: the loop needs only "given this conversation and
    these tools, what next". A backend owns its own message format
    internally -- the loop never inspects it.
    """

    name: str

    def start(self, system_prompt: str, tools: Sequence[ToolSpec]) -> None:
        """Prepare for a conversation. Called once."""
        ...

    def send_user_message(self, text: str) -> ModelResponse:
        """Add a user turn and get the model's reply."""
        ...

    def send_tool_results(
        self, results: Sequence["ToolResult"]
    ) -> ModelResponse:
        """Return tool output and get the model's next reply."""
        ...


@dataclass
class ToolResult:
    """The outcome of running one tool call."""

    call_id: str
    name: str
    content: Dict[str, Any]

    @property
    def is_error(self) -> bool:
        return "error" in self.content


@dataclass
class Turn:
    """A record of one user message and everything it produced.

    Kept for the CLI's benefit -- it lets the interface show which tools
    ran without the loop having to print anything itself.
    """

    user_message: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    tool_results: List[ToolResult] = field(default_factory=list)
    reply: str = ""
    rounds: int = 0


class DispatcherAgent:
    """Runs the tool-use loop against a model backend.

    Args:
        backend: The model driving the conversation.
        tools: Tools to expose. Defaults to the full registry.
        system_prompt: Overrides the default dispatcher persona.
        max_rounds: Ceiling on tool rounds per user message. A model
            that keeps calling tools without concluding would otherwise
            loop forever; this turns that into a reported failure.
        on_tool_call / on_tool_result: Optional callbacks so a CLI can
            show progress while the loop runs. The loop itself prints
            nothing.
    """

    def __init__(
        self,
        backend: ModelBackend,
        tools: Optional[Sequence[ToolSpec]] = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_rounds: int = 8,
        on_tool_call: Optional[Callable[[ToolCall], None]] = None,
        on_tool_result: Optional[Callable[[ToolResult], None]] = None,
    ):
        self.backend = backend
        self.tools = list(tools) if tools is not None else list(TOOLS)
        self.system_prompt = system_prompt
        self.max_rounds = max_rounds
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result

        self.turns: List[Turn] = []
        self.backend.start(system_prompt, self.tools)

    def ask(self, message: str) -> Turn:
        """Send a user message and run the loop until the model is done.

        THE LOOP, IN FULL
        -----------------
        Everything below this docstring is the agent. Note what is NOT
        here: no routing logic, no decision about which tool suits the
        request, no interpretation of results. The model makes those
        choices; this code only carries messages back and forth and runs
        whatever it is asked to run.
        """
        turn = Turn(user_message=message)
        self.turns.append(turn)

        # First pass: hand the user's message to the model. It replies
        # with prose, or asks for tools.
        response = self.backend.send_user_message(message)

        while response.wants_tools:
            turn.rounds += 1

            if turn.rounds > self.max_rounds:
                # The model is not converging. Better to say so than to
                # spin: an unbounded loop burns tokens and never returns.
                turn.reply = (
                    f"Stopped after {self.max_rounds} rounds of tool calls "
                    "without reaching an answer."
                )
                return turn

            # Run every requested tool. A model may ask for several at
            # once (e.g. resolving two airports); all results go back
            # together, in one message, which is what both APIs expect.
            results: List[ToolResult] = []
            for call in response.tool_calls:
                turn.tool_calls.append(call)
                if self.on_tool_call:
                    self.on_tool_call(call)

                # dispatch() never raises -- failures come back as dicts
                # the model can read and recover from.
                result = ToolResult(
                    call_id=call.id,
                    name=call.name,
                    content=dispatch(call.name, call.arguments),
                )
                results.append(result)
                turn.tool_results.append(result)
                if self.on_tool_result:
                    self.on_tool_result(result)

            # Feed the results back and see what the model wants next.
            # This is the only place the loop repeats.
            response = self.backend.send_tool_results(results)

        turn.reply = response.text
        return turn

    # -- introspection, for the CLI and for tests ------------------------

    @property
    def total_tool_calls(self) -> int:
        return sum(len(turn.tool_calls) for turn in self.turns)

    def tool_call_names(self) -> List[str]:
        """Every tool called so far, in order. Useful in tests to assert
        the model actually used the tools rather than inventing an
        answer."""
        return [call.name for turn in self.turns for call in turn.tool_calls]
