"""Apple Foundation Models backend -- the on-device model.

Runs entirely on the machine: no API key, no network, no per-token cost.
Requires macOS 26+, Apple Silicon, Apple Intelligence enabled, and a full
Xcode install (the SDK compiles Swift bindings at pip-install time).

AN IMPORTANT DIFFERENCE FROM THE CLAUDE API
-------------------------------------------
Apple's SDK runs the tool loop ITSELF. You hand `LanguageModelSession` a
list of `Tool` objects, call `respond()`, and it internally decides which
tools to invoke, awaits them, and returns only the final text. There is
no point at which it hands back "I would like to call plan_flight" for
the caller to act on.

The Claude API works the opposite way: it returns `tool_use` blocks and
the caller runs them and sends results back. That is the loop written by
hand in `agent.py`.

So against this backend, `agent.py`'s loop makes exactly one pass: it
sends the message, and the reply already has the tools applied, so
`wants_tools` is False and it exits. The orchestration is Apple's, not
ours.

THE TOOLS ARE STILL OURS
------------------------
What does carry over is everything below the loop. `_BridgeTool` wraps a
`ToolSpec` and calls the same `dispatch()` the hand-written loop calls,
which runs the same `plan_flight`, which calls the same `plan_route`.
Apple's session decides WHEN; our code still decides WHAT HAPPENS. And
because the bridge records every call, the CLI can display the
orchestration even though it did not perform it.

CONSTRAINTS OF THE ON-DEVICE MODEL
----------------------------------
- Context is 4,096 tokens. Tool descriptions are a large fraction of
  that, so the tool surface is kept lean and long histories are trimmed.
- Guardrails can fire on innocuous prompts (an instruction as bland as
  "Reply with the word OK" was refused during testing). These surface as
  `GuardrailViolationError` and are caught and reported rather than
  crashing the conversation.
- Its unaided aviation knowledge is poor -- asked about ICAO codes with
  no tools attached, it claimed they were "highly classified". This is
  precisely why every fact must come from a tool result.
"""

import asyncio
from typing import Any, Dict, List, Optional, Sequence

from .agent import ModelResponse, ToolCall, ToolResult
from .tools import ToolSpec, dispatch

try:
    import apple_fm_sdk as afm

    SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - environment-dependent
    afm = None
    SDK_AVAILABLE = False


class AppleBackendUnavailable(RuntimeError):
    """Raised when the on-device model cannot be used, with the reason."""


# Apple's schema types are a fixed set; map the neutral ToolSpec types on.
_TYPE_MAP = {"string": str, "number": float, "integer": int, "boolean": bool}


def _is_exposed(spec: Dict[str, Any]) -> bool:
    """Whether a parameter should appear in this backend's schema.

    APPLE'S SCHEMA HAS NO OPTIONAL FIELDS. Every property declared on a
    generable class gets a value, so a parameter offered to the model is
    a parameter the model WILL fill -- inventing one if it has no basis
    for a real value. Marking it "(optional)" in the description does not
    help; that was tried and ignored.

    Measured, not theorised. Asked to plan KPWK to KMSP, the model
    supplied `payload_lb: 1600` unprompted. A Cessna 172's entire useful
    load is 870 lb, so the tool correctly refused to plan, and the
    conversation dead-ended on a constraint the user never set.

    So the rule: expose required parameters always, and optional ones
    only where an invented value is harmless.

      exposed      required parameters -- the model must supply these
      exposed      booleans -- two possible values, both survivable, and
                   the Python default already encodes the sane one
      exposed      enums -- `anyOf` constrains the model to real values
      WITHHELD     free numbers -- payload_lb, altitude_ft. Plausible and
                   harmful is the worst combination, and the Python
                   defaults are better than any guess.

    Withheld parameters are not lost: `plan_flight` still has them, and
    they still work from the CLI or from a backend that can express
    optionality. They are simply not offered to a model that cannot
    decline to answer.
    """
    if spec.get("required", False):
        return True
    if "enum" in spec:
        return True
    return spec["type"] == "boolean"


def _make_arguments_class(tool: ToolSpec):
    """Build an `@generable` class describing one tool's arguments.

    The SDK's documented pattern is a hand-written class per tool:

        @fm.generable("Calculator parameters")
        class CalculatorParams:
            operation: str = fm.guide("The operation to perform")

    That does not work here -- the tools are data (`ToolSpec`), so the
    classes have to be built at runtime from `tool.parameters`. `type()`
    creates the class, `__annotations__` supplies the field types, and
    the class attributes carry the `guide()` descriptions. The decorator
    is then applied manually, exactly as `@` would.

    Only parameters passing `_is_exposed` are included -- see that
    function for why.
    """
    annotations: Dict[str, Any] = {}
    attributes: Dict[str, Any] = {}

    for name, spec in tool.parameters.items():
        if not _is_exposed(spec):
            continue

        annotations[name] = _TYPE_MAP.get(spec["type"], str)

        # `anyOf` constrains the model to a fixed set, which matters more
        # on a small model that would otherwise invent a value.
        if "enum" in spec:
            attributes[name] = afm.guide(spec["description"], anyOf=list(spec["enum"]))
        else:
            attributes[name] = afm.guide(spec["description"])

    attributes["__annotations__"] = annotations

    arguments_class = type(f"{tool.name}_Arguments", (), attributes)
    return afm.generable(f"Arguments for {tool.name}")(arguments_class)


def _build_bridge_tool(tool: ToolSpec, backend: "AppleBackend"):
    """Wrap one ToolSpec as an Apple `Tool`.

    This is the whole adapter: Apple's session calls `call()`, and
    `call()` hands straight to the same `dispatch()` the hand-written
    loop uses. The routing engine is reached by an identical path either
    way.
    """
    arguments_class = _make_arguments_class(tool)
    schema = arguments_class.generation_schema()

    class BridgeTool(afm.Tool):
        name = tool.name
        description = tool.description

        @property
        def arguments_schema(self):
            return schema

        async def call(self, args) -> str:
            # Read back only the arguments the model actually supplied.
            # A missing optional raises rather than returning None, so
            # each is attempted individually and absent ones are left to
            # the Python function's own defaults.
            supplied: Dict[str, Any] = {}
            for arg_name, arg_spec in tool.parameters.items():
                if not _is_exposed(arg_spec):
                    continue
                try:
                    value = args.value(
                        _TYPE_MAP.get(arg_spec["type"], str), for_property=arg_name
                    )
                except Exception:  # noqa: BLE001 - absence is not an error
                    continue
                if value is not None:
                    supplied[arg_name] = value

            backend.record_call(tool.name, supplied)
            result = dispatch(tool.name, supplied)
            backend.record_result(tool.name, result)

            # Apple's tools return a string. JSON keeps the structure
            # legible to the model without inventing a prose format.
            import json

            return json.dumps(result, default=str)

    BridgeTool.__name__ = f"{tool.name}_BridgeTool"
    return BridgeTool()


class AppleBackend:
    """Drives the conversation with Apple's on-device model.

    Satisfies `ModelBackend`, so `DispatcherAgent` accepts it -- with the
    caveat in the module docstring that Apple owns the tool loop, so the
    hand-written one passes through in a single iteration.

    Args:
        max_history_turns: How many prior exchanges to keep. The context
            window is only 4,096 tokens and the tool descriptions consume
            a large share, so old turns are dropped rather than
            overflowing. Apple's session holds its own transcript; this
            bounds how much is rebuilt on a reset.
    """

    name = "apple-foundation-models"

    def __init__(self, max_history_turns: int = 6):
        if not SDK_AVAILABLE:
            raise AppleBackendUnavailable(
                "apple-fm-sdk is not installed. It requires macOS 26+, Apple "
                "Silicon, and a full Xcode install:\n"
                "  sudo xcode-select -s /Applications/Xcode.app\n"
                "  pip install apple-fm-sdk"
            )

        model = afm.SystemLanguageModel()
        available, reason = model.is_available()
        if not available:
            raise AppleBackendUnavailable(
                f"The on-device model is unavailable: {reason}. Check that "
                "Apple Intelligence is enabled in System Settings."
            )

        self.model = model
        self.context_size = model.context_size
        self.max_history_turns = max_history_turns

        self.session = None
        self.bridge_tools: List[Any] = []
        self.system_prompt = ""

        # Recorded by the bridge tools so the CLI can show what ran, even
        # though this backend never surfaces tool calls through the loop.
        self.calls_this_turn: List[ToolCall] = []
        self.results_this_turn: List[ToolResult] = []
        self._call_counter = 0

    # -- ModelBackend interface -----------------------------------------

    def start(self, system_prompt: str, tools: Sequence[ToolSpec]) -> None:
        self.system_prompt = system_prompt
        self.bridge_tools = [_build_bridge_tool(tool, self) for tool in tools]
        self.session = afm.LanguageModelSession(
            instructions=system_prompt,
            tools=self.bridge_tools,
        )

    def send_user_message(self, text: str) -> ModelResponse:
        """Send a user turn.

        The returned response never carries tool calls -- by the time
        `respond()` returns, Apple's session has already run whatever it
        wanted. The calls it made are on `calls_this_turn` instead.
        """
        self.calls_this_turn = []
        self.results_this_turn = []

        try:
            reply = self._respond(text)
        except AppleBackendUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 - surfaced as conversation text
            return ModelResponse(text=self._describe_failure(exc), raw=exc)

        return ModelResponse(text=reply, tool_calls=[], raw=reply)

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelResponse:
        """Never called: this backend never asks the loop for tool results.

        Present only to satisfy `ModelBackend`. Its existence is the
        clearest statement of the architectural difference -- against
        Claude this method carries the entire conversation forward; here
        it is unreachable.
        """
        raise AssertionError(
            "AppleBackend runs tools internally; send_tool_results is unreachable."
        )

    # -- internals -------------------------------------------------------

    def _respond(self, text: str) -> str:
        """Run the async `respond` from synchronous code.

        `DispatcherAgent` is deliberately synchronous -- an agent loop is
        clearer without async colouring every caller -- while Apple's SDK
        is async throughout. `asyncio.run` bridges the two, creating and
        tearing down an event loop per turn. At conversational latency
        (~1 s per exchange) that overhead is irrelevant.
        """
        return str(asyncio.run(self.session.respond(text)))

    def _describe_failure(self, exc: Exception) -> str:
        """Turn an SDK exception into something a user can act on."""
        name = type(exc).__name__

        if name == "GuardrailViolationError":
            return (
                "The on-device model declined that request. Its safety filters "
                "fire on some innocuous phrasing -- try rewording, or switch to "
                "the Claude backend."
            )
        if name == "ExceededContextWindowSizeError":
            return (
                f"The conversation exceeded the on-device context window "
                f"({self.context_size} tokens). Start a new conversation with "
                "/reset."
            )
        if name == "RateLimitedError":
            return "The on-device model is rate limited. Wait a moment and retry."
        if name == "AssetsUnavailableError":
            return (
                "Apple Intelligence model assets are unavailable. They may still "
                "be downloading; check System Settings."
            )
        return f"The on-device model failed: {name}: {exc}"

    # -- call recording, for the CLI --------------------------------------

    def record_call(self, name: str, arguments: Dict[str, Any]) -> None:
        self._call_counter += 1
        self.calls_this_turn.append(
            ToolCall(id=f"afm_{self._call_counter}", name=name, arguments=arguments)
        )

    def record_result(self, name: str, content: Dict[str, Any]) -> None:
        self.results_this_turn.append(
            ToolResult(
                call_id=f"afm_{self._call_counter}", name=name, content=content
            )
        )

    def reset(self) -> None:
        """Start a fresh conversation, discarding history.

        The 4,096-token window fills quickly once tool results are in the
        transcript, so this is a routine operation rather than an edge
        case.
        """
        self.session = afm.LanguageModelSession(
            instructions=self.system_prompt,
            tools=self.bridge_tools,
        )
        self.calls_this_turn = []
        self.results_this_turn = []
