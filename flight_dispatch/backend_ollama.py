"""Ollama backend -- a larger local model, and the one that drives the loop.

Runs a model on this machine through Ollama's local HTTP server. Free, no
API key, no data leaving the machine, and with a 128K context window
instead of the on-device model's 4,096.

THE IMPORTANT DIFFERENCE FROM THE APPLE BACKEND
-----------------------------------------------
Apple's SDK runs the tool loop itself: you hand `LanguageModelSession` a
list of tools, call `respond()`, and it decides, executes and returns
only the final text. Against that backend `agent.py`'s loop makes exactly
one pass and exits, so the orchestration written there never actually
orchestrates.

Ollama works the other way, like the Claude API. A reply comes back
either as prose or as a list of `tool_calls` for the CALLER to execute
and send back. So this backend is the one that exercises the hand-written
loop -- `send_user_message` and `send_tool_results` both return, the loop
runs the tools, and it repeats until the model stops asking.

That makes this the more honest demonstration of the agent, and it is
worth more than the extra context.

WHY THE CONTEXT WINDOW HAS TO BE SET EXPLICITLY
-----------------------------------------------
Ollama defaults `num_ctx` to 2,048 tokens -- SMALLER than Apple's 4,096 --
and silently truncates rather than erroring. Pulling a model advertised
as 128K and getting 2,048 is a trap worth naming: `DEFAULT_NUM_CTX` below
is why this backend is an improvement rather than a regression.

WHAT IS SHARED WITH EVERY OTHER BACKEND
---------------------------------------
The tools. `ToolSpec.json_schema()` already emits exactly the format
Ollama wants, because that format is OpenAI's and so is Claude's. The
tool functions, their descriptions, and `dispatch()` are untouched --
this file converts message shapes and nothing else.
"""

import json
from typing import Any, Dict, List, Optional, Sequence

from .agent import ModelResponse, ToolCall, ToolResult
from .tools import ToolSpec

try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:  # pragma: no cover - requests is a project dependency
    requests = None
    REQUESTS_AVAILABLE = False


DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "llama3.1"

# Ollama's own default is 2,048, which is smaller than the on-device
# model this backend exists to improve on, and it truncates silently.
# 32,768 is large enough that the context stops shaping the design and
# small enough to stay comfortable in 16 GB of RAM -- the full 128K would
# need considerably more.
DEFAULT_NUM_CTX = 32768

# Generation is slow on a laptop, and a flight plan can involve several
# rounds. Long enough not to abandon a working answer, short enough that
# a wedged server does not hang the CLI forever.
REQUEST_TIMEOUT_S = 300


class OllamaUnavailable(RuntimeError):
    """Raised when Ollama cannot be used, with what to do about it."""


class OllamaBackend:
    """Drives the agent loop against a model served by Ollama.

    Args:
        model: Ollama model name, e.g. "llama3.1" or "qwen2.5".
        host: Where the Ollama server is listening.
        num_ctx: Context window in tokens. See DEFAULT_NUM_CTX -- leaving
            this to Ollama's default would make the window smaller than
            Apple's.
        temperature: 0 by default. Tool selection is a decision, not a
            creative act, and a deterministic backend is far easier to
            debug against a scripted test.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        host: str = DEFAULT_HOST,
        num_ctx: int = DEFAULT_NUM_CTX,
        temperature: float = 0.0,
        session: Optional[Any] = None,
    ):
        if not REQUESTS_AVAILABLE:
            raise OllamaUnavailable(
                "The `requests` package is required.\n"
                "Install it with:  pip install requests"
            )

        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.session = session or requests.Session()

        self.name = f"ollama:{model}"
        self.context_size = num_ctx

        self._check_available()

        # The whole conversation, resent on every call. Ollama is
        # stateless per request, exactly like the Claude API -- which is
        # why history lives here rather than in a session object as it
        # does for Apple.
        self.messages: List[Dict[str, Any]] = []
        self.tool_schemas: List[Dict[str, Any]] = []

        self._next_id = 0

    # -- availability ----------------------------------------------------

    def _check_available(self) -> None:
        """Fail early, with the command that fixes it.

        A connection error thrown from the middle of a conversation is a
        stack trace; the same condition caught here is one line telling
        the user to start the server. `AppleBackendUnavailable` exists
        for the same reason.
        """
        try:
            response = self.session.get(f"{self.host}/api/tags", timeout=5)
            response.raise_for_status()
            installed = {
                model.get("name", "").split(":")[0]
                for model in response.json().get("models", [])
            }
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.host}.\n\n"
                "Start it with:   ollama serve\n"
                f"Or use the on-device model:   --backend apple\n\n"
                f"({type(exc).__name__})"
            ) from exc

        if installed and self.model.split(":")[0] not in installed:
            raise OllamaUnavailable(
                f"Model {self.model!r} is not installed.\n\n"
                f"Pull it with:   ollama pull {self.model}\n"
                f"Installed:      {', '.join(sorted(installed)) or 'none'}"
            )

    # -- ModelBackend protocol -------------------------------------------

    def start(self, system_prompt: str, tools: Sequence[ToolSpec]) -> None:
        """Begin a conversation.

        `json_schema()` needs no adaptation: Ollama takes OpenAI-style
        function schemas, and that is what ToolSpec already renders for
        the Claude API. The neutral ToolSpec paying off exactly as
        intended.
        """
        self.messages = [{"role": "system", "content": system_prompt}]
        self.tool_schemas = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.json_schema(),
                },
            }
            for tool in tools
        ]

    def send_user_message(self, text: str) -> ModelResponse:
        self.messages.append({"role": "user", "content": text})
        return self._chat()

    def send_tool_results(self, results: Sequence[ToolResult]) -> ModelResponse:
        """Return tool output to the model and get its next reply.

        THIS METHOD IS WHY THIS BACKEND EXISTS. On the Apple backend it
        is unreachable -- asserted so in its tests -- because Apple's
        session runs tools internally. Here the loop in `agent.py` calls
        it after every round, which is the loop doing its job.
        """
        for result in results:
            self.messages.append(
                {
                    "role": "tool",
                    "content": json.dumps(result.content),
                    # Ollama echoes the function name rather than a call
                    # id. Pairing is positional within a round, which is
                    # safe here because the results are appended in the
                    # order the calls were made -- unlike Apple, which
                    # runs tools concurrently and needed ids.
                    "name": result.name,
                }
            )
        return self._chat()

    # -- internals -------------------------------------------------------

    def _chat(self) -> ModelResponse:
        """One round trip, appending the reply to the history."""
        payload = {
            "model": self.model,
            "messages": self.messages,
            "tools": self.tool_schemas,
            "stream": False,
            "options": {
                "num_ctx": self.num_ctx,
                "temperature": self.temperature,
            },
        }

        try:
            response = self.session.post(
                f"{self.host}/api/chat", json=payload, timeout=REQUEST_TIMEOUT_S
            )
            response.raise_for_status()
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise OllamaUnavailable(
                f"Ollama request failed: {type(exc).__name__}. "
                "Is `ollama serve` still running?"
            ) from exc

        message = body.get("message", {}) or {}

        # Keep the assistant turn in history exactly as it came back,
        # tool calls included. Dropping them would leave the tool results
        # that follow with nothing to answer.
        #
        # `role` is defaulted rather than assumed: every message in the
        # history is resent on the next request, and one without a role
        # would be rejected. Ollama supplies it, but a reply that did not
        # would poison the conversation from that point on rather than
        # failing where the mistake was.
        message.setdefault("role", "assistant")
        self.messages.append(message)

        return ModelResponse(
            text=message.get("content", "") or "",
            tool_calls=self._read_tool_calls(message),
            raw=body,
        )

    def _read_tool_calls(self, message: Dict[str, Any]) -> List[ToolCall]:
        """Convert Ollama's tool calls into the loop's own type.

        Arguments arrive as a dict from well-behaved models and
        occasionally as a JSON string from less careful ones, so both are
        accepted. A malformed one becomes an empty dict rather than an
        exception: `dispatch()` will then answer with a readable error
        about the missing argument, which the model can recover from --
        the same errors-as-data reasoning used throughout the tools.
        """
        calls = []
        for raw in message.get("tool_calls") or []:
            function = raw.get("function", {}) or {}
            arguments = function.get("arguments", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except ValueError:
                    arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}

            self._next_id += 1
            calls.append(
                ToolCall(
                    id=f"call_{self._next_id}",
                    name=function.get("name", ""),
                    arguments=arguments,
                )
            )
        return calls

    # -- diagnostics -----------------------------------------------------

    def context_usage(self) -> Optional[dict]:
        """Roughly how full the context window is, by role.

        Estimated from the message JSON at four characters per token, the
        same approximation `backend_apple` uses, so the meter reads
        consistently whichever backend is running. It is an estimate and
        is labelled as one wherever it is shown.
        """
        if not self.messages:
            return None

        by_role: Dict[str, int] = {}
        total = 0

        # The tool schemas are sent on every request and are a large
        # fixed cost -- on the Apple backend they were a third of the
        # window before anything was said, which is only visible if they
        # are counted.
        if self.tool_schemas:
            schema_tokens = len(json.dumps(self.tool_schemas)) // 4
            by_role["tools"] = schema_tokens
            total += schema_tokens

        for message in self.messages:
            size = len(json.dumps(message)) // 4
            role = message.get("role", "?")
            by_role[role] = by_role.get(role, 0) + size
            total += size

        return {
            "total": total,
            "limit": self.num_ctx,
            "percent": 100.0 * total / self.num_ctx,
            "by_role": by_role,
        }

    def reset(self) -> None:
        """Start a fresh conversation, keeping the system prompt."""
        self.messages = self.messages[:1] if self.messages else []
        self._next_id = 0
