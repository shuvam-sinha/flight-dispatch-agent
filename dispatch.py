"""Interactive flight dispatch assistant.

    python dispatch.py

Ask for a flight in plain language and watch the agent call real routing,
weather and airspace tools to answer. Every number it reports comes from
a tool result -- the model does no computation itself.

Defaults to Apple's on-device Foundation Models: free, private, no API
key, no network for the model itself.

This file is a thin shell, like plan_route.py. It reads input, prints
output, and owns no agent logic -- that lives in flight_dispatch/agent.py
so it can be driven from a test, a notebook, or a web server just as
easily.
"""

import argparse
import sys
from typing import List, Optional

from flight_dispatch.agent import DispatcherAgent, ToolCall, ToolResult
from flight_dispatch.tools import TOOLS, TOOLS_BY_NAME, ToolSpec

# Terminal colours. Kept minimal and disabled when piping, so the output
# stays readable in a file or a pipe.
_COLOUR = sys.stdout.isatty()


def _paint(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _COLOUR else text


def dim(text: str) -> str:
    return _paint(text, "2")


def bold(text: str) -> str:
    return _paint(text, "1")


def red(text: str) -> str:
    return _paint(text, "31")


def cyan(text: str) -> str:
    return _paint(text, "36")


# The on-device model has a 4,096-token context, and tool schemas are a
# large share of it. This subset covers the common cases -- planning a
# flight and asking what airspace is in the way -- while leaving the
# model room to think. Use --all-tools for the full surface.
LEAN_TOOL_NAMES = ("find_airport", "plan_flight", "check_airspace")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Talk to the flight dispatch agent.",
        epilog="Try: 'plan a flight from Chicago Executive to Minneapolis in a Cirrus'",
    )
    parser.add_argument(
        "--backend",
        default="apple",
        choices=["apple"],
        help="Model backend (default: apple, on-device)",
    )
    parser.add_argument(
        "--all-tools",
        action="store_true",
        help="Expose all 5 tools instead of the lean set. Uses more context.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide tool calls. They are shown by default -- watching them is the point.",
    )
    parser.add_argument(
        "--ask",
        metavar="QUESTION",
        help="Ask one question and exit, instead of starting a session.",
    )
    return parser.parse_args(argv)


def build_backend(name: str):
    """Construct a backend, or exit with an actionable message."""
    if name == "apple":
        from flight_dispatch.backend_apple import (
            AppleBackend,
            AppleBackendUnavailable,
        )

        try:
            return AppleBackend()
        except AppleBackendUnavailable as exc:
            raise SystemExit(f"{red('On-device model unavailable')}\n\n{exc}")

    raise SystemExit(f"Unknown backend: {name}")


def choose_tools(all_tools: bool) -> List[ToolSpec]:
    if all_tools:
        return list(TOOLS)
    return [TOOLS_BY_NAME[name] for name in LEAN_TOOL_NAMES]


def paired_calls(turn, backend):
    """Tool calls with their results, correctly matched.

    Two sources, because the backends differ. A backend that drives the
    hand-written loop puts calls and results on the Turn; Apple runs
    tools itself and records them on the backend instead.

    Never zip the two lists. Apple executes tools concurrently, so they
    can be in different orders -- a "Minneapolis" lookup once displayed
    Chicago Executive's result because of exactly that.
    """
    if turn.tool_calls:
        by_id = {result.call_id: result for result in turn.tool_results}
        return [
            (call, by_id[call.id]) for call in turn.tool_calls if call.id in by_id
        ]
    if hasattr(backend, "paired_calls"):
        return backend.paired_calls()
    return []


def format_arguments(call: ToolCall) -> str:
    """Render arguments compactly, hiding defaults the model filled in.

    Apple's schema has no optional fields, so the model supplies a value
    for every boolean whether the user cared or not. Printing
    `use_wind=True avoid_airspace=True save_map=False` on every call is
    noise; only non-default values are interesting.
    """
    hidden = {"use_wind": True, "avoid_airspace": True, "save_map": False}
    parts = [
        f"{key}={value!r}"
        for key, value in call.arguments.items()
        if hidden.get(key, object()) != value
    ]
    return ", ".join(parts)


def summarise_result(result: ToolResult) -> str:
    """One line describing what a tool returned."""
    content = result.content

    if result.is_error:
        return red(f"error: {content['error']}")

    if result.name == "plan_flight":
        pieces = [content.get("route", "")]
        if "route_distance_nm" in content:
            pieces.append(f"{content['route_distance_nm']} nm")
        if "ete" in content:
            pieces.append(content["ete"])
        if "fuel_required_gal" in content:
            pieces.append(f"{content['fuel_required_gal']} gal")
        return "  ".join(str(p) for p in pieces if p)

    if result.name == "find_airport":
        if "icao" in content:
            return f"{content['icao']} — {content['name']}"
        count = content.get("match_count", 0)
        first = content.get("matches", [{}])[0].get("icao", "?")
        return f"{count} match(es), first {first}"

    if result.name == "list_aircraft":
        return f"{content.get('count', 0)} aircraft"

    if result.name == "check_airspace":
        crossings = content.get("direct_course_crossings", [])
        return (
            f"{content.get('active_volumes_in_region', 0)} active, "
            f"{len(crossings)} on the direct course"
        )

    if result.name == "get_winds_aloft":
        return (
            f"{content.get('wind_speed_kt')} kt from "
            f"{content.get('wind_from_degrees_true')}°"
        )

    return "ok"


HELP = """\
Commands
  /reset     start a fresh conversation (frees the context window)
  /tools     list the tools currently available to the model
  /help      this message
  /quit      exit  (Ctrl-D also works)

Examples
  plan a flight from Chicago Executive to Minneapolis in a Cirrus
  what if I flew a 737 instead?
  what restricted airspace is between KLAX and KSLC?
  plan KPWK to KMSP and save the map
"""


def run_session(agent: DispatcherAgent, backend, tools, show_tools: bool) -> int:
    print(bold("Flight Dispatch Assistant"))
    print(dim(f"  {backend.name} · {len(tools)} tools · /help for commands"))
    if hasattr(backend, "context_size"):
        print(
            dim(
                f"  context {backend.context_size:,} tokens — use /reset if a "
                "conversation runs long"
            )
        )
    print()

    while True:
        try:
            message = input(bold("> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        if not message:
            continue

        lowered = message.lower()
        if lowered in ("/quit", "/exit"):
            return 0
        if lowered == "/help":
            print(HELP)
            continue
        if lowered == "/tools":
            for tool in tools:
                print(f"  {bold(tool.name)}")
                print(dim(f"    {tool.description[:100]}..."))
            print()
            continue
        if lowered == "/reset":
            if hasattr(backend, "reset"):
                backend.reset()
                agent.turns.clear()
                print(dim("  conversation cleared\n"))
            else:
                print(dim("  this backend cannot reset\n"))
            continue

        print()
        try:
            turn = agent.ask(message)
        except KeyboardInterrupt:
            print(dim("\n  interrupted\n"))
            continue

        if show_tools:
            for call, result in paired_calls(turn, backend):
                arguments = format_arguments(call)
                print(dim(f"  → {call.name}({arguments})"))
                print(dim(f"    {summarise_result(result)}"))
            if turn.tool_calls or getattr(backend, "calls_this_turn", []):
                print()

        print(turn.reply)
        print()


def main(argv=None) -> int:
    args = parse_args(argv)

    backend = build_backend(args.backend)
    tools = choose_tools(args.all_tools)
    agent = DispatcherAgent(backend, tools=tools)

    if args.ask:
        turn = agent.ask(args.ask)
        pairs = paired_calls(turn, backend)
        if not args.quiet and pairs:
            for call, result in pairs:
                print(dim(f"  → {call.name}({format_arguments(call)})"))
                print(dim(f"    {summarise_result(result)}"))
            print()
        print(turn.reply)
        return 0

    return run_session(agent, backend, tools, show_tools=not args.quiet)


if __name__ == "__main__":
    sys.exit(main())
