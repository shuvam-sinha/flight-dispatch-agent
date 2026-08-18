# Flight Dispatch Agent

An AI flight-dispatch assistant for pilots/dispatchers: given a natural-language
mission request, an agent orchestrates real routing, weather, and airspace tools
to produce a flight plan, plus a grounded preflight checklist and an interactive
map.

All routing, weather and airspace logic lives in deterministic Python functions that
work independently of any model. The model decides which tool to call and
synthesises the results — every number it reports comes from real tool output,
and every checklist item from a document that can be opened.

## Status

**Working end to end.** A\* routing over a waypoint mesh built from real navaid
data, with flight time as the cost function, live global winds aloft, FAA
restricted-airspace avoidance, a virtual routing grid for oceanic legs, a
conversational agent running against two local model backends, a
retrieval-grounded preflight checklist, and a self-contained HTML report.

## Demo

**The agent, in the terminal.** Plain-language requests, the tool calls each one
triggers, and answers where every number came from a Python function.

<!-- VIDEO 1: replace this line with the github.com/user-attachments/assets/... URL -->

**The dispatch report.** KJFK to EGLL in a 787: the route, the map with the
restricted airspace it routed around, the computed figures, and a preflight
checklist where every item cites a procedure document.

<!-- VIDEO 2: replace this line with the github.com/user-attachments/assets/... URL -->

The commands and prompts being run are in [Usage](docs/usage.md).

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

```bash
python -m unittest discover tests      # 598 tests
```

## Documentation

- [Usage](docs/usage.md) — Running it, from the agent or the CLI
- [How it works](docs/how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Layout](docs/layout.md) — Which file does what, and what depends on what
- [Design notes](docs/design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Bugs worth recording](docs/bugs.md) — Nine fixed failures and the reasoning behind each
- [Scope](docs/scope.md) — What this models, and what it does not
- [Verification](docs/verification.md) — How each claim is checked
