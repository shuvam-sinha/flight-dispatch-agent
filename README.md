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

```
you> plan a flight from Chicago Executive to Minneapolis in a Cirrus

  · find_airport(query="Chicago Executive")   -> KPWK  Chicago Executive
  · find_airport(query="Minneapolis")         -> KMSP  Minneapolis–Saint Paul Intl
  · plan_flight(origin="KPWK", dest="KMSP", aircraft="sr22")
        KPWK LNR LSE RGK KMSP — 285.3 nm, 1h44m, 42.2 gal

KPWK to KMSP is 285 nm direct. Routing via Lone Rock, La Crosse and Red
Wing gives 285.3 nm — essentially the direct line — and a Cirrus SR22
covers it in about 1 hour 44 minutes on 42.2 gallons including reserve.
```

Every number in that reply came out of a Python function, not the model.

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
