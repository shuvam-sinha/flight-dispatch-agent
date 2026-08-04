# Flight Dispatch Agent

An AI flight-dispatch assistant for pilots/dispatchers: given a natural-language
mission request, an agent orchestrates real routing, weather, and airspace tools
to produce a flight plan, plus a grounded preflight checklist and an interactive
map.

This is a simplified planning engine, not an ATC-accurate flight-filing system —
see "Scope and limitations" below.

## Status

**Checkpoint 1 complete**: airport/navaid data ingestion + naive route lookup,
verified against real OurAirports data.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage (CP1)

```bash
python plan_route.py --origin KPWK --dest KMSP
```

```
KPWK (Chicago Executive Airport)  ->  KMSP (Minneapolis–Saint Paul International)

  KPWK    42.1142   -87.9015  departure
  LVV     42.6988   -88.5932    46.6 nm  319T    46.6 nm total
  MSN     43.1448   -89.3397    42.4 nm  309T    89.0 nm total
  HBW     43.6552   -90.3331    53.1 nm  306T   142.0 nm total
  ODI     43.9124   -91.4676    51.5 nm  288T   193.6 nm total
  RGK     44.5889   -92.4921    59.9 nm  313T   253.5 nm total
  KMSP    44.8801   -93.2217    35.7 nm  300T   289.2 nm total

Waypoints:       7 (5 intermediate)
Direct distance: 285.0 nm
Route distance:  289.2 nm
```

`--corridor-nm` sets how far off the direct course a navaid may sit to be
eligible; `--max-waypoints` caps the intermediate waypoint count.

Note that CP1 does no path search — it samples navaids that happen to lie near
the direct course, so on short legs it can produce visible zigzags. CP2 replaces
this with A* over a waypoint mesh graph.

Run the tests with:

```bash
python -m unittest discover tests
```

## Architecture (grows with each checkpoint)

- CP1: data ingestion (OurAirports airports/navaids) + naive route
  - `flight_dispatch/geo.py` — great-circle distance, bearing, cross/along-track
  - `flight_dispatch/data_loader.py` — CSV parsing, ICAO lookup, bounding-box filter
  - `flight_dispatch/route.py` — corridor sampling, `RoutePlan`
- CP2: waypoint mesh graph + A* pathfinding
- CP3: wind-adjusted routing (NOAA winds aloft) + restricted airspace avoidance (FAA GeoJSON)
- CP4: dispatcher agent (tool-use loop, multi-turn conversation)
- CP5: RAG-based checklist agent (grounded in POH/FAA reference material)
- CP6: combined JSON/HTML report with interactive map

## Scope and limitations

This project models routing, wind, and static restricted airspace using real
public data. It intentionally does not model published airways (V/J/Q routes),
SIDs/STARs, ATC-assigned routings, or live air traffic — see the project plan
for details on what's in scope vs. future work.
