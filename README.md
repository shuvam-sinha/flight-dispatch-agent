# Flight Dispatch Agent

An AI flight-dispatch assistant for pilots/dispatchers: given a natural-language
mission request, an agent orchestrates real routing, weather, and airspace tools
to produce a flight plan, plus a grounded preflight checklist and an interactive
map.

This is a simplified planning engine, not an ATC-accurate flight-filing system —
see "Scope and limitations" below.

## Status

**Checkpoint 2 complete**: A\* shortest-path routing over a waypoint mesh graph
built from real OurAirports navaid data. Verified against exhaustive brute-force
search and against Dijkstra's algorithm.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage

```bash
python plan_route.py --origin KPWK --dest KMSP
```

```
KPWK (Chicago Executive Airport)  ->  KMSP (Minneapolis–Saint Paul International)

  KPWK    42.1142   -87.9015  departure
  LNR     43.2944   -90.1331   121.3 nm  306T   121.3 nm total
  LSE     43.8761   -91.2560    60.0 nm  306T   181.3 nm total
  KMSP    44.8801   -93.2217   103.7 nm  306T   285.0 nm total

Waypoints:       4 (2 intermediate)
Direct distance: 285.0 nm
Route distance:  285.0 nm (100.0% of direct)
Mesh graph:      279 nodes, 12692 edges; A* expanded 3
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--origin` / `--dest` | required | ICAO codes, case-insensitive |
| `--radius-nm` | 150 | Mesh edge connection radius |
| `--margin-nm` | 100 | How far either side of course to gather navaids |
| `--map [PATH]` | — | Write an interactive HTML map |
| `--naive` | off | Use the CP1 corridor sampler instead of A\* |

`--naive` is kept for comparison. On KPWK→KORD it flies 21.8 nm for an 8.1 nm
trip; A\* flies it direct.

Run the tests with:

```bash
python -m unittest discover tests
```

## Architecture (grows with each checkpoint)

**CP1 — data ingestion**
- `flight_dispatch/models.py` — `Airport` / `Navaid`, frozen dataclasses
- `flight_dispatch/geo.py` — great-circle distance, bearing, cross/along-track,
  great-circle interpolation, antimeridian-safe bounding boxes
- `flight_dispatch/data_loader.py` — CSV parsing, ICAO lookup, region filters

**CP2 — routing**
- `flight_dispatch/graph.py` — waypoint mesh: nodes are navaids plus the two
  airports, edges connect anything within `--radius-nm`, with a k-nearest floor
  and component bridging so the graph is always connected. Built per request,
  not precomputed globally.
- `flight_dispatch/search.py` — hand-written A\* over `heapq`, with a
  great-circle heuristic. Takes an optional `cost_function`, which is how CP3
  will fold in wind and block edges crossing restricted airspace.
- `flight_dispatch/route.py` — `plan_route` (A\*) and `naive_route` (CP1)
- `flight_dispatch/mapping.py` — folium/Leaflet HTML map (early CP6 slice)

**Planned**
- CP3: wind-adjusted routing (NOAA winds aloft) + restricted airspace (FAA GeoJSON)
- CP4: dispatcher agent (tool-use loop, multi-turn conversation)
- CP5: RAG-based checklist agent (grounded in POH/FAA reference material)
- CP6: combined JSON/HTML report with interactive map

### Why the mesh radius is 150 nm

Tuned by measurement on KJFK→KLAX (2,146 nm direct, 1,604 navaids in region):

| radius | waypoints | route | % of direct | nodes expanded |
| ---: | ---: | ---: | ---: | ---: |
| 75 | 39 | 2197.2 | 102.4% | 796 |
| 150 | 21 | 2154.9 | 100.4% | 436 |
| 300 | 10 | 2146.2 | 100.0% | 67 |

A tight radius forces many short hops that each drift off the great circle, and
the deviations accumulate. A loose radius converges on the direct line but
returns a flight plan with almost no waypoints in it. 150 nm is the knee.

## Verification

- A\* checked against **exhaustive brute-force search** on 400 random graphs —
  zero mismatches
- A\* checked against **Dijkstra** on real routes — identical costs, far fewer
  nodes expanded (KPWK→KMSP: 3 vs 216)
- 89 unit tests

## Scope and limitations

This project models routing over real navaid data. It is deliberately not a
real-world-accurate dispatch system.

**Not modeled:**
- Published airways (V/J/Q routes), SIDs/STARs, or ATC-assigned routings — real
  dispatchers file preferred routes that ATC will accept, which is a different
  problem from shortest-path
- Named intersections/fixes, the most common waypoint type in real flight plans.
  OurAirports doesn't publish them; the **FAA NASR 28-day subscription** does,
  free, and ingesting it is the clearest future-work item
- Terrain and minimum en-route altitudes
- ETOPS / diversion-range constraints
- Convective weather, icing, turbulence
- Live air traffic

**Oceanic routes are unrealistic.** Navaids are ground stations, and there is no
ground mid-Atlantic. KJFK→EGLL currently routes via Greenland and Iceland at 133%
of the direct distance, because those are the only waypoints that exist. Real
oceanic flights use named lat/lon fixes on organized track systems. CP3 will add
a virtual routing grid, which fixes this as a side effect. Land routes are
100–100.4% of direct.

**Performance.** Mesh construction is O(n²) and dominates runtime — roughly 2 s
for a 1,606-node transcontinental graph, of which the A\* search itself is a few
milliseconds.
