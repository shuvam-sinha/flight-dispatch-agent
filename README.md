# Flight Dispatch Agent

An AI flight-dispatch assistant for pilots/dispatchers: given a natural-language
mission request, an agent orchestrates real routing, weather, and airspace tools
to produce a flight plan, plus a grounded preflight checklist and an interactive
map.

The core design principle is that **the LLM never does computation**. All
routing, weather and airspace logic lives in deterministic Python functions that
work independently of any model. The LLM decides which tool to call and
synthesises the results — every number it reports comes from real tool output.

## Status

**Checkpoints 1–3 complete.** A\* routing over a waypoint mesh built from real
navaid data, with flight time as the cost function, live global winds aloft, and
FAA restricted-airspace avoidance.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage

```bash
python plan_route.py --origin KLAX --dest KSLC --aircraft sr22 \
    --wind live --avoid-airspace --map route.html
```

```
KLAX (Los Angeles International)  ->  KSLC (Salt Lake City International)

  KLAX    33.9425  -118.4080  departure
  DAG     34.9635  -116.2790   119.6 nm  057T   119.6 nm total
  LAS     36.0808  -115.1630    93.4 nm  039T   213.0 nm total
  MMM     36.7693  -114.2770    66.1 nm  043T   279.1 nm total
  MLF     38.3597  -113.0130   126.2 nm  035T   405.3 nm total
  TVY     40.6107  -112.3480   138.6 nm  013T   543.9 nm total
  KSLC    40.7889  -111.9799    19.9 nm  057T   536.9 nm total

Waypoints:       8 (6 intermediate)
Direct distance: 512.6 nm
Route distance:  536.9 nm (104.7% of direct)
ETE:             2h53m (avg 186 kt GS, net tailwind +6 kt)
Fuel required:   61.8 gal (incl. 45 min reserve)
Airspace:        avoided 95 active prohibited/restricted/warning volumes
Mesh graph:      149 nodes, 2747 edges; A* expanded 44
```

| Flag | Default | Effect |
| --- | --- | --- |
| `--origin` / `--dest` | required | ICAO codes, case-insensitive |
| `--aircraft` | `c172` | One of 47 profiles; see `--list-aircraft` |
| `--payload` | typical occupancy | Payload in pounds |
| `--wind` | off | `live` for Open-Meteo, or `DDD/SS` (e.g. `270/40`) |
| `--altitude` | aircraft's own | Cruise altitude in feet |
| `--avoid-airspace` | off | Route around FAA prohibited/restricted/warning areas |
| `--radius-nm` | 150 | Mesh edge connection radius |
| `--map [PATH]` | — | Interactive HTML map with airspace overlay |
| `--naive` | off | CP1's corridor sampler, for comparison |

```bash
python -m unittest discover tests      # 183 tests
```

## What each checkpoint demonstrates

### CP1 — data ingestion

Loads 85,825 airports and 11,009 navaids from OurAirports CSVs. The geometry
module (`geo.py`) holds no project types, so everything downstream reuses it:
haversine distance, initial bearing, signed cross/along-track decomposition,
great-circle interpolation, and antimeridian-safe bounding boxes.

### CP2 — A\* over a waypoint mesh

`graph.py` builds a mesh per request over the region between origin and
destination: nodes are navaids plus the two airports, edges connect anything
within the radius. A k-nearest floor prevents isolated nodes, and component
bridging joins clusters separated by open water, so the graph is always
connected.

`search.py` is a hand-written A\* over `heapq`. The great-circle heuristic is
admissible by construction — no route between two points is shorter than the
straight line — so the result is provably optimal.

The payoff over CP1's corridor sampling is immediate: `KPWK→KORD` went from
21.8 nm (267% of direct) to 8.1 nm.

**Verification:** matched against exhaustive brute-force search on 400 random
graphs with zero mismatches, and against Dijkstra on real routes — identical
costs, far fewer nodes expanded (KPWK→KMSP: 3 vs 216).

### CP3 — wind and airspace

**Cost becomes time, not distance:**

```
CP2:  cost = distance_nm
CP3:  cost = distance_nm / ground_speed_kt
```

Ground speed is not airspeed. The aircraft flies through air that is itself
moving, so the shortest route stops being the fastest one. On a banded wind
field, `KPWK→KMSP` chose a route 8.6 nm longer that arrives 13 minutes sooner.

Changing the cost units meant the heuristic had to change too — distance in
nautical miles is not a lower bound on hours. `a_star` takes a `distance_to_cost`
converter, and the time version divides by TAS plus the strongest wind in the
graph, which is the best ground speed physically achievable and therefore a
valid lower bound.

**Winds:** live global forecasts from Open-Meteo. Coordinates are batched,
snapped to a 0.5° grid and cached, so a 78,707-edge mesh collapses to 955 grid
cells and 10 HTTP requests. `WindSource` is a narrow protocol, so alternative
backends drop in without the router knowing.

**Airspace:** 780 blocking FAA volumes, indexed with an STRtree. An edge
crossing prohibited airspace costs `math.inf`, and A\* routes around it — no
avoidance algorithm required, which is what the `cost_function` hook was for.
Altitude bands are respected: 501 volumes are active at 8,000 ft, 235 at
41,000 ft.

```
KLAX→KSLC   unrestricted   crosses 9 restricted areas    515.8 nm
            avoiding       crosses 0                     536.6 nm
```

**Aircraft:** 47 profiles from a Cessna 172 to an A380, with fuel and payload
trading against maximum takeoff weight — the constraint a payload-range diagram
describes. A 787-8 with 248 passengers can load 27,379 of its 33,340 gallons,
giving ~7,140 nm against a published ~7,300.

### Planned

- **CP4** — dispatcher agent: tool schemas over the CP1–CP3 functions, an agent
  loop, and multi-turn conversation
- **CP5** — RAG checklist agent: a small embedded corpus, cosine retrieval, and
  cited output
- **CP6** — combined JSON/HTML dispatch report

## Design notes

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

### Why Open-Meteo for winds

NOAA's FD product on aviationweather.gov is the bulletin pilots read at
preflight, and it was the first choice. It is fixed-width text covering ~218 US
stations, with point data rather than a grid. This project routes globally and
CP3's value shows most on long-haul flights, so gridded model data was the
better fit — the same GFS/ECMWF runs that FD is derived from, served as JSON at
any coordinate. `WindSource` exists so that decision can be revisited without
touching the router.

### Why navaids, not airports, as intermediate waypoints

Real en-route waypoints are navaids and named intersections. Airports appear in
a flight plan as origin, destination and alternate — you don't route *over* one
you aren't landing at. So the mesh uses navaids, with airports only at the ends.

## Scope

This models routing, winds aloft, and static special-use airspace using real
public data, at planning-estimate fidelity.

Routes are built over a proximity mesh of navaids rather than published airways
(V/J/Q routes) between named intersections — those aren't in OurAirports. The
**FAA NASR 28-day subscription** publishes both, free, and ingesting it is the
clearest next step for route realism.

Winds and airspace are real and functional. Airspace coverage is US-only (FAA
data); routing geometry is worldwide. Aircraft performance uses one cruise TAS
and one burn rate per type — adding altitude, temperature and weight dependence
would sharpen the fuel figures, and making cruise altitude a search dimension
would let the router trade climb fuel against better winds, which is how real
wind-optimal routing works.

Oceanic routes use ground-based navaids, so they route via landmasses; a virtual
routing grid over water — the approach real oceanic track systems use — is a
planned enhancement.

## Verification

- A\* matched against **exhaustive brute-force search** on 400 random graphs —
  zero mismatches
- A\* matched against **Dijkstra** on real routes — identical costs, far fewer
  expansions
- Airspace avoidance verified leg-by-leg: 9 crossings before, 0 after
- **183 unit tests**
