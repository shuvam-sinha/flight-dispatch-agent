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

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
```

## Usage

### Talk to it

```bash
python dispatch.py
```

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

Tool calls are printed as they happen, because watching them is the point:
every number in the prose above came out of a Python function, not the model.

| Flag | Default | Effect |
| --- | --- | --- |
| `--backend` | asks | `apple` (on-device) or `ollama` (local, 32K context) |
| `--model` | `llama3.1` | Ollama model name |
| `--all-tools` | off | Expose all 5 tools instead of the lean 3 |
| `--ask "..."` | — | One-shot question, then exit |
| `--quiet` | off | Hide tool calls |

`/reset` clears the conversation, `/tools` lists the tool surface, `/help` and
`/quit` do what they say.

### Or drive the router directly

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
| `--naive` | off | The earlier corridor sampler, for comparison |
| `--no-grid` | grid on | Disable virtual oceanic waypoints |
| `--report [PATH]` | — | Full dispatch report as HTML and JSON |
| `--checklist` | off | Add a retrieved, cited preflight checklist to the report |

```bash
python -m unittest discover tests      # 598 tests
```

## Layout

Each layer depends only on the ones above it. `geo.py` imports nothing from the
project; `agent.py` knows nothing about aviation.

```
geo.py          great-circle maths — the floor everything stands on
models.py       Airport, Navaid, RoutePlan
data_loader.py  OurAirports CSVs, region filtering
  ↓
graph.py        per-request waypoint mesh
search.py       hand-written A* over heapq
grid.py         virtual lat/lon waypoints where navaids run out
  ↓
cost.py         distance → time, via the wind triangle
wind_openmeteo.py   live gridded forecasts, batched and cached
airspace.py     FAA volumes, STRtree-indexed; blocking edges cost infinity
aircraft.py     47 performance profiles
phases.py       climb / cruise / descent split
route.py        plan_route() — ties the above into a RoutePlan
  ↓
retrieval.py    procedure corpus, embeddings, cosine search
report.py       the HTML/JSON dispatch report
  ↓
tools.py        the six tools, as ToolSpecs with English descriptions
agent.py        the loop; ModelBackend protocol
backend_apple.py    on-device Foundation Models
backend_ollama.py   a larger local model — and the one the loop drives
  ↓
plan_route.py   CLI          dispatch.py   conversational REPL
```

## How it works

### The data

85,825 airports and 11,009 navaids from OurAirports, plus 780 FAA special-use
airspace volumes. `geo.py` holds no project types, so everything above reuses
it: haversine distance, initial bearing, signed cross/along-track
decomposition, great-circle interpolation and destination-point projection, and
antimeridian-safe bounding boxes.

### Routing: A\* over a waypoint mesh

`graph.py` builds a mesh per request over the region between origin and
destination — nodes are navaids plus the two airports, edges connect anything
within the radius. A k-nearest floor prevents isolated nodes, and component
bridging joins clusters separated by open water, so the graph is always
connected.

`search.py` is a hand-written A\* over `heapq`. The great-circle heuristic is
admissible by construction — no route between two points is shorter than the
straight line — so the result is provably optimal.

**Verification:** matched against exhaustive brute-force search on 400 random
graphs with zero mismatches, and against Dijkstra on real routes — identical
costs, far fewer nodes expanded (KPWK→KMSP: 3 vs 216).

### Cost is time, not distance

```
distance-only:  cost = distance_nm
wind-aware:     cost = distance_nm / ground_speed_kt
```

Ground speed is not airspeed. The aircraft flies through air that is itself
moving, so the shortest route stops being the fastest one. On a banded wind
field, `KPWK→KMSP` chose a route 8.6 nm longer that arrives 13 minutes sooner.

Changing the cost units meant the heuristic had to change too — distance in
nautical miles is not a lower bound on hours. `a_star` takes a
`distance_to_cost` converter, and the time version divides by TAS plus the
strongest wind in the graph, which is the best ground speed physically
achievable and therefore a valid lower bound.

Wind is sampled per leg and per segment within a leg, with the course
recomputed each time: a 150 nm edge can start in a headwind and end in a
crosswind, and on a great circle the heading changes continuously.

### Airspace avoidance, without an avoidance algorithm

There is no code anywhere that steers a route around a restricted area. The
behaviour falls out of the cost function.

780 FAA volumes are loaded as polygons and indexed with an STRtree, so asking
"does this leg cross anything" is a bounding-box lookup rather than 780
intersection tests. Volumes are filtered by altitude first: 501 are active at
8,000 ft and 235 at 41,000 ft, so a jet is not routed around a range that tops
out at 12,000.

Then one line does the work:

```python
if index.blocks(a.lat, a.lon, b.lat, b.lon):
    return math.inf
```

An edge crossing prohibited airspace costs infinity. A\* skips any edge whose
cost is not finite, so such a leg is never added to the frontier and can never
appear in a result. The search does not know what airspace *is* — it knows some
edges are unaffordable, and it was already built to prefer cheap ones.

The airspace cost **wraps** the wind cost rather than replacing it:

```
distance  →  time (wind)  →  math.inf if blocked
```

so the answer is the *fastest* route that is also legal, not merely a legal one.

That is what the `cost_function` hook existed for. Adding airspace avoidance
required no change to `search.py` at all — which is the same property that let
the agent be added later without touching the router.

```
KLAX→KSLC   unrestricted   crosses 9 restricted areas    515.8 nm
            avoiding       crosses 0                     536.6 nm
```

Verified leg by leg rather than by trusting the count: every segment of the
avoiding route was re-tested against the index, and none intersects a blocking
volume.

### Aircraft

47 profiles from a Cessna 172 to an A380, with fuel and payload trading against
maximum takeoff weight — the constraint a payload-range diagram describes. A
787-8 with 248 passengers can load 27,379 of its 33,340 gallons, giving ~7,140
nm against a published ~7,300.

### The agent

The routing engine was written and tested before any model existed, and adding
the agent on top of it took **3,314 added lines and 17 deleted ones**. Nothing
in `geo.py`, `search.py`, `graph.py`, `cost.py` or `airspace.py` had to change —
the engine does not know a model is calling it. That was the bet of the
architecture, and it paid.

Six tools wrap the engine:

| Tool | What it does |
| --- | --- |
| `find_airport` | ICAO code, name, or city → airport, ranked by significance |
| `list_aircraft` | The 47 profiles, with cruise speed, altitude, seats, range |
| `plan_flight` | Full plan: route, waypoints, ETE, fuel, optional map and report |
| `get_winds_aloft` | Forecast wind at one point and altitude |
| `check_airspace` | Restricted areas on the direct course between two airports |
| `find_procedures` | Real procedure documents relevant to this flight |

A `ToolSpec` holds a name, an English description, a parameter schema and a
Python callable. `dispatch()` looks up the name, coerces arguments to their
declared types, calls the function, and — critically — **returns errors as data
rather than raising**. A model that gets `{"error": "unknown aircraft 'b737'"}`
can correct itself on the next turn; an exception just kills the conversation.

`agent.py` is the loop, hand-written and about 40 lines at its core: send the
conversation, check for tool calls, execute them, append the results, send
again, repeat until the model answers in prose. `max_rounds` turns a model that
never converges into a reported failure rather than an unbounded spend. A tool
called twice with identical arguments, failing both times, gets an escalated
error — the same message returned twice reads as the same situation, so the
model tries the same thing again.

`ModelBackend` is a three-method `Protocol`, so the loop is tested against a
`ScriptedBackend` with no model and no network, and runs against two real
backends selected with `--backend`.

### Retrieval, so the checklist cannot be invented

A checklist has three possible sources and two of them are bad. Hardcoded, it
is a dictionary that cannot adapt to the flight. Written by the model, it is
confident, plausible and invented.

So: retrieve first, write only from what was retrieved, cite every item. It is
the rule the routing tools already follow, applied to text instead of numbers —
`plan_flight` guarantees every number came from a computation, `find_procedures`
guarantees every procedure came from a document.

Fifteen procedure documents live in `data/procedures` as plain markdown, so
adding one changes the checklist with no code change and no retraining.
Embeddings come from `nomic-embed-text` through the same local server the model
backend uses, cached to disk by content hash. Cosine similarity over a list of
fifteen vectors is exact and takes microseconds; a vector database here would be
a dependency and a running service to search fewer items than a phone book page.

Similarity alone was not enough. Asked for a Cessna departure, vector search
ranked `night-flight` **first** — it is dense with light-aircraft language and
genuinely was the most similar document, but nothing said the flight was at
night. Seven of the fifteen documents now declare a precondition and are
excluded before ranking unless it holds, with the conditions derived from real
data: cruise altitude, field elevation, and the routing grid for open water.

```
777  KSFO→KEWR   high-altitude-cruise · fuel-reserves · engine-failure
c172 KPWK→KMSP   engine-failure · preflight-inspection · crosswind-landing
b738 KDEN→KMCI   density-altitude · mountain-flying · high-altitude-cruise
```

### The report

Everything else is reachable only through a terminal, and a route that bends
around restricted airspace is a number in a sentence until you can see it bend.
`report.py` produces one self-contained HTML file — route, map, figures,
checklist with citations — plus the same content as JSON.

**The report enforces the citation rule rather than asking for it.** An item
citing nothing does not enter the checklist, and neither does one citing a
document that does not exist — a citation to nothing is worse than no citation,
because it looks like provenance. Rejected items appear in their own section
rather than being dropped: a silently cleaned report would look fully grounded
while the model had been inventing.

## Design notes

### Why A\*, and not something else

The problem is a shortest path through a weighted graph with no negative
weights, which narrows the field quickly.

**Breadth-first search** finds the fewest *edges*, not the shortest distance —
it would happily return a three-hop route that is 400 nm longer than a
four-hop one. **Depth-first** does not find shortest paths at all.

**Bellman-Ford** handles negative edge weights, which cannot occur here: a leg
cannot have negative length, and with wind it cannot have negative time either,
because a headwind stronger than the aircraft's TAS makes the leg impossible
rather than instantaneous. Paying `O(VE)` for a capability the problem cannot
use is a poor trade.

**Floyd-Warshall** computes every pair of shortest paths. One route is needed,
and the graph is rebuilt per request, so this is the wrong problem in the wrong
shape.

**Greedy best-first** is fast and wrong: it follows the heuristic alone and
returns the first path it stumbles into, with no guarantee it is the shortest.

That leaves **Dijkstra**, which is correct, and **A\***, which is Dijkstra plus
an estimate of the distance still to go. The estimate is what makes it worth
choosing here, and this problem hands one over for free: the great-circle
distance from any waypoint to the destination. It is **admissible** by
construction — no route between two points can be shorter than the straight line
between them — so A\* returns exactly the path Dijkstra would, having looked at
far less of the graph.

```
KPWK→KMSP    Dijkstra expanded 216 nodes    A* expanded 3    identical route
```

Admissibility is the whole argument. A heuristic that overestimates makes A\*
fast and wrong; one that never overestimates makes it fast and provably optimal.
When the cost became time rather than distance, the heuristic had to be
rewritten to stay admissible — nautical miles are not a lower bound on hours —
which is why `a_star` takes a `distance_to_cost` converter rather than assuming
the units.

Written by hand rather than imported from `networkx`, for the same reason the
agent loop is: the algorithm is the thing worth demonstrating. It is about 90
lines over `heapq`, and it was verified against exhaustive brute-force search on
400 random graphs and against Dijkstra on real routes.

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
wind routing shows its value most on long-haul flights, so gridded model data
was the better fit — the same GFS/ECMWF runs that FD is derived from, served as JSON at
any coordinate. `WindSource` exists so that decision can be revisited without
touching the router.

### Why navaids, not airports, as intermediate waypoints

Real en-route waypoints are navaids and named intersections. Airports appear in
a flight plan as origin, destination and alternate — you don't route *over* one
you aren't landing at. So the mesh uses navaids, with airports only at the ends.

### Why an on-device model

Apple Foundation Models runs locally: no API key, no per-token cost, no data
leaving the machine, and it works on a plane with no wifi — which is a pleasing
property for a flight-planning tool. The trade is a **4,096-token context**,
which forces real engineering discipline rather than letting a large window
paper over sloppy tool design. Two constraints followed from it:

- **Tool results must be small.** `list_aircraft` originally returned 1,853
  tokens — 45% of the entire window for one call. Rewritten as one compact line
  per aircraft, it costs 737.
- **Candidate lists are for choosing from, not for using.** Dumping a real
  session's transcript showed one `find_airport("Chicago")` at 1,206 characters
  — **41% of the entire conversation**, and the largest single thing in it. It
  returned eight matches, each with latitude and longitude, and the model picked
  one and carried the other seven for the rest of the conversation. Capping at
  three and dropping the coordinates cut it to 247 characters, a 72% reduction.
  Nothing downstream wanted those coordinates: `plan_flight` takes ICAO codes,
  and `get_winds_aloft` gets a position by looking the chosen code up again for
  36 tokens.
- **Tool schemas must be small.** The five tools together crowd out the model's
  own reasoning, so `dispatch.py` exposes a lean three by default and
  `--all-tools` opts into the rest.

`ModelBackend` exists precisely so this choice is reversible — and it now has
two implementations, selected with `--backend`.

### Why a second backend, and what it proved

Apple's SDK runs the tool loop itself, so `agent.py`'s hand-written loop makes
one pass and exits. The orchestration was written, tested and documented — and
never actually orchestrated. Ollama returns `tool_calls` for the caller to
execute — the way a hosted chat API does — so the loop finally drives a real
conversation. `send_tool_results` is asserted *unreachable* in the Apple tests
and is exercised on every round here.

`ToolSpec.json_schema()` needed no adaptation: Ollama takes OpenAI-style
function schemas, which is the format `ToolSpec` already rendered. The backend
converts message shapes and nothing else.

Running identical tools against a 3B and an 8B model settled several open
questions in one evening:

| | Apple (3B) | Ollama (8B) |
| --- | --- | --- |
| Tool schemas | 1,355 tokens = **33%** of the window | 1,366 = **4%** |
| Asked for wind at 34,000 ft | planned a flight instead, twice | called `get_winds_aloft` correctly |
| 30-waypoint oceanic route | `27N023W` → `27NN023W` | copied intact |
| Conversation depth | dead at turn 3 | 14% used after 7 |

So the tool-selection failures were **model capacity**, not description quality.
One number still slipped in transcription (`18871.7` → `18771.7`), which says
the "supply the phrasing, never make the model derive it" work was correct
engineering rather than a workaround for a small model.

The new backend also exposed three bugs the first one had hidden:

- **Stringified arguments.** Ollama sends `payload_lb='300'`, which reached the
  weight arithmetic as text and raised a `TypeError`. Worse, `avoid_airspace='true'`
  never raised — a non-empty string is truthy, so `'false'` would have been too,
  quietly turning airspace avoidance *on* when the model asked for it off.
  Coercion now happens once in `dispatch()`, against each parameter's declared
  type, so every tool and every future backend inherits it.
- **A fix in the wrong layer.** Withholding `altitude_ft` was implemented inside
  `backend_apple._is_exposed`, which is Apple's rule — needed because its schema
  cannot express optionality. JSON Schema can, so Ollama rebuilt the parameter
  and volunteered 30,000 ft for a Cirrus that tops out at 17,500. A decision
  about the *tool* now lives in the tool.
- **Tool calls written as text.** llama3.1 sometimes writes
  `{"name": "plan_flight", "parameters": {...}}` into its reply instead of
  emitting a call — both observed cases followed a tool result. The loop saw
  prose, concluded the model was finished, and handed raw JSON to the user.
  Recognised and executed now, guarded so that only a name matching an exposed
  tool is run.

### Why the tool descriptions are prose, not code

The description strings in `tools.py` are not documentation; they are prompt
text, and they are the only thing the model reads when deciding what to call.
"Call this FIRST whenever the user names an airport in words rather than an
ICAO code" is an instruction to a reader, and it changed behaviour more than any
schema change did. Getting a tool used correctly turned out to be a writing
problem as much as an engineering one.

### Why flight time is not distance over cruise speed

A flight has three phases, and only the middle one is flown at cruise speed:

```
ft
|          ______________________________
|         /                              \
|        /            cruise              \
|       /                                  \
|      / climb                      descent \
+-----+------------------------------------- +-----> nm
   origin                                  destination
```

`phases.py` subtracts the climb and descent distances from the route and flies
the remainder at cruise. KORD→KMIA in a 777-300ER went from 2h07m to 2h18m, and
fuel from 7,063 to 7,516 gallons — because a jet burns about 1.6× cruise flow
climbing and a third of it descending, which a single flat burn rate misses in
both directions.

Two decisions worth noting.

**It runs after the search, not inside the cost function.** Climb and descent
depend on the route's total length and the two field elevations — not on which
waypoints A\* picks. They are identical for every candidate path, so they cannot
change which route wins. Folding them into the edge cost would have slowed the
search down to compute a constant.

**Short flights level off lower.** A 777 cannot reach FL370 in 150 nm and still
come down, and naively subtracting both phases yields a negative cruise leg and
a flight that lands before it departs. Instead the aircraft levels off at the
altitude where climb distance plus descent distance exactly equals the route.
Both are linear in height, so it inverts directly rather than needing a search.
KORD→KMDW, 13 nm, tops out at 3,000 ft — which is what actually happens.

Climb and descent rates are per *category*, not per type. Real per-type climb
schedules live in manufacturer performance manuals, which are not public data,
and inventing 47 sets of them would be dressing a guess up as a specification.
Speeds are fractions of each aircraft's own cruise TAS, so a Cessna's climb
speed stays sensible for a Cessna.

Fixing this also exposed a smaller bug: the CLI compared average ground speed
against cruise TAS to report net head- or tailwind. Once climb and descent were
in the average, every flight reported a headwind. The comparison now uses the
cruise segment, which is the part the wind acted on.

Remaining gap to a published schedule is taxi and padding. ETE is airborne time
and is labelled as such.

### Why the routing grid exists

Navaids are radio transmitters on the ground, so coverage ends where the ground
does. Overland routes came out at 100–105% of direct on every continent, but
KJFK→EGLL ran **132.7%**, crawling up through Greenland and Iceland because
those were the only waypoints in existence.

Real oceanic flights don't use ground navaids — they use named lat/lon points,
where `56N020W` means 56° north, 20° west. `grid.py` generates those: a lattice
of columns spaced 150 nm along the great circle, each carrying five lanes offset
±200 nm perpendicular, with the perpendicular bearing recomputed per column
because the course changes continuously along a great circle.

The rule that keeps it honest is `fill_navaid_gaps`: a grid point within 120 nm
of a real navaid is dropped. A generated fix over Nebraska corresponds to
nothing on any chart, so overland routes keep naming real VORs and only oceanic
legs get lat/lon fixes — which is exactly how a real flight plan reads.

| Route | Before | After |
| --- | ---: | ---: |
| KJFK → EGLL | 132.7% of direct | **101.0%** |
| LPPT → TNCM | 146.9% | **102.3%** |
| KSFO → EGLL | 107.0% | **103.4%** |
| KPWK → KMSP | 100.0% | unchanged |
| KJFK → KLAX | 100.4% | unchanged |

The last two rows matter as much as the first three.

Grid points are plain `Navaid` instances with `navaid_type="GRID"`, so
`build_mesh`, `a_star`, the cost functions and the map renderer all work
unchanged.

## Bugs worth recording

**All of these are fixed** — open limitations are in Scope below. They are here
because the reasoning was worth keeping, and because most were found by using
the thing rather than by testing it. Agents fail in ways unit tests do not
reach.

**Along-track distance was unsigned.** The textbook `acos` form returns 0–π, so
a navaid *behind* the aircraft reported positive forward progress and the early
corridor sampler selected waypoints in the wrong direction. Fixed by recovering
the sign from `cos(θ₁₃ − θ₁₂)`.

**The region was computed wrong, twice.** Sampling only the endpoints for a
bounding box cut 1.35° off KJFK→EGLL, silently discarding navaids the route
needed — a great circle bulges poleward between its ends. And PANC→RJTT searched
293° of longitude instead of 77°, pulling 5,410 navaids instead of 182, because
the box straddled the antimeridian. Both are now handled in `geo.py`, so nothing
downstream has to think about them.

**A connected graph is not the same as no isolated nodes.** The k-nearest floor
guarantees every node has neighbours, which sounds like the same guarantee and
is not: KJFK→EGLL built 695 nodes of which only 350 were reachable, split into
clusters by open water. `_bridge_components` joins them with a union-find pass.

**`find_airport` was wrong in five different ways.** It sorted by name length,
so a Mexican airstrip literally named "San Francisco" outranked SFO — and the
agent planned a flight from it. Ranking now uses real signals: airport type,
scheduled service, IATA code, city match, then total runway area. That last one
replaced longest-single-runway, which ranked Al Maktoum above Dubai
International on a runway 174 ft longer at an airport with almost no traffic.
Matching was wrong too: `New York JFK` matched nothing (city and code live in
different columns), `Sao Paulo` matched nothing (accents), and `JFK` matched
nothing (IATA codes aren't substrings of names). Across 45 major world cities
the ranker now returns the expected airport 44 times.

**The airspace result was narrated backwards.** The router avoided all 95
restricted areas near a route, and the reply said *"Route includes prohibited
and restricted airspace."* Every number was correct; the safety claim came out
inverted. The cause was two fields the model had to assemble itself —
`airspace_avoidance_applied: true` and `restricted_volumes_considered: 95` —
where "considered" reads equally well as *taken into account* and *included in*.
It now returns one sentence that cannot be read the other way.

This generalised the founding rule. "The model never does computation" was not
enough: `95` is a number reported faithfully, and *through* versus *around* is
an interpretation built from it. **The model never derives anything.** No test
caught this, because every test checked the router, and the router was never
wrong.

**A schema constraint cascaded three times.** Apple's format cannot express an
optional parameter, so every parameter offered gets filled — the model invented
`payload_lb: 1600` for a Cessna 172, exceeding its 870 lb useful load, and both
plans were refused for a reason nobody asked for. Withholding free numerics
fixed that and broke aircraft selection, because "in a Cirrus" could no longer
reach the tool; an enum of the 47 keys fixed that. Then the same reasoning
applied to altitude left `get_winds_aloft` unable to be given one at all: asked
for the wind at 35,000 ft it answered at its 8,000 ft default and labelled the
answer 35,000. The rule that came out of it is narrower than any of the three
attempts: **expose a parameter where it is the question, withhold it where the
aircraft already knows the answer.**

**A single flight plan could exhaust the weather API's quota.** Live winds
failed on every long route for three sessions. Open-Meteo meters *work*, not
requests — roughly locations × variables × days — and at a fixed 0.5° grid a
transcontinental plan wanted ~1,936 units against a 600-per-minute allowance. No
retry schedule could have helped. The cell count is now capped and the grid
resolution follows, so short routes keep fine resolution and long ones coarsen.

**The model mistyped numbers it had to reformat.** `27N023W` became `27NN023W`
in a 30-waypoint oceanic route; `1h01m` became "1 hour 4 minutes". In the same
reply, distance, fuel, altitude and airspace count were all exact, and the
`wind` and `restricted_airspace` sentences were copied verbatim. The compact
token was the only thing the model had to *rewrite* rather than repeat. So the
tool supplies the phrasing — `ete_spoken: "1 hour 1 minute"` — the same move as
the compass point, which exists because 239° came back as "from the northeast".

**The map drew the wrong line.** The dashed reference course was drawn from two
points, and Leaflet joins two points with a line that is straight *on screen* —
a rhumb line, not a great circle. San Francisco to Dubai showed a "direct
course" labelled 7,030 nm running across Africa, beside a route labelled 7,290
nm that appeared to detour thousands of miles over the Arctic for nothing. The
route was right; the line it was compared against was a different path. The
course is now sampled at 64 points along the actual great circle.

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

Oceanic legs use generated lat/lon waypoints on the same naming convention real
oceanic tracks use. The lattice is static, whereas real organised track systems
are republished daily to follow the jet stream; feeding the live winds into the
grid layout is the natural next step.

The on-device model's 4,096-token context holds roughly two conversational turns
with the lean tool set; the local 8B model has 32,768 and does not run out.
`ModelBackend` is the seam where a larger model drops in.

The procedure corpus is fifteen documents of general aviation practice, written
for this project. Retrieval selects *which* apply to a flight and the report
anchors them to that flight's computed figures, but the documents themselves say
the same thing wherever they appear — retrieval selects text, it does not write
it. Aircraft-specific manual excerpts would make the checklist concretely
different between a 172 and a 777 rather than differing only in which topics
appear.

## Verification

- A\* matched against **exhaustive brute-force search** on 400 random graphs —
  zero mismatches
- A\* matched against **Dijkstra** on real routes — identical costs, far fewer
  expansions
- Airspace avoidance verified leg-by-leg: 9 crossings before, 0 after
- Oceanic routing measured before and after the grid on five routes, including
  two overland controls that must *not* change
- The agent loop tested against a `ScriptedBackend` — no model, no network, and
  end to end against the Ollama backend, which is the one that drives it
- **The tool layer proved to be an adapter, not new logic**: routes planned via
  `dispatch("plan_flight")` compared against direct `plan_route()` calls on a
  domestic hop, a transcontinental route and an ocean crossing — identical
  waypoints, distance, time, fuel and phase profile
- Retrieval ranking tested with hand-chosen vectors, so the ordering is verified
  with no model, no network and no embedding service; live tests confirm that
  "ice on the wing" reaches the icing document and "the engine has failed"
  reaches engine failure
- The report refuses uncited checklist items, verified against a real generated
  answer containing one invented item and one citation to a document that does
  not exist
- **598 unit tests**
