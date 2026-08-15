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

**Checkpoints 1–4 complete.** A\* routing over a waypoint mesh built from real
navaid data, with flight time as the cost function, live global winds aloft, FAA
restricted-airspace avoidance, a virtual routing grid for oceanic legs, and a
conversational dispatcher agent running on-device.

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
| `--backend` | `apple` | Model backend |
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
| `--naive` | off | CP1's corridor sampler, for comparison |
| `--no-grid` | grid on | Disable virtual oceanic waypoints |

```bash
python -m unittest discover tests      # 410 tests
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
tools.py        the five tools, as ToolSpecs with English descriptions
agent.py        the loop; ModelBackend protocol
backend_apple.py    on-device Foundation Models
  ↓
plan_route.py   CLI          dispatch.py   conversational REPL
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

### CP4 — the dispatcher agent

Adding a conversational agent on top of CP1–CP3 took **3,314 added lines and 17
deleted ones**. Nothing in `geo.py`, `search.py`, `graph.py`, `cost.py` or
`airspace.py` had to change — the routing engine does not know a model is
calling it. That was the whole bet of the architecture, and it paid.

**The tool surface** (`tools.py`) wraps the CP1–CP3 functions in five tools:

| Tool | What it does |
| --- | --- |
| `find_airport` | ICAO code, name, or city → airport, ranked by significance |
| `list_aircraft` | The 47 profiles, with cruise speed, altitude, seats, range |
| `plan_flight` | Full plan: route, waypoints, ETE, fuel |
| `get_winds_aloft` | Forecast wind at one point and altitude |
| `check_airspace` | Restricted areas on the direct course between two airports |

A `ToolSpec` holds a name, an English description, a parameter schema and a
Python callable. `dispatch()` looks up the name, validates arguments, calls the
function, and — critically — **returns errors as data rather than raising**. A
model that gets `{"error": "unknown aircraft 'b737'"}` can correct itself on the
next turn; an exception just kills the conversation.

**The agent loop** (`agent.py`) is hand-written, ~40 lines at its core: send the
conversation, check for tool calls, execute them, append the results, send
again, repeat until the model answers in prose. `ModelBackend` is a three-method
`Protocol`, so the loop is tested against a `ScriptedBackend` with no model and
no network.

**The backend** is Apple's on-device Foundation Models — free, private, no API
key, and the model itself never touches the network.

### Planned

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

`ModelBackend` exists precisely so this choice is reversible — a Claude API
backend is a third implementation of the same three methods, and the loop above
it does not change.

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

Four of these came from running the assistant conversationally rather than from
a test. Agents fail in ways unit tests do not reach.

**Along-track distance was unsigned.** The textbook `acos` form returns 0–π, so
a navaid *behind* the aircraft reported positive forward progress and CP1's
corridor sampler happily selected waypoints in the wrong direction. Fixed by
recovering the sign from `cos(θ₁₃ − θ₁₂)`.

**The bounding box missed the great-circle bulge.** Sampling only the endpoints
cut 1.35° off KJFK→EGLL, silently discarding navaids the route needed.
`route_bounding_box` now samples 32 points along the arc.

**The antimeridian.** PANC→RJTT searched 293° of longitude instead of 77°,
pulling 5,410 navaids instead of 182. Fixed with `unwrap_longitudes`.

**A connected graph is not the same as no isolated nodes.** The k-nearest floor
guarantees every node has neighbours, which is not the same guarantee: KJFK→EGLL
built 695 nodes of which only 350 were reachable, split into clusters by open
water. `_bridge_components` joins them with a union-find pass.

**The model invented a payload.** Apple's schema format has no optional fields,
so "(optional)" written in prose was invisible — the model supplied
`payload_lb: 1600` for a Cessna 172, exceeding its 870 lb useful load, and both
plans were correctly refused for a reason the user never asked for. Free
numerics are now withheld from the schema entirely; only required fields, enums
and booleans are exposed.

**Withholding then broke aircraft selection**, because "in a Cirrus" could no
longer reach the tool. Fixed by making `aircraft` an enum of all 47 keys, with
model names inlined as hints (`sr22 (Cirrus SR22)`) for when the lean tool set
drops `list_aircraft`.

**Tool calls displayed against the wrong results.** A transcript showed
"Minneapolis" printed above Chicago Executive's result. Apple's SDK runs tools
concurrently; a shared counter plus a positional `zip` mispaired them. Fixed
with an id per invocation and explicit pairing.

**A 2,093-character URL inside one error message** consumed half the context
window. `_short_error` now truncates at the URL boundary and caps at 180 chars.

**`find_airport("San Francisco")` returned a Mexican airstrip** — and the agent
planned a flight from it, which is the failure mode that matters: a wrong answer
delivered fluently. Matches had been sorted by name length. Ranking now uses
real significance signals: airport type, scheduled service, IATA code,
municipality match, then longest runway (which meant finally opening the
`runways.csv` that had been sitting in the data directory since CP1).

**The size tiebreaker measured the wrong thing.** Among airports tied on type,
scheduled service, IATA code and city, longest-single-runway decided — and it
loses to anywhere that built one long strip and little else. Dubai
International, ~90 million passengers a year, ranked below Al Maktoum, which is
nearly empty with a runway 174 ft longer. Tokyo Haneda lost to Narita the same
way. Total runway area — every runway, times its width, excluding closed ones —
tracks how much traffic an airport can handle. On fifteen multi-airport cities:
longest runway 12 correct, total area 14.

**Accents.** `_normalise` stripped punctuation but not diacritics, so
`Sao Paulo` never matched Guarulhos' municipality `São Paulo` and fell through
to a hotel helipad that spells it without one. NFKD decomposition splits an
accented character into base letter plus combining mark; dropping the marks
leaves the base letters. Zürich, Málaga and Köln were the same bug.

**`keywords` was never read.** OurAirports keeps alternate names there —
`Londres` for Heathrow, `Ciudad de México` for Benito Juárez, plus IATA
metropolitan area codes (`LON`, `NYC`, `CHI`). Adding it to the search text
costs nothing and makes local-language queries work.

Across 45 major world cities the ranker now returns the expected airport 44
times. The exception is **Mexico City**: Felipe Ángeles is a converted air force
base with four runways and 8.9M sq ft of pavement against Benito Juárez's 3.8M —
more concrete, almost no traffic. No runway-derived metric separates them and
this dataset carries no passenger figures, so it is a documented miss with a
test pinning the current behaviour. Naming the airport (`Benito Juarez`, `MEX`,
`AICM`, `Ciudad de Mexico`) reaches it correctly.

**`find_airport("New York JFK")` found nothing at all** — the phrase is a
substring of no field, because the city and the code live in different columns.
Fixing it exposed a family of related misses, and the search now tries, in
order: exact ICAO, exact IATA, phrase, all-words-anywhere, best partial match,
and finally a spacing-blind compare so `OHare` reaches `O'Hare`. Each pass
*widens* rather than replaces, because phrase search can succeed on the wrong
thing: "Los Angeles airport" is contained in "Hilton Los Angeles Airport
Helipad" but not in "Los Angeles International Airport". All candidates go to
the ranker, which knows a large airport outranks a helipad.

**The airspace result was narrated backwards.** The router avoided all 95
restricted areas near a route — and the reply said *"Route includes prohibited
and restricted airspace."* Every number was correct; the safety claim came out
inverted. The cause was two fields the model had to assemble itself:

```python
"airspace_avoidance_applied": True
"restricted_volumes_considered": 95
```

"Considered" reads equally well as *taken into account* and *included in*, and
beside a count of 95 the second reading is the more natural one. The fix was to
return the conclusion as a sentence — `"Routed clear of 95 active volumes. The
route crosses none of them."` — which cannot be read the other way.

This generalised the project's founding rule. "The LLM never does computation"
was not enough: `95` is a number the model reported faithfully, and *through*
versus *around* is an interpretation built from the identical number. So the
rule is now **the LLM never does computation, and never decides what a result
means either.** Any field a reasonable reader could draw the opposite conclusion
from should be a sentence, not a value.

No test caught this, because every test checked the router, and the router was
never wrong.

**An unspecified aircraft was filled in silently.** Asked to fly KJFK to EGLL
with no type named, `plan_flight` defaulted to a Cessna 172 and returned a
straight-faced plan: 22h15m and 195 gallons, in an aircraft holding 56. The
default now reports itself, and the schema tells the model to omit the parameter
rather than guess.

The range warning was wrong in a subtler way: it said *"a fuel stop is
required."* The discriminator turned out not to be how far short the aircraft
falls — a 172 crossing the United States needs four stops, and that is a trip
people genuinely make. What makes the Atlantic different is that there is
nowhere to land, and the oceanic waypoints already record exactly that. Overland
shortfalls suggest fuel stops; oceanic ones say the aircraft cannot fly the
route.

**Live winds were read at the wrong hour.** `forecast_hour=0` was documented as
"roughly now." Open-Meteo's hourly series starts at 00:00 UTC of the current
day, so index 0 is midnight — accurate at 00:30 UTC and twenty-three hours stale
by late evening. The data was genuinely live, from the current model run; it was
being read at the wrong point in it. The response carries its own timestamps, so
the index is now looked up rather than assumed.

**The wind altitude could not be asked for.** Asked for the wind at 35,000 ft
over Denver, the agent answered *12 kt from 239°, 11.9°C* and called it the
35,000 ft wind. The true value was **26 kt from 223°, −41°C**. The temperature
gives it away — nothing at FL350 is ever +12°.

The model was not being careless. `_is_exposed` withholds free numerics from
Apple's schema, which was the fix for the invented `payload_lb: 1600`. The side
effect was that `altitude_ft` could not be passed to `get_winds_aloft` **at
all** — so it answered at its 8,000 ft default, truthfully, and the model
attached the user's altitude to it. `altitude_ft` is now an enum of ten
altitudes, which passes the filter the same way `aircraft` does, with values
chosen to land on distinct pressure levels so no two options return identical
data.

The schema fix alone was not enough, because the altitude was **already** a
field in the result and the model skipped it — a bare number beside other bare
numbers is easy to skip. The wind result now names its own altitude in the same
sentence as the wind:

```
At 34,000 ft: wind from 223 degrees true (southwest) at 26 kt, temperature -41C.
```

Same principle as the airspace fix: a result that states what it is cannot be
relabelled. The compass point is there for a related reason — the model rendered
239° as "from the northeast", which is the opposite side of the compass.

**Then the same enum broke `plan_flight` within the hour.** Given *"plan a flight
from KJFK to KLAX in a 737"*, the model volunteered `altitude_ft='30000'` — which
nobody asked for, and a 737-800 cruises at 35,000. The follow-up *"what about in
a 172?"* carried that 30,000 into an aircraft with a **14,000 ft service
ceiling**, and the plan was refused outright. Before the enum existed, the model
could not set the parameter and both flights planned fine.

So the rule is narrower than "expose it": **expose an altitude where the altitude
is the question, and withhold it where the aircraft already knows the answer.**
Wind without an altitude is meaningless, so `get_winds_aloft` keeps its enum.
An aircraft profile's own cruise altitude is better than anything the model will
invent, so `plan_flight` goes back to withheld. `check_airspace` keeps the enum —
altitude is a genuine query dimension there — but its result now names the
altitude in the same sentence as the count, because a wrong altitude there fails
*silently* rather than erroring.

Worth noting what went right: the tool returned `"cannot cruise at 30,000 ft;
its service ceiling is 14,000 ft"` as **data**, and the model explained the
problem and offered alternatives instead of crashing. That is the errors-as-data
design doing its job.

**The model mistyped numbers it had to reformat.** Two cases, both from real
transcripts. On a South Atlantic route the oceanic fix `27N023W` was written as
`27NN023W`. On KSFO-KLAS an ETE of `1h01m` was reported as "1 hour 4 minutes".

The pattern is sharp: in that same KSFO reply, distance, fuel, altitude, wind
and airspace count were all exact, and the `wind` and `restricted_airspace`
sentences were copied word for word. The compact token `1h01m` was the only
thing the model had to *rewrite* rather than repeat -- and rewriting is where
the error entered.

So the tool supplies the phrasing rather than leaving it to be derived:
`ete_spoken: "1 hour 1 minute"` alongside the compact `ete`. Same move as the
compass point, which exists because 239 degrees came back as "from the
northeast". The remaining error surface here is narrow and worth naming
precisely: the model is no longer inventing facts, it is making typos.

**A wind fetch failure destroyed the whole plan.** Now the route is replanned in
still air and returned with a `wind_note` saying so. A degraded answer beats no
answer.

**Open-Meteo returned 429s** on batches of 100 coordinates. Reduced to 50, with
retry and backoff honouring `Retry-After`.

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
are republished daily to follow the jet stream; feeding CP3's live winds into
the grid layout is the natural next step.

The on-device model's 4,096-token context holds roughly two conversational turns
with the lean tool set. The `ModelBackend` protocol is the seam where a
larger-context model drops in.

## Verification

- A\* matched against **exhaustive brute-force search** on 400 random graphs —
  zero mismatches
- A\* matched against **Dijkstra** on real routes — identical costs, far fewer
  expansions
- Airspace avoidance verified leg-by-leg: 9 crossings before, 0 after
- Oceanic routing measured before and after the grid on five routes, including
  two overland controls that must *not* change
- The agent loop tested against a `ScriptedBackend` — no model, no network
- **The tool layer proved to be an adapter, not new logic**: routes planned via
  `dispatch("plan_flight")` compared against direct `plan_route()` calls on a
  domestic hop, a transcontinental route and an ocean crossing — identical
  waypoints, distance, time, fuel and phase profile
- **410 unit tests**
