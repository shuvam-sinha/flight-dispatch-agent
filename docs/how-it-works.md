# How it works

### The data

72,417 airports and 10,841 navaids from OurAirports — out of 85,825 and 11,008
rows, the rest lacking a usable identifier or coordinates — plus 780 FAA
special-use airspace volumes. `geo.py` holds no project types, so everything above reuses
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

---

[← Back to the README](../README.md)

**Other pages**

- [Usage](usage.md) — Running it, from the agent or the CLI
- [Layout](layout.md) — Which file does what, and what depends on what
- [Design notes](design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Bugs worth recording](bugs.md) — Nine fixed failures and the reasoning behind each
- [Scope](scope.md) — What this models, and what it does not
- [Verification](verification.md) — How each claim is checked
