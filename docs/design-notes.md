# Design notes

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

---

[← Back to the README](../README.md)

**Other pages**

- [Usage](usage.md) — Running it, from the agent or the CLI
- [How it works](how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Layout](layout.md) — Which file does what, and what depends on what
- [Bugs worth recording](bugs.md) — Nine fixed failures and the reasoning behind each
- [Scope](scope.md) — What this models, and what it does not
- [Verification](verification.md) — How each claim is checked
