# Project Brief: Flight Dispatch Agent

This document exists so a new session (human or Claude) can pick up this
project with zero back-and-forth. Read this fully before doing anything else.

## Who's building this and why

Builder is a rising junior in Computer Engineering (UIUC), preparing for
software/AI internship applications. This is a portfolio project with a hard
deadline mindset: 2.5 weeks, weekdays only, ~4-5 hours/day (~55-65 hours
total). The explicit goal is to produce evidence of two things recruiters
screen for: (1) real software engineering fundamentals (algorithms, data
modeling, systems design) and (2) hands-on experience with how modern
LLM/agentic applications are actually built (tool use, agent orchestration,
RAG) — not just "I called a chatbot API."

Given the time budget, scope has been deliberately cut multiple times
throughout planning. Do not re-expand scope without the user explicitly
asking. If time runs short, the checklist agent (CP5) is the safest thing to
cut back to "future work," since it's the most isolated component.

## What the project is

A flight-dispatch planning assistant for **pilots and flight dispatchers** —
someone planning a specific general-aviation flight before departure. It is
explicitly **not**:
- A consumer flight-booking tool (not "search and buy a ticket," no Expedia-style UX)
- An airline operations/scheduling system
- An ATC-accurate flight-filing system

You give it a natural-language request (e.g., "plan a flight from KPWK to
KORD, avoid restricted airspace, tell me fuel and time"), and it returns a
route, fuel/time estimate, a preflight checklist, and an interactive map —
built by an LLM agent orchestrating a set of real Python tools, not by the
LLM inventing numbers.

## Core design principle (do not violate this)

**The LLM never does computation.** It only decides which tool to call, with
what arguments, in what order, and it synthesizes/narrates results in
natural language. All routing math, weather data, airspace checks, and
retrieval happen in plain Python functions that exist independent of any
LLM. This is the difference the project is explicitly built to demonstrate:
an LLM that's grounded in real tool output vs. one that hallucinates
plausible-sounding answers.

## Honest scope / what NOT to claim

This is a simplified planning engine, not a real-world-accurate one. Be
upfront about this in the README and in any demo:
- Routes are built over a **waypoint mesh graph** (airports + navaids
  connected by proximity), not real published airways (V/J/Q routes) — those
  aren't in free public data (OurAirports doesn't have them).
- No SIDs/STARs, no ATC-assigned/preferred routings, no cost-index
  optimization like real airline dispatch uses.
- No live/typical air-traffic modeling (that's the optional stretch goal,
  CP7 — only attempt if CP1-6 finish early, using ADS-B data e.g. OpenSky
  Network).
- Single fixed aircraft performance profile — no multi-aircraft fleet
  database, no "recommend best aircraft" advisory engine. (These were in an
  earlier, much larger 10-step plan and were cut for time.)
- Weather and restricted-airspace avoidance ARE real and functional — those
  are genuinely modeled, not simplified away.

## Tech stack, and why each piece was chosen

- **Python 3.10+** — chosen over C++ specifically because this project is
  data/API/geospatial-integration heavy, not performance-critical; Python
  gets to a working end-to-end system much faster.
- **`csv` (stdlib)** — parses OurAirports data (`airports.csv`,
  `navaids.csv`, `runways.csv`), downloaded via `scripts/download_data.py`.
  These are NOT committed to git (see `.gitignore`); each session re-runs
  the download script.
- **`shapely`** — geospatial polygon intersection, used to detect whether a
  route crosses FAA restricted/special-use airspace (GeoJSON polygons).
- **`requests`** — HTTP client for NOAA aviationweather.gov (winds aloft)
  and for pulling FAA GeoJSON airspace data.
- **Custom A\* (using `heapq`)** — pathfinding over the waypoint mesh graph.
  Written by hand, not via a graph library like networkx, since implementing
  it is itself part of the algorithmic value of the project.
- **Apple Foundation Models Python SDK** (`github.com/apple/python-apple-fm-sdk`,
  confirmed real, official Apple repo) — the FIRST LLM backend used for the
  dispatcher agent (CP4), chosen so the user can build/test the agent loop
  for free, on-device, with no API cost. Provides a `Tool` class and
  `LanguageModelSession` for tool-use, functionally equivalent in concept
  to Claude's tool-use API. Requires macOS 26+, Xcode 26+, Apple Silicon,
  and Apple Intelligence enabled in System Settings — **already confirmed
  compatible** on the user's machine (macOS 26.2, M3 chip), but Xcode
  version and Apple Intelligence toggle should still be double-checked
  before starting CP4 (`xcodebuild -version`, System Settings).
- **Anthropic Claude API** — the SECOND LLM backend, swapped in after the
  agent loop works against Apple's SDK. Only the model-client-facing part
  of the loop changes (different tool schema format, different response
  parsing); all the actual tool functions from CP1-3 stay untouched, since
  they don't know or care which LLM is calling them.
- **Embeddings + in-memory cosine similarity (no vector DB)** — used for
  the RAG checklist agent (CP5). Deliberately NOT using a vector database;
  the corpus is ~10-20 small text chunks (POH excerpts, FAA procedures),
  so an in-memory list + cosine similarity is the right-sized solution.
  Note this explicitly in the README as "kept simple on purpose" so it
  doesn't read as an oversight in review.
- **`folium`** — generates an interactive Leaflet map (HTML) showing the
  route and restricted-airspace overlay. Chosen over matplotlib/cartopy
  (static, less demo-friendly) and over Plotly (more code for similar
  payoff).
- **git** — one commit per checkpoint (see below), ideally tagged
  (`cp1-route-skeleton`, `cp2-astar`, etc.) so there's a clean before/after
  history to walk an interviewer through.

## The 6 checkpoints (+ 1 stretch goal)

Each checkpoint should end in a working, demoable, committed state — not
just code that compiles. If a session is picking up mid-project, check
which checkpoint's "definition of done" is met to know where things stand.

### CP1: Data ingestion + basic route skeleton
**Goal:** prove the data layer works end to end, nothing algorithmic yet.
**Definition of done:** `python plan_route.py --origin KPWK --dest KORD`
prints an ordered list of real waypoints (airports + any navaids near the
straight-line path) with real coordinates pulled from OurAirports data.
**Explicitly NOT in scope for CP1:** graph construction, A*, weighting,
wind, airspace. Resist scope creep here.
**Status as of last session:** scaffolded but NOT yet verified running.
Files created: `README.md`, `requirements.txt`, `.gitignore`,
`data/README.md`, `scripts/download_data.py`, `flight_dispatch/__init__.py`,
`flight_dispatch/models.py` (Airport, Navaid dataclasses),
`flight_dispatch/data_loader.py` (CSV parsing, ICAO-keyed airport lookup),
`flight_dispatch/route.py` (haversine distance, cross-track/along-track
great-circle math, `naive_route()` which picks navaids within a corridor of
the origin-dest great-circle line), `plan_route.py` (CLI entrypoint),
`tests/test_route.py` (unit tests for haversine, cross/along-track math,
and naive_route behavior).
**Immediate next steps:** create a venv, `pip install -r requirements.txt`,
run `python scripts/download_data.py` to pull real CSVs, run
`python -m unittest discover tests` to verify tests pass, then run
`plan_route.py` against a real ICAO pair to confirm it works end to end.
Then commit.

### CP2: A* route planner over waypoint mesh
**Goal:** replace the naive corridor-based route with a real shortest-path
search.
**Definition of done:** build a waypoint mesh graph (nodes = airports +
navaids within a bounding box around origin/dest, edges = connections
between nodes within some distance threshold — build this lazily per
request, not as one global precomputed graph, since a global US-wide mesh
would be too large/slow). Implement A* with a great-circle-distance
heuristic. `plan_route.py` (or its successor) returns a genuine shortest
path over this graph, still using plain distance as the cost (no wind
yet).

### CP3: Wind + airspace aware routing
**Goal:** this is where the route becomes realistic, not just geometric.
**Definition of done:**
- Integrate NOAA aviationweather.gov winds-aloft data; adjust edge costs
  by converting true airspeed to ground speed using wind vectors at the
  aircraft's cruising altitude.
- Load FAA special-use-airspace GeoJSON; use shapely to detect when a graph
  edge crosses a restricted polygon, and block/penalize those edges so A*
  routes around them.
- Add a single fixed `AircraftProfile` (cruise speed, service ceiling,
  basic fuel-burn constant) — this is the ONE aircraft type used throughout
  the project (no fleet database, no aircraft-selection engine).
- Output now includes wind-adjusted ETE (estimated time en route) and a
  fuel estimate, and the route visibly bends around restricted zones when
  a direct path would cross one.

### CP4: Dispatcher agent with tool use
**Goal:** let a user ask for a flight plan in natural language instead of
calling functions directly; also support a real back-and-forth
conversation, not just one-shot requests.
**Definition of done:**
- Wrap the CP1-3 functions as tool schemas.
- Build the agent loop: send message -> check if model wants to call a
  tool -> execute the matching Python function -> send the result back as
  a tool result -> repeat until the model produces a final answer.
- **Start against Apple's Foundation Models Python SDK** (free, on-device)
  to build and debug this loop without API cost. Once it's working,
  **swap the model-client layer to Claude API** — expect to rewrite the
  client-facing part of the loop (tool schema format, response parsing)
  but NOT the tools themselves.
- **Maintain conversation history across turns**: keep a running list of
  all messages (system/user/assistant/tool) and pass the full history on
  every call, so the user can do things like "plan KPWK to KORD" then
  follow up with "what if I used a different aircraft" and have the agent
  respond in context, without restating the whole request.
- A natural-language request should produce the same underlying route as a
  direct CP3 call, but produced via the agent's tool orchestration, plus
  the agent should handle at least a simple follow-up question in the same
  session.

### CP5: RAG-based checklist agent
**Goal:** replace what would otherwise be a hardcoded rule-based checklist
with a retrieval-grounded one — this is the strongest "I understand RAG"
proof point in the project.
**Definition of done:**
- Assemble a small local corpus (10-20 short text chunks): aircraft POH
  excerpts, generic FAA procedures/checklist templates, basic emergency
  procedures. Public domain / your own summaries, stored as local text
  files.
- Embed the corpus once (pick any convenient embedding
  model/library — sentence-transformers, or an API-based embedding model;
  not yet decided as of last session).
- At request time, build a query from the flight's actual conditions
  (aircraft type, METAR weather, phase of flight), embed it, retrieve the
  top-k most similar chunks via cosine similarity (plain Python, in-memory
  — no vector database; the corpus is intentionally small).
- Have the agent generate a checklist using ONLY the retrieved chunks as
  grounding, and require it to cite which chunk each item came from.
- This should work as a standalone demo independent of the routing
  pipeline (i.e., testable without needing a full flight plan first).

### CP6: Final report, map, and polish
**Goal:** turn working parts into a finished, demoable product.
**Definition of done:**
- Combine everything into a structured JSON + HTML report: route,
  waypoints, ETE, fuel estimate, checklist (with citations), and a
  natural-language dispatch briefing.
- Embed a `folium` interactive map in the HTML report, showing the route
  as a polyline and restricted airspace polygons overlaid (so the
  airspace-avoidance behavior is visually obvious, not just numerically
  correct).
- Test against several real ICAO pairs; handle edge cases gracefully (bad
  ICAO code, no route found, a tool call failing, agent giving up).
- Write/update the README with an architecture diagram or clear written
  walkthrough (data layer -> A* -> wind/airspace -> dispatcher agent -> RAG
  checklist agent -> report), plus the "honest scope / limitations"
  section from above.
- Prepare a short demo script/flow for interviews.

### CP7 (stretch, optional): Traffic-aware routing
**Only attempt if CP1-6 are done with time remaining.** Pull an ADS-B
snapshot for the relevant departure time window (e.g., OpenSky Network's
free API) and add traffic density as a soft cost penalty in A*, similar to
how restricted airspace is penalized. This directly answers "does the
route account for typical air traffic at that time of day," which the
project does NOT claim to do otherwise — be careful not to imply this
capability exists unless CP7 is actually done.

## How to explain this project simply

**One-liner:** "An AI flight-dispatch assistant — you give it a plain-
language flight request, and an LLM agent orchestrates a set of tools
(routing, weather, airspace, checklist generation) to produce a full
flight plan with a map, grounded entirely in real data rather than
LLM-invented numbers."

**To a technical audience, using LLM terminology:** the routing/weather/
airspace logic are deterministic Python functions exposed to an LLM as
tools via function calling. A dispatcher agent plans and executes a
multi-step tool-call sequence and grounds every claim in actual tool
output. A separate RAG component embeds a small corpus of aviation
reference documents and retrieves relevant chunks at query time so the
checklist agent generates cited, grounded output instead of hallucinating
one. The LLM is deliberately kept out of the computation path — it only
orchestrates and synthesizes.

**Division of labor, if asked "did you build this or did the AI build it
for you":** the builder writes 100% of the code — every tool, the agent
orchestration loop, the RAG pipeline, all error handling. The LLM's role
inside the finished system is narrow: at runtime, given the user's request
and the available tool schemas, it decides which tool to call and
synthesizes a final answer. It doesn't write the code and doesn't perform
the routing/weather/airspace computations itself.

## Repo location and current file state

Project lives at `~/Desktop/Projects/flight-dispatch-agent` (moved here
from `~/work/` at the user's request — Desktop/Projects is the correct
home for this and future projects).

As of the last session, CP1 files exist but have NOT been runtime-verified
(no venv created, no dependencies installed, no data downloaded, tests not
run). That is the very next action to take.
