# Layout

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

---

[← Back to the README](../README.md)

**Other pages**

- [Usage](usage.md) — Running it, from the agent or the CLI
- [How it works](how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Design notes](design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Bugs worth recording](bugs.md) — Nine fixed failures and the reasoning behind each
- [Scope](scope.md) — What this models, and what it does not
- [Verification](verification.md) — How each claim is checked
