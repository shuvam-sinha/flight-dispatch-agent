# Bugs worth recording

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
agent planned a flight from it.

Ranking is now a tuple compared element by element, so it is strict priority
rather than a weighted score with constants to tune. Every field is a real
column in the data:

```python
(airport_type,        # large_airport (1,172) before small_airport (42,674)
 not scheduled_service,   # 4,357 carry commercial service
 not iata_code,           # 9,052 have one; an airstrip does not
 not municipality_match,  # "San Francisco" the city, not the name
 -total_runway_area,      # every open runway, length × width
 len(name))               # last resort, all it was ever suited for
```

The first match wins on the first field that separates them. Worked through:

```
"chicago"
KORD  (0, False, False, False, -12,634,250, 36)
KMDW  (0, False, False, False,  -3,002,640, 36)   ← ties four fields, loses on area
KRFD  (1, False, False, False,  -2,730,300, 38)   ← medium_airport, out on field one
```

Runway *area* replaced longest-single-runway, which ranked Al Maktoum above
Dubai International on a strip 174 ft longer at an airport with almost no
traffic. Counting every runway tracks how much aircraft a field can actually
handle: 12 of 15 multi-airport cities came out right on longest runway, 14 on
total area.

Matching was wrong too: `New York JFK` matched nothing (city and code live in
different columns), `Sao Paulo` matched nothing (accents), and `JFK` matched
nothing (IATA codes aren't substrings of names). Search now widens through exact
ICAO, exact IATA, phrase, all-words-anywhere, best partial match, and a
spacing-blind compare — each pass *widening* rather than replacing, because
"Los Angeles airport" is contained in "Hilton Los Angeles Airport Helipad" and
not in "Los Angeles International Airport".

Across 45 major world cities the ranker now returns the expected airport 44
times. The exception is Mexico City, where Felipe Ángeles has four runways and
almost no traffic against Benito Juárez's two — no runway metric separates them,
and this dataset carries no passenger figures.

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

---

[← Back to the README](../README.md)

**Other pages**

- [Usage](usage.md) — Running it, from the agent or the CLI
- [How it works](how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Layout](layout.md) — Which file does what, and what depends on what
- [Design notes](design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Scope](scope.md) — What this models, and what it does not
- [Verification](verification.md) — How each claim is checked
