# Usage

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

---

[← Back to the README](../README.md)

**Other pages**

- [How it works](how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Layout](layout.md) — Which file does what, and what depends on what
- [Design notes](design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Bugs worth recording](bugs.md) — Nine fixed failures and the reasoning behind each
- [Scope](scope.md) — What this models, and what it does not
- [Verification](verification.md) — How each claim is checked
