# Verification

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

---

[← Back to the README](../README.md)

**Other pages**

- [Usage](usage.md) — Running it, from the agent or the CLI
- [How it works](how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Layout](layout.md) — Which file does what, and what depends on what
- [Design notes](design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Bugs worth recording](bugs.md) — Nine fixed failures and the reasoning behind each
- [Scope](scope.md) — What this models, and what it does not
