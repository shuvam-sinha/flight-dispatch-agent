# Scope

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

---

[← Back to the README](../README.md)

**Other pages**

- [Usage](usage.md) — Running it, from the agent or the CLI
- [How it works](how-it-works.md) — Data, routing, wind, airspace, the agent, retrieval, the report
- [Layout](layout.md) — Which file does what, and what depends on what
- [Design notes](design-notes.md) — Why A*, why these thresholds, why an on-device model
- [Bugs worth recording](bugs.md) — Nine fixed failures and the reasoning behind each
- [Verification](verification.md) — How each claim is checked
