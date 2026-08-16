"""The dispatch report: everything the project knows, in one file.

WHY THIS EXISTS
---------------
Every part of this system so far is reachable only through a terminal.
Nobody who is not already interested will run a CLI, and a route that
bends around restricted airspace is a number in a sentence until you can
see it bend.

So the last piece is one self-contained HTML file -- route, map, fuel,
airspace, checklist with citations -- plus the same content as JSON for
anything that wants to consume it rather than read it.

CITATIONS ARE ENFORCED HERE, NOT REQUESTED
------------------------------------------
The system prompt asks the model to cite every checklist item. Asked to
plan a flight and give a checklist in one request, it once ignored that
and wrote eight items from memory. An instruction is a request.

This module treats an uncited item as an item that does not go in the
report. It is not dropped silently -- that would hide the failure -- but
listed separately as rejected, so a reader sees both what was grounded
and what was not. The report cannot lie about its own provenance.

NO TEMPLATE ENGINE
------------------
The HTML is built with f-strings. Jinja would be a dependency and a
second language for one page whose structure never varies. `folium`
already renders the map to HTML, and that is embedded directly.
"""

import html
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from .retrieval import CITATION_PATTERN

# Items shorter than this are almost certainly a fragment rather than a
# procedure -- a stray bullet, a heading the model bulleted by accident.
MIN_ITEM_CHARS = 12


@dataclass(frozen=True)
class ChecklistItem:
    """One checklist line and the procedures it claims to come from."""

    text: str
    citations: List[str] = field(default_factory=list)

    @property
    def is_grounded(self) -> bool:
        return bool(self.citations)


@dataclass
class DispatchReport:
    """A complete flight briefing, ready to render.

    Built from a `RoutePlan` and, optionally, a checklist the agent
    produced. Everything numeric comes from the plan; everything textual
    in the checklist comes from the corpus. Nothing here computes.
    """

    plan: Any  # RoutePlan -- untyped to keep this module import-light
    checklist: List[ChecklistItem] = field(default_factory=list)
    rejected: List[str] = field(default_factory=list)
    sources: List[Dict[str, str]] = field(default_factory=list)
    figures: Dict[str, Any] = field(default_factory=dict)
    briefing: str = ""
    generated_at: str = ""

    def __post_init__(self):
        if not self.generated_at:
            self.generated_at = datetime.now(timezone.utc).strftime(
                "%Y-%m-%d %H:%M UTC"
            )

    # -- data ------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """The report as plain data.

        Deliberately the same shape as the HTML shows, so the two cannot
        drift: a reader and a program see the same report.
        """
        plan = self.plan
        aircraft = plan.aircraft

        report: Dict[str, Any] = {
            "generated_at": self.generated_at,
            "origin": {
                "icao": plan.origin.icao,
                "name": plan.origin.name,
                "elevation_ft": plan.origin.elevation_ft,
            },
            "destination": {
                "icao": plan.dest.icao,
                "name": plan.dest.name,
                "elevation_ft": plan.dest.elevation_ft,
            },
            "route": " ".join(w.ident for w in plan.waypoints),
            "waypoints": [
                {
                    "ident": w.ident,
                    "latitude": round(w.lat, 4),
                    "longitude": round(w.lon, 4),
                }
                for w in plan.waypoints
            ],
            "direct_distance_nm": round(plan.direct_distance_nm, 1),
            "route_distance_nm": round(plan.total_distance_nm, 1),
            "efficiency_percent": round(
                100.0 * plan.total_distance_nm / plan.direct_distance_nm, 1
            )
            if plan.direct_distance_nm
            else None,
        }

        if aircraft is not None:
            report["aircraft"] = {
                "key": aircraft.key,
                "name": aircraft.name,
                "cruise_tas_kt": aircraft.cruise_tas_kt,
                "cruise_altitude_ft": aircraft.cruise_altitude_ft,
            }

        if plan.ete_hours is not None:
            hours = int(plan.ete_hours)
            minutes = int(round((plan.ete_hours - hours) * 60))
            report["ete"] = f"{hours}h{minutes:02d}m"
            report["ete_hours"] = round(plan.ete_hours, 2)
            report["fuel_required_gal"] = (
                round(plan.fuel_required_gal, 1)
                if plan.fuel_required_gal is not None
                else None
            )
            report["within_aircraft_range"] = plan.is_within_range()

        if plan.phases is not None:
            phases = plan.phases
            report["profile"] = {
                "climb_minutes": round(phases.climb_time_hours * 60),
                "climb_distance_nm": round(phases.climb_distance_nm),
                "cruise_distance_nm": round(phases.cruise_distance_nm),
                "cruise_altitude_ft": round(phases.cruise_altitude_ft),
                "descent_minutes": round(phases.descent_time_hours * 60),
                "descent_distance_nm": round(phases.descent_distance_nm),
                "reached_planned_altitude": phases.reached_planned_altitude,
            }

        if plan.airspace_avoided is not None:
            report["restricted_airspace_avoided"] = plan.airspace_avoided

        if plan.grid_waypoints_used:
            report["oceanic_waypoints"] = plan.grid_waypoints_used

        report["checklist"] = [
            {"text": item.text, "citations": item.citations}
            for item in self.checklist
        ]
        report["sources"] = self.sources
        report["figures"] = self.figures

        # THE HONEST FIELD. Present even when empty, so its absence never
        # means "nothing was rejected" by accident.
        report["rejected_uncited_items"] = self.rejected

        if self.briefing:
            report["briefing"] = self.briefing

        return report

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    # -- presentation ----------------------------------------------------

    def to_html(self, airspace=None) -> str:
        """The whole report as one self-contained HTML page.

        The map is rendered by folium and embedded in an iframe, so the
        file needs no server, no assets and no network beyond the tile
        layer.
        """
        data = self.to_dict()
        return _render_html(data, self._map_html(airspace))

    def _map_html(self, airspace=None) -> str:
        """The folium map as an embeddable fragment, or "" if it fails.

        A missing map should cost the map, not the report. folium is an
        optional dependency and tile rendering is the one part of this
        that can fail for reasons outside the project.
        """
        try:
            from .mapping import route_map

            return route_map(self.plan, airspace=airspace)._repr_html_()
        except Exception:  # noqa: BLE001 - the report survives without it
            return ""

    def write(self, path: str, airspace=None) -> Dict[str, str]:
        """Write both files, returning where each landed."""
        html_path = path if path.endswith(".html") else f"{path}.html"
        json_path = html_path[: -len(".html")] + ".json"

        with open(html_path, "w", encoding="utf-8") as handle:
            handle.write(self.to_html(airspace=airspace))
        with open(json_path, "w", encoding="utf-8") as handle:
            handle.write(self.to_json())

        return {"html": html_path, "json": json_path}


def parse_checklist(text: str, index=None) -> "tuple[List[ChecklistItem], List[str]]":
    """Split a generated checklist into grounded items and rejected ones.

    THE ENFORCEMENT. An item citing nothing does not go in the report,
    and neither does one citing a document that does not exist -- a
    citation to nothing is worse than no citation, because it looks like
    provenance.

    Rejected items are returned rather than discarded. Dropping them
    silently would hide exactly the failure this exists to catch: the
    report would look clean while the model had been inventing.

    Prose around the list -- a preamble, a caveat, a closing note -- is
    neither an item nor a rejection. A checklist is a list, and the unit
    a reader trusts or distrusts is the item.
    """
    items: List[ChecklistItem] = []
    rejected: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not _looks_like_an_item(stripped):
            continue

        citations = CITATION_PATTERN.findall(stripped)
        if index is not None:
            citations = [c for c in citations if index.by_id(c)]

        if citations:
            items.append(ChecklistItem(text=_clean_item(stripped), citations=citations))
        else:
            rejected.append(_clean_item(stripped))

    return items, rejected


def _looks_like_an_item(line: str) -> bool:
    """Whether a line is a checklist item rather than surrounding prose."""
    import re

    if len(line) < MIN_ITEM_CHARS:
        return False
    return bool(re.match(r"^([-*+]|\d+[.)])\s+\S", line))


def _clean_item(line: str) -> str:
    """Strip the bullet and the citation markers, keeping the instruction."""
    import re

    without_bullet = re.sub(r"^([-*+]|\d+[.)])\s+", "", line)
    without_citations = CITATION_PATTERN.sub("", without_bullet)
    return re.sub(r"\s+", " ", without_citations).strip(" .:-") + "."


def build_report(
    plan,
    checklist_text: str = "",
    procedures: Optional[Sequence[Dict[str, Any]]] = None,
    briefing: str = "",
    index=None,
    figures: Optional[Dict[str, Any]] = None,
) -> DispatchReport:
    """Assemble a report from a plan and whatever the agent produced.

    `procedures` is `find_procedures`'s output, kept so the report can
    show what the checklist was written FROM -- a citation is only worth
    something if the source travels with it.
    """
    items, rejected = parse_checklist(checklist_text, index=index)

    sources = [
        {
            "id": procedure["id"],
            "title": procedure.get("title", procedure["id"]),
            "category": procedure.get("category", ""),
            "text": procedure.get("text", ""),
        }
        for procedure in (procedures or [])
    ]

    return DispatchReport(
        plan=plan,
        checklist=items,
        rejected=rejected,
        sources=sources,
        figures=dict(figures or {}),
        briefing=briefing,
    )


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

_STYLE = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body {
  font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  margin: 0; padding: 2rem 1.5rem; max-width: 60rem; margin-inline: auto;
  color: #14181d; background: #fbfbfc;
}
h1 { font-size: 1.6rem; margin: 0 0 .2rem; letter-spacing: -.02em; }
h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .09em;
     color: #5b6672; margin: 2.2rem 0 .7rem; font-weight: 600; }
.sub { color: #5b6672; margin: 0 0 1.6rem; }
.grid { display: grid; gap: .8rem;
        grid-template-columns: repeat(auto-fit, minmax(9rem, 1fr)); }
.stat { background: #fff; border: 1px solid #e3e7eb; border-radius: 8px;
        padding: .7rem .85rem; }
.stat .label { font-size: .7rem; text-transform: uppercase;
               letter-spacing: .06em; color: #6b7681; }
.stat .value { font-size: 1.25rem; font-weight: 600; margin-top: .15rem;
               font-variant-numeric: tabular-nums; }
.route { font-family: ui-monospace, "SF Mono", Menlo, monospace;
         font-size: .82rem; background: #fff; border: 1px solid #e3e7eb;
         border-radius: 8px; padding: .8rem; word-spacing: .35em;
         line-height: 1.9; }
.map { border: 1px solid #e3e7eb; border-radius: 8px; overflow: hidden;
       height: 30rem; }
.map iframe { width: 100%; height: 100%; border: 0; display: block; }
ol.checklist { padding-left: 1.2rem; }
ol.checklist li { margin: .5rem 0; }
.cite { font-size: .72rem; font-family: ui-monospace, Menlo, monospace;
        background: #eef3f8; color: #2c5d8f; border-radius: 4px;
        padding: .1rem .35rem; margin-left: .35rem; white-space: nowrap; }
.warn { background: #fff6e8; border: 1px solid #f0d9b0; border-radius: 8px;
        padding: .75rem .9rem; }
.warn .cite { background: #f6e3c6; color: #8a5a12; }
details { background: #fff; border: 1px solid #e3e7eb; border-radius: 8px;
          padding: .6rem .85rem; margin-bottom: .5rem; }
summary { cursor: pointer; font-weight: 600; font-size: .9rem; }
details p { color: #3f4a55; white-space: pre-wrap; margin: .6rem 0 .2rem; }
footer { margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid #e3e7eb;
         color: #8792a0; font-size: .78rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e6eaee; background: #14181d; }
  .stat, .route, details, .map { background: #1b2027; border-color: #2c333c; }
  .sub, .stat .label, footer { color: #98a3b0; }
  .cite { background: #1e3348; color: #8fbde8; }
  .warn { background: #2b2418; border-color: #4d3f24; }
  .warn .cite { background: #4d3f24; color: #e0bd7c; }
  details p { color: #c3ccd6; }
}
"""


_FIGURE_UNITS = {
    "gal": ("fuel", "reserve"),
    "lb": ("load", "payload", "weight"),
    "ft": ("altitude", "ceiling", "elevation"),
    "nm": ("distance", "range"),
    "h": ("hours",),
    "min": ("minutes",),
}


def _label(key: str) -> str:
    """A field name as a heading a person would write."""
    return key.replace("_", " ").replace(" ft", "").replace(" gal", "").replace(
        " lb", ""
    ).replace(" nm", "").replace(" hours", "").replace(" minutes", "").strip().capitalize()


def _figure(key: str, value: Any) -> str:
    """A number with the unit its field name implies."""
    if not isinstance(value, (int, float)):
        return str(value)

    for unit, markers in _FIGURE_UNITS.items():
        if any(marker in key for marker in markers):
            return f"{value:,g} {unit}"
    return f"{value:,g}"


def _stat(label: str, value: str) -> str:
    return (
        f'<div class="stat"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div></div>'
    )


def _render_html(data: Dict[str, Any], map_html: str) -> str:
    origin, dest = data["origin"], data["destination"]
    title = f"{origin['icao']} to {dest['icao']}"

    stats = [_stat("Route", f"{data['route_distance_nm']:,.0f} nm")]
    if data.get("efficiency_percent") is not None:
        stats.append(_stat("Of direct", f"{data['efficiency_percent']}%"))
    if "ete" in data:
        stats.append(_stat("Time en route", data["ete"]))
    if data.get("fuel_required_gal") is not None:
        stats.append(_stat("Fuel", f"{data['fuel_required_gal']:,.0f} gal"))
    if "profile" in data:
        stats.append(
            _stat("Cruise", f"{data['profile']['cruise_altitude_ft']:,} ft")
        )
    if data.get("restricted_airspace_avoided") is not None:
        stats.append(
            _stat("Airspace avoided", str(data["restricted_airspace_avoided"]))
        )

    sections = [
        f"<h1>{html.escape(title)}</h1>",
        f'<p class="sub">{html.escape(origin["name"])} &rarr; '
        f'{html.escape(dest["name"])}'
        + (
            f' &middot; {html.escape(data["aircraft"]["name"])}'
            if "aircraft" in data
            else ""
        )
        + "</p>",
        f'<div class="grid">{"".join(stats)}</div>',
        "<h2>Route</h2>",
        f'<div class="route">{html.escape(data["route"])}</div>',
    ]

    if data.get("within_aircraft_range") is False:
        sections.append(
            '<h2>Range</h2><div class="warn">This route exceeds the '
            "aircraft's endurance. A fuel stop is required.</div>"
        )

    if map_html:
        sections += ["<h2>Map</h2>", f'<div class="map">{map_html}</div>']

    if data["checklist"]:
        rows = "".join(
            f"<li>{html.escape(item['text'])}"
            + "".join(
                f'<span class="cite">{html.escape(c)}</span>'
                for c in item["citations"]
            )
            + "</li>"
            for item in data["checklist"]
        )
        sections += [
            "<h2>Preflight checklist</h2>",
            f'<ol class="checklist">{rows}</ol>',
        ]

    # THE REJECTED SECTION IS NOT OPTIONAL. Dropping uncited items
    # silently would leave a report that looks fully grounded while the
    # model had been inventing.
    if data["rejected_uncited_items"]:
        rows = "".join(
            f"<li>{html.escape(text)}</li>"
            for text in data["rejected_uncited_items"]
        )
        sections += [
            "<h2>Excluded &mdash; no source</h2>",
            '<div class="warn">These items were produced without citing any '
            "procedure in the corpus, so they are not part of the checklist "
            f"above.<ul>{rows}</ul></div>",
        ]

    # RENDERED HERE BECAUSE THE MODEL WOULD NOT USE THEM. find_procedures
    # returns these alongside the procedures precisely so a rule can be
    # anchored to this flight -- "carry 45 minutes of reserve, which is
    # 1,852 gal against 47,890 of capacity". Handed the numbers, an 8B
    # model quoted the documents verbatim and used none of them.
    #
    # So the report does it instead. Deterministic, and the flight-
    # specific half of the checklist stops depending on the model
    # noticing a field.
    if data.get("figures"):
        cells = "".join(
            _stat(_label(key), _figure(key, value))
            for key, value in data["figures"].items()
        )
        sections += [
            "<h2>Figures for this aircraft and route</h2>",
            f'<div class="grid">{cells}</div>',
        ]

    if data["sources"]:
        blocks = "".join(
            f"<details><summary>{html.escape(source['id'])} &mdash; "
            f"{html.escape(source['title'])}</summary>"
            f"<p>{html.escape(source['text'])}</p></details>"
            for source in data["sources"]
        )
        sections += ["<h2>Sources</h2>", blocks]

    if data.get("briefing"):
        sections += [
            "<h2>Briefing</h2>",
            f'<div class="route" style="word-spacing:normal;line-height:1.6">'
            f"{html.escape(data['briefing'])}</div>",
        ]

    sections.append(
        f"<footer>Generated {html.escape(data['generated_at'])} &middot; "
        "every figure computed from real navaid, wind and airspace data; "
        "every checklist item cited to a procedure document.</footer>"
    )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{html.escape(title)} &middot; dispatch report</title>"
        f"<style>{_STYLE}</style></head><body>"
        + "".join(sections)
        + "</body></html>"
    )
