"""FAA special-use airspace, and the geometry of avoiding it.

WHAT THIS IS
------------
Some airspace you may not enter, or may not enter at certain times. The
FAA publishes it as GeoJSON polygons with altitude bands:

  P   Prohibited     never, at any time (P-56 over the White House)
  R   Restricted     only with permission from the controlling agency;
                     live fire, artillery, missile testing
  W   Warning        over international waters, same hazards as restricted
  A   Alert          high volume of unusual activity, entry legal
  MOA Military Ops   military training; entry legal for VFR but unwise
  D   Danger         international equivalent of a warning area

Only the first two are hard no-go for a civil flight. The rest are
advisory, which is why `BLOCKED_TYPES` is smaller than the dataset.

HOW AVOIDANCE WORKS
-------------------
There is no special avoidance algorithm. A* already minimises whatever
the cost function returns, so an edge crossing prohibited airspace simply
returns `math.inf` and the search routes around it -- for the same reason
it routes around anything expensive. Blocking every route makes A* report
no path, which is correct: sometimes there genuinely isn't one.

That is the payoff of the cost_function hook: airspace avoidance is a
cost function, not new search code.

ALTITUDE MATTERS
----------------
A restricted area topping out at 8,000 ft does not affect an airliner at
FL350, and one starting at FL180 does not affect a Cessna at 8,000. Each
polygon carries a lower and upper bound, so only airspace that actually
overlaps the planned cruise altitude is considered. Ignoring this would
block far more routes than reality does.

WHAT IS NOT MODELLED
--------------------
- TFRs (Temporary Flight Restrictions) -- these appear at a day's notice
  for VIP movement, wildfires and stadiums, and are published separately.
  A real dispatcher checks them; this does not.
- Times of use. Many restricted areas are only hot during published
  hours, and the TIMESOFUSE field is free text ("0800-2200 DAILY",
  "BY NOTAM"). Everything is treated as always active, which is
  conservative -- it may route around airspace that is currently cold.
- Class B/C/D controlled airspace, which requires a clearance rather than
  being prohibited.
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .geo import great_circle_point

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
AIRSPACE_GEOJSON = DATA_DIR / "special_use_airspace.geojson"

# Types a civil flight must not enter without specific authorisation.
# Alert areas and MOAs are legal to transit, so they are advisory here
# rather than blocking.
BLOCKED_TYPES: Set[str] = {"P", "R", "W"}

ADVISORY_TYPES: Set[str] = {"MOA", "A", "D"}

TYPE_NAMES = {
    "P": "Prohibited",
    "R": "Restricted",
    "W": "Warning",
    "A": "Alert",
    "D": "Danger",
    "MOA": "Military Operations Area",
}

# The FAA uses this sentinel for "no defined limit".
UNLIMITED_SENTINEL = -9998
UNLIMITED_FT = 999999.0


class AirspaceDataError(FileNotFoundError):
    """Raised when the airspace GeoJSON has not been downloaded."""


@dataclass(frozen=True)
class SpecialUseAirspace:
    """One special-use airspace volume.

    Attributes:
        name: Designator, e.g. "R-2508" or "P-56A".
        type_code: One of the codes in TYPE_NAMES.
        lower_ft: Floor in feet MSL. 0 means surface.
        upper_ft: Ceiling in feet MSL.
        state: US state, for display.
        times_of_use: Free text, e.g. "CONTINUOUS" or "BY NOTAM". Not
            parsed -- see the module docstring.
        geometry: The shapely polygon. Typed loosely to avoid importing
            shapely at module scope.
    """

    name: str
    type_code: str
    lower_ft: float
    upper_ft: float
    state: Optional[str]
    times_of_use: Optional[str]
    geometry: object

    @property
    def is_blocking(self) -> bool:
        return self.type_code in BLOCKED_TYPES

    @property
    def type_name(self) -> str:
        return TYPE_NAMES.get(self.type_code, self.type_code)

    def active_at(self, altitude_ft: float) -> bool:
        """Whether this volume spans a given altitude."""
        return self.lower_ft <= altitude_ft <= self.upper_ft

    def describe(self) -> str:
        ceiling = (
            "unlimited" if self.upper_ft >= UNLIMITED_FT else f"{self.upper_ft:,.0f} ft"
        )
        floor = "surface" if self.lower_ft <= 0 else f"{self.lower_ft:,.0f} ft"
        return f"{self.name} ({self.type_name}, {floor} to {ceiling})"


def _altitude_to_ft(value: Optional[str], uom: Optional[str], code: Optional[str]) -> float:
    """Normalise one of the FAA's altitude fields to feet MSL.

    The data mixes three conventions in string fields:
      UOM "FT"   plain feet
      UOM "FL"   flight level -- hundreds of feet, so FL180 is 18,000 ft
      UNLTD      no ceiling, flagged by a -9998 sentinel

    Flight levels are pressure altitudes, which only equal true altitude
    on a standard day. Treating them as feet MSL is the usual planning
    simplification and is what the altitude comparison here needs.
    """
    if code == "UNLTD" or value is None:
        return UNLIMITED_FT

    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNLIMITED_FT

    if number == UNLIMITED_SENTINEL:
        return UNLIMITED_FT

    # Flight levels, and the "STD" code used when UOM is absent, are both
    # expressed in hundreds of feet.
    if uom == "FL" or (uom is None and code == "STD"):
        return number * 100.0

    return number


def load_airspace(
    path: Path = AIRSPACE_GEOJSON,
    types: Optional[Iterable[str]] = None,
) -> List[SpecialUseAirspace]:
    """Load special-use airspace polygons from the FAA GeoJSON.

    Args:
        types: Type codes to keep. Defaults to the blocking types only,
            since advisory airspace does not affect routing.

    Raises:
        AirspaceDataError: if the file has not been downloaded.
    """
    from shapely.geometry import shape  # imported lazily; heavy dependency

    if not path.exists():
        raise AirspaceDataError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    allowed = BLOCKED_TYPES if types is None else set(types)

    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)

    volumes: List[SpecialUseAirspace] = []

    for feature in payload.get("features", []):
        properties = feature.get("properties") or {}
        type_code = (properties.get("TYPE_CODE") or "").strip().upper()
        if type_code not in allowed:
            continue

        geometry = feature.get("geometry")
        if not geometry:
            continue

        polygon = shape(geometry)
        # A handful of published polygons are self-intersecting. buffer(0)
        # is the standard shapely repair; without it, intersects() raises.
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty:
            continue

        volumes.append(
            SpecialUseAirspace(
                name=(properties.get("NAME") or "unnamed").strip(),
                type_code=type_code,
                lower_ft=_altitude_to_ft(
                    properties.get("LOWER_VAL"),
                    properties.get("LOWER_UOM"),
                    properties.get("LOWER_CODE"),
                ),
                upper_ft=_altitude_to_ft(
                    properties.get("UPPER_VAL"),
                    properties.get("UPPER_UOM"),
                    properties.get("UPPER_CODE"),
                ),
                state=properties.get("STATE"),
                times_of_use=properties.get("TIMESOFUSE"),
                geometry=polygon,
            )
        )

    return volumes


class AirspaceIndex:
    """Spatially indexed airspace, for fast edge-crossing tests.

    WHY AN INDEX
    ------------
    A transcontinental mesh has ~78,000 edges, and there are ~750 blocking
    volumes. Testing every edge against every polygon is 58 million
    intersection tests -- minutes of work.

    shapely's STRtree is an R-tree: it narrows "which polygons could
    possibly touch this line" to a handful using bounding boxes, and only
    those get the exact test. That turns the problem into something that
    runs in well under a second.
    """

    def __init__(
        self,
        volumes: Sequence[SpecialUseAirspace],
        altitude_ft: Optional[float] = None,
    ):
        """
        Args:
            volumes: The airspace to index.
            altitude_ft: If given, only volumes spanning this altitude are
                indexed. This is the filter that stops a surface-to-8,000ft
                restricted area from blocking an airliner at FL350.
        """
        from shapely.strtree import STRtree

        self.altitude_ft = altitude_ft
        self.volumes: List[SpecialUseAirspace] = [
            volume
            for volume in volumes
            if altitude_ft is None or volume.active_at(altitude_ft)
        ]
        self._tree = (
            STRtree([volume.geometry for volume in self.volumes])
            if self.volumes
            else None
        )

    def __len__(self) -> int:
        return len(self.volumes)

    def crossings(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> List[SpecialUseAirspace]:
        """Which indexed volumes a leg passes through.

        The leg is approximated as a sequence of straight segments in
        lat/lon space rather than a true great circle. Over a 150 nm edge
        the difference is under a nautical mile, far below the precision
        of the airspace boundaries themselves.
        """
        if self._tree is None:
            return []

        from shapely.geometry import LineString

        # Sample the great circle so long edges do not cut corners across
        # airspace that a straight lat/lon line would miss.
        points = [
            great_circle_point(i / 8, lat1, lon1, lat2, lon2) for i in range(9)
        ]
        line = LineString([(lon, lat) for lat, lon in points])

        hits = []
        for index in self._tree.query(line):
            volume = self.volumes[int(index)]
            if line.intersects(volume.geometry):
                hits.append(volume)
        return hits

    def blocks(self, lat1: float, lon1: float, lat2: float, lon2: float) -> bool:
        """Whether a leg is unflyable because of airspace."""
        return any(volume.is_blocking for volume in self.crossings(lat1, lon1, lat2, lon2))

    def containing(self, lat: float, lon: float) -> List[SpecialUseAirspace]:
        """Volumes containing a single point."""
        if self._tree is None:
            return []

        from shapely.geometry import Point

        point = Point(lon, lat)
        return [
            self.volumes[int(i)]
            for i in self._tree.query(point)
            if self.volumes[int(i)].geometry.contains(point)
        ]


def make_airspace_cost(
    index: AirspaceIndex,
    base_cost_function=None,
    penalty_factor: float = math.inf,
):
    """Wrap a cost function so that airspace-crossing edges are blocked.

    COMPOSITION, NOT REPLACEMENT
    ----------------------------
    This takes the wind-based cost function and layers airspace on top,
    rather than reimplementing it. The route that comes out is therefore
    the fastest one that is also legal -- both constraints at once, which
    is the point of expressing them as costs over the same graph.

    Args:
        index: Airspace filtered to the cruise altitude.
        base_cost_function: The cost to wrap. Defaults to plain distance.
        penalty_factor: What a crossing edge costs. Infinity forbids it
            outright. A large finite number instead makes it a strong
            preference -- useful for advisory airspace like MOAs, which
            are legal to enter but better avoided.
    """

    def cost(graph, from_index: int, to_index: int, base_nm: float) -> float:
        a, b = graph.nodes[from_index], graph.nodes[to_index]

        if index.blocks(a.lat, a.lon, b.lat, b.lon):
            if math.isinf(penalty_factor):
                return math.inf
            base = (
                base_nm
                if base_cost_function is None
                else base_cost_function(graph, from_index, to_index, base_nm)
            )
            return base * penalty_factor

        if base_cost_function is None:
            return base_nm
        return base_cost_function(graph, from_index, to_index, base_nm)

    return cost


def airspace_near_route(
    volumes: Sequence[SpecialUseAirspace],
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    margin_deg: float = 3.0,
) -> List[SpecialUseAirspace]:
    """Volumes whose bounding box falls near a route's bounding box.

    A cheap prefilter before building the index, on the same principle as
    the navaid region filter: a Chicago-to-Minneapolis flight cannot be
    affected by a restricted area over Nevada.
    """
    min_lat = min(lat1, lat2) - margin_deg
    max_lat = max(lat1, lat2) + margin_deg
    min_lon = min(lon1, lon2) - margin_deg
    max_lon = max(lon1, lon2) + margin_deg

    nearby = []
    for volume in volumes:
        west, south, east, north = volume.geometry.bounds
        if east >= min_lon and west <= max_lon and north >= min_lat and south <= max_lat:
            nearby.append(volume)
    return nearby
