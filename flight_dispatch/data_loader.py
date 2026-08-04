"""Loading of OurAirports reference data.

WHERE THE DATA COMES FROM
-------------------------
OurAirports (ourairports.com) is a free, community-maintained database of
the world's airports and navigation aids, published as plain CSV. This
project uses two of its files:

  airports.csv  ~12 MB, ~80,000 rows -- every airport, heliport and
                airstrip on Earth, with coordinates.
  navaids.csv   ~1.5 MB, ~11,000 rows -- ground-based radio navigation
                beacons (VORs, NDBs and friends).

The CSVs are NOT committed to git -- they are large, they change, and
they are trivially re-downloadable. Run `scripts/download_data.py` to
fetch them. See `data/README.md` and `.gitignore`.

A NOTE ON DEFENSIVENESS
-----------------------
This is real-world public data, so it is messy: blank coordinates, blank
ICAO codes, airports that closed decades ago. Every loader here silently
skips rows it cannot use rather than crashing. Being strict would mean
one bad row out of 80,000 takes down the whole program.
"""

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .geo import bounding_box
from .models import Airport, Navaid

# __file__ is this file; .parents[1] walks up from flight_dispatch/ to the
# project root. Building paths this way means the code works no matter
# which directory you run Python from.
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

AIRPORTS_CSV = DATA_DIR / "airports.csv"
NAVAIDS_CSV = DATA_DIR / "navaids.csv"

# Navaid types worth routing over.
#
# A VOR is a radio beacon that tells an aircraft its bearing from the
# station -- the classic backbone of pre-GPS navigation, and still what
# published airway waypoints are built from. VORTAC and VOR-DME are VORs
# with extra equipment bolted on. NDBs are older and less precise, but GA
# aircraft still use them, so they are included.
#
# Plain "DME" is excluded on purpose: a DME only tells you your DISTANCE
# from the station, not your direction, so it cannot define a position on
# its own and is not a valid standalone waypoint.
ROUTABLE_NAVAID_TYPES = frozenset({"VOR", "VOR-DME", "VORTAC", "TACAN", "NDB", "NDB-DME"})


class MissingDataError(FileNotFoundError):
    """Raised when a required OurAirports CSV has not been downloaded.

    Subclasses FileNotFoundError so that code which only cares that a file
    is missing can catch the built-in, while code that wants to give the
    "run the download script" hint can catch this specific type.
    """


def load_airports(path: Path = AIRPORTS_CSV) -> Dict[str, Airport]:
    """Load airports into a dict keyed by ICAO code.

    A dict, not a list, because the CLI's core operation is "look up KORD"
    -- an O(1) hash lookup instead of scanning 80,000 rows.

    ICAO codes are the four-letter identifiers pilots use (KORD, EGLL).
    OurAirports stores them in two different columns, annoyingly:

      icao_code  the official code, but blank for many small fields
      ident      OurAirports' own primary key, which IS the ICAO code
                 wherever one exists, and a made-up local code otherwise

    So: prefer icao_code, fall back to ident.

    Closed airports are skipped. They remain in the dataset for historical
    reasons, but you cannot file a flight plan to a runway that is now a
    parking lot.
    """
    airports: Dict[str, Airport] = {}

    for row in _read_rows(path):
        if row.get("type", "").strip() == "closed":
            continue

        # `or` chains through falsy values, so a blank icao_code falls
        # through to ident, and if both are blank we get "".
        icao = (row.get("icao_code") or row.get("ident") or "").strip().upper()

        lat = _safe_float(row.get("latitude_deg"))
        lon = _safe_float(row.get("longitude_deg"))

        # No identifier or no position means the row is useless to us.
        if not icao or lat is None or lon is None:
            continue

        airports[icao] = Airport(
            icao=icao,
            name=(row.get("name") or "").strip(),
            lat=lat,
            lon=lon,
            elevation_ft=_safe_float(row.get("elevation_ft")),
        )

    return airports


def load_navaids(
    path: Path = NAVAIDS_CSV,
    types: Optional[Iterable[str]] = ROUTABLE_NAVAID_TYPES,
) -> List[Navaid]:
    """Load navaids, by default restricted to routable types.

    A list, not a dict, because navaid identifiers are NOT globally unique
    (several countries have an "OB") and because the access pattern is
    "scan them all looking for ones near my course", not "look up one by
    name".

    Args:
        types: Which navaid types to keep. Pass None to keep every row.
    """
    # Normalise to uppercase once, up front, rather than per row.
    allowed = frozenset(t.upper() for t in types) if types is not None else None

    navaids: List[Navaid] = []

    for row in _read_rows(path):
        navaid_type = (row.get("type") or "").strip().upper()
        if allowed is not None and navaid_type not in allowed:
            continue

        lat = _safe_float(row.get("latitude_deg"))
        lon = _safe_float(row.get("longitude_deg"))
        ident = (row.get("ident") or "").strip().upper()

        if not ident or lat is None or lon is None:
            continue

        navaids.append(
            Navaid(
                ident=ident,
                name=(row.get("name") or "").strip(),
                navaid_type=navaid_type,
                lat=lat,
                lon=lon,
            )
        )

    return navaids


def navaids_in_bounds(
    navaids: Iterable[Navaid],
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
    margin_nm: float = 50.0,
) -> List[Navaid]:
    """Narrow a navaid list to a padded rectangle around two points.

    A cheap prefilter. The dataset is global, but a Chicago-to-Minneapolis
    flight can only care about navaids in the upper Midwest. Four float
    comparisons per navaid discard the other ~99%, before the expensive
    trigonometry in `naive_route` ever runs.

    The margin is generous on purpose: a filter like this must never throw
    away something a later stage would have wanted. Over-including costs a
    few microseconds; under-including silently corrupts the route.

    This matters more in CP2 than here -- mesh graph construction is
    roughly O(n^2) in the node count, so cutting n early is the difference
    between a fast search and an unusable one.
    """
    min_lat, min_lon, max_lat, max_lon = bounding_box(lat1, lon1, lat2, lon2, margin_nm)

    return [
        navaid
        for navaid in navaids
        if min_lat <= navaid.lat <= max_lat and min_lon <= navaid.lon <= max_lon
    ]


def _read_rows(path: Path) -> Iterable[dict]:
    """Yield CSV rows as dicts, with a useful error if the file is missing.

    `csv.DictReader` reads the header line and turns every subsequent row
    into a dict keyed by column name, so the loaders can say
    row["latitude_deg"] instead of tracking column numbers.

    This is a GENERATOR (it uses `yield from`), meaning rows are produced
    one at a time and the file is never fully loaded into memory. On a
    12 MB CSV that is a nice-to-have; the habit pays off on bigger data.

    The explicit `.exists()` check exists purely for the error message: a
    bare FileNotFoundError tells a new user nothing, whereas naming the
    download script tells them exactly what to do next.
    """
    if not path.exists():
        raise MissingDataError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )

    # newline="" is what the csv module documentation requires -- it lets
    # csv handle line endings itself, so quoted fields containing newlines
    # (airport names do sometimes) parse correctly.
    with open(path, newline="", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


def _safe_float(value: Optional[str]) -> Optional[float]:
    """Parse a CSV field as a float, returning None for blanks and junk.

    CSV has no types -- everything arrives as a string. Real data contains
    empty cells and the occasional malformed value, and float("") raises.
    Returning None lets callers decide whether a missing value is fatal
    (no coordinates: skip the row) or tolerable (no elevation: fine).
    """
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None
