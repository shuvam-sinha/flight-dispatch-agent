import tempfile
import unittest
from pathlib import Path

from flight_dispatch.data_loader import (
    MissingDataError,
    load_airports,
    load_navaids,
    navaids_in_bounds,
)
from flight_dispatch.models import Navaid

AIRPORTS_CSV = """\
id,ident,type,name,latitude_deg,longitude_deg,elevation_ft,icao_code
1,KPWK,small_airport,Chicago Executive,42.1142,-87.9014,647,KPWK
2,KORD,large_airport,O'Hare Intl,41.9786,-87.9048,672,KORD
3,XX01,small_airport,No ICAO Code Field,40.0,-88.0,500,
4,KOLD,closed,Closed Field,41.0,-88.5,600,KOLD
5,BAD1,small_airport,Missing Coords,,,300,BAD1
6,BAD2,small_airport,Junk Coords,north,west,,BAD2
"""

NAVAIDS_CSV = """\
id,ident,name,type,latitude_deg,longitude_deg
1,OBK,Northbrook,VOR-DME,42.2225,-87.9997
2,DPA,Dupage,VORTAC,41.8942,-88.2586
3,LOWB,Low Beacon,NDB,41.5,-88.0
4,ORDI,O'Hare DME,DME,41.9,-87.9
5,NOPOS,No Position,VOR,,
"""


class LoaderTestCase(unittest.TestCase):
    """Writes the sample CSVs to a temp dir for the duration of a test."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        root = Path(self._dir.name)

        self.airports_path = root / "airports.csv"
        self.navaids_path = root / "navaids.csv"
        self.airports_path.write_text(AIRPORTS_CSV, encoding="utf-8")
        self.navaids_path.write_text(NAVAIDS_CSV, encoding="utf-8")


class TestLoadAirports(LoaderTestCase):
    def test_keys_on_icao_code(self):
        airports = load_airports(self.airports_path)
        self.assertIn("KPWK", airports)
        self.assertEqual(airports["KPWK"].name, "Chicago Executive")
        self.assertAlmostEqual(airports["KPWK"].lat, 42.1142)
        self.assertAlmostEqual(airports["KPWK"].elevation_ft, 647.0)

    def test_falls_back_to_ident_when_icao_code_blank(self):
        airports = load_airports(self.airports_path)
        self.assertIn("XX01", airports)

    def test_skips_closed_airports(self):
        self.assertNotIn("KOLD", load_airports(self.airports_path))

    def test_skips_rows_with_unusable_coordinates(self):
        airports = load_airports(self.airports_path)
        self.assertNotIn("BAD1", airports)
        self.assertNotIn("BAD2", airports)

    def test_missing_file_names_the_download_script(self):
        with self.assertRaises(MissingDataError) as ctx:
            load_airports(Path(self._dir.name) / "nope.csv")
        self.assertIn("download_data.py", str(ctx.exception))


class TestLoadNavaids(LoaderTestCase):
    def test_loads_routable_types_only_by_default(self):
        idents = {navaid.ident for navaid in load_navaids(self.navaids_path)}
        self.assertEqual(idents, {"OBK", "DPA", "LOWB"})

    def test_types_none_keeps_everything_positioned(self):
        idents = {n.ident for n in load_navaids(self.navaids_path, types=None)}
        self.assertIn("ORDI", idents)
        self.assertNotIn("NOPOS", idents)  # still dropped: no coordinates

    def test_explicit_type_filter(self):
        navaids = load_navaids(self.navaids_path, types={"NDB"})
        self.assertEqual([n.ident for n in navaids], ["LOWB"])

    def test_parses_fields(self):
        obk = next(n for n in load_navaids(self.navaids_path) if n.ident == "OBK")
        self.assertEqual(obk.name, "Northbrook")
        self.assertEqual(obk.navaid_type, "VOR-DME")
        self.assertAlmostEqual(obk.lon, -87.9997)


class TestNavaidsInBounds(unittest.TestCase):
    def setUp(self):
        self.near = Navaid("NEAR", "Near", "VOR", 42.0, -88.0)
        self.far = Navaid("FAR", "Far", "VOR", 20.0, -110.0)

    def test_keeps_only_navaids_inside_the_padded_box(self):
        result = navaids_in_bounds(
            [self.near, self.far], 42.1142, -87.9014, 41.9786, -87.9048, margin_nm=50.0
        )
        self.assertEqual([n.ident for n in result], ["NEAR"])

    def test_margin_pulls_in_more_navaids(self):
        outside = Navaid("EDGE", "Edge", "VOR", 42.9, -88.0)
        tight = navaids_in_bounds(
            [outside], 42.1142, -87.9014, 41.9786, -87.9048, margin_nm=5.0
        )
        loose = navaids_in_bounds(
            [outside], 42.1142, -87.9014, 41.9786, -87.9048, margin_nm=120.0
        )
        self.assertEqual(tight, [])
        self.assertEqual([n.ident for n in loose], ["EDGE"])


class TestRunwayArea(unittest.TestCase):
    """The size proxy behind airport ranking."""

    @classmethod
    def setUpClass(cls):
        from flight_dispatch.data_loader import load_runway_area

        cls.area = load_runway_area()

    def test_major_airports_are_present(self):
        for icao in ("KORD", "EGLL", "KJFK", "OMDB"):
            self.assertGreater(self.area.get(icao, 0), 0, icao)

    def test_a_hub_outranks_its_secondary_airport(self):
        # The pairs that longest-single-runway got wrong.
        self.assertGreater(self.area["OMDB"], self.area["OMDW"])  # Dubai
        self.assertGreater(self.area["RJTT"], self.area["RJAA"])  # Tokyo
        # And the ones it got right, which must not regress.
        self.assertGreater(self.area["KORD"], self.area["KMDW"])  # Chicago
        self.assertGreater(self.area["EGLL"], self.area["EGKK"])  # London
        self.assertGreater(self.area["KJFK"], self.area["KLGA"])  # New York

    def test_area_counts_every_runway_not_just_the_longest(self):
        # O'Hare has 8 open runways to Midway's 5, and the gap should be
        # far wider than any single-runway comparison would show.
        self.assertGreater(self.area["KORD"], 3 * self.area["KMDW"])

    def test_airports_with_no_runway_rows_are_absent(self):
        self.assertNotIn("ZZZZ", self.area)


if __name__ == "__main__":
    unittest.main()
