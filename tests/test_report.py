"""Tests for the dispatch report.

The report is where the citation rule stops being a request. The system
prompt asks the model to cite every checklist item; asked to plan a
flight and give a checklist in one request, it once ignored that and
wrote eight items from memory. These tests pin the behaviour that does
not depend on the model complying.

No network, no model, no embedding service: reports are built from a
RoutePlan and a string.
"""

import json
import tempfile
import unittest
from pathlib import Path

from flight_dispatch.aircraft import get_aircraft
from flight_dispatch.data_loader import (
    load_airports,
    load_navaids,
    navaids_near_route,
)
from flight_dispatch.report import (
    ChecklistItem,
    DispatchReport,
    build_report,
    parse_checklist,
)
from flight_dispatch.retrieval import Chunk, ProcedureIndex
from flight_dispatch.route import plan_route


def _index() -> ProcedureIndex:
    """A tiny index whose ids are known, so citation checks are exact."""
    chunks = [
        Chunk(id="fuel-reserves", title="Fuel reserves", category="planning", text="f"),
        Chunk(id="icing", title="Airframe icing", category="weather", text="i"),
    ]
    return ProcedureIndex(chunks, [[1.0], [1.0]])


class TestChecklistParsing(unittest.TestCase):
    """THE ENFORCEMENT.

    An item citing nothing does not enter the report, and neither does
    one citing a document that does not exist -- a citation to nothing is
    worse than no citation, because it looks like provenance.
    """

    def parse(self, text):
        return parse_checklist(text, index=_index())

    def test_a_cited_item_is_kept(self):
        items, rejected = self.parse("1. Carry 45 minutes of reserve [fuel-reserves].")
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0].citations, ["fuel-reserves"])
        self.assertEqual(rejected, [])

    def test_an_uncited_item_is_rejected(self):
        # The exact failure: "File a flight plan with air traffic
        # control" appears in no document in the corpus.
        items, rejected = self.parse("1. File a flight plan with air traffic control.")
        self.assertEqual(items, [])
        self.assertEqual(len(rejected), 1)

    def test_a_citation_to_nothing_is_rejected(self):
        items, rejected = self.parse("1. Check the flux capacitor [not-a-real-doc].")
        self.assertEqual(items, [])
        self.assertEqual(len(rejected), 1)

    def test_rejected_items_are_returned_not_discarded(self):
        # Dropping them silently would hide the failure: the report would
        # look fully grounded while the model had been inventing.
        _, rejected = self.parse("- Something invented entirely.")
        self.assertIn("Something invented entirely.", rejected)

    def test_prose_is_neither_kept_nor_rejected(self):
        items, rejected = self.parse(
            "Based on the retrieved procedures, here is the checklist:\n"
            "1. Carry 45 minutes of reserve [fuel-reserves].\n"
            "Note: NOTAMs are not covered by these procedures."
        )
        self.assertEqual(len(items), 1)
        self.assertEqual(rejected, [])

    def test_mixed_grounding_separates_correctly(self):
        # The dangerous case: mostly cited, one invented item hidden
        # among them.
        items, rejected = self.parse(
            "1. Carry 45 minutes of reserve [fuel-reserves].\n"
            "2. Set the transponder to 1200.\n"
            "3. Leave icing conditions promptly [icing]."
        )
        self.assertEqual(len(items), 2)
        self.assertEqual(len(rejected), 1)

    def test_bullets_and_numbers_both_count(self):
        items, _ = self.parse(
            "- Carry reserve fuel [fuel-reserves]\n* Avoid ice [icing]"
        )
        self.assertEqual(len(items), 2)

    def test_citation_markers_are_stripped_from_the_text(self):
        items, _ = self.parse("1. Carry 45 minutes of reserve [fuel-reserves].")
        self.assertNotIn("[", items[0].text)
        self.assertIn("reserve", items[0].text)

    def test_without_an_index_any_citation_is_accepted(self):
        # Used when the corpus is not to hand; the shape is still checked
        # even though the id cannot be resolved.
        items, _ = parse_checklist("1. Do the thing [some-doc].")
        self.assertEqual(len(items), 1)

    def test_an_empty_checklist_is_empty_not_an_error(self):
        self.assertEqual(self.parse(""), ([], []))


class TestReportContent(unittest.TestCase):
    """Slow -- builds a real mesh once."""

    @classmethod
    def setUpClass(cls):
        airports = load_airports()
        navaids = load_navaids()
        origin, dest = airports["KPWK"], airports["KMSP"]
        near = navaids_near_route(
            navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        cls.plan = plan_route(
            origin, dest, near, aircraft=get_aircraft("sr22"), use_grid=True
        )

    def report(self, checklist=""):
        return build_report(self.plan, checklist_text=checklist, index=_index())

    def test_every_number_comes_from_the_plan(self):
        data = self.report().to_dict()
        self.assertAlmostEqual(
            data["route_distance_nm"], self.plan.total_distance_nm, places=1
        )
        self.assertAlmostEqual(
            data["direct_distance_nm"], self.plan.direct_distance_nm, places=1
        )

    def test_the_route_string_matches_the_waypoints(self):
        data = self.report().to_dict()
        self.assertEqual(
            data["route"], " ".join(w.ident for w in self.plan.waypoints)
        )
        self.assertEqual(len(data["waypoints"]), len(self.plan.waypoints))

    def test_the_phase_profile_is_carried_through(self):
        profile = self.report().to_dict()["profile"]
        self.assertEqual(
            profile["cruise_altitude_ft"],
            round(self.plan.phases.cruise_altitude_ft),
        )

    def test_the_rejected_field_is_always_present(self):
        # Present even when empty, so its absence never means "nothing
        # was rejected" by accident.
        self.assertIn("rejected_uncited_items", self.report().to_dict())

    def test_json_round_trips(self):
        data = json.loads(self.report().to_json())
        self.assertEqual(data["origin"]["icao"], "KPWK")
        self.assertEqual(data["destination"]["icao"], "KMSP")

    def test_html_is_a_complete_document(self):
        page = self.report("1. Carry reserve [fuel-reserves].").to_html()
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertTrue(page.rstrip().endswith("</html>"))

    def test_html_shows_the_checklist_and_its_citations(self):
        page = self.report("1. Carry 45 minutes of reserve [fuel-reserves].").to_html()
        self.assertIn("reserve", page)
        self.assertIn("fuel-reserves", page)

    def test_html_shows_rejected_items_rather_than_hiding_them(self):
        page = self.report("1. File a flight plan with ATC.").to_html()
        self.assertIn("no source", page.lower())
        self.assertIn("File a flight plan", page)

    def test_html_has_no_rejected_section_when_nothing_was_rejected(self):
        page = self.report("1. Carry reserve [fuel-reserves].").to_html()
        self.assertNotIn("no source", page.lower())

    def test_html_escapes_content(self):
        page = self.report("1. Do <script>alert(1)</script> [fuel-reserves].").to_html()
        self.assertNotIn("<script>alert", page)

    def test_writing_produces_both_files(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = self.report().write(str(Path(directory) / "report"))
            self.assertTrue(Path(paths["html"]).is_file())
            self.assertTrue(Path(paths["json"]).is_file())

    def test_a_missing_map_costs_the_map_not_the_report(self):
        # folium is optional and tile rendering can fail for reasons
        # outside this project.
        from unittest.mock import patch

        report = self.report("1. Carry reserve [fuel-reserves].")
        with patch.object(report, "_map_html", return_value=""):
            page = report.to_html()
        self.assertIn("reserve", page)
        self.assertNotIn("<h2>Map</h2>", page)


class TestReportWithoutAnAircraft(unittest.TestCase):
    """A distance-only plan still produces a report -- CP2 had no
    aircraft, no time and no fuel, and the report predates none of it."""

    @classmethod
    def setUpClass(cls):
        airports = load_airports()
        navaids = load_navaids()
        origin, dest = airports["KPWK"], airports["KMSP"]
        near = navaids_near_route(
            navaids, origin.lat, origin.lon, dest.lat, dest.lon, margin_nm=100
        )
        cls.plan = plan_route(origin, dest, near)

    def test_it_renders(self):
        page = build_report(self.plan).to_html()
        self.assertIn("KPWK", page)

    def test_time_and_fuel_are_simply_absent(self):
        data = build_report(self.plan).to_dict()
        self.assertNotIn("ete", data)
        self.assertNotIn("aircraft", data)


class TestChecklistItem(unittest.TestCase):
    def test_grounded_requires_a_citation(self):
        self.assertTrue(ChecklistItem("x", ["fuel-reserves"]).is_grounded)
        self.assertFalse(ChecklistItem("x", []).is_grounded)


class TestGeneratedAt(unittest.TestCase):
    def test_a_timestamp_is_added(self):
        report = DispatchReport(plan=None)
        self.assertIn("UTC", report.generated_at)

    def test_an_explicit_timestamp_is_kept(self):
        report = DispatchReport(plan=None, generated_at="2020-01-01 00:00 UTC")
        self.assertEqual(report.generated_at, "2020-01-01 00:00 UTC")


if __name__ == "__main__":
    unittest.main()


class TestFigureUnitsAndLabels(unittest.TestCase):
    """THE BUG THIS FIXES.

    Units were guessed from substrings, and "reserve" matched gallons
    before "minutes" matched minutes -- so `reserve_minutes: 45` rendered
    as "45 gal", a 45-gallon reserve in an aircraft carrying 2,835.
    Stripping the suffix for the heading then collapsed `reserve_gal` and
    `reserve_minutes` onto the same word, printed twice with different
    numbers and no way to tell which was which.
    """

    def figure(self, key, value):
        from flight_dispatch.report import _figure

        return _figure(key, value)

    def label(self, key):
        from flight_dispatch.report import _label

        return _label(key)

    def test_the_reserve_that_was_wrong(self):
        self.assertEqual(self.figure("reserve_minutes", 45), "45 min")
        self.assertEqual(self.figure("reserve_gal", 2835), "2,835 gal")

    def test_units_come_from_the_suffix(self):
        for key, value, expected in (
            ("usable_fuel_gal", 84535, "84,535 gal"),
            ("useful_load_lb", 657300, "657,300 lb"),
            ("cruise_altitude_ft", 39000, "39,000 ft"),
            ("direct_distance_nm", 7030.4, "7,030.4 nm"),
            ("endurance_hours", 20.4, "20.4 h"),
        ):
            with self.subTest(key):
                self.assertEqual(self.figure(key, value), expected)

    def test_a_name_with_no_unit_gets_none(self):
        self.assertEqual(self.figure("waypoint_count", 22), "22")

    def test_non_numeric_values_pass_through(self):
        self.assertEqual(self.figure("note_text", "hello"), "hello")

    def test_labels_are_unique_across_the_real_figures(self):
        # The collision that produced two headings reading "Reserve".
        from flight_dispatch.tools import _flight_figures
        from flight_dispatch.aircraft import get_aircraft
        from flight_dispatch.data_loader import load_airports

        airports = load_airports()
        figures = _flight_figures(
            get_aircraft("a388"), airports["KSFO"], airports["OMDB"]
        )
        labels = [self.label(key) for key in figures]
        self.assertEqual(len(labels), len(set(labels)), sorted(labels))

    def test_the_two_reserves_are_distinguishable(self):
        self.assertNotEqual(self.label("reserve_gal"), self.label("reserve_minutes"))

    def test_labels_read_as_headings(self):
        self.assertEqual(self.label("cruise_altitude_ft"), "Cruise altitude")
        self.assertEqual(self.label("origin_elevation_ft"), "Origin elevation")
