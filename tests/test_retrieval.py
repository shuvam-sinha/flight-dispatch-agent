"""Tests for the procedure corpus and retrieval.

The maths and the corpus are pure and always run. Nothing here needs
Ollama, a network or an embedding model: `ProcedureIndex` takes vectors
directly, so the search can be tested with numbers chosen to make the
expected answer obvious. Where the embeddings came from is a separate
concern from whether the ranking is right.

Tests that do need a live embedding model are marked and skipped when it
is not installed, in the same style as test_backend_apple.py.
"""

import json
import tempfile
import unittest
from pathlib import Path

from flight_dispatch.retrieval import (
    CORPUS_DIR,
    Chunk,
    ProcedureIndex,
    RetrievalUnavailable,
    cosine_similarity,
    load_corpus,
)


def _embedding_model_ready() -> bool:
    try:
        import requests

        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        installed = {
            model.get("name", "").split(":")[0]
            for model in response.json().get("models", [])
        }
        return "nomic-embed-text" in installed
    except Exception:  # noqa: BLE001
        return False


MODEL_READY = _embedding_model_ready()
requires_embeddings = unittest.skipUnless(
    MODEL_READY, "nomic-embed-text not installed (ollama pull nomic-embed-text)"
)


class TestCorpus(unittest.TestCase):
    """The corpus is plain markdown so it can be edited without code."""

    @classmethod
    def setUpClass(cls):
        cls.chunks = load_corpus()

    def test_the_corpus_is_small_on_purpose(self):
        # Fifteen documents is why there is no vector database. If this
        # ever grows past a few hundred, the retrieval design should be
        # revisited rather than the test loosened.
        self.assertGreaterEqual(len(self.chunks), 10)
        self.assertLess(len(self.chunks), 100)

    def test_every_chunk_has_a_title_and_body(self):
        for chunk in self.chunks:
            with self.subTest(chunk.id):
                self.assertTrue(chunk.title)
                self.assertGreater(len(chunk.text.split()), 40)

    def test_headers_are_stripped_from_the_body(self):
        # The heading and category line are metadata; embedding them
        # would put the word "category" in every vector.
        for chunk in self.chunks:
            with self.subTest(chunk.id):
                self.assertNotIn("category:", chunk.text)
                self.assertFalse(chunk.text.startswith("#"))

    def test_ids_are_unique_and_citable(self):
        ids = [chunk.id for chunk in self.chunks]
        self.assertEqual(len(ids), len(set(ids)))
        # A citation should point at a file a reader can open.
        for chunk_id in ids:
            self.assertTrue((CORPUS_DIR / f"{chunk_id}.md").is_file(), chunk_id)

    def test_categories_are_populated(self):
        categories = {chunk.category for chunk in self.chunks}
        self.assertGreater(len(categories), 2)
        self.assertNotIn("general", categories)  # the fallback, unused

    def test_digest_changes_with_the_text(self):
        # The cache key. If it did not change, editing a procedure would
        # leave the old vector in place and the edit would do nothing.
        a = Chunk(id="x", title="x", category="x", text="one")
        b = Chunk(id="x", title="x", category="x", text="two")
        self.assertNotEqual(a.digest, b.digest)

    def test_a_missing_corpus_says_so(self):
        with self.assertRaises(RetrievalUnavailable):
            load_corpus(Path("/nonexistent/procedures"))


class TestCosineSimilarity(unittest.TestCase):
    """The one piece of maths in the pipeline, written out rather than
    imported so it is visible."""

    def test_identical_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 2, 3], [1, 2, 3]), 1.0)

    def test_orthogonal_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)

    def test_opposite_vectors(self):
        self.assertAlmostEqual(cosine_similarity([1, 0], [-1, 0]), -1.0)

    def test_magnitude_does_not_matter(self):
        # THE REASON IT IS COSINE AND NOT DISTANCE. Two texts about the
        # same subject should match whether one is three sentences and
        # the other three paragraphs.
        self.assertAlmostEqual(
            cosine_similarity([1, 2, 3], [10, 20, 30]), 1.0
        )

    def test_a_zero_vector_matches_nothing(self):
        self.assertEqual(cosine_similarity([0, 0], [1, 1]), 0.0)

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            cosine_similarity([1, 2], [1, 2, 3])


def _index(**vectors_by_id) -> ProcedureIndex:
    """Build an index from hand-chosen vectors, so the expected ranking
    is obvious by inspection rather than by whatever a model produced."""
    chunks = [
        Chunk(id=name, title=name, category="test", text=f"body of {name}")
        for name in vectors_by_id
    ]
    return ProcedureIndex(chunks, list(vectors_by_id.values()))


class TestSearch(unittest.TestCase):
    def test_the_closest_chunk_ranks_first(self):
        index = _index(icing=[1.0, 0.0], fuel=[0.0, 1.0])
        best = index.search([0.9, 0.1], min_similarity=0.0)[0]
        self.assertEqual(best.chunk.id, "icing")

    def test_results_are_ordered_by_score(self):
        index = _index(a=[1.0, 0.0], b=[0.7, 0.7], c=[0.0, 1.0])
        scores = [m.score for m in index.search([1.0, 0.0], min_similarity=-1.0)]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_top_k_limits_the_result(self):
        index = _index(a=[1.0, 0.0], b=[0.9, 0.1], c=[0.8, 0.2], d=[0.7, 0.3])
        self.assertEqual(len(index.search([1.0, 0.0], top_k=2)), 2)

    def test_weak_matches_are_dropped_not_padded(self):
        # Every chunk returned is licence for the model to write about
        # that subject, so a barely-related document is an invitation to
        # put mountain flying in a checklist for a flight across Kansas.
        index = _index(relevant=[1.0, 0.0], unrelated=[0.0, 1.0])
        matches = index.search([1.0, 0.0], top_k=4, min_similarity=0.4)
        self.assertEqual([m.chunk.id for m in matches], ["relevant"])

    def test_nothing_relevant_returns_nothing(self):
        index = _index(a=[1.0, 0.0])
        self.assertEqual(index.search([0.0, 1.0], min_similarity=0.4), [])

    def test_by_id_finds_a_chunk(self):
        index = _index(icing=[1.0, 0.0])
        self.assertIsNotNone(index.by_id("icing"))
        self.assertIsNone(index.by_id("not-a-procedure"))

    def test_chunks_and_vectors_must_correspond(self):
        with self.assertRaises(ValueError):
            ProcedureIndex(
                [Chunk(id="a", title="a", category="t", text="a")], [[1.0], [2.0]]
            )


class TestEmbeddingCache(unittest.TestCase):
    """Embedding is the slow part; the cache is keyed by content hash so
    editing one document re-embeds that one and leaves the rest."""

    def test_a_corrupt_cache_is_ignored_rather_than_fatal(self):
        from flight_dispatch.retrieval import _read_cache

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
            handle.write("{not json")
            path = Path(handle.name)
        try:
            self.assertEqual(_read_cache(path), {})
        finally:
            path.unlink()

    def test_a_missing_cache_is_empty_not_an_error(self):
        from flight_dispatch.retrieval import _read_cache

        self.assertEqual(_read_cache(Path("/nonexistent/cache.json")), {})

    def test_round_trip(self):
        from flight_dispatch.retrieval import _read_cache, _write_cache

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.json"
            _write_cache(path, {"abc": [1.0, 2.0]})
            self.assertEqual(_read_cache(path), {"abc": [1.0, 2.0]})


class TestQueryConstruction(unittest.TestCase):
    """The query is built from facts, not from the user's phrasing.

    Otherwise retrieval would depend on how well the model described the
    flight, and a forgotten detail would silently drop a procedure.
    """

    def query(self, aircraft_key, origin=None, dest=None, phase="preflight"):
        from flight_dispatch.aircraft import get_aircraft
        from flight_dispatch.data_loader import load_airports
        from flight_dispatch.tools import _checklist_query

        airports = load_airports()
        return _checklist_query(
            get_aircraft(aircraft_key),
            airports.get(origin) if origin else None,
            airports.get(dest) if dest else None,
            phase,
        )

    def test_a_light_aircraft_and_a_jet_ask_different_questions(self):
        self.assertNotEqual(self.query("c172"), self.query("b789"))

    def test_a_jet_asks_about_altitude(self):
        self.assertIn("oxygen", self.query("b789"))
        self.assertNotIn("oxygen", self.query("c172"))

    def test_a_high_field_asks_about_density_altitude(self):
        # KDEN is at 5,400 ft.
        self.assertIn("density altitude", self.query("b738", "KDEN", "KMCI"))
        self.assertNotIn("density altitude", self.query("b738", "KJFK", "KBOS"))

    def test_a_long_route_asks_about_diversion(self):
        self.assertIn("diversion", self.query("b789", "KJFK", "EGLL"))
        self.assertNotIn("diversion", self.query("c172", "KPWK", "KMSP"))

    def test_the_phase_is_included(self):
        self.assertIn("emergency", self.query("c172", phase="emergency"))


class TestFindProceduresTool(unittest.TestCase):
    """Error paths, which need no embedding model."""

    def call(self, **kwargs):
        from flight_dispatch.tools import dispatch

        return dispatch("find_procedures", kwargs)

    def test_unknown_aircraft(self):
        self.assertIn("error", self.call(aircraft="tardis"))

    def test_unknown_phase_lists_the_valid_ones(self):
        result = self.call(aircraft="c172", phase="teleporting")
        self.assertIn("valid_phases", result)

    def test_the_tool_is_registered(self):
        from flight_dispatch.tools import TOOLS_BY_NAME

        self.assertIn("find_procedures", TOOLS_BY_NAME)

    def test_the_description_forbids_invented_items(self):
        # It is prompt text, and it is the only thing stopping the model
        # padding a checklist from memory.
        from flight_dispatch.tools import TOOLS_BY_NAME

        description = TOOLS_BY_NAME["find_procedures"].description.lower()
        self.assertIn("cite", description)
        self.assertIn("never add", description)


@requires_embeddings
class TestLiveRetrieval(unittest.TestCase):
    """Actually embeds. Needs `ollama pull nomic-embed-text`."""

    @classmethod
    def setUpClass(cls):
        cls.index = ProcedureIndex.build()

    def search(self, text, conditions=None):
        from flight_dispatch.retrieval import embed_texts

        return self.index.search(
            embed_texts([text])[0], conditions=conditions or []
        )

    def test_the_corpus_embeds(self):
        self.assertEqual(len(self.index), len(load_corpus()))
        self.assertGreater(len(self.index.vectors[0]), 100)

    def test_a_query_finds_the_obviously_right_document(self):
        matches = self.search(
            "flying into freezing rain and cloud, ice on the wing",
            conditions=["icing"],
        )
        self.assertIn("icing", [m.chunk.id for m in matches])

    def test_the_same_query_finds_nothing_when_the_condition_is_unmet(self):
        # Real embeddings, real corpus: the icing document is the most
        # similar thing there is, and it is still excluded because
        # nothing said the flight would meet icing.
        matches = self.search("flying into freezing rain and cloud, ice on the wing")
        self.assertNotIn("icing", [m.chunk.id for m in matches])

    def test_a_different_query_finds_a_different_document(self):
        matches = self.search("the engine has failed and I need to land")
        self.assertIn("engine-failure", [m.chunk.id for m in matches])

    def test_retrieval_discriminates(self):
        # The point of retrieval: two different flights get different
        # source material, without a line of routing logic.
        icing = {m.chunk.id for m in self.search("ice accretion in cloud")}
        fuel = {m.chunk.id for m in self.search("how much fuel must I carry in reserve")}
        self.assertNotEqual(icing, fuel)

    def test_every_citation_resolves(self):
        # The guarantee that makes citations worth anything.
        for match in self.search("preflight checks before a light aircraft flight"):
            self.assertIsNotNone(self.index.by_id(match.chunk.id))


if __name__ == "__main__":
    unittest.main()


class TestPreconditions(unittest.TestCase):
    """THE BUG THESE COVER.

    Asked for a Cessna 172 departure, plain vector search ranked
    `night-flight` FIRST at 0.619 -- above the preflight inspection. It
    deserved to: it is dense with light-aircraft VFR language and was
    genuinely the most similar document. It simply did not apply, because
    nothing said the flight was at night. The checklist told a pilot to
    carry a flashlight and expect the black-hole illusion, possibly at
    noon.

    So similarity is filtered by applicability first.
    """

    def index(self):
        return ProcedureIndex(
            [
                Chunk(id="always", title="a", category="t", text="a"),
                Chunk(id="at-night", title="b", category="t", text="b",
                      applies_when="night"),
            ],
            [[1.0, 0.0], [1.0, 0.0]],  # equally similar to anything
        )

    def test_a_conditional_chunk_is_excluded_when_unmet(self):
        found = self.index().search([1.0, 0.0], conditions=[])
        self.assertEqual([m.chunk.id for m in found], ["always"])

    def test_a_conditional_chunk_is_included_when_met(self):
        found = self.index().search([1.0, 0.0], conditions=["night"])
        self.assertEqual({m.chunk.id for m in found}, {"always", "at-night"})

    def test_unconditional_chunks_always_compete(self):
        found = self.index().search([1.0, 0.0], conditions=["something-else"])
        self.assertIn("always", [m.chunk.id for m in found])

    def test_filtering_happens_before_ranking(self):
        # Not merely reordered: an inapplicable chunk must not occupy a
        # top_k slot that a relevant one could have used.
        index = ProcedureIndex(
            [
                Chunk(id="night", title="n", category="t", text="n",
                      applies_when="night"),
                Chunk(id="relevant", title="r", category="t", text="r"),
            ],
            [[1.0, 0.0], [0.9, 0.1]],  # the conditional one scores higher
        )
        found = index.search([1.0, 0.0], top_k=1, conditions=[])
        self.assertEqual([m.chunk.id for m in found], ["relevant"])

    def test_the_corpus_declares_its_conditions(self):
        conditions = ProcedureIndex(load_corpus(), [[1.0]] * len(load_corpus())
                                    ).conditions_available()
        self.assertIn("night", conditions)
        self.assertIn("overwater", conditions)

    def test_unconditional_documents_exist(self):
        # If everything were conditional, a flight with no known
        # conditions would retrieve nothing at all.
        unconditional = [c for c in load_corpus() if not c.applies_when]
        self.assertGreaterEqual(len(unconditional), 5)


class TestConditionsFromFlightFacts(unittest.TestCase):
    """Conditions are derived from real data, never assumed."""

    def conditions(self, aircraft_key, origin=None, dest=None):
        from flight_dispatch.aircraft import get_aircraft
        from flight_dispatch.data_loader import load_airports
        from flight_dispatch.tools import _flight_conditions

        airports = load_airports()
        return _flight_conditions(
            get_aircraft(aircraft_key),
            airports.get(origin) if origin else None,
            airports.get(dest) if dest else None,
        )

    def test_a_jet_is_high_altitude(self):
        self.assertIn("high-altitude", self.conditions("b789"))
        self.assertNotIn("high-altitude", self.conditions("c172"))

    def test_a_high_field_is_high_elevation(self):
        self.assertIn("high-elevation", self.conditions("b738", "KDEN", "KMCI"))
        self.assertNotIn("high-elevation", self.conditions("b738", "KJFK", "KBOS"))

    def test_an_ocean_crossing_is_overwater(self):
        self.assertIn("overwater", self.conditions("b789", "KJFK", "EGLL"))

    def test_a_domestic_route_is_not_overwater(self):
        # The first version of this test asked merely whether ANY grid
        # point existed, and declared a flight across Wisconsin oceanic.
        self.assertNotIn("overwater", self.conditions("sr22", "KPWK", "KMSP"))
        self.assertNotIn("overwater", self.conditions("b738", "KJFK", "KLAX"))

    def test_night_is_never_assumed(self):
        # Nothing in the system records time of day, so the condition is
        # never satisfied and night procedures stay out unless a caller
        # supplies it. A checklist item that does not apply is noise a
        # pilot has to filter, and filtering is what a checklist avoids.
        for args in (("c172",), ("b789", "KJFK", "EGLL"), ("b738", "KDEN", "KMCI")):
            self.assertNotIn("night", self.conditions(*args))


class TestCitationVerification(unittest.TestCase):
    """THE FAILURE THIS CATCHES.

    Asked to plan a flight from KSFO to KEWR on a 777 AND give a
    checklist, the model planned the route and never called
    find_procedures. It then wrote eight items from memory -- "file a
    flight plan with air traffic control", "obtain clearance for takeoff"
    -- none of it in the corpus and none of it cited. Asked again for an
    A320 it produced the IDENTICAL eight items, which is the tell:
    nothing derived them from anything.

    It had been instructed not to, in the system prompt, twice. An
    instruction is a request; this is a test.
    """

    def index(self):
        return ProcedureIndex(
            [
                Chunk(id="fuel-reserves", title="f", category="planning", text="f"),
                Chunk(id="preflight-inspection", title="p", category="preflight", text="p"),
            ],
            [[1.0], [1.0]],
        )

    def check(self, text):
        from flight_dispatch.retrieval import verify_citations

        return verify_citations(text, self.index())

    def test_a_grounded_checklist_is_clean(self):
        result = self.check(
            "1. Drain every fuel sump [preflight-inspection].\n"
            "2. Carry 45 minutes of reserve [fuel-reserves]."
        )
        self.assertTrue(result.is_clean)
        self.assertEqual(len(result.supported), 2)

    def test_the_777_answer_is_caught(self):
        result = self.check(
            "1. Perform a pre-flight inspection of the aircraft.\n"
            "2. File a flight plan with air traffic control.\n"
            "3. Obtain clearance for takeoff and departure."
        )
        self.assertFalse(result.is_clean)
        self.assertEqual(len(result.unsupported), 3)

    def test_a_citation_to_nothing_is_caught(self):
        # Worse than no citation: it looks like provenance and is not.
        result = self.check("1. Check the widget [not-a-procedure].")
        self.assertEqual(result.unknown, ["not-a-procedure"])
        self.assertFalse(result.is_clean)

    def test_prose_around_the_list_needs_no_citation(self):
        # A preamble or a caveat is not an assertion about procedure.
        result = self.check(
            "Based on the retrieved procedures, here is the checklist:\n"
            "1. Drain every fuel sump [preflight-inspection].\n"
            "Note: NOTAMs are not covered by these procedures."
        )
        self.assertTrue(result.is_clean)

    def test_bullets_count_as_items(self):
        result = self.check("- Check the oil.\n* Check the fuel.")
        self.assertEqual(len(result.unsupported), 2)

    def test_mixed_grounding_is_not_clean(self):
        # The dangerous case: mostly cited, with one invented item
        # hidden among them.
        result = self.check(
            "1. Drain every fuel sump [preflight-inspection].\n"
            "2. Set the transponder to 1200 before departure.\n"
            "3. Carry 45 minutes of reserve [fuel-reserves]."
        )
        self.assertFalse(result.is_clean)
        self.assertEqual(len(result.unsupported), 1)
        self.assertEqual(len(result.supported), 2)

    def test_several_citations_on_one_line(self):
        result = self.check(
            "1. Fuel and weight both matter [fuel-reserves] [preflight-inspection]."
        )
        self.assertEqual(len(result.supported), 2)

    def test_an_empty_answer_is_vacuously_clean(self):
        self.assertTrue(self.check("").is_clean)


class TestChecklistReminder(unittest.TestCase):
    """The reminder that arrives where the failure happened.

    The system prompt already forbade an uncited checklist and was
    ignored -- by the time the plan came back, the checklist had become
    an afterthought. A note in the result arrives at that moment instead
    of thousands of tokens earlier.
    """

    def plan(self):
        from flight_dispatch.tools import dispatch

        return dispatch(
            "plan_flight",
            {"origin": "KPWK", "dest": "KMSP", "aircraft": "sr22", "use_wind": False},
        )

    def test_a_plan_says_it_contains_no_checklist(self):
        note = self.plan()["checklist_note"]
        self.assertIn("find_procedures", note)
        self.assertIn("no checklist", note)

    def test_the_note_forbids_writing_one_from_memory(self):
        self.assertIn("from memory", self.plan()["checklist_note"])


class TestFlightFigures(unittest.TestCase):
    """Retrieval selects text; it does not write it, so `fuel-reserves`
    reads identically for a Cessna hop and a 777 to Newark. The document
    states the RULE; these numbers say where THIS flight sits against it.

    Both halves are grounded -- the rule in a procedure document, the
    figures in the aircraft profile and the airport records -- so
    anchoring one to the other invents nothing.
    """

    def figures(self, aircraft_key, origin=None, dest=None):
        from flight_dispatch.aircraft import get_aircraft
        from flight_dispatch.data_loader import load_airports
        from flight_dispatch.tools import _flight_figures

        airports = load_airports()
        return _flight_figures(
            get_aircraft(aircraft_key),
            airports.get(origin) if origin else None,
            airports.get(dest) if dest else None,
        )

    def test_figures_differ_by_aircraft(self):
        self.assertNotEqual(self.figures("c172"), self.figures("b77w"))

    def test_fuel_figures_match_the_profile(self):
        from flight_dispatch.aircraft import get_aircraft

        profile = get_aircraft("b77w")
        figures = self.figures("b77w")
        self.assertEqual(figures["usable_fuel_gal"], round(profile.usable_fuel_gal))
        self.assertEqual(figures["reserve_gal"], round(profile.reserve_gal))

    def test_route_figures_appear_only_with_both_airports(self):
        self.assertNotIn("direct_distance_nm", self.figures("c172"))
        self.assertIn("direct_distance_nm", self.figures("c172", "KPWK", "KMSP"))

    def test_distance_matches_the_geometry(self):
        from flight_dispatch.data_loader import load_airports
        from flight_dispatch.geo import haversine_nm

        airports = load_airports()
        origin, dest = airports["KSFO"], airports["KEWR"]
        self.assertAlmostEqual(
            self.figures("b77w", "KSFO", "KEWR")["direct_distance_nm"],
            haversine_nm(origin.lat, origin.lon, dest.lat, dest.lon),
            places=1,
        )

    def test_elevation_comes_from_the_airport_record(self):
        self.assertGreater(
            self.figures("b738", "KDEN", "KMCI")["origin_elevation_ft"], 5000
        )

    def test_no_route_planning_happens(self):
        # These must stay cheap: building a mesh and running A* to write
        # a checklist would cost seconds and duplicate plan_flight.
        import time

        start = time.time()
        self.figures("b77w", "KSFO", "KEWR")
        self.assertLess(time.time() - start, 0.5)

    def test_the_tool_returns_them(self):
        from flight_dispatch.tools import dispatch

        result = dispatch(
            "find_procedures",
            {"aircraft": "b77w", "origin": "KSFO", "dest": "KEWR"},
        )
        self.assertIn("figures", result)
        self.assertIn("reserve_gal", result["figures"])

    def test_the_note_forbids_inventing_other_numbers(self):
        from flight_dispatch.tools import dispatch

        result = dispatch("find_procedures", {"aircraft": "c172"})
        self.assertIn("ONLY the numbers", result["note"])
