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

    def search(self, text):
        from flight_dispatch.retrieval import embed_texts

        return self.index.search(embed_texts([text])[0])

    def test_the_corpus_embeds(self):
        self.assertEqual(len(self.index), len(load_corpus()))
        self.assertGreater(len(self.index.vectors[0]), 100)

    def test_a_query_finds_the_obviously_right_document(self):
        matches = self.search("flying into freezing rain and cloud, ice on the wing")
        self.assertIn("icing", [m.chunk.id for m in matches])

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
