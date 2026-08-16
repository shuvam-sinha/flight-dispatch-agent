"""Retrieval over a small corpus of aviation procedures.

WHY THIS EXISTS
---------------
A preflight checklist has three possible sources and two of them are bad.

Hardcode it, and it is a dictionary that cannot adapt to the flight.
Ask the model to write one, and it produces something confident and
plausible with invented fuel figures, speeds and procedures -- the exact
failure this project is built against.

So: retrieve first, then write only from what was retrieved, and cite
every item. It is the same rule the routing tools follow, applied to text
instead of numbers. `plan_flight` guarantees every number came from a
computation; retrieval guarantees every procedure came from a document.

NO VECTOR DATABASE, DELIBERATELY
--------------------------------
The corpus is fifteen short documents. Cosine similarity over a list of
vectors is exact, takes microseconds, and has no operational surface. A
vector database here would be a dependency, a running service and a
migration path, all to search fewer items than a phone book page. Size
the tool to the problem.

The interface below is the one a vector store would expose anyway --
embed, search, top-k -- so swapping in FAISS or pgvector at a hundred
thousand documents would be a contained change.

EMBEDDINGS COME FROM OLLAMA
---------------------------
`nomic-embed-text` runs locally through the Ollama server that the
`backend_ollama` model already uses: free, private, no API key. The
alternative, sentence-transformers, would pull in PyTorch -- roughly 2 GB
-- to embed fifteen paragraphs.

Vectors are cached to disk keyed by a hash of the text, so the corpus is
embedded once and re-embedded only when a document actually changes.
"""

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_DIR = DATA_DIR / "procedures"
CACHE_FILE = DATA_DIR / "procedure_embeddings.json"

OLLAMA_HOST = "http://localhost:11434"
EMBEDDING_MODEL = "nomic-embed-text"
EMBEDDING_TIMEOUT_S = 60

# Chunks returned per query. Small on purpose: the checklist is written
# from these, and burying three relevant procedures among seven
# irrelevant ones invites the model to pad the list with material that
# does not apply to the flight.
DEFAULT_TOP_K = 4

# Below this cosine similarity a chunk is not really about the query.
# Returning weak matches is worse than returning fewer: every retrieved
# chunk is licence for the model to write about it.
MIN_SIMILARITY = 0.4


class RetrievalUnavailable(RuntimeError):
    """Raised when the corpus cannot be embedded, with the fix."""


@dataclass(frozen=True)
class Chunk:
    """One retrievable document.

    Attributes:
        id: Stable identifier, used for citation. Derived from the
            filename, so a citation points at a file a reader can open.
        title: Human-readable heading.
        category: Loose grouping -- preflight, weather, emergency. Not
            used for ranking; useful for filtering and for display.
        applies_when: A condition this document requires, or "" when it
            applies to every flight. See `ProcedureIndex.search`.
        text: The body, which is what gets embedded.
    """

    id: str
    title: str
    category: str
    text: str
    applies_when: str = ""

    @property
    def digest(self) -> str:
        """Hash of the text, so the cache invalidates when it changes."""
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:16]


def load_corpus(directory: Path = CORPUS_DIR) -> List[Chunk]:
    """Read every procedure document from disk.

    The format is deliberately plain -- a markdown heading, a category
    line, then prose -- so the corpus can be edited by anyone without
    touching code. Adding a document changes the checklist with no code
    change and no retraining, which is the whole point of retrieval.
    """
    if not directory.is_dir():
        raise RetrievalUnavailable(
            f"No procedure corpus at {directory}.\n"
            "Expected markdown files, one procedure each."
        )

    chunks: List[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        raw = path.read_text(encoding="utf-8")

        title_match = re.search(r"^#\s+(.+)$", raw, re.MULTILINE)
        category_match = re.search(r"^category:\s*(.+)$", raw, re.MULTILINE)
        applies_match = re.search(r"^applies_when:\s*(.+)$", raw, re.MULTILINE)

        body = raw
        for pattern in (r"^#\s+.+$", r"^category:\s*.+$", r"^applies_when:\s*.+$"):
            body = re.sub(pattern, "", body, count=1, flags=re.MULTILINE)

        chunks.append(
            Chunk(
                id=path.stem,
                title=title_match.group(1).strip() if title_match else path.stem,
                category=(
                    category_match.group(1).strip() if category_match else "general"
                ),
                applies_when=(
                    applies_match.group(1).strip() if applies_match else ""
                ),
                text=body.strip(),
            )
        )

    if not chunks:
        raise RetrievalUnavailable(f"No documents found in {directory}.")
    return chunks


def embed_texts(
    texts: Sequence[str],
    model: str = EMBEDDING_MODEL,
    host: str = OLLAMA_HOST,
) -> List[List[float]]:
    """Embed strings with a local model served by Ollama.

    Raises `RetrievalUnavailable` with the command that fixes it, rather
    than a connection error from deep inside a search -- the same
    treatment `backend_ollama` gives a missing server.
    """
    import requests  # local import: retrieval maths needs no HTTP client

    vectors: List[List[float]] = []
    for text in texts:
        try:
            response = requests.post(
                f"{host}/api/embeddings",
                json={"model": model, "prompt": text},
                timeout=EMBEDDING_TIMEOUT_S,
            )
            response.raise_for_status()
            vector = response.json().get("embedding")
        except Exception as exc:  # noqa: BLE001 - any failure means unusable
            raise RetrievalUnavailable(
                f"Cannot embed with {model!r} at {host}.\n\n"
                "Start Ollama with:   ollama serve\n"
                f"Install the model:   ollama pull {model}\n\n"
                f"({type(exc).__name__})"
            ) from exc

        if not vector:
            raise RetrievalUnavailable(f"{model!r} returned no embedding.")
        vectors.append(vector)

    return vectors


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine of the angle between two vectors, in [-1, 1].

    The measure is the ANGLE, not the distance: two texts about icing
    should match whether one is three sentences and the other is three
    paragraphs, and dividing by both magnitudes removes length from the
    comparison.

    Written out rather than imported so the one piece of maths in the RAG
    pipeline is visible. numpy would be faster and, at fifteen documents,
    indistinguishable.
    """
    if len(a) != len(b):
        raise ValueError(f"vectors differ in length: {len(a)} vs {len(b)}")

    dot = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(y * y for y in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    return dot / (magnitude_a * magnitude_b)


@dataclass(frozen=True)
class Match:
    """A retrieved chunk and how well it matched."""

    chunk: Chunk
    score: float


class ProcedureIndex:
    """The corpus, embedded once and searched in memory.

    Construct it with `build()`, which uses the disk cache. The class
    itself takes vectors directly so the search logic can be tested with
    no model, no network and no Ollama -- the retrieval maths is separate
    from where the numbers came from.
    """

    def __init__(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]):
        if len(chunks) != len(vectors):
            raise ValueError("every chunk needs exactly one vector")
        self.chunks = list(chunks)
        self.vectors = [list(vector) for vector in vectors]

    def __len__(self) -> int:
        return len(self.chunks)

    @classmethod
    def build(
        cls,
        directory: Path = CORPUS_DIR,
        cache_file: Optional[Path] = CACHE_FILE,
        model: str = EMBEDDING_MODEL,
    ) -> "ProcedureIndex":
        """Load the corpus, embedding only what the cache does not hold.

        Keyed by a hash of each document's text, so editing one procedure
        re-embeds that one and leaves the rest alone.
        """
        chunks = load_corpus(directory)
        cache = _read_cache(cache_file)

        missing = [chunk for chunk in chunks if chunk.digest not in cache]
        if missing:
            fresh = embed_texts([chunk.text for chunk in missing], model=model)
            for chunk, vector in zip(missing, fresh):
                cache[chunk.digest] = vector
            _write_cache(cache_file, cache)

        return cls(chunks, [cache[chunk.digest] for chunk in chunks])

    def search(
        self,
        query_vector: Sequence[float],
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = MIN_SIMILARITY,
        conditions: Optional[Sequence[str]] = None,
    ) -> List[Match]:
        """The best-matching APPLICABLE chunks, most similar first.

        SIMILARITY IS NOT APPLICABILITY. Asked for a Cessna 172
        departure, plain vector search ranked `night-flight` FIRST at
        0.619 -- above the preflight inspection. It deserved to: it is
        dense with light-aircraft VFR language and genuinely was the most
        similar document. It simply did not apply, because nothing said
        the flight was at night. The checklist told a pilot to carry a
        flashlight and expect the black-hole illusion, possibly at noon.

        The scores made the problem plain: across the light-aircraft
        documents they ran 0.579 to 0.619, barely discriminating, because
        they are all about light aircraft.

        So a document may declare a precondition, and one whose condition
        is not known to hold is excluded BEFORE ranking rather than left
        to compete on similarity. Metadata filter first, vector search
        second -- the usual shape of hybrid retrieval.

        Unconditional documents -- preflight, fuel reserves, weight and
        balance -- carry no `applies_when` and always compete.

        Weak matches are then dropped rather than padded out to `top_k`.
        Every chunk returned becomes licence for the model to write about
        that subject, so a barely-related document is not a neutral
        addition -- it is an invitation to put mountain flying in the
        checklist for a flight across Kansas.
        """
        active = set(conditions or ())

        scored = [
            Match(chunk=chunk, score=cosine_similarity(query_vector, vector))
            for chunk, vector in zip(self.chunks, self.vectors)
            if not chunk.applies_when or chunk.applies_when in active
        ]
        scored.sort(key=lambda match: -match.score)
        return [match for match in scored if match.score >= min_similarity][:top_k]

    def conditions_available(self) -> List[str]:
        """Every precondition the corpus declares.

        Lets a caller check that a condition it believes it is supplying
        actually gates something -- a typo would otherwise exclude a
        document silently and for ever.
        """
        return sorted(
            {chunk.applies_when for chunk in self.chunks if chunk.applies_when}
        )

    def by_id(self, chunk_id: str) -> Optional[Chunk]:
        """Look up a chunk by citation id, for verifying a citation."""
        for chunk in self.chunks:
            if chunk.id == chunk_id:
                return chunk
        return None


CITATION_PATTERN = re.compile(r"\[([a-z0-9][a-z0-9-]*)\]")


@dataclass(frozen=True)
class CitationCheck:
    """The result of auditing a generated answer's citations.

    Attributes:
        cited: Every id the text claims to cite.
        unknown: Cited ids that match no document. A citation to a
            document that does not exist is worse than no citation --
            it looks like provenance and is not.
        unsupported: Lines that read like checklist items but carry no
            citation at all. These are where invented material appears.
        supported: Cited ids that do resolve.
    """

    cited: List[str]
    unknown: List[str]
    unsupported: List[str]
    supported: List[str]

    @property
    def is_clean(self) -> bool:
        return not self.unknown and not self.unsupported


def verify_citations(text: str, index: "ProcedureIndex") -> CitationCheck:
    """Audit a generated checklist against the corpus it should have used.

    WHY A MECHANICAL CHECK AND NOT JUST AN INSTRUCTION. Asked to plan a
    flight AND give a checklist in one request, the model planned the
    route and then wrote eight items from memory -- "file a flight plan
    with air traffic control", "obtain clearance for takeoff" -- none of
    it in the corpus, none of it cited. It had been instructed not to,
    in the system prompt, twice. Asked again for a different aircraft it
    produced the identical eight items.

    An instruction is a request. This is a test: given the answer and the
    index, it reports which citations resolve, which do not, and which
    lines assert something without citing anything at all. A caller can
    then refuse to render the answer rather than pass it on.

    Line-based rather than sentence-based, because a checklist is a list
    and the unit a reader trusts or distrusts is the item.
    """
    cited: List[str] = []
    unknown: List[str] = []
    unsupported: List[str] = []
    supported: List[str] = []

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        ids = CITATION_PATTERN.findall(stripped)
        if ids:
            for chunk_id in ids:
                cited.append(chunk_id)
                (supported if index.by_id(chunk_id) else unknown).append(chunk_id)
            continue

        # An item is a numbered or bulleted line. Prose around the list --
        # a preamble, a caveat, a closing note -- is not an assertion
        # about procedure and does not need a source.
        if re.match(r"^([-*+]|\d+[.)])\s+\S", stripped):
            unsupported.append(stripped)

    return CitationCheck(
        cited=cited, unknown=unknown, unsupported=unsupported, supported=supported
    )


def _read_cache(path: Optional[Path]) -> Dict[str, List[float]]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        # A corrupt cache is a reason to re-embed, not to fail.
        return {}


def _write_cache(path: Optional[Path], cache: Dict[str, List[float]]) -> None:
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
    except OSError:
        # Losing the cache costs time on the next run and nothing else.
        pass
