"""Search the regulation corpus: the backend of the agent's search tool.

Two interchangeable retrievers share the ``Retriever`` protocol:

- ``QdrantRetriever``: semantic search over the on-disk index (NVIDIA embeddings).
- ``BM25Retriever``: keyword search over ``chunks.jsonl``, with no network or
  index needed. It is the fallback, because the Qdrant index is gitignored (D8).

``open_retriever`` picks one and logs why. See D13.

Manual check, from anywhere::

    uv run python -m energy.rag.retrieve "pregunta" [--k 5] [--bm25]
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from energy import config
from energy.rag.chunk import (
    DEFAULT_CHUNKS_PATH,
    Chunk,
    embedding_text,
    read_chunks,
    section_name,
)
from energy.rag.index import COLLECTION, QDRANT_PATH, make_embeddings

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)

# Tuned for the nemotron embedder in the workshop (CLAUDE.md, Retrieval).
SCORE_THRESHOLD = config.SCORE_THRESHOLD
BM25_K1 = 1.5
BM25_B = 0.75
_STOPWORDS = frozenset(
    "de la el en y a los las del que por para con un una se es al lo su o como más".split()
)
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class Hit:
    """A search result: a chunk's metadata and text plus its relevance score.

    Attributes:
        doc_id: Manifest id of the source document.
        title: Full document title.
        publisher: Issuing body.
        article: Article number as a string, or ``None``.
        in_force: ``False`` if the rule was replaced.
        superseded_by: Name of the replacing rule, if any.
        chunk_index: Position of the chunk within its document.
        text: Chunk text.
        score: Cosine similarity (Qdrant) or BM25 score; not comparable across retrievers.
    """

    doc_id: str
    title: str
    publisher: str
    article: str | None
    in_force: bool
    superseded_by: str
    chunk_index: int
    text: str
    score: float

    @property
    def citation(self) -> str:
        """Human-readable source: ``"<title>, Art. <n>"``, ``"<title>, Anexo <n>"`` or the title."""
        if self.article:
            return f"{self.title}, {section_name(self.article, short=True)}"
        return self.title

    @classmethod
    def from_payload(cls, payload: dict[str, Any], score: float) -> Hit:
        """Build a hit from a Qdrant payload (a serialized ``Chunk``)."""
        return cls(
            doc_id=payload["doc_id"],
            title=payload["title"],
            publisher=payload["publisher"],
            article=payload["article"],
            in_force=payload["in_force"],
            superseded_by=payload["superseded_by"],
            chunk_index=payload["chunk_index"],
            text=payload["text"],
            score=score,
        )


class Retriever(Protocol):
    """Anything the agent can search the corpus with."""

    name: str

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` hits, best first."""
        ...


class QdrantRetriever:
    """Semantic search over a Qdrant collection built by ``energy.rag.index``."""

    name = "qdrant-nvidia"

    def __init__(
        self,
        client: QdrantClient,
        embed_query: Callable[[str], list[float]],
        collection: str = COLLECTION,
        score_threshold: float = SCORE_THRESHOLD,
    ) -> None:
        """Store the dependencies.

        Args:
            client: An open Qdrant client; the caller owns it.
            embed_query: Maps a query to a vector (``make_embeddings().embed_query``).
            collection: Collection to search.
            score_threshold: Minimum cosine score; hits below it are dropped, so
                out-of-corpus questions return nothing.
        """
        self._client = client
        self._embed_query = embed_query
        self._collection = collection
        self._threshold = score_threshold

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` chunks scoring at least the threshold, best first."""
        points = self._client.query_points(
            self._collection,
            query=self._embed_query(query),
            limit=k,
            score_threshold=self._threshold,
            with_payload=True,
        ).points
        return [Hit.from_payload(p.payload or {}, p.score) for p in points]


def tokenize(text: str) -> list[str]:
    """Lowercase, strip accents, split into words and drop Spanish stopwords."""
    folded = unicodedata.normalize("NFKD", text.lower())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    return [t for t in _WORD.findall(folded) if t not in _STOPWORDS]


class BM25Retriever:
    """Okapi BM25 over ``embedding_text`` of each chunk.

    Implemented here to avoid a dependency for a fallback (D13). The query is
    tokenized the same way as the documents.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = BM25_K1, b: float = BM25_B) -> None:
        """Index ``chunks`` in memory."""
        self._chunks = chunks
        self._k1 = k1
        self._b = b
        self._tf = [Counter(tokenize(embedding_text(c))) for c in chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = (sum(self._len) / len(chunks)) if chunks else 0.0
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(chunks)
        # The "+1" inside the log keeps IDF positive for very common terms.
        self._idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}

    def search(self, query: str, k: int = 5) -> list[Hit]:
        """Return up to ``k`` chunks with a positive BM25 score, best first."""
        terms = [t for t in dict.fromkeys(tokenize(query)) if t in self._idf]
        scored: list[tuple[float, int]] = []
        for i, tf in enumerate(self._tf):
            norm = self._k1 * (1 - self._b + self._b * self._len[i] / self._avg_len)
            score = sum(
                self._idf[t] * tf[t] * (self._k1 + 1) / (tf[t] + norm) for t in terms if t in tf
            )
            if score > 0:
                scored.append((score, i))
        scored.sort(key=lambda s: (-s[0], s[1]))
        hits = []
        for score, i in scored[:k]:
            c = self._chunks[i]
            hits.append(
                Hit(
                    doc_id=c.doc_id,
                    title=c.title,
                    publisher=c.publisher,
                    article=c.article,
                    in_force=c.in_force,
                    superseded_by=c.superseded_by,
                    chunk_index=c.chunk_index,
                    text=c.text,
                    score=score,
                )
            )
        return hits


def open_retriever(
    qdrant_path: Path = QDRANT_PATH,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    embeddings_factory: Callable[[], Any] = make_embeddings,
) -> Retriever:
    """Open the best available retriever.

    Uses Qdrant when ``NVIDIA_API_KEY`` is set, the index folder exists and it
    opens with the expected collection. Otherwise falls back to BM25 over
    ``chunks_path`` and logs the reason, so the app keeps running offline.
    Call it once and cache the result: a local Qdrant path takes a file lock.

    Args:
        qdrant_path: Folder of the on-disk index.
        chunks_path: ``chunks.jsonl`` for the BM25 fallback.
        embeddings_factory: Creates the embeddings client; tests inject a fake.

    Returns:
        A ``Retriever``; check its ``name`` to know which one.
    """
    reason = ""
    if not os.environ.get("NVIDIA_API_KEY"):
        reason = "NVIDIA_API_KEY is not set"
    elif not Path(qdrant_path).exists():
        reason = f"index folder {qdrant_path} does not exist"
    else:
        client = None
        try:
            from qdrant_client import QdrantClient

            client = QdrantClient(path=str(qdrant_path))
            if not client.collection_exists(COLLECTION):
                raise RuntimeError(f"collection {COLLECTION!r} not found")
            return QdrantRetriever(client, embeddings_factory().embed_query)
        except Exception as exc:  # any failure must degrade to BM25, not crash the app
            reason = f"could not open Qdrant ({type(exc).__name__}: {exc})"
            if client is not None:
                client.close()
    logger.warning("Falling back to BM25: %s", reason)
    return BM25Retriever(read_chunks(chunks_path))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: print the ranked hits for a question."""
    config.load_env()  # entry point only, never at import time
    # Windows consoles default to cp1252 and garble Spanish accents.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search the regulation corpus.")
    parser.add_argument("query")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bm25", action="store_true", help="force the BM25 retriever")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.bm25:
        retriever: Retriever = BM25Retriever(read_chunks(args.chunks))
    else:
        retriever = open_retriever(args.qdrant_path, args.chunks)
    hits = retriever.search(args.query, k=args.k)
    print(f"[{retriever.name}] {len(hits)} hits")
    for hit in hits:
        status = "in force" if hit.in_force else f"NOT in force (superseded by {hit.superseded_by})"
        snippet = hit.text[:200].replace("\n", " ")
        print(f"{hit.score:.3f}  {hit.citation}  [{status}]\n    {snippet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
