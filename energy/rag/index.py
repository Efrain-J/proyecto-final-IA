"""Build the on-disk Qdrant index from ``data/corpus/chunks.jsonl``.

Each chunk is embedded with NVIDIA ``nvidia/nemotron-3-embed-1b`` (the setup
proven in the workshop) and stored with its metadata as the point payload, so
a search hit carries everything a citation needs. The index lives in
``data/qdrant/`` (gitignored, D8); the app only opens it. See D13.

Run from the repo root, after ``energy.rag.chunk``::

    uv run python -m energy.rag.index

Needs ``NVIDIA_API_KEY`` (environment or ``.env``). Rebuilding replaces the
collection, so the command is safe to rerun.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from energy.rag.chunk import DEFAULT_CHUNKS_PATH, Chunk, embedding_text, read_chunks

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
COLLECTION = "regulacion"
QDRANT_PATH = Path("data/qdrant")

_RETRIES = 4


def make_embeddings() -> Any:
    """Create the NVIDIA embeddings client.

    ``truncate="END"`` cuts over-long inputs instead of failing the call.
    The SDK is imported here so importing this module stays cheap and offline.

    Returns:
        A ``NVIDIAEmbeddings`` with ``embed_documents`` and ``embed_query``
        (the latter sets the query input type).
    """
    import warnings

    from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

    # The SDK doesn't know this model's type and warns "inference may fail";
    # it works (verified in D13), so silence only that message.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*type is unknown.*")
        return NVIDIAEmbeddings(model=EMBED_MODEL, truncate="END")


def _embed_with_retry(
    embed_documents: Callable[[list[str]], list[list[float]]], texts: list[str]
) -> list[list[float]]:
    """Call the embedder, retrying with backoff (rate limits, transient errors)."""
    for attempt in range(_RETRIES):
        try:
            return embed_documents(texts)
        except Exception:  # SDK errors vary (HTTP 429, timeouts); retry any
            if attempt == _RETRIES - 1:
                raise
            time.sleep(2.0 * 2**attempt)
    raise AssertionError("unreachable")


def build_index(
    chunks: list[Chunk],
    client: QdrantClient,
    embed_documents: Callable[[list[str]], list[list[float]]],
    *,
    collection: str = COLLECTION,
    batch_size: int = 64,
) -> int:
    """Embed ``chunks`` and store them in a freshly created cosine collection.

    Args:
        chunks: Chunks to index; each becomes one point whose payload is the
            chunk's fields.
        client: An open Qdrant client (on disk, or ``":memory:"`` in tests).
        embed_documents: Maps texts to vectors; injected so tests need no network.
        collection: Collection name. An existing one is deleted first.
        batch_size: Texts embedded and upserted per call.

    Returns:
        Number of points stored.

    Raises:
        ValueError: If ``chunks`` is empty (the dimension cannot be probed).
    """
    from qdrant_client.models import Distance, PointStruct, VectorParams

    if not chunks:
        raise ValueError("No chunks to index")

    # Probe the dimension from the first batch rather than hardcoding it.
    first = _embed_with_retry(embed_documents, [embedding_text(c) for c in chunks[:batch_size]])
    dim = len(first[0])

    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE)
    )

    total = 0
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = (
            first
            if start == 0
            else _embed_with_retry(embed_documents, [embedding_text(c) for c in batch])
        )
        client.upsert(
            collection,
            points=[
                PointStruct(id=start + i, vector=vec, payload=asdict(chunk))
                for i, (chunk, vec) in enumerate(zip(batch, vectors, strict=True))
            ],
        )
        total += len(batch)
    return total


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: embed ``chunks.jsonl`` into ``data/qdrant``.

    Returns:
        ``0`` on success, ``1`` if the API key or the chunks file is missing.
    """
    from dotenv import load_dotenv
    from qdrant_client import QdrantClient

    load_dotenv()  # entry point only, never at import time
    parser = argparse.ArgumentParser(description="Build the Qdrant index of the RAG corpus.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--qdrant-path", type=Path, default=QDRANT_PATH)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args(argv)

    if not os.environ.get("NVIDIA_API_KEY"):
        print("NVIDIA_API_KEY is not set (environment or .env).", file=sys.stderr)
        return 1
    if not args.chunks.exists():
        print(f"{args.chunks} not found; run `python -m energy.rag.chunk` first.", file=sys.stderr)
        return 1

    chunks = read_chunks(args.chunks)
    embeddings = make_embeddings()
    client = QdrantClient(path=str(args.qdrant_path))
    try:
        start = time.perf_counter()
        count = build_index(chunks, client, embeddings.embed_documents, batch_size=args.batch_size)
        elapsed = time.perf_counter() - start
        dim = client.get_collection(COLLECTION).config.params.vectors.size
    finally:
        client.close()
    print(f"indexed {count} points, dimension {dim}, {elapsed:.1f}s -> {args.qdrant_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
