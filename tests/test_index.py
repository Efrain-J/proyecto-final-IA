"""Tests for the Qdrant index builder (fake embedder, in-memory client)."""
from __future__ import annotations

from dataclasses import asdict

import pytest
from qdrant_client import QdrantClient

from energy.rag.index import COLLECTION, build_index


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def many_chunks(make_chunk, n):
    return [
        make_chunk(doc_id=f"d{i}", chunk_index=i, text=f"Fragmento número {i} sobre tema{i} y energía.")
        for i in range(n)
    ]


def test_build_index_returns_count_and_populates_collection(client, sample_chunks, fake_embed_documents):
    n = build_index(sample_chunks, client, fake_embed_documents)
    assert n == len(sample_chunks)
    assert client.count(COLLECTION).count == len(sample_chunks)


def test_vector_size_is_probed_from_embedder(client, sample_chunks, fake_embed_documents, fake_dim):
    build_index(sample_chunks, client, fake_embed_documents)
    params = client.get_collection(COLLECTION).config.params.vectors
    assert params.size == fake_dim


def test_payload_contains_every_chunk_field(client, sample_chunks, fake_embed_documents):
    build_index(sample_chunks, client, fake_embed_documents)
    points, _ = client.scroll(COLLECTION, limit=10, with_payload=True)
    payloads = {p.payload["doc_id"]: p.payload for p in points}
    for chunk in sample_chunks:
        payload = payloads[chunk.doc_id]
        for key, value in asdict(chunk).items():
            assert payload[key] == value


def test_build_index_twice_does_not_duplicate(client, sample_chunks, fake_embed_documents):
    build_index(sample_chunks, client, fake_embed_documents)
    build_index(sample_chunks, client, fake_embed_documents)
    assert client.count(COLLECTION).count == len(sample_chunks)


def test_small_batch_size_still_indexes_everything(client, make_chunk, fake_embed_documents):
    chunks = many_chunks(make_chunk, 7)
    n = build_index(chunks, client, fake_embed_documents, batch_size=2)
    assert n == 7
    assert client.count(COLLECTION).count == 7
