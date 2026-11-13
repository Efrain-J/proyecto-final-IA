"""Tests for the retrievers (fake embedder, in-memory Qdrant, no network)."""
from __future__ import annotations

from dataclasses import asdict

import pytest
from qdrant_client import QdrantClient

from energy.rag.chunk import write_chunks
from energy.rag.index import build_index
from energy.rag.retrieve import BM25Retriever, Hit, QdrantRetriever, open_retriever


# --- Hit ---


def test_hit_citation_with_article():
    hit = Hit("d", "Resolución X", "CREG", "12", True, "", 0, "texto", 0.5)
    assert hit.citation == "Resolución X, Art. 12"


def test_hit_citation_without_article():
    hit = Hit("d", "Glosario XM", "XM", None, True, "", 0, "texto", 0.5)
    assert hit.citation == "Glosario XM"


def test_hit_from_payload_round_trip(make_chunk):
    chunk = make_chunk(article="5", in_force=False, superseded_by="CREG 2 de 2026", chunk_index=3)
    hit = Hit.from_payload(asdict(chunk), 0.77)
    assert hit.score == 0.77
    for key, value in asdict(chunk).items():
        assert getattr(hit, key) == value


# --- QdrantRetriever ---


@pytest.fixture
def indexed(sample_chunks, fake_embed_documents):
    client = QdrantClient(":memory:")
    build_index(sample_chunks, client, fake_embed_documents)
    yield client
    client.close()


def test_qdrant_retriever_name(indexed, fake_embed_query):
    assert QdrantRetriever(indexed, fake_embed_query).name == "qdrant-nvidia"


def test_qdrant_retriever_returns_matching_chunk_first(indexed, fake_embed_query):
    retriever = QdrantRetriever(indexed, fake_embed_query, score_threshold=0.0)
    hits = retriever.search("despacho ideal sistema sin restricciones de red")
    assert hits
    assert hits[0].doc_id == "xmglosario"
    assert hits[0].score >= hits[-1].score


def test_qdrant_retriever_high_threshold_filters_all(indexed, fake_embed_query):
    retriever = QdrantRetriever(indexed, fake_embed_query, score_threshold=0.999)
    assert retriever.search("receta de arepas con queso") == []


def test_qdrant_retriever_k_limits_results(indexed, fake_embed_query):
    retriever = QdrantRetriever(indexed, fake_embed_query, score_threshold=-1.0)
    assert len(retriever.search("energía sistema precio", k=2)) <= 2
    assert len(retriever.search("energía sistema precio", k=1)) == 1


# --- BM25Retriever ---


@pytest.fixture
def bm25(sample_chunks):
    return BM25Retriever(sample_chunks)


def test_bm25_name(bm25):
    assert bm25.name == "bm25"


def test_bm25_keyword_match_ranks_first(bm25):
    hits = bm25.search("precio de bolsa generadores")
    assert hits[0].doc_id == "xmbolsa"


def test_bm25_is_accent_insensitive(make_chunk):
    retriever = BM25Retriever([
        make_chunk(doc_id="a", text="La energía eléctrica se transa en bolsa cada hora del día."),
        make_chunk(doc_id="b", text="Documento sobre otro asunto completamente distinto aquí."),
    ])
    hits = retriever.search("energia")
    assert hits
    assert hits[0].doc_id == "a"


def test_bm25_no_overlap_returns_empty(bm25):
    assert bm25.search("zzzqqq xyzzy") == []


def test_bm25_scores_positive(bm25):
    hits = bm25.search("sistema energía precio despacho")
    assert hits
    assert all(h.score > 0 for h in hits)


def test_bm25_limits_to_k(bm25):
    assert len(bm25.search("sistema energía precio despacho potencia", k=2)) <= 2


# --- open_retriever ---


def test_open_retriever_falls_back_to_bm25_without_key(tmp_path, monkeypatch, sample_chunks):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(sample_chunks, chunks_path)

    def forbidden():
        pytest.fail("embeddings must not be created without a key")

    retriever = open_retriever(
        qdrant_path=tmp_path / "qdrant", chunks_path=chunks_path,
        embeddings_factory=forbidden,
    )
    assert retriever.name == "bm25"


def test_open_retriever_falls_back_when_index_missing(
    tmp_path, monkeypatch, sample_chunks, fake_embed_query
):
    monkeypatch.setenv("NVIDIA_API_KEY", "fake-key")
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(sample_chunks, chunks_path)

    class FakeEmbeddings:
        def embed_query(self, text):
            return fake_embed_query(text)

        def embed_documents(self, texts):
            return [fake_embed_query(t) for t in texts]

    retriever = open_retriever(
        qdrant_path=tmp_path / "does-not-exist", chunks_path=chunks_path,
        embeddings_factory=FakeEmbeddings,
    )
    assert retriever.name == "bm25"


def test_hit_citation_for_annex():
    hit = Hit("d", "Resolución X", "CREG", "Anexo 1", True, "", 0, "texto", 1.0)
    assert hit.citation == "Resolución X, Anexo 1"
