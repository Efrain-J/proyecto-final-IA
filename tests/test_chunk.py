"""Tests for corpus chunking (no network, no real documents)."""
from __future__ import annotations

import json

from energy.data.corpus import Source
from energy.rag.chunk import (
    Chunk,
    chunk_corpus,
    chunk_document,
    embedding_text,
    main,
    read_chunks,
    split_articles,
    split_text,
    write_chunks,
)

PREAMBLE = "Considerando que la comisión tiene facultades para regular el servicio."


def make_source(sid="doc", **kw):
    return Source(sid, "Resolución CREG X", "CREG", "https://x.test/" + sid, "html", **kw)


def long_text(n_chars=3000):
    sentence = "La comisión regula la actividad de generación de energía eléctrica. "
    return (sentence * (n_chars // len(sentence) + 1))[:n_chars]


# --- split_articles ---


def test_split_articles_preamble_is_none_article():
    md = f"{PREAMBLE}\n\n## ARTÍCULO 1o. OBJETO. Se regula algo.\n\nMás texto."
    sections = split_articles(md)
    assert sections[0][0] is None
    assert PREAMBLE in sections[0][1]
    assert sections[1][0] == "1"


def test_split_articles_keeps_heading_line_and_number():
    md = "## ARTÍCULO 2o. ÁMBITO DE APLICACIÓN. Aplica a todos.\n\nSegundo párrafo."
    sections = split_articles(md)
    assert len(sections) == 1
    article, text = sections[0]
    assert article == "2"
    assert "## ARTÍCULO 2o. ÁMBITO" in text
    assert "Segundo párrafo." in text


def test_split_articles_accepts_unaccented_heading():
    sections = split_articles("## ARTICULO 7. DEFINICIONES. Texto.")
    assert sections[0][0] == "7"


def test_split_articles_multiple_sections_in_order():
    md = "## ARTÍCULO 1. A. uno\n\n## ARTÍCULO 2. B. dos\n\n## ARTÍCULO 12. C. tres"
    assert [a for a, _ in split_articles(md)] == ["1", "2", "12"]


def test_split_articles_lowercase_mention_does_not_split():
    md = "Según lo dispuesto en el artículo 5 de la ley, se aplica.\n\nartículo 6 también."
    sections = split_articles(md)
    assert len(sections) == 1
    assert sections[0][0] is None


def test_split_articles_no_headings_gives_single_none_item():
    md = "Documento sin artículos.\n\nSolo párrafos."
    sections = split_articles(md)
    assert len(sections) == 1
    assert sections[0][0] is None
    assert "Solo párrafos." in sections[0][1]


# --- split_text ---


def test_split_text_respects_size():
    pieces = split_text(long_text(5000), size=300, overlap=30)
    assert len(pieces) > 1
    assert all(len(p) <= 300 for p in pieces)


def test_split_text_drops_short_pieces():
    text = long_text(2000) + "\n\nfin."
    pieces = split_text(text, size=300, overlap=30)
    assert all(len(p) >= 30 for p in pieces)


def test_split_text_all_short_returns_empty():
    assert split_text("corto") == []


# --- chunk_document ---


def test_chunk_document_long_article_split_with_same_article():
    md = "## ARTÍCULO 3. TÍTULO. " + long_text(3000)
    chunks = chunk_document(make_source(), md, size=900, overlap=120)
    assert len(chunks) > 1
    assert {c.article for c in chunks} == {"3"}


def test_chunk_document_without_articles_has_none_article():
    chunks = chunk_document(make_source("xm"), long_text(2000), size=900, overlap=120)
    assert chunks
    assert all(c.article is None for c in chunks)


def test_chunk_document_copies_source_metadata():
    source = make_source(in_force=False, superseded_by="CREG 101 113 de 2026")
    md = "## ARTÍCULO 1. OBJETO. " + long_text(500)
    chunks = chunk_document(source, md, size=900, overlap=120)
    assert chunks
    for c in chunks:
        assert c.doc_id == "doc"
        assert c.title == "Resolución CREG X"
        assert c.publisher == "CREG"
        assert c.in_force is False
        assert c.superseded_by == "CREG 101 113 de 2026"


def test_chunk_document_index_sequential_across_articles():
    md = "## ARTÍCULO 1. A. " + long_text(2000) + "\n\n## ARTÍCULO 2. B. " + long_text(2000)
    chunks = chunk_document(make_source(), md, size=900, overlap=120)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.article for c in chunks} == {"1", "2"}


def test_chunk_document_min_length():
    md = PREAMBLE + "\n\n## ARTÍCULO 1. A. " + long_text(2500) + "\n\n## ARTÍCULO 2. Corto."
    chunks = chunk_document(make_source(), md, size=900, overlap=120)
    assert all(len(c.text) >= 30 for c in chunks)


# --- embedding_text ---


def test_embedding_text_with_article(make_chunk):
    chunk = make_chunk(title="Resolución X", article="12", text="Contenido.")
    assert embedding_text(chunk) == "Resolución X — Artículo 12\n\nContenido."


def test_embedding_text_without_article(make_chunk):
    chunk = make_chunk(title="Glosario XM", article=None, text="Contenido.")
    assert embedding_text(chunk) == "Glosario XM\n\nContenido."


# --- write_chunks / read_chunks ---


def test_jsonl_round_trip_with_accents(tmp_path, make_chunk):
    chunks = [
        make_chunk(article="2", text="Ámbito de aplicación: energía, año, ñandú."),
        make_chunk(
            chunk_index=1, in_force=False, superseded_by="CREG 1 de 2026",
            text="Segundo fragmento con acentos: eléctrica.",
        ),
    ]
    path = tmp_path / "out" / "chunks.jsonl"
    write_chunks(chunks, path)
    assert read_chunks(path) == chunks


def test_write_chunks_one_json_object_per_line(tmp_path, make_chunk):
    chunks = [make_chunk(chunk_index=i) for i in range(3)]
    path = tmp_path / "chunks.jsonl"
    write_chunks(chunks, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert [p["chunk_index"] for p in parsed] == [0, 1, 2]
    assert set(parsed[0]) == {
        "doc_id", "title", "publisher", "article", "in_force",
        "superseded_by", "chunk_index", "text",
    }


# --- chunk_corpus ---


def test_chunk_corpus_records_missing_file(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "present.md").write_text("## ARTÍCULO 1. A. " + long_text(500), encoding="utf-8")
    sources = [make_source("present"), make_source("absent")]
    chunks, failures = chunk_corpus(sources, clean_dir)
    assert chunks
    assert {c.doc_id for c in chunks} == {"present"}
    assert "absent" in failures
    assert "present" not in failures


# --- main ---


def _write_manifest(path, ids):
    rows = ["id,title,publisher,url,format,selector,in_force,superseded_by"]
    for sid in ids:
        rows.append(f"{sid},Título {sid},CREG,https://x.test/{sid},html,,yes,")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def test_main_writes_deterministic_jsonl(tmp_path):
    clean_dir = tmp_path / "clean"
    clean_dir.mkdir()
    (clean_dir / "a.md").write_text("## ARTÍCULO 1. A. " + long_text(2500), encoding="utf-8")
    (clean_dir / "b.md").write_text(long_text(1500), encoding="utf-8")
    manifest = tmp_path / "sources.csv"
    _write_manifest(manifest, ["a", "b"])
    out = tmp_path / "chunks.jsonl"
    argv = ["--manifest", str(manifest), "--clean-dir", str(clean_dir), "--out", str(out)]

    assert main(argv) == 0
    first = out.read_bytes()
    assert first
    chunks = read_chunks(out)
    assert {c.doc_id for c in chunks} == {"a", "b"}
    assert all(isinstance(c, Chunk) for c in chunks)

    assert main(argv) == 0
    assert out.read_bytes() == first


def test_split_articles_gives_annexes_their_own_section():
    md = (
        "## ARTÍCULO 8o. VIGENCIA. Rige desde su publicación.\n\n"
        "Firmas del director.\n\n"
        "## ANEXO 1.\n\nELEGIBILIDAD PARA COMERCIALIZACION.\n\n"
        "## ANEXO.\n\nCONDICIONES DE LAS GARANTÍAS."
    )

    sections = split_articles(md)

    assert [label for label, _ in sections] == ["8", "Anexo 1", "Anexo"]
    assert "ELEGIBILIDAD" in sections[1][1]
    assert "ELEGIBILIDAD" not in sections[0][1]


def test_embedding_text_names_annex_sections(make_chunk):
    chunk = make_chunk(article="Anexo 1")
    assert embedding_text(chunk).startswith(f"{chunk.title} — Anexo 1\n\n")
