"""Tests for the corpus cleaning step (no network, no real documents)."""
from __future__ import annotations

import csv

import pytest

from energy.data.corpus import Source
from energy.rag import clean
from energy.rag.clean import (
    CleaningError,
    CleanResult,
    clean_corpus,
    clean_document,
    extract_main_html,
    main,
    mark_annexes,
    mark_articles,
    normalize,
    strip_links,
    to_markdown,
)

NAV_HTML = (
    "<html><body>"
    "<nav><ul><li>Nuestra empresa</li><li>Menú</li></ul></nav>"
    '<div class="panel-documento"><p>Contenido de la resolución.</p></div>'
    "</body></html>"
)


def html_source(sid="doc", selector="div.panel-documento"):
    return Source(sid, "Title " + sid, "CREG", "https://x.test/" + sid, "html", selector)


def write_raw(raw_dir, source, text):
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"{source.id}.{source.format}"
    path.write_text(text, encoding="utf-8")
    return path


# ------------------------------------------------------------------- errors
def test_cleaning_error_is_value_error():
    assert issubclass(CleaningError, ValueError)


# ------------------------------------------------------- extract_main_html
def test_extract_main_html_keeps_content_and_drops_nav():
    out = extract_main_html(NAV_HTML.encode("utf-8"), "div.panel-documento")
    assert "Contenido de la resolución." in out
    assert "Nuestra empresa" not in out
    assert "Menú" not in out


def test_extract_main_html_id_selector():
    html = (
        "<html><body><nav>Nuestra empresa</nav>"
        '<div id="some-id"><p>Texto principal</p></div></body></html>'
    )
    out = extract_main_html(html.encode("utf-8"), "#some-id")
    assert "Texto principal" in out
    assert "Nuestra empresa" not in out


def test_extract_main_html_no_match_raises():
    with pytest.raises(CleaningError):
        extract_main_html(NAV_HTML.encode("utf-8"), "#does-not-exist")


def test_latin1_bytes_survive_extract_and_markdown():
    html = (
        '<html><head><meta charset="iso-8859-1"></head><body>'
        '<div class="panel-documento"><p>RESOLUCIÓN sobre energía renovable</p></div>'
        "</body></html>"
    ).encode("latin-1")
    fragment = extract_main_html(html, "div.panel-documento")
    md = to_markdown(fragment.encode("utf-8"), ".html")
    assert "energía" in md
    assert "RESOLUCIÓN" in md


# -------------------------------------------------------------- to_markdown
def test_to_markdown_html_paragraph():
    md = to_markdown(b"<html><body><p>Hola mundo</p></body></html>", ".html")
    assert "Hola mundo" in md


# -------------------------------------------------------------- strip_links
@pytest.mark.parametrize(
    "text,expected",
    [
        ("[142](ley_0142_1994.htm#0)", "142"),
        ("plain text only", "plain text only"),
        ("see [note] and (parens)", "see [note] and (parens)"),
        ("Ley [142](a.htm#0) y Ley [143](b.htm#1) fin", "Ley 142 y Ley 143 fin"),
    ],
)
def test_strip_links(text, expected):
    assert strip_links(text) == expected


def test_strip_links_removes_images():
    out = strip_links("before ![logo](img.png) after")
    assert "logo" not in out
    assert "img.png" not in out
    assert "![" not in out
    assert out.startswith("before") and out.endswith("after")


# ---------------------------------------------------------------- normalize
@pytest.mark.parametrize("ch", ["​", "‌", "‍", "﻿"])
def test_normalize_removes_zero_width(ch):
    assert normalize(f"frecu{ch}entes") == "frecuentes"


def test_normalize_strips_trailing_spaces_per_line():
    assert normalize("uno   \ndos \t\ntres") == "uno\ndos\ntres"


@pytest.mark.parametrize("n", [3, 4, 7])
def test_normalize_collapses_blank_runs_to_two_newlines(n):
    assert normalize("a" + "\n" * n + "b") == "a\n\nb"


def test_normalize_keeps_double_newline():
    assert normalize("a\n\nb") == "a\n\nb"


def test_normalize_strips_document_edges():
    assert normalize("  \n\n hola \n\n  ") == "hola"


# ------------------------------------------------------------ mark_articles
@pytest.mark.parametrize(
    "line,heading",
    [
        ("ARTÍCULO 1o. OBJETO. Definir reglas.", "## ARTÍCULO 1o. OBJETO."),
        ("ARTICULO 2. ALCANCE. Aplica a todos.", "## ARTICULO 2. ALCANCE."),
        ("ARTÍCULO 3°. DEFINICIONES. Para efectos.", "## ARTÍCULO 3°. DEFINICIONES."),
        ("ARTÍCULO 4º. VIGENCIA. Rige desde.", "## ARTÍCULO 4º. VIGENCIA."),
    ],
)
def test_mark_articles_variants(line, heading):
    out = mark_articles(f"intro\n\n{line}\n\notro")
    assert any(ln.startswith(heading) for ln in out.splitlines())
    assert "intro" in out and "otro" in out


@pytest.mark.parametrize("prefix", ["**", "#", "## "])
def test_mark_articles_single_prefix_when_already_prefixed(prefix):
    out = mark_articles(f"{prefix}ARTÍCULO 1o. OBJETO. Definir reglas.")
    (line,) = [ln for ln in out.splitlines() if "ARTÍCULO 1o." in ln]
    assert line.startswith("## ARTÍCULO 1o.")
    assert not line.startswith("## ## ")
    assert not line.startswith("## **")


@pytest.mark.parametrize(
    "text",
    [
        "Conforme al artículo 365 de la Constitución, el Estado presta servicios.",
        "Texto previo según el ARTÍCULO 5o. de la ley.",
        "Un párrafo sin ningún artículo numerado.",
    ],
)
def test_mark_articles_leaves_non_headings_unchanged(text):
    assert mark_articles(text) == text


def test_mark_articles_keeps_quoted_amended_article_inside_amending_one():
    text = (
        "ARTÍCULO 79. ALCANCE.\n\n"
        "ARTÍCULO 80. MODIFICACIÓN DEL ARTÍCULO 6 DE LA RESOLUCIÓN CREG 055 DE 1994. "
        "quedará así:\n\n"
        "ARTÍCULO 6o. Ofertas de precio en la bolsa de energía.\n\n"
        "ARTÍCULO 81. VIGENCIA."
    )

    lines = mark_articles(text).split("\n\n")

    assert lines[0].startswith("## ARTÍCULO 79.")
    assert lines[1].startswith("## ARTÍCULO 80.")
    assert lines[2] == "ARTÍCULO 6o. Ofertas de precio en la bolsa de energía."
    assert lines[3].startswith("## ARTÍCULO 81.")


# ----------------------------------------------------------- clean_document
def test_clean_document_html(tmp_path):
    html = (
        "<html><body><nav>Nuestra empresa Menú</nav>"
        '<div class="panel-documento">'
        "<p>ARTÍCULO 1o. OBJETO. Ver [Ley 142](ley_0142_1994.htm#0) y frecu​entes.</p>"
        "</div></body></html>"
    )
    source = html_source()
    path = write_raw(tmp_path, source, html)
    out = clean_document(source, path)
    assert "Nuestra empresa" not in out
    assert "Menú" not in out
    assert "](" not in out
    assert "​" not in out
    assert "frecuentes" in out
    assert any(ln.startswith("## ARTÍCULO 1o. OBJETO.") for ln in out.splitlines())


def test_clean_document_pdf_applies_strip_links_and_normalize(tmp_path, monkeypatch):
    monkeypatch.setattr(
        clean,
        "to_markdown",
        lambda *a, **k: "  Ver [anexo](a.pdf) frecu​entes   \n\n\n\nfin  ",
    )
    source = Source("p", "Pdf", "XM", "https://x.test/p.pdf", "pdf")
    path = tmp_path / "p.pdf"
    path.write_bytes(b"%PDF-fake")
    out = clean_document(source, path)
    assert out == "Ver anexo frecuentes\n\nfin"


# ------------------------------------------------------------- clean_corpus
def two_article_html(tag):
    return (
        '<html><body><nav>Menú</nav><div class="panel-documento">'
        f"<p>ARTÍCULO 1o. OBJETO. Contenido {tag}.</p>"
        f"<p>ARTÍCULO 2o. ALCANCE. Mas {tag}.</p>"
        "</div></body></html>"
    )


def test_clean_corpus_writes_files_and_results(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    sources = [html_source("a"), html_source("b")]
    for s in sources:
        write_raw(raw, s, two_article_html(s.id))
    results, failures = clean_corpus(sources, raw, out)
    assert failures == {}
    assert [r.id for r in results] == ["a", "b"]
    for r in results:
        assert isinstance(r, CleanResult)
        text = (out / f"{r.id}.md").read_text(encoding="utf-8")
        assert r.chars == len(text)
        assert r.articles == 2


def test_clean_corpus_missing_raw_file_is_a_failure_others_written(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    good, missing = html_source("good"), html_source("missing")
    write_raw(raw, good, two_article_html("g"))
    results, failures = clean_corpus([missing, good], raw, out)
    assert "missing" in failures
    assert [r.id for r in results] == ["good"]
    assert (out / "good.md").exists()
    assert not (out / "missing.md").exists()


def test_clean_corpus_selector_mismatch_is_a_failure(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    good, bad = html_source("good"), html_source("bad", selector="#nope")
    write_raw(raw, good, two_article_html("g"))
    write_raw(raw, bad, two_article_html("b"))
    results, failures = clean_corpus([bad, good], raw, out)
    assert list(failures) == ["bad"]
    assert [r.id for r in results] == ["good"]
    assert (out / "good.md").exists()


# --------------------------------------------------------------------- main
def write_manifest_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "title", "publisher", "url", "format", "selector"])
        w.writerows(rows)
    return path


def test_main_writes_files_and_returns_zero(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    manifest = write_manifest_csv(
        tmp_path / "s.csv",
        [["a", "A", "CREG", "https://x.test/a", "html", "div.panel-documento"]],
    )
    write_raw(raw, html_source("a"), two_article_html("a"))
    argv = ["--manifest", str(manifest), "--raw-dir", str(raw), "--clean-dir", str(out)]
    assert main(argv) == 0
    assert (out / "a.md").exists()


def test_main_returns_nonzero_when_a_document_fails(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "clean"
    manifest = write_manifest_csv(
        tmp_path / "s.csv",
        [
            ["a", "A", "CREG", "https://x.test/a", "html", "div.panel-documento"],
            ["b", "B", "CREG", "https://x.test/b", "html", "#nope"],
        ],
    )
    write_raw(raw, html_source("a"), two_article_html("a"))
    write_raw(raw, html_source("b"), two_article_html("b"))
    argv = ["--manifest", str(manifest), "--raw-dir", str(raw), "--clean-dir", str(out)]
    assert main(argv) != 0
    assert (out / "a.md").exists()


# ------------------------------------------------------------- mark_annexes
@pytest.mark.parametrize(
    "line,heading",
    [
        ("ANEXO 1.", "## ANEXO 1."),
        ("ANEXO NO. 1.", "## ANEXO 1."),
        ("ANEXO.", "## ANEXO."),
        ("ANEXO IV.", "## ANEXO IV."),
    ],
)
def test_mark_annexes_marks_bare_annex_lines(line, heading):
    assert mark_annexes(f"texto previo\n\n{line}\n\ncontenido") == (
        f"texto previo\n\n{heading}\n\ncontenido"
    )


@pytest.mark.parametrize(
    "text",
    [
        "**ANEXO CO.4.**",
        "El ANEXO 1. establece las condiciones.",
        "ver el anexo 1.",
    ],
)
def test_mark_annexes_leaves_quoted_or_inline_annexes(text):
    assert mark_annexes(text) == text
