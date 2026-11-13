import httpx
import pytest

from energy.data import corpus
from energy.data.corpus import (
    DownloadSummary,
    Source,
    download_sources,
    load_manifest,
    target_path,
    text_length,
)

HEADER = "id,title,publisher,url,format,notes\n"


def write_manifest(path, rows, header=HEADER):
    path.write_text(header + "".join(r + "\n" for r in rows), encoding="utf-8")
    return path


# ---------------------------------------------------------------- load_manifest
def test_load_manifest_parses_valid_rows(tmp_path):
    m = write_manifest(
        tmp_path / "s.csv",
        [
            "a,Title A,CREG,https://x.test/a.html,html,some note",
            "b,Title B,XM,https://x.test/b.pdf,pdf,",
        ],
    )
    assert load_manifest(m) == [
        Source("a", "Title A", "CREG", "https://x.test/a.html", "html"),
        Source("b", "Title B", "XM", "https://x.test/b.pdf", "pdf"),
    ]


def test_load_manifest_handles_commas_in_quoted_fields(tmp_path):
    m = write_manifest(
        tmp_path / "s.csv",
        ['a,"Title, with comma",CREG,"https://x.test/a?x=1,2,3",html,"note, with comma"'],
    )
    (src,) = load_manifest(m)
    assert src.url == "https://x.test/a?x=1,2,3"
    assert src.title == "Title, with comma"


def test_source_is_frozen():
    src = Source("a", "T", "P", "https://x.test", "html")
    with pytest.raises(Exception):
        src.id = "other"


def test_load_manifest_missing_column_raises(tmp_path):
    m = write_manifest(
        tmp_path / "s.csv",
        ["a,T,P,html"],
        header="id,title,publisher,format\n",
    )
    with pytest.raises(ValueError):
        load_manifest(m)


def test_load_manifest_duplicate_id_raises(tmp_path):
    m = write_manifest(
        tmp_path / "s.csv",
        ["a,T1,P,https://x.test/1,html,", "a,T2,P,https://x.test/2,pdf,"],
    )
    with pytest.raises(ValueError):
        load_manifest(m)


def test_load_manifest_bad_format_raises(tmp_path):
    m = write_manifest(tmp_path / "s.csv", ["a,T,P,https://x.test/1,docx,"])
    with pytest.raises(ValueError):
        load_manifest(m)


def test_load_manifest_empty_url_raises(tmp_path):
    m = write_manifest(tmp_path / "s.csv", ["a,T,P,,html,"])
    with pytest.raises(ValueError):
        load_manifest(m)


# ------------------------------------------------------------------ target_path
@pytest.mark.parametrize("fmt", ["html", "pdf"])
def test_target_path_is_id_dot_format(tmp_path, fmt):
    src = Source("creg_1", "T", "P", "https://x.test", fmt)
    assert target_path(tmp_path, src) == tmp_path / f"creg_1.{fmt}"


# -------------------------------------------------------------- download_sources
SOURCES = [
    Source("a", "A", "P", "https://x.test/a.html", "html"),
    Source("b", "B", "P", "https://x.test/b.pdf", "pdf"),
    Source("c", "C", "P", "https://x.test/c.html", "html"),
]


def make_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def ok_handler(calls):
    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, content=b"BYTES:" + request.url.path.encode())

    return handler


def dl(sources, raw_dir, client, **kw):
    return download_sources(sources, raw_dir, delay=0, backoff_base=0, client=client, **kw)


def test_download_sources_writes_bytes_to_target_path(tmp_path):
    summary = dl(SOURCES, tmp_path, make_client(ok_handler([])))
    for s in SOURCES:
        expected = b"BYTES:" + s.url.split("x.test")[1].encode()
        assert target_path(tmp_path, s).read_bytes() == expected
    assert sorted(summary.downloaded) == ["a", "b", "c"]
    assert summary.failed == {}


def test_download_sources_creates_raw_dir(tmp_path):
    raw = tmp_path / "deep" / "raw"
    dl(SOURCES[:1], raw, make_client(ok_handler([])))
    assert target_path(raw, SOURCES[0]).exists()


def test_download_sources_skips_existing_files(tmp_path):
    calls = []
    client = make_client(ok_handler(calls))
    dl(SOURCES, tmp_path, client)
    calls.clear()
    summary = dl(SOURCES, tmp_path, client)
    assert calls == []
    assert sorted(summary.skipped) == ["a", "b", "c"]
    assert summary.downloaded == []


def test_download_sources_force_redownloads(tmp_path):
    calls = []
    client = make_client(ok_handler(calls))
    dl(SOURCES, tmp_path, client)
    calls.clear()
    summary = dl(SOURCES, tmp_path, client, force=True)
    assert len(calls) == 3
    assert sorted(summary.downloaded) == ["a", "b", "c"]


def test_download_sources_persistent_404_lands_in_failed_without_raising(tmp_path):
    def handler(request):
        if request.url.path == "/b.pdf":
            return httpx.Response(404, text="nope")
        return httpx.Response(200, content=b"ok")

    summary = dl(SOURCES, tmp_path, make_client(handler))
    assert list(summary.failed) == ["b"]
    assert sorted(summary.downloaded) == ["a", "c"]
    assert not target_path(tmp_path, SOURCES[1]).exists()


def test_download_sources_retries_transient_500(tmp_path):
    attempts = {"n": 0}

    def handler(request):
        attempts["n"] += 1
        if attempts["n"] == 1:
            return httpx.Response(500)
        return httpx.Response(200, content=b"finally")

    summary = dl(SOURCES[:1], tmp_path, make_client(handler))
    assert attempts["n"] == 2
    assert summary.downloaded == ["a"]
    assert target_path(tmp_path, SOURCES[0]).read_bytes() == b"finally"


def test_download_sources_failure_leaves_no_tmp_files(tmp_path):
    summary = dl(SOURCES[:1], tmp_path, make_client(lambda r: httpx.Response(404)))
    assert "a" in summary.failed
    assert not list(tmp_path.glob("*.tmp"))


# ------------------------------------------------------------------ text_length
def test_text_length_contains_paragraph_text(tmp_path):
    text = "Resolucion de almacenamiento con baterias en el sistema interconectado nacional."
    f = tmp_path / "doc.html"
    f.write_text(f"<html><body><h1>Titulo</h1><p>{text}</p></body></html>", encoding="utf-8")
    assert text_length(f) >= len(text)


def test_text_length_empty_body_is_small(tmp_path):
    f = tmp_path / "empty.html"
    f.write_text("<html><head><title>x</title></head><body></body></html>", encoding="utf-8")
    assert text_length(f) < 100


# ------------------------------------------------------------------------- main
def _setup_main(tmp_path, monkeypatch, failed):
    manifest = write_manifest(
        tmp_path / "s.csv",
        ["a,A,P,https://x.test/a.html,html,", "b,B,P,https://x.test/b.html,html,"],
    )
    raw = tmp_path / "raw"

    def fake_download_sources(sources, raw_dir, **kwargs):
        raw_dir.mkdir(parents=True, exist_ok=True)
        ok = []
        for s in sources:
            if s.id in failed:
                continue
            target_path(raw_dir, s).write_text("<html><body><p>hola mundo</p></body></html>")
            ok.append(s.id)
        return DownloadSummary(downloaded=ok, skipped=[], failed=dict(failed))

    monkeypatch.setattr(corpus, "download_sources", fake_download_sources)
    return ["--manifest", str(manifest), "--raw-dir", str(raw)]


def test_main_returns_nonzero_when_a_download_failed(tmp_path, monkeypatch):
    argv = _setup_main(tmp_path, monkeypatch, failed={"b": "HTTP 404"})
    assert corpus.main(argv) != 0


def test_main_returns_zero_when_everything_succeeds(tmp_path, monkeypatch):
    argv = _setup_main(tmp_path, monkeypatch, failed={})
    assert corpus.main(argv) == 0
