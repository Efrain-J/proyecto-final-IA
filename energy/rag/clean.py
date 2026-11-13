"""Turn the raw corpus (HTML/PDF) into clean Markdown ready for chunking.

For HTML: keep only the main content node (the manifest's ``selector``, which
drops the site menus), convert it to Markdown with markitdown, reduce links to
their text, normalize whitespace, and mark CREG article headings. PDFs skip
the selector step. See D11 for the reasoning.

Run from the repo root, after adding or changing a document::

    uv run python -m energy.rag.clean

Inputs:
    ``data/corpus/sources.csv`` and ``data/corpus/raw/<id>.<format>``.

Outputs:
    ``data/corpus/clean/<id>.md`` (committed, so anyone can read exactly what
    the agent can retrieve). Always regenerated: the step is cheap and deterministic.
"""

from __future__ import annotations

import argparse
import io
import re
from dataclasses import dataclass
from pathlib import Path

from bs4 import BeautifulSoup

from energy.data.corpus import Source, load_manifest, target_path
from energy.data.io import write_bytes_atomic

MIN_CHARS = 1000

_ZERO_WIDTH = re.compile("[\u200b\u200c\u200d\ufeff]")
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_ARTICLE = re.compile(
    r"^[ \t]*(?:[#*_]+[ \t]*)?"
    r"((?:ART[ÍI]CULO)\s+(\d+)\s*[o°º]?\s*\.)",
    re.MULTILINE,
)
_HEADING = re.compile(r"^## ART[ÍI]CULO", re.MULTILINE)


class CleaningError(ValueError):
    """Raised when a document cannot be cleaned."""


@dataclass(frozen=True)
class CleanResult:
    """Outcome of cleaning one document.

    Attributes:
        id: Manifest id of the document.
        chars: Length of the clean Markdown text.
        articles: Number of ``## ARTÍCULO`` headings marked (0 for non-CREG).
    """

    id: str
    chars: int
    articles: int


def extract_main_html(html: bytes, selector: str) -> str:
    """Return the HTML of the first node matching ``selector``.

    Fails loudly rather than falling back to the whole page, because the whole
    page would put site menus into the index.

    Args:
        html: Raw page bytes. BeautifulSoup detects the encoding; the CREG
            pages are ISO-8859-1.
        selector: CSS selector, e.g. ``div.panel-documento`` or ``#block-xm-content``.

    Returns:
        The matched node serialized as HTML (a Unicode string).

    Raises:
        CleaningError: If nothing matches ``selector``.
    """
    soup = BeautifulSoup(html, "html.parser")
    node = soup.select_one(selector)
    if node is None:
        raise CleaningError(f"Selector {selector!r} matched nothing")
    return str(node)


def to_markdown(data: bytes, extension: str) -> str:
    """Convert raw bytes to Markdown with markitdown.

    Args:
        data: Document bytes (UTF-8 for HTML coming from ``extract_main_html``).
        extension: ``".html"`` or ``".pdf"``; the leading dot is optional.

    Returns:
        The converted Markdown text.
    """
    from markitdown import MarkItDown, StreamInfo

    ext = extension if extension.startswith(".") else f".{extension}"
    result = MarkItDown().convert_stream(io.BytesIO(data), stream_info=StreamInfo(extension=ext))
    return result.text_content


def strip_links(md: str) -> str:
    """Drop images and replace ``[text](url)`` with ``text``.

    CREG pages link every cited law (``[142](ley_0142_1994.htm#0)``); the URLs
    are noise for retrieval, so only the visible text is kept.
    """
    return _LINK.sub(r"\1", _IMAGE.sub("", md))


def normalize(md: str) -> str:
    """Remove zero-width characters, trailing spaces and extra blank lines.

    Zero-width characters appear inside words on the XM pages ("frecu\\u200bentes")
    and would break keyword matching. Runs of 3+ newlines collapse to one
    blank line, and the whole text is stripped.
    """
    md = _ZERO_WIDTH.sub("", md)
    md = "\n".join(line.rstrip() for line in md.split("\n"))
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def mark_articles(md: str) -> str:
    """Turn lines starting with ``ARTÍCULO <n>.`` into ``## `` headings.

    Only headings whose number is higher than the previous marked one are
    marked. A resolution that amends another one quotes the amended article
    (e.g. art. 80 of CREG 101 113 rewrites "ARTÍCULO 6o." of CREG 055/1994);
    those quoted articles break the ascending order and must stay inside the
    amending article, so citations point to the right article number.
    """
    last = 0

    def _mark(match: re.Match[str]) -> str:
        nonlocal last
        number = int(match.group(2))
        if number <= last:
            return match.group(0)
        last = number
        return f"## {match.group(1)}"

    return _ARTICLE.sub(_mark, md)


def clean_document(source: Source, raw_path: Path) -> str:
    """Clean one raw document into Markdown text.

    Args:
        source: Manifest entry; ``format`` picks the pipeline and HTML needs a ``selector``.
        raw_path: The downloaded file.

    Returns:
        Clean Markdown. Article headings are marked for HTML (CREG) documents only.

    Raises:
        CleaningError: If an HTML source has no selector or the selector matches nothing.
    """
    data = Path(raw_path).read_bytes()
    if source.format == "html":
        if not source.selector:
            raise CleaningError(f"No selector for HTML source {source.id}")
        html = extract_main_html(data, source.selector)
        md = to_markdown(html.encode("utf-8"), ".html")
        return mark_articles(normalize(strip_links(md)))
    return normalize(strip_links(to_markdown(data, ".pdf")))


def clean_corpus(
    sources: list[Source], raw_dir: Path, clean_dir: Path
) -> tuple[list[CleanResult], dict[str, str]]:
    """Clean every source; failures are recorded and processing continues.

    Args:
        sources: Documents from the manifest.
        raw_dir: Folder with the downloaded files.
        clean_dir: Destination for ``<id>.md`` (written atomically, UTF-8).

    Returns:
        A ``CleanResult`` for each document written, and a dict mapping the id
        of each failed document to its error message.
    """
    results: list[CleanResult] = []
    failures: dict[str, str] = {}
    for source in sources:
        raw_path = target_path(raw_dir, source)
        try:
            if not raw_path.exists():
                raise FileNotFoundError(f"Raw file missing: {raw_path}")
            text = clean_document(source, raw_path)
            write_bytes_atomic(Path(clean_dir) / f"{source.id}.md", text.encode("utf-8"))
        except Exception as exc:  # per-document isolation: keep going
            failures[source.id] = f"{type(exc).__name__}: {exc}"
            continue
        results.append(CleanResult(source.id, len(text), len(_HEADING.findall(text))))
    return results, failures


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: regenerate the clean corpus and print per-document stats.

    Returns:
        ``0`` if every document was cleaned, ``1`` if any failed.
    """
    parser = argparse.ArgumentParser(description="Clean the RAG corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/sources.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/corpus/raw"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/corpus/clean"))
    args = parser.parse_args(argv)

    sources = load_manifest(args.manifest)
    results, failures = clean_corpus(sources, args.raw_dir, args.clean_dir)
    for r in results:
        warning = f"  WARNING: under {MIN_CHARS} chars" if r.chars < MIN_CHARS else ""
        print(f"{r.id}: {r.chars} chars, {r.articles} articles{warning}")
    for sid, msg in failures.items():
        print(f"{sid}: FAILED ({msg})")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
