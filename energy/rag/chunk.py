"""Split the clean corpus into chunks that carry their citation metadata.

CREG documents are first split on their ``## ARTÍCULO n.`` headings (D11), so a
chunk never mixes two articles and every piece knows which article it came
from. Long articles are then split by size with a small overlap. Documents
without article headings (XM) are split directly. See D13.

Run from the repo root, after ``energy.rag.clean``::

    uv run python -m energy.rag.chunk [--size 900] [--overlap 120]

Inputs:
    ``data/corpus/sources.csv`` and ``data/corpus/clean/<id>.md``.

Outputs:
    ``data/corpus/chunks.jsonl`` (committed): one JSON ``Chunk`` per line.
    The step is deterministic, so rerunning gives a byte-identical file.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

from energy.data.corpus import Source, load_manifest
from energy.data.io import write_bytes_atomic

CHUNK_SIZE = 900
CHUNK_OVERLAP = 120
MIN_CHUNK_CHARS = 30
DEFAULT_CHUNKS_PATH = Path("data/corpus/chunks.jsonl")

# Same headings that ``energy.rag.clean.mark_articles`` / ``mark_annexes`` write
# (D11). Annexes get their own sections; otherwise annex text would be cited
# as the last article (D13).
_ARTICLE_HEADING = re.compile(
    r"(?m)^## (?:ART[ÍI]CULO\s+(\d+)|ANEXO(?:\s+([0-9]+|[IVX]+))?\s*\.)"
)


def _section_label(match: re.Match[str]) -> str:
    """``"12"`` for ``## ARTÍCULO 12``, ``"Anexo 1"`` / ``"Anexo"`` for annexes."""
    if match.group(1):
        return match.group(1)
    return f"Anexo {match.group(2)}" if match.group(2) else "Anexo"
_SEPARATORS = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]


@dataclass(frozen=True)
class Chunk:
    """A retrievable piece of a document, with what a citation needs.

    Attributes:
        doc_id: Manifest id of the source document.
        title: Full document title.
        publisher: Issuing body (``CREG``, ``XM``).
        article: Article number as a string (``"12"``), an annex label
            (``"Anexo 1"``), or ``None`` if the document has no article
            structure or the text precedes article 1.
        in_force: ``False`` if the rule was replaced.
        superseded_by: Name of the replacing rule, if any.
        chunk_index: Position of the chunk within its document, from 0.
        text: Chunk text, as displayed to the user and the LLM.
    """

    doc_id: str
    title: str
    publisher: str
    article: str | None
    in_force: bool
    superseded_by: str
    chunk_index: int
    text: str


def split_articles(md: str) -> list[tuple[str | None, str]]:
    """Split Markdown on ``## ARTÍCULO n`` and ``## ANEXO n.`` headings.

    The heading line stays in its section's text because it carries the
    first paragraph of the article. Annex sections are labeled ``"Anexo n"``.

    Args:
        md: Clean Markdown of one document.

    Returns:
        ``(article, text)`` pairs in document order. Text before the first
        heading becomes ``(None, text)`` if it is not empty. A document
        without headings yields a single ``(None, md)`` (or nothing if empty).
    """
    matches = list(_ARTICLE_HEADING.finditer(md))
    if not matches:
        text = md.strip()
        return [(None, text)] if text else []
    sections: list[tuple[str | None, str]] = []
    preamble = md[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md)
        sections.append((_section_label(match), md[match.start() : end].strip()))
    return sections


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split ``text`` into pieces of about ``size`` characters.

    Args:
        text: Text to split.
        size: Target maximum characters per piece.
        overlap: Characters shared by consecutive pieces.

    Returns:
        Stripped pieces, without those shorter than ``MIN_CHUNK_CHARS``
        (headings or page residue that would only add retrieval noise).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, separators=_SEPARATORS
    )
    pieces = (p.strip() for p in splitter.split_text(text))
    return [p for p in pieces if len(p) >= MIN_CHUNK_CHARS]


def chunk_document(
    source: Source, md: str, *, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[Chunk]:
    """Chunk one document, copying citation metadata from its manifest entry.

    Args:
        source: Manifest entry (title, publisher, ``in_force``, ``superseded_by``).
        md: Clean Markdown of the document.
        size: Target maximum characters per chunk.
        overlap: Characters shared by consecutive chunks.

    Returns:
        Chunks with ``chunk_index`` sequential from 0. Every piece of an
        article keeps that article's number.
    """
    chunks: list[Chunk] = []
    for article, section in split_articles(md):
        for piece in split_text(section, size, overlap):
            chunks.append(
                Chunk(
                    doc_id=source.id,
                    title=source.title,
                    publisher=source.publisher,
                    article=article,
                    in_force=source.in_force,
                    superseded_by=source.superseded_by,
                    chunk_index=len(chunks),
                    text=piece,
                )
            )
    return chunks


def embedding_text(chunk: Chunk) -> str:
    """Text that is embedded (and BM25-indexed) for ``chunk``.

    Prefixing the title and article means a continuation piece of a long
    article is still findable by "Resolución CREG 131, artículo 2", while
    ``chunk.text`` stays clean for display (D13).
    """
    header = chunk.title
    if chunk.article:
        header += f" — {section_name(chunk.article)}"
    return f"{header}\n\n{chunk.text}"


def section_name(article: str, *, short: bool = False) -> str:
    """Display name of a section label: ``"Artículo 12"`` / ``"Art. 12"`` / ``"Anexo 1"``."""
    if article.startswith("Anexo"):
        return article
    return f"Art. {article}" if short else f"Artículo {article}"


def chunk_corpus(
    sources: list[Source],
    clean_dir: Path,
    *,
    size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> tuple[list[Chunk], dict[str, str]]:
    """Chunk every source; failures are recorded and processing continues.

    Args:
        sources: Documents from the manifest.
        clean_dir: Folder with ``<id>.md`` files.
        size: Target maximum characters per chunk.
        overlap: Characters shared by consecutive chunks.

    Returns:
        All chunks in manifest order, and a dict mapping the id of each
        missing or failed document to its error message.
    """
    chunks: list[Chunk] = []
    failures: dict[str, str] = {}
    for source in sources:
        try:
            md = (Path(clean_dir) / f"{source.id}.md").read_text(encoding="utf-8")
            chunks.extend(chunk_document(source, md, size=size, overlap=overlap))
        except Exception as exc:  # per-document isolation: keep going
            failures[source.id] = f"{type(exc).__name__}: {exc}"
    return chunks, failures


def write_chunks(chunks: list[Chunk], path: Path) -> None:
    """Write ``chunks`` as UTF-8 JSONL (one object per line), atomically."""
    lines = (json.dumps(asdict(c), ensure_ascii=False) for c in chunks)
    body = "".join(line + "\n" for line in lines)
    write_bytes_atomic(Path(path), body.encode("utf-8"))


def read_chunks(path: Path) -> list[Chunk]:
    """Read chunks written by ``write_chunks``."""
    text = Path(path).read_text(encoding="utf-8")
    return [Chunk(**json.loads(line)) for line in text.splitlines() if line.strip()]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: chunk the clean corpus and print per-document stats.

    Returns:
        ``0`` if every document was chunked, ``1`` if any failed.
    """
    parser = argparse.ArgumentParser(description="Chunk the clean RAG corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/sources.csv"))
    parser.add_argument("--clean-dir", type=Path, default=Path("data/corpus/clean"))
    parser.add_argument("--out", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args(argv)

    sources = load_manifest(args.manifest)
    chunks, failures = chunk_corpus(sources, args.clean_dir, size=args.size, overlap=args.overlap)
    write_chunks(chunks, args.out)

    for source in sources:
        count = sum(1 for c in chunks if c.doc_id == source.id)
        print(f"{source.id}: {count} chunks")
    for sid, msg in failures.items():
        print(f"{sid}: FAILED ({msg})")
    if chunks:
        sizes = [len(c.text) for c in chunks]
        print(
            f"total: {len(chunks)} chunks; chars min/median/max = "
            f"{min(sizes)}/{statistics.median(sizes):.0f}/{max(sizes)}"
        )
        creg = [c for c in chunks if c.publisher == "CREG"]
        if creg:
            share = sum(1 for c in creg if c.article) / len(creg)
            print(f"CREG chunks with article: {share:.1%} of {len(creg)}")
    print(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
