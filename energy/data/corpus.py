"""Download the RAG corpus (CREG resolutions, XM documents) listed in a CSV manifest.

The manifest ``data/corpus/sources.csv`` is the single source of document
metadata: title, publisher and URL, plus the cleaning ``selector`` and the
``in_force`` / ``superseded_by`` fields that later become chunk metadata for
citations (see D6, D11). Adding a document means adding a row and rerunning.

Run from the repo root::

    uv run python -m energy.data.corpus [--force]

Outputs:
    ``data/corpus/raw/<id>.<format>``: each document as downloaded (existing
    files are skipped). Prints each document's converted text length as a
    sanity check: pages rendered by JavaScript come back nearly empty.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from energy.data.io import write_bytes_atomic

USER_AGENT = "trabajo-final-ia/0.1 (academic project)"
REQUIRED_COLUMNS = ("id", "title", "publisher", "url", "format")
VALID_FORMATS = {"html", "pdf"}
MIN_TEXT_CHARS = 1000


@dataclass(frozen=True)
class Source:
    """One document of the corpus, as described by a manifest row.

    Attributes:
        id: Short unique key; also the file stem in ``raw/`` and ``clean/``.
        title: Full document title, used in citations.
        publisher: Issuing body (``CREG``, ``XM``).
        url: Download URL.
        format: ``"html"`` or ``"pdf"``.
        selector: CSS selector of the main content, for HTML (see D11).
        in_force: ``False`` if the rule was replaced.
        superseded_by: Name of the replacing rule, if any.
    """

    id: str
    title: str
    publisher: str
    url: str
    format: str
    selector: str = ""
    in_force: bool = True
    superseded_by: str = ""


@dataclass
class DownloadSummary:
    """Outcome of ``download_sources``.

    Attributes:
        downloaded: Ids fetched in this run.
        skipped: Ids whose file already existed.
        failed: Id to last error message, for documents that failed every retry.
    """

    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


_TRUE = {"yes", "true"}
_FALSE = {"no", "false"}


def _parse_in_force(value: str, source_id: str) -> bool:
    """Parse a yes/no/true/false cell; empty means in force."""
    text = value.strip().lower()
    if not text or text in _TRUE:
        return True
    if text in _FALSE:
        return False
    raise ValueError(f"Invalid in_force {value!r} for {source_id}")


def load_manifest(path: Path) -> list[Source]:
    """Read and validate the manifest CSV.

    The columns ``id, title, publisher, url, format`` are required.
    ``selector``, ``in_force`` and ``superseded_by`` are optional and default
    to ``""``, ``True`` and ``""``. Other columns, such as ``notes``, are ignored.

    Args:
        path: Manifest CSV (UTF-8).

    Returns:
        One ``Source`` per row, in file order.

    Raises:
        ValueError: On a missing required column, duplicate id, unknown
            format, empty url, or an ``in_force`` value other than yes/no/true/false.
    """
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        missing = [c for c in REQUIRED_COLUMNS if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"Manifest is missing columns: {missing}")
        sources: list[Source] = []
        seen: set[str] = set()
        for row in reader:
            source = Source(
                *(row[c].strip() for c in REQUIRED_COLUMNS),
                selector=(row.get("selector") or "").strip(),
                in_force=_parse_in_force(row.get("in_force") or "", row["id"].strip()),
                superseded_by=(row.get("superseded_by") or "").strip(),
            )
            if source.id in seen:
                raise ValueError(f"Duplicate id in manifest: {source.id}")
            if source.format not in VALID_FORMATS:
                raise ValueError(f"Unknown format {source.format!r} for {source.id}")
            if not source.url:
                raise ValueError(f"Empty url for {source.id}")
            seen.add(source.id)
            sources.append(source)
    return sources


def target_path(raw_dir: Path, source: Source) -> Path:
    """Local path where ``source`` is stored: ``raw_dir/<id>.<format>``."""
    return Path(raw_dir) / f"{source.id}.{source.format}"


def download_sources(
    sources: list[Source],
    raw_dir: Path,
    *,
    force: bool = False,
    retries: int = 3,
    timeout: float = 60.0,
    delay: float = 1.0,
    backoff_base: float = 1.0,
    client: httpx.Client | None = None,
) -> DownloadSummary:
    """Download every source sequentially; failures are recorded, never raised.

    Sequential with a polite ``delay``, because the corpus is small and the
    sources are government sites. One failed document doesn't stop the rest.

    Args:
        sources: Documents to fetch, usually from ``load_manifest``.
        raw_dir: Destination folder.
        force: Re-download documents whose file already exists.
        retries: Attempts per document.
        timeout: Per-request timeout, in seconds (only for the default client).
        delay: Seconds to wait between downloads.
        backoff_base: Seconds before the first retry; doubles each attempt.
        client: Optional ``httpx.Client``; tests pass one with a ``MockTransport``.
            When omitted, a client with redirects and a generic User-Agent is
            created and closed here.

    Returns:
        Which ids were downloaded, skipped or failed.
    """
    summary = DownloadSummary()
    owns_client = client is None
    if client is None:
        client = httpx.Client(
            timeout=timeout, follow_redirects=True, headers={"User-Agent": USER_AGENT}
        )
    try:
        fetched_any = False
        for source in sources:
            path = target_path(raw_dir, source)
            if path.exists() and not force:
                summary.skipped.append(source.id)
                continue
            if fetched_any and delay > 0:
                time.sleep(delay)
            fetched_any = True
            error = "no attempts made"
            for attempt in range(retries):
                try:
                    response = client.get(source.url, follow_redirects=True)
                    response.raise_for_status()
                    write_bytes_atomic(path, response.content)
                    summary.downloaded.append(source.id)
                    break
                except httpx.HTTPError as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    if attempt < retries - 1 and backoff_base > 0:
                        time.sleep(backoff_base * 2**attempt)
            else:
                summary.failed[source.id] = error
    finally:
        if owns_client:
            client.close()
    return summary


def text_length(path: Path) -> int:
    """Character count of the markitdown conversion of ``path``."""
    from markitdown import MarkItDown

    return len(MarkItDown().convert(str(path)).text_content)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: download the corpus and report each document's text length.

    Returns:
        ``0`` if every document is available, ``1`` if any download failed.
    """
    parser = argparse.ArgumentParser(description="Download the RAG corpus.")
    parser.add_argument("--manifest", type=Path, default=Path("data/corpus/sources.csv"))
    parser.add_argument("--raw-dir", type=Path, default=Path("data/corpus/raw"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    sources = load_manifest(args.manifest)
    summary = download_sources(sources, args.raw_dir, force=args.force)
    print(f"downloaded: {summary.downloaded}")
    print(f"skipped: {summary.skipped}")
    print(f"failed: {summary.failed}")
    for source in sources:
        path = target_path(args.raw_dir, source)
        if not path.exists():
            continue
        try:
            length = text_length(path)
        except Exception as exc:  # markitdown raises many converter-specific errors
            print(f"{source.id}: conversion failed ({type(exc).__name__}: {exc})")
            continue
        warning = "  WARNING: under 1000 chars" if length < MIN_TEXT_CHARS else ""
        print(f"{source.id}: {length} chars{warning}")
    return 1 if summary.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
