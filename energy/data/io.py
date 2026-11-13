"""Atomic file-writing helpers shared by the download and cleaning steps.

Every generated file in ``data/`` is written through these functions, so an
interrupted run never leaves a half-written file behind. Resumable downloads
rely on this: a file that exists is complete, so it can be skipped.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` atomically.

    The bytes go to a temporary sibling (``<name>.tmp``), which then replaces
    ``path`` with ``os.replace``. Parent directories are created as needed.

    Args:
        path: Destination file. An existing file is overwritten.
        data: Raw bytes to write.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Serialize ``obj`` as UTF-8 JSON and write it atomically to ``path``.

    Non-ASCII characters are kept as-is (``ensure_ascii=False``) so Spanish
    text stays readable in the file.

    Args:
        path: Destination file.
        obj: Any JSON-serializable object.
    """
    write_bytes_atomic(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"))
