"""Atomic file-writing helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Write ``data`` to ``path`` via a temporary sibling file and ``os.replace``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any) -> None:
    """Serialize ``obj`` as UTF-8 JSON and write it atomically to ``path``."""
    write_bytes_atomic(path, json.dumps(obj, ensure_ascii=False).encode("utf-8"))
