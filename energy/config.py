"""Repo-root-anchored paths and shared settings (O8, D12).

Every path is absolute, anchored at the repo root (the folder holding
``pyproject.toml``), so any entry point works from any working directory.
Settings that more than one module needs live here so they are defined once.

Importing this module has no side effects: it reads no files and does not touch
``os.environ``. Entry points (``main()`` functions, ``app.py``) call
``load_env()`` explicitly to read ``.env``.

Imports nothing from ``energy`` (the dependency root, see docs/architecture.md)
and uses only the standard library; ``dotenv`` is imported inside ``load_env``.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_XM_DIR = DATA_DIR / "raw" / "xm"
PRICES_CSV = DATA_DIR / "processed" / "prices_hourly.csv"
CORPUS_MANIFEST = DATA_DIR / "corpus" / "sources.csv"
CORPUS_RAW_DIR = DATA_DIR / "corpus" / "raw"
CORPUS_CLEAN_DIR = DATA_DIR / "corpus" / "clean"
CHUNKS_PATH = DATA_DIR / "corpus" / "chunks.jsonl"
QDRANT_DIR = DATA_DIR / "qdrant"
POLICIES_DIR = ROOT / "artifacts" / "policies"
RESULTS_DIR = ROOT / "results"
ENV_FILE = ROOT / ".env"

USER_AGENT = "trabajo-final-ia/0.1 (academic project)"
MIN_TEXT_CHARS = 1000  # below this a downloaded/cleaned document is suspicious
EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
QDRANT_COLLECTION = "regulacion"
RETRIEVAL_K = 5
SCORE_THRESHOLD = 0.25  # tuned for the nemotron embedder

# LLM agent (D16)
GROQ_MODELS = ("openai/gpt-oss-120b", "openai/gpt-oss-20b")  # tried in order
T_MAX = 5  # tool rounds before the agent must answer
HISTORY_MESSAGES = 6  # earlier chat messages kept in the prompt
LLM_TIMEOUT_S = 30


def load_env() -> bool:
    """Load ``ENV_FILE`` into ``os.environ`` without overriding existing variables.

    Call only from entry points, never at import time, so tests never see real keys.

    Returns:
        True if the file existed, False otherwise.
    """
    from dotenv import load_dotenv

    if not ENV_FILE.is_file():
        return False
    load_dotenv(ENV_FILE, override=False)
    return True
