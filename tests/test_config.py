"""Tests for energy.config: root-anchored paths and opt-in .env loading (O8, D12)."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pytest

from energy import config
from energy.data.prices import load_prices
from energy.rag.chunk import DEFAULT_CHUNKS_PATH, read_chunks

PATH_NAMES = (
    "DATA_DIR", "RAW_XM_DIR", "PRICES_CSV", "CORPUS_MANIFEST", "CORPUS_RAW_DIR",
    "CORPUS_CLEAN_DIR", "CHUNKS_PATH", "QDRANT_DIR", "POLICIES_DIR", "RESULTS_DIR",
    "ENV_FILE",
)


def test_root_contains_project_markers():
    assert (config.ROOT / "pyproject.toml").is_file()
    assert (config.ROOT / "energy").is_dir()


@pytest.mark.parametrize("name", PATH_NAMES)
def test_paths_are_absolute_and_under_root(name):
    path = getattr(config, name)
    assert path.is_absolute()
    assert path.is_relative_to(config.ROOT)


def test_committed_data_files_exist():
    assert config.PRICES_CSV.is_file()
    assert config.CHUNKS_PATH.is_file()


def test_load_env_sets_missing_variable(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CFG_TEST_NEW=from_file\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.delenv("CFG_TEST_NEW", raising=False)
    assert config.load_env() is True
    assert os.environ["CFG_TEST_NEW"] == "from_file"
    monkeypatch.delenv("CFG_TEST_NEW")  # loaded outside monkeypatch's tracking


def test_load_env_does_not_override_existing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("CFG_TEST_KEEP=from_file\n", encoding="utf-8")
    monkeypatch.setattr(config, "ENV_FILE", env)
    monkeypatch.setenv("CFG_TEST_KEEP", "from_env")
    assert config.load_env() is True
    assert os.environ["CFG_TEST_KEEP"] == "from_env"


def test_load_env_missing_file_returns_false(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "ENV_FILE", tmp_path / "absent.env")
    assert config.load_env() is False


def test_import_does_not_touch_environ():
    before = dict(os.environ)
    importlib.reload(config)
    assert dict(os.environ) == before


def test_data_loaders_work_from_another_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert not load_prices().empty
    assert read_chunks(DEFAULT_CHUNKS_PATH)
