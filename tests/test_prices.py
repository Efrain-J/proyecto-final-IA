"""Tests for `energy.data.prices`: loading the hourly price CSV and the scenario split.

Offline: CSVs are written to `tmp_path`; split frames are built in memory.
Run with `uv run pytest tests/test_prices.py`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from energy.data.prices import (
    DEFAULT_PRICES_PATH,
    TRAIN_LAST_MONTH,
    Split,
    load_prices,
    scenario_split,
)


def _write_csv(path: Path, timestamps, prices) -> Path:
    lines = ["timestamp,price_cop_kwh"] + [f"{t},{p}" for t, p in zip(timestamps, prices)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _hourly(start: str, end: str) -> pd.DataFrame:
    ts = pd.date_range(start, end, freq="h")
    return pd.DataFrame({"timestamp": ts, "price_cop_kwh": np.arange(len(ts), dtype=float)})


def test_constants():
    assert TRAIN_LAST_MONTH == 9
    assert Path(DEFAULT_PRICES_PATH) == Path("data/processed/prices_hourly.csv")


def test_load_prices_columns_and_dtypes(tmp_path):
    ts = pd.date_range("2025-01-01", periods=5, freq="h").strftime("%Y-%m-%d %H:%M:%S")
    path = _write_csv(tmp_path / "p.csv", ts, [100, 110.5, 120, 130, 140])
    frame = load_prices(path)
    assert list(frame.columns) == ["timestamp", "price_cop_kwh"]
    assert pd.api.types.is_datetime64_any_dtype(frame["timestamp"])
    assert pd.api.types.is_float_dtype(frame["price_cop_kwh"])
    assert len(frame) == 5
    assert frame["price_cop_kwh"].iloc[1] == pytest.approx(110.5)


def test_load_prices_sorts_and_resets_index(tmp_path):
    ts = list(pd.date_range("2025-01-01", periods=4, freq="h").strftime("%Y-%m-%d %H:%M:%S"))
    order = [2, 0, 3, 1]
    path = _write_csv(tmp_path / "p.csv", [ts[i] for i in order], [i * 10.0 for i in order])
    frame = load_prices(path)
    assert frame["timestamp"].is_monotonic_increasing
    assert list(frame.index) == [0, 1, 2, 3]
    # Each price must stay attached to its own timestamp after sorting.
    assert frame["timestamp"].iloc[0] == pd.Timestamp(ts[0])
    assert frame["price_cop_kwh"].tolist() == [0.0, 10.0, 20.0, 30.0]


def test_load_prices_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_prices(tmp_path / "nope.csv")


def test_load_prices_missing_column(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("timestamp,other\n2025-01-01 00:00:00,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prices(path)


def test_scenario_split_boundaries():
    frame = _hourly("2025-01-01 00:00", "2025-12-31 23:00")
    split = scenario_split(frame, 2025)
    assert isinstance(split, Split)
    assert split.train["timestamp"].iloc[0] == pd.Timestamp("2025-01-01 00:00")
    assert split.train["timestamp"].iloc[-1] == pd.Timestamp("2025-09-30 23:00")
    assert split.test["timestamp"].iloc[0] == pd.Timestamp("2025-10-01 00:00")
    assert split.test["timestamp"].iloc[-1] == pd.Timestamp("2025-12-31 23:00")


def test_scenario_split_sizes_and_no_overlap():
    frame = _hourly("2025-01-01 00:00", "2025-12-31 23:00")
    split = scenario_split(frame, 2025)
    assert len(split.train) == 273 * 24
    assert len(split.test) == 92 * 24
    assert split.train["timestamp"].max() < split.test["timestamp"].min()


def test_scenario_split_resets_indexes():
    frame = _hourly("2025-01-01 00:00", "2025-12-31 23:00")
    split = scenario_split(frame, 2025)
    assert list(split.train.index[:3]) == [0, 1, 2]
    assert split.train.index[-1] == len(split.train) - 1
    assert split.test.index[0] == 0
    assert split.test.index[-1] == len(split.test) - 1


def test_scenario_split_picks_requested_year_only():
    frame = _hourly("2024-01-01 00:00", "2025-12-31 23:00")
    split = scenario_split(frame, 2024)
    assert split.train["timestamp"].iloc[0] == pd.Timestamp("2024-01-01 00:00")
    assert split.test["timestamp"].iloc[-1] == pd.Timestamp("2024-12-31 23:00")
    assert (split.train["timestamp"].dt.year == 2024).all()
    assert (split.test["timestamp"].dt.year == 2024).all()


def test_scenario_split_absent_year():
    frame = _hourly("2025-01-01 00:00", "2025-12-31 23:00")
    with pytest.raises(ValueError):
        scenario_split(frame, 2023)
