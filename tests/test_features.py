"""Tests for `energy.rl.features`: relative price, hour blocks and the discretizer.

Offline and pure numpy/pandas. Run with `uv run pytest tests/test_features.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from energy.rl.features import (
    N_HOUR_BLOCKS,
    N_PRICE_LEVELS,
    TRAILING_HOURS,
    Discretizer,
    add_relative_price,
    hour_block,
)


@pytest.fixture
def random_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    ts = pd.date_range("2025-01-01", periods=400, freq="h")
    return pd.DataFrame({"timestamp": ts, "price_cop_kwh": rng.uniform(100, 400, len(ts))})


def test_constants():
    assert (TRAILING_HOURS, N_HOUR_BLOCKS, N_PRICE_LEVELS) == (168, 12, 4)


def test_add_relative_price_returns_copy_with_new_columns(random_prices):
    out = add_relative_price(random_prices)
    assert "trailing_mean" in out.columns and "rel_price" in out.columns
    assert "rel_price" not in random_prices.columns
    assert len(out) == len(random_prices)
    assert np.array_equal(out["price_cop_kwh"], random_prices["price_cop_kwh"])


def test_first_row_uses_own_price(random_prices):
    out = add_relative_price(random_prices)
    assert out["rel_price"].iloc[0] == pytest.approx(1.0)
    assert out["trailing_mean"].iloc[0] == pytest.approx(random_prices["price_cop_kwh"].iloc[0])


def test_trailing_mean_short_history(random_prices):
    p = random_prices["price_cop_kwh"].to_numpy()
    out = add_relative_price(random_prices)
    assert out["trailing_mean"].iloc[5] == pytest.approx(p[0:5].mean())
    assert out["rel_price"].iloc[5] == pytest.approx(p[5] / p[0:5].mean())


def test_trailing_mean_full_window_excludes_current_hour(random_prices):
    p = random_prices["price_cop_kwh"].to_numpy()
    out = add_relative_price(random_prices)
    assert out["trailing_mean"].iloc[200] == pytest.approx(p[32:200].mean())
    assert out["rel_price"].iloc[200] == pytest.approx(p[200] / p[32:200].mean())


def test_custom_window(random_prices):
    p = random_prices["price_cop_kwh"].to_numpy()
    out = add_relative_price(random_prices, window=24)
    assert out["trailing_mean"].iloc[30] == pytest.approx(p[6:30].mean())


def test_no_look_ahead(random_prices):
    base = add_relative_price(random_prices)
    changed = random_prices.copy()
    changed.loc[250, "price_cop_kwh"] *= 10
    out = add_relative_price(changed)
    assert np.array_equal(out["rel_price"].to_numpy()[:250], base["rel_price"].to_numpy()[:250])
    assert out["rel_price"].iloc[250] != pytest.approx(base["rel_price"].iloc[250])


@pytest.mark.parametrize(
    "hour,block", [(0, 0), (1, 0), (2, 1), (18, 9), (19, 9), (22, 11), (23, 11)]
)
def test_hour_block_values(hour, block):
    ts = pd.Series(pd.to_datetime([f"2025-03-04 {hour:02d}:00"]))
    assert hour_block(ts)[0] == block


def test_hour_block_dtype_and_range():
    ts = pd.Series(pd.date_range("2025-01-01", periods=48, freq="h"))
    blocks = hour_block(ts)
    assert blocks.dtype == np.int8
    assert len(blocks) == 48
    assert set(blocks.tolist()) == set(range(N_HOUR_BLOCKS))


def test_discretizer_fit_gives_quartiles():
    rng = np.random.default_rng(0)
    rel = rng.lognormal(0.0, 0.3, 4000)
    disc = Discretizer.fit(rel)
    assert len(disc.edges) == 3
    assert disc.edges == pytest.approx(tuple(np.percentile(rel, [25, 50, 75])))


def test_discretizer_bins_are_balanced_and_in_range():
    rng = np.random.default_rng(1)
    rel = rng.lognormal(0.0, 0.3, 4000)
    disc = Discretizer.fit(rel)
    bins = disc.transform(rel)
    assert bins.dtype == np.int8
    assert bins.min() >= 0 and bins.max() <= 3
    for level in range(4):
        assert np.mean(bins == level) == pytest.approx(0.25, abs=0.03)


def test_discretizer_edge_value_goes_to_upper_bin():
    disc = Discretizer(edges=(0.9, 1.0, 1.1))
    out = disc.transform(np.array([0.5, 0.9, 0.95, 1.0, 1.1, 5.0]))
    assert out.tolist() == [0, 1, 1, 2, 3, 3]
