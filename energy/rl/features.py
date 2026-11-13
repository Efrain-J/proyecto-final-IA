"""State features for the battery MDP (D14): relative price, hour block, discretizer.

Pure numpy/pandas helpers, no CLI. The relative price is computed once over the whole
series, before splitting, so each split gets its warm-up from the hours just before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRAILING_HOURS = 168
N_HOUR_BLOCKS = 12
N_PRICE_LEVELS = 4


def add_relative_price(frame: pd.DataFrame, window: int = TRAILING_HOURS) -> pd.DataFrame:
    """Add ``trailing_mean`` and ``rel_price`` (price / trailing mean) columns.

    The trailing mean at hour ``t`` is the mean of ``price[max(0, t - window):t]``: past
    hours only, so there is no look-ahead. Row 0 has no past and uses its own price.

    Args:
        frame: Frame with ``price_cop_kwh``, in time order.
        window: Trailing window in hours.

    Returns:
        A copy of ``frame`` with the two new columns.
    """
    price = frame["price_cop_kwh"].to_numpy(dtype=float)
    n = len(price)
    prefix = np.concatenate(([0.0], np.cumsum(price)))
    t = np.arange(n)
    lo = np.maximum(0, t - window)
    mean = (prefix[t] - prefix[lo]) / np.maximum(t - lo, 1)
    if n:
        mean[0] = price[0]
    out = frame.copy()
    out["trailing_mean"] = mean
    out["rel_price"] = price / mean
    return out


def hour_block(timestamps: pd.Series | pd.Index) -> np.ndarray:
    """Hour of day in pairs (0-1 -> 0, ..., 22-23 -> 11), as int8."""
    return (pd.DatetimeIndex(timestamps).hour.to_numpy() // 2).astype(np.int8)


@dataclass(frozen=True)
class Discretizer:
    """Maps a relative price to one of 4 quartile bins (D14)."""

    edges: tuple[float, float, float]

    @classmethod
    def fit(cls, rel_prices: np.ndarray) -> Discretizer:
        """Fit the 25th/50th/75th percentile edges on (training) relative prices."""
        q1, q2, q3 = np.percentile(np.asarray(rel_prices, dtype=float), [25, 50, 75])
        return cls(edges=(float(q1), float(q2), float(q3)))

    def transform(self, rel_prices: np.ndarray) -> np.ndarray:
        """Bin values into 0..3 (int8); a value on an edge goes to the upper bin."""
        bins = np.searchsorted(np.asarray(self.edges), np.asarray(rel_prices), side="right")
        return bins.astype(np.int8)
