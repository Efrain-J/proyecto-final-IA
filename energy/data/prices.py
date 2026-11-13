"""Hourly bolsa prices for Part 1: loading the CSV and the within-year train/test split.

Reads ``data/processed/prices_hourly.csv`` (written by ``energy.data.xm_prices``).
Nothing is downloaded here. Used as a library by ``energy.rl``; there is no CLI.
The split follows D5: within each year, January-September train, October-December test.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_PRICES_PATH = Path("data/processed/prices_hourly.csv")
TRAIN_LAST_MONTH = 9  # D5: train Jan-Sep, test Oct-Dec
_COLUMNS = ["timestamp", "price_cop_kwh"]


@dataclass(frozen=True)
class Split:
    """Train and test parts of one calendar year (both with a reset index)."""

    train: pd.DataFrame
    test: pd.DataFrame


def load_prices(path: Path | str = DEFAULT_PRICES_PATH) -> pd.DataFrame:
    """Load the hourly price CSV.

    Args:
        path: CSV with ``timestamp`` and ``price_cop_kwh`` (COP/kWh) columns.

    Returns:
        Frame with those two columns, sorted by time, with a fresh integer index.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If a required column is missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    header = pd.read_csv(path, nrows=0).columns
    missing = [col for col in _COLUMNS if col not in header]
    if missing:
        raise ValueError(f"{path} is missing the columns {missing}")
    frame = pd.read_csv(path, usecols=_COLUMNS, parse_dates=["timestamp"])[_COLUMNS]
    frame["price_cop_kwh"] = frame["price_cop_kwh"].astype(float)
    return frame.sort_values("timestamp").reset_index(drop=True)


def scenario_split(frame: pd.DataFrame, year: int) -> Split:
    """Cut one calendar year into train (Jan-Sep) and test (Oct-Dec), see D5.

    Args:
        frame: Hourly prices; may span several years.
        year: Calendar year of the scenario.

    Returns:
        The two parts, each with a reset index.

    Raises:
        ValueError: If ``year`` has no rows in ``frame``.
    """
    in_year = frame[frame["timestamp"].dt.year == year]
    if in_year.empty:
        raise ValueError(f"No price data for {year}")
    is_train = in_year["timestamp"].dt.month <= TRAIN_LAST_MONTH
    return Split(
        train=in_year[is_train].reset_index(drop=True),
        test=in_year[~is_train].reset_index(drop=True),
    )
