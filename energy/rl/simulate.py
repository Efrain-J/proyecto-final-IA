"""Replay a dispatch policy over a date range and report the savings (D14, D15, A2, A3, A5).

Library, used by the training report, the agent tool and the dashboard. The plant's
total load is unknown (A2), so savings are in COP only: ``-grid_kwh * price``. The
battery starts empty; impossible actions (including discharging while the factory is
off) become idle and are counted. Prices without shift columns get a seeded random
shift schedule over the whole series (A5).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from energy.data.prices import load_prices
from energy.rl.battery import BatteryPreset, get_preset, transition_tables, valid_mask
from energy.rl.factory import TEST_SHIFT_SEEDS, ShiftMix, add_shifts, get_mix
from energy.rl.features import add_relative_price
from energy.rl.policies import (
    NightDay,
    NoBattery,
    PerfectForesight,
    Policy,
    PriceThreshold,
    ShiftSchedule,
    factory_columns,
    threshold_path,
)
from energy.rl.qlearning import TabularPolicy, policy_stem

logger = logging.getLogger(__name__)

DEFAULT_POLICIES_DIR = Path("artifacts/policies")
POLICY_IDS = (
    "no_battery",
    "night_day",
    "shift_schedule",
    "price_threshold",
    "perfect_foresight",
    "q_learning",
)


@dataclass(frozen=True)
class SimulationSummary:
    """Totals of one simulation. ``cycles`` = delivered / (capacity * eta)."""

    policy: str
    preset_id: str
    start: date
    end: date
    hours: int
    savings_cop: float
    bought_kwh: float
    delivered_kwh: float
    cycles: float
    invalid_actions: int
    shift_mix: str | None = None  # None when the prices already had shift columns
    shift_seed: int | None = None


@dataclass
class SimulationResult:
    """Summary plus the hourly trace (timestamp, price, action, soc, grid, savings)."""

    summary: SimulationSummary
    hourly: pd.DataFrame


def resolve_policy(
    policy_id: str, preset_id: str, year: int, policies_dir: Path | str = DEFAULT_POLICIES_DIR
) -> Policy:
    """Build a policy from its id.

    A missing Q-table or threshold file logs a warning and falls back to
    ``ShiftSchedule`` (architecture fallback matrix), so the app keeps running.

    Raises:
        ValueError: If ``policy_id`` is unknown.
    """
    if policy_id == "no_battery":
        return NoBattery()
    if policy_id == "night_day":
        return NightDay()
    if policy_id == "shift_schedule":
        return ShiftSchedule()
    if policy_id == "perfect_foresight":
        return PerfectForesight()
    if policy_id == "price_threshold":
        path = threshold_path(policies_dir, preset_id, year)
        try:
            return PriceThreshold.load(path)
        except (OSError, KeyError, ValueError) as exc:  # JSONDecodeError is a ValueError
            logger.warning("No tuned thresholds at %s (%s); falling back to shift_schedule", path, exc)
            return ShiftSchedule()
    if policy_id == "q_learning":
        stem = policy_stem(policies_dir, preset_id, year)
        try:
            return TabularPolicy.load(stem)
        except (FileNotFoundError, OSError, KeyError, ValueError) as exc:
            logger.warning("No Q-table at %s (%s); falling back to shift_schedule", stem, exc)
            return ShiftSchedule()
    raise ValueError(f"Unknown policy {policy_id!r}; valid: {list(POLICY_IDS)}")


def simulate(
    policy: Policy | str,
    start: date,
    end: date,
    preset: BatteryPreset | str,
    prices: pd.DataFrame | None = None,
    policies_dir: Path | str = DEFAULT_POLICIES_DIR,
    mix: str | ShiftMix = "default",
    shift_seed: int = TEST_SHIFT_SEEDS[0],
) -> SimulationResult:
    """Replay ``policy`` from ``start`` 00:00 to ``end`` 23:00 (inclusive dates).

    Args:
        policy: A policy object or an id from ``POLICY_IDS`` (resolved with
            ``year = start.year``).
        start: First day.
        end: Last day.
        preset: Battery (or its id).
        prices: Hourly prices; loaded from the default CSV if None. ``rel_price`` is added
            over the whole frame if missing, so the slice has its warm-up history.
        policies_dir: Where saved Q-tables and thresholds live.
        mix: Shift mix (id or object) used if ``prices`` has no shift columns (A5).
        shift_seed: Seed of that shift pattern; the default is the first test pattern.

    Returns:
        Summary and hourly trace.

    Raises:
        ValueError: If ``start > end``, any part of the range is outside the data, or
            the policy id or the shift mix is unknown.
    """
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    preset = get_preset(preset)
    if prices is None:
        prices = load_prices()
    if "rel_price" not in prices.columns:
        prices = add_relative_price(prices)
    added_shifts = not {"shift_type", "factory_on"} <= set(prices.columns)
    if added_shifts:  # over the whole series, so a day's shift does not depend on the slice
        prices = add_shifts(prices, mix, shift_seed)
    if isinstance(policy, str):
        policy = resolve_policy(policy, preset.id, start.year, policies_dir)

    first, last = pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23)
    ts = prices["timestamp"]
    if ts.empty or first < ts.min() or last > ts.max():
        raise ValueError(f"{start}..{end} is outside the price data")
    frame = prices[(ts >= first) & (ts <= last)].reset_index(drop=True)

    requested = np.asarray(policy.actions(frame, preset), dtype=np.int64)
    next_level, grid_table = transition_tables(preset)
    masks = {on: valid_mask(preset, on) for on in (False, True)}
    shift_type, factory_on = factory_columns(frame)
    n = len(frame)
    actions = np.zeros(n, dtype=np.int8)
    soc = np.zeros(n, dtype=np.int8)
    grid = np.zeros(n)
    level, invalid = 0, 0
    for t in range(n):
        a = int(requested[t])
        if not masks[bool(factory_on[t])][level, a]:
            invalid += 1
            a = 0
        actions[t] = a
        grid[t] = grid_table[level, a]
        level = int(next_level[level, a])
        soc[t] = level

    price = frame["price_cop_kwh"].to_numpy(dtype=float)
    savings = -grid * price
    hourly = pd.DataFrame(
        {
            "timestamp": frame["timestamp"],
            "price_cop_kwh": price,
            "action": actions,
            "soc_level": soc,
            "grid_kwh": grid,
            "savings_cop": savings,
            "cum_savings_cop": np.cumsum(savings),
            "shift_type": shift_type,
            "factory_on": factory_on,
        }
    )
    delivered = float(-grid[grid < 0].sum())
    summary = SimulationSummary(
        policy=policy.name,
        preset_id=preset.id,
        start=start,
        end=end,
        hours=n,
        savings_cop=float(savings.sum()),
        bought_kwh=float(grid[grid > 0].sum()),
        delivered_kwh=delivered,
        cycles=delivered / (preset.capacity_kwh * preset.one_way_efficiency),
        invalid_actions=invalid,
        shift_mix=get_mix(mix).id if added_shifts else None,
        shift_seed=shift_seed if added_shifts else None,
    )
    return SimulationResult(summary=summary, hourly=hourly)
