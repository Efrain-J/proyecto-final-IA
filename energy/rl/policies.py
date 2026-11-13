"""Dispatch policies: the ``Policy`` protocol and the non-learned ones (D14, D15).

``NoBattery`` and ``NightDay`` are simple baselines; ``ShiftSchedule`` and the tuned
``PriceThreshold`` are the rules Q-learning must beat (D15); ``PerfectForesight`` is the
non-causal upper bound. Each policy returns the whole action sequence for a frame in
one call (speed measure 7). A frame without ``shift_type``/``factory_on`` means the
factory always runs, which keeps the D14 behaviour. Library only, no CLI.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from energy.data.io import write_json_atomic
from energy.rl.battery import Action, BatteryPreset, get_preset, transition_tables, valid_mask
from energy.rl.factory import SHIFT_AFTERNOON, SHIFT_FULL, SHIFT_MORNING, SHIFT_OFF

# Quantile grid for ``PriceThreshold.tune`` (D15): (cheap quantiles, expensive quantiles).
# Steps of 5% include the 10%-spaced deciles the plan names.
DEFAULT_GRID: tuple[tuple[float, ...], tuple[float, ...]] = (
    tuple(float(q) for q in np.arange(0.10, 0.501, 0.05)),
    tuple(float(q) for q in np.arange(0.50, 0.901, 0.05)),
)


class Policy(Protocol):
    """A dispatch policy: one action per row of ``frame``."""

    name: str

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        """Return int8 actions, one per row.

        Args:
            frame: Has ``timestamp``, ``price_cop_kwh`` and ``rel_price``, and optionally
                ``shift_type`` and ``factory_on`` (absent means always on).
            preset: Battery being dispatched.
        """
        ...


def factory_columns(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(shift_type int8, factory_on bool)`` of ``frame``.

    Missing columns default to a factory that always runs (full-day shift), so frames
    from before D15 keep working.
    """
    n = len(frame)
    if "shift_type" in frame.columns:
        shift_type = frame["shift_type"].to_numpy(dtype=np.int8)
    else:
        shift_type = np.full(n, SHIFT_FULL, dtype=np.int8)
    if "factory_on" in frame.columns:
        factory_on = frame["factory_on"].to_numpy(dtype=np.bool_)
    else:
        factory_on = np.ones(n, dtype=np.bool_)
    return shift_type, factory_on


@dataclass(frozen=True)
class NoBattery:
    """Baseline: never use the battery."""

    name: str = "no_battery"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        return np.zeros(len(frame), dtype=np.int8)


@dataclass(frozen=True)
class NightDay:
    """Baseline: charge from ``charge_start`` and discharge from ``discharge_start``.

    Each window lasts ``duration_h`` hours. It ignores the shifts and may issue
    impossible actions (e.g. discharging while the factory is off); ``simulate`` turns
    those into idle.
    """

    charge_start: int = 2
    discharge_start: int = 18
    name: str = "night_day"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        hours = pd.DatetimeIndex(frame["timestamp"]).hour.to_numpy()
        out = np.zeros(len(frame), dtype=np.int8)
        for i in range(preset.duration_h):
            out[hours == (self.charge_start + i) % 24] = Action.CHARGE
            out[hours == (self.discharge_start + i) % 24] = Action.DISCHARGE
        return out


@dataclass(frozen=True)
class ShiftSchedule:
    """Baseline (D15): the fixed schedule adapted to the day's shift.

    Charges from 02:00 for ``duration_h`` hours on days the factory runs. Discharges from
    18:00 on afternoon and full days, and during the last ``duration_h`` hours of the
    morning shift (``14 - d .. 13``) on morning days. Off days do nothing. Where the
    windows overlap (long batteries) the discharge wins, and windows are cut at midnight.
    Without shift columns it is ``NightDay``.
    """

    name: str = "shift_schedule"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        hours = pd.DatetimeIndex(frame["timestamp"]).hour.to_numpy()
        shift_type, factory_on = factory_columns(frame)
        d = preset.duration_h
        evening = factory_on & np.isin(shift_type, (SHIFT_AFTERNOON, SHIFT_FULL))
        morning = factory_on & (shift_type == SHIFT_MORNING)
        out = np.zeros(len(frame), dtype=np.int8)
        for i in range(d):
            out[(hours == 2 + i) & (shift_type != SHIFT_OFF)] = Action.CHARGE
        for i in range(d):
            out[(hours == 18 + i) & evening] = Action.DISCHARGE
            out[(hours == 14 - d + i) & morning] = Action.DISCHARGE
        return out


def threshold_path(policies_dir: Path | str, preset_id: str, year: int) -> Path:
    """Path of a tuned threshold file: ``<dir>/threshold_<preset>_<year>.json``."""
    return Path(policies_dir) / f"threshold_{preset_id}_{year}.json"


@dataclass(frozen=True)
class PriceThreshold:
    """Rule (D15): charge when cheap, discharge when expensive and the factory runs.

    ``cheap`` and ``expensive`` are relative-price cut-offs (strict comparisons). The
    policy tracks the charge level itself, so it never issues an impossible action.
    """

    cheap: float
    expensive: float
    name: str = "price_threshold"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        rel = frame["rel_price"].to_numpy(dtype=float).tolist()
        factory_on = factory_columns(frame)[1].tolist()
        top = preset.duration_h
        out = [0] * len(rel)
        level = 0
        for t, price in enumerate(rel):
            if price < self.cheap and level < top:
                out[t] = Action.CHARGE
                level += 1
            elif price > self.expensive and factory_on[t] and level > 0:
                out[t] = Action.DISCHARGE
                level -= 1
        return np.asarray(out, dtype=np.int8)

    @classmethod
    def tune(
        cls,
        frames: list[pd.DataFrame],
        preset: BatteryPreset | str,
        grid: tuple[tuple[float, ...], tuple[float, ...]] = DEFAULT_GRID,
    ) -> PriceThreshold:
        """Pick the cut-offs that maximize the mean savings over ``frames``.

        Candidates are quantiles of the pooled ``rel_price`` (cheap quantile below the
        expensive one); the first best pair in grid order wins, so it is deterministic.

        Args:
            frames: Training frames with ``price_cop_kwh``, ``rel_price`` and the shift
                columns, each replayed from an empty battery.
            preset: Battery (or its id).
            grid: ``(cheap quantiles, expensive quantiles)``.

        Raises:
            ValueError: If the grid has no pair with expensive > cheap.
        """
        preset = get_preset(preset)
        pooled = np.concatenate([f["rel_price"].to_numpy(dtype=float) for f in frames])
        eta = preset.one_way_efficiency
        grid_by_action = np.array([0.0, preset.power_kw / eta, -preset.power_kw * eta])
        prices = [f["price_cop_kwh"].to_numpy(dtype=float) for f in frames]
        best_value, best = -np.inf, None
        for q_cheap in grid[0]:
            for q_expensive in grid[1]:
                if q_expensive <= q_cheap:
                    continue
                rule = cls(
                    float(np.quantile(pooled, q_cheap)), float(np.quantile(pooled, q_expensive))
                )
                # The rule only issues valid actions, so savings need no replay.
                value = float(
                    np.mean(
                        [
                            -(grid_by_action[rule.actions(f, preset)] * p).sum()
                            for f, p in zip(frames, prices)
                        ]
                    )
                )
                if value > best_value:
                    best_value, best = value, rule
        if best is None:
            raise ValueError("grid has no pair with expensive > cheap")
        return best

    def save(self, path: Path | str) -> None:
        """Write the cut-offs as ``{"cheap": ..., "expensive": ...}`` atomically."""
        write_json_atomic(Path(path), {"cheap": self.cheap, "expensive": self.expensive})

    @classmethod
    def load(cls, path: Path | str) -> PriceThreshold:
        """Read a file written by ``save``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(cheap=float(data["cheap"]), expensive=float(data["expensive"]))


@dataclass(frozen=True)
class PerfectForesight:
    """Upper bound: backward DP over hours x charge levels with the true prices.

    Not causal, so it is never a deployable policy. Maximizes
    ``sum(-grid_kwh * price)`` starting empty; ties go to idle. Discharging is masked
    out in the hours the factory is off (D15).
    """

    name: str = "perfect_foresight"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        price = frame["price_cop_kwh"].to_numpy(dtype=float)
        n = len(price)
        next_level, grid_kwh = transition_tables(preset)
        factory_on = factory_columns(frame)[1]
        invalid = {on: ~valid_mask(preset, on) for on in (False, True)}
        best = np.zeros((n, preset.n_levels), dtype=np.int8)
        value = np.zeros(preset.n_levels)  # value after the last hour
        for t in range(n - 1, -1, -1):
            q = -grid_kwh * price[t] + value[next_level]  # [levels, actions]
            q[invalid[bool(factory_on[t])]] = -np.inf
            best[t] = np.argmax(q, axis=1)  # first max wins, and IDLE is index 0
            value = q[np.arange(preset.n_levels), best[t]]
        out = np.zeros(n, dtype=np.int8)
        level = 0
        for t in range(n):
            out[t] = best[t, level]
            level = int(next_level[level, out[t]])
        return out
