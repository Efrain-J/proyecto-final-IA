"""Dispatch policies: the ``Policy`` protocol and the three non-learned ones (D14).

``NoBattery`` and ``NightDay`` are the baselines; ``PerfectForesight`` is the
non-causal upper bound. Each policy returns the whole action sequence for a frame in
one call (speed measure 7). Library only, no CLI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from energy.rl.battery import Action, BatteryPreset, transition_tables, valid_mask


class Policy(Protocol):
    """A dispatch policy: one action per row of ``frame``."""

    name: str

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        """Return int8 actions, one per row.

        Args:
            frame: Has ``timestamp``, ``price_cop_kwh`` and ``rel_price``.
            preset: Battery being dispatched.
        """
        ...


@dataclass(frozen=True)
class NoBattery:
    """Baseline: never use the battery."""

    name: str = "no_battery"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        return np.zeros(len(frame), dtype=np.int8)


@dataclass(frozen=True)
class NightDay:
    """Baseline: charge from ``charge_start`` and discharge from ``discharge_start``.

    Each window lasts ``duration_h`` hours. It may issue impossible actions (e.g. on a
    frame that starts mid-schedule); ``simulate`` turns those into idle.
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
class PerfectForesight:
    """Upper bound: backward DP over hours x charge levels with the true prices.

    Not causal, so it is never a deployable policy. Maximizes
    ``sum(-grid_kwh * price)`` starting empty; ties go to idle.
    """

    name: str = "perfect_foresight"

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        price = frame["price_cop_kwh"].to_numpy(dtype=float)
        n = len(price)
        next_level, grid_kwh = transition_tables(preset)
        invalid = ~valid_mask(preset)
        best = np.zeros((n, preset.n_levels), dtype=np.int8)
        value = np.zeros(preset.n_levels)  # value after the last hour
        for t in range(n - 1, -1, -1):
            q = -grid_kwh * price[t] + value[next_level]  # [levels, actions]
            q[invalid] = -np.inf
            best[t] = np.argmax(q, axis=1)  # first max wins, and IDLE is index 0
            value = q[np.arange(preset.n_levels), best[t]]
        out = np.zeros(n, dtype=np.int8)
        level = 0
        for t in range(n):
            out[t] = best[t, level]
            level = int(next_level[level, out[t]])
        return out

