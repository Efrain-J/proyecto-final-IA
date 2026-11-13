"""Synthetic factory shift schedule for the battery MDP (D15, A5).

The battery may discharge only while the factory runs (no export), so the factory's
on/off pattern matters. There is no real shift data, so each calendar day gets one
random shift type from a named mix (A5). Library only, no CLI; the generator is seeded
and deterministic, and ``simulate`` / ``energy.rl.train`` call it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

SHIFT_OFF, SHIFT_MORNING, SHIFT_AFTERNOON, SHIFT_FULL = 0, 1, 2, 3
N_SHIFT_TYPES = 4

# Hours of the day (0-23) in which the factory runs, per shift type. Shift edges fall on
# even hours so the 2-hour state blocks never straddle one (D15).
SHIFT_HOURS: dict[int, range] = {
    SHIFT_OFF: range(0),
    SHIFT_MORNING: range(6, 14),
    SHIFT_AFTERNOON: range(14, 22),
    SHIFT_FULL: range(6, 22),
}

# Seeds for the held-out test shift patterns; never used in training or tuning.
TEST_SHIFT_SEEDS = (1000, 1001, 1002, 1003, 1004)

# [shift_type, hour] -> factory is running
_ON_TABLE = np.zeros((N_SHIFT_TYPES, 24), dtype=np.bool_)
for _shift, _hours in SHIFT_HOURS.items():
    _ON_TABLE[_shift, list(_hours)] = True


@dataclass(frozen=True)
class ShiftMix:
    """Probabilities of (off, morning, afternoon, full) by kind of day (A5)."""

    id: str
    weekday: tuple[float, float, float, float]
    saturday: tuple[float, float, float, float]
    sunday: tuple[float, float, float, float]

    def __post_init__(self) -> None:
        for kind in ("weekday", "saturday", "sunday"):
            probs = getattr(self, kind)
            if len(probs) != N_SHIFT_TYPES or abs(sum(probs) - 1.0) > 1e-9 or min(probs) < 0:
                raise ValueError(
                    f"{self.id!r}.{kind} must be {N_SHIFT_TYPES} probabilities summing to 1, "
                    f"got {probs}"
                )


_SATURDAY = (0.60, 0.40, 0.0, 0.0)
_SUNDAY = (1.0, 0.0, 0.0, 0.0)
MIXES: dict[str, ShiftMix] = {
    "default": ShiftMix("default", (0.10, 0.30, 0.15, 0.45), _SATURDAY, _SUNDAY),
    "mostly_double": ShiftMix("mostly_double", (0.05, 0.10, 0.10, 0.75), _SATURDAY, _SUNDAY),
    "mostly_single": ShiftMix("mostly_single", (0.10, 0.45, 0.35, 0.10), _SATURDAY, _SUNDAY),
}


def get_mix(mix: str | ShiftMix) -> ShiftMix:
    """Return a mix from its id (or pass a mix through).

    Raises:
        ValueError: If the id is unknown; the message lists the valid ones.
    """
    if isinstance(mix, ShiftMix):
        return mix
    try:
        return MIXES[mix]
    except KeyError:
        raise ValueError(f"Unknown shift mix {mix!r}; valid: {list(MIXES)}") from None


def train_shift_seeds(job_seed: int, k: int) -> list[int]:
    """Seeds of the ``k`` training shift patterns of a job (distinct from the test ones)."""
    return [job_seed + 1 + i for i in range(k)]


def add_shifts(
    frame: pd.DataFrame, mix: str | ShiftMix = "default", seed: int = 0
) -> pd.DataFrame:
    """Return a copy of ``frame`` with ``shift_type`` (int8) and ``factory_on`` (bool).

    One shift type is drawn per calendar day, in date order, from
    ``np.random.default_rng(seed)``; the same frame, mix and seed always give the same
    schedule. Call it on the whole series before slicing so a day's shift does not
    depend on where the slice starts.

    Args:
        frame: Frame with a ``timestamp`` column.
        mix: Mix object or id (A5).
        seed: Seed of this shift pattern.

    Raises:
        ValueError: If the mix id is unknown.
    """
    mix = get_mix(mix)
    stamps = pd.DatetimeIndex(frame["timestamp"])
    days, day_of_row = np.unique(stamps.normalize().to_numpy(), return_inverse=True)
    weekday = pd.DatetimeIndex(days).day_of_week.to_numpy()
    by_kind = np.array([mix.weekday, mix.saturday, mix.sunday])
    probs = by_kind[np.where(weekday < 5, 0, weekday - 4)]  # 5 -> saturday, 6 -> sunday
    cumulative = np.cumsum(probs, axis=1)
    cumulative /= cumulative[:, -1:]
    draws = np.random.default_rng(seed).random(len(days))
    shift_of_day = (draws[:, None] >= cumulative[:, :-1]).sum(axis=1).astype(np.int8)

    shift_type = shift_of_day[day_of_row.reshape(-1)]
    out = frame.copy()
    out["shift_type"] = shift_type
    out["factory_on"] = _ON_TABLE[shift_type, stamps.hour.to_numpy()]
    return out
