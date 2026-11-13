"""Battery physics for the dispatch MDP (D9, D14, D15).

``step`` is the single definition of the physics; ``transition_tables`` and
``valid_mask`` derive lookup arrays from it so training and replay never re-implement
it. Library only, no CLI. Every battery starts empty (D9); wear is ignored (A3).
Discharging is possible only while the factory runs (D15, A5); ``transition_tables`` stays
pure physics and the factory flag lives in ``step`` and ``valid_mask``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np

N_ACTIONS = 3


class Action(IntEnum):
    """Dispatch action; the values index the Q-table directly."""

    IDLE = 0
    CHARGE = 1
    DISCHARGE = 2


@dataclass(frozen=True)
class BatteryPreset:
    """A fixed battery size (D9). The state of charge is a level 0..duration_h."""

    id: str
    label: str
    capacity_kwh: float
    power_kw: float
    round_trip_efficiency: float = 0.9

    def __post_init__(self) -> None:
        ratio = self.capacity_kwh / self.power_kw
        if abs(ratio - round(ratio)) > 1e-9 or round(ratio) < 1:
            raise ValueError(
                f"capacity/power must be a whole number of hours, got {ratio} for {self.id!r}"
            )

    @property
    def duration_h(self) -> int:
        """Hours of discharge at full power."""
        return round(self.capacity_kwh / self.power_kw)

    @property
    def n_levels(self) -> int:
        """Number of state-of-charge levels (0..duration_h)."""
        return self.duration_h + 1

    @property
    def one_way_efficiency(self) -> float:
        """Efficiency of one charge or one discharge (sqrt of round trip)."""
        return math.sqrt(self.round_trip_efficiency)


PRESETS: dict[str, BatteryPreset] = {
    f"{h}h": BatteryPreset(id=f"{h}h", label=f"{h} h", capacity_kwh=100.0 * h, power_kw=100.0)
    for h in (4, 8, 12)  # D15 (replaces D9's 2/3/4 h)
}


def get_preset(preset: str | BatteryPreset) -> BatteryPreset:
    """Return a preset from its id (or pass a preset through).

    Raises:
        ValueError: If the id is unknown; the message lists the valid ones.
    """
    if isinstance(preset, BatteryPreset):
        return preset
    try:
        return PRESETS[preset]
    except KeyError:
        raise ValueError(f"Unknown preset {preset!r}; valid: {sorted(PRESETS)}") from None


def step(
    level: int, action: Action | int, preset: BatteryPreset, factory_on: bool = True
) -> tuple[int, float]:
    """Apply one hour of ``action`` at ``level``.

    An impossible action (discharge when empty or while the factory is off, charge when
    full) is treated as idle.

    Args:
        level: Current state-of-charge level.
        action: Dispatch action.
        preset: Battery.
        factory_on: Whether the factory runs this hour; discharging needs it (D15).

    Returns:
        ``(next_level, grid_kwh)``, where ``grid_kwh`` is the change in energy bought
        from the grid this hour: ``+P/eta`` charging, ``-P*eta`` discharging.

    Raises:
        ValueError: If ``level`` is outside ``0..duration_h``.
    """
    if not 0 <= level <= preset.duration_h:
        raise ValueError(f"level {level} outside 0..{preset.duration_h}")
    eta = preset.one_way_efficiency
    if action == Action.CHARGE and level < preset.duration_h:
        return level + 1, preset.power_kw / eta
    if action == Action.DISCHARGE and level > 0 and factory_on:
        return level - 1, -preset.power_kw * eta
    return level, 0.0


def transition_tables(preset: BatteryPreset) -> tuple[np.ndarray, np.ndarray]:
    """Lookup tables ``(next_level int8, grid_kwh float64)``, each ``[n_levels, 3]``."""
    shape = (preset.n_levels, N_ACTIONS)
    next_level = np.zeros(shape, dtype=np.int8)
    grid_kwh = np.zeros(shape, dtype=np.float64)
    for level in range(preset.n_levels):
        for action in Action:
            next_level[level, action], grid_kwh[level, action] = step(level, action, preset)
    return next_level, grid_kwh


def valid_mask(preset: BatteryPreset, factory_on: bool = True) -> np.ndarray:
    """Boolean ``[n_levels, 3]``: True where the action is possible at that level.

    With ``factory_on=False`` discharging is impossible at every level (D15).
    """
    mask = np.ones((preset.n_levels, N_ACTIONS), dtype=np.bool_)
    if not factory_on:
        mask[:, Action.DISCHARGE] = False
    mask[0, Action.DISCHARGE] = False
    mask[preset.duration_h, Action.CHARGE] = False
    return mask
