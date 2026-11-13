"""The user's battery and period, which drive the agent by default (D16, option A).

``UserSettings`` is a frozen, validated dataclass. The loop fills the ``simulate_battery``
arguments the model omits from it, and the system prompt states it. Validation raises
``ValueError`` with a Spanish message so the dashboard can show it with ``st.error``.
Library only; see ``energy.agent.loop``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from energy import config
from energy.rl.battery import PRESETS

# Span of the committed XM prices.
DATA_FIRST_DAY = date(2024, 1, 1)
DATA_LAST_DAY = date(2025, 12, 31)


@dataclass(frozen=True)
class UserSettings:
    """Battery and period chosen by the user.

    Attributes:
        preset: Battery preset id (``4h``, ``8h`` or ``12h``).
        start: First day, inclusive.
        end: Last day, inclusive; same calendar year as ``start`` (Q-tables are per year).

    Raises:
        ValueError: On an unknown preset, ``start`` after ``end``, dates in different
            years or dates outside the price data.
    """

    preset: str = config.DEFAULT_PRESET
    start: date = config.DEFAULT_START
    end: date = config.DEFAULT_END

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"Batería desconocida {self.preset!r}; usa una de {sorted(PRESETS)}.")
        if self.start > self.end:
            raise ValueError("La fecha de inicio no puede ser posterior a la fecha final.")
        if self.start.year != self.end.year:
            raise ValueError("El periodo debe estar dentro de un mismo año calendario.")
        if self.start < DATA_FIRST_DAY or self.end > DATA_LAST_DAY:
            raise ValueError(f"Hay precios solo entre {DATA_FIRST_DAY} y {DATA_LAST_DAY}.")
