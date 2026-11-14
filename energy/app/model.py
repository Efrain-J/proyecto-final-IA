"""Pure logic behind the dashboard: policy runs, size comparison and KPIs (no streamlit).

Used by `app.py`. Inputs: a price frame with `rel_price` (see
`energy.rl.features.add_relative_price`), a battery preset id, a date range and a shift
mix. Outputs: frozen dataclasses and DataFrames ready to plot. Offline; reads the saved
policies in `config.POLICIES_DIR`.

Every figure is the mean over the 5 test shift calendars (D15, A5). This mirrors the
agent tool `simulate_battery` in `energy/agent/tools.py`, so the dashboard and the chat
report the same numbers.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from energy import config
from energy.rl.battery import PRESETS
from energy.rl.factory import TEST_SHIFT_SEEDS, add_shifts
from energy.rl.simulate import resolve_policy, simulate

# No perfect foresight (an oracle) and no "no battery" (always 0): decided with the user.
APP_POLICIES: tuple[str, ...] = ("q_learning", "price_threshold", "shift_schedule", "night_day")

MIX_LABELS: dict[str, str] = {
    "default": "Mixto",
    "mostly_double": "Principalmente turnos dobles",
    "mostly_single": "Principalmente un turno",
}

# Test quarter of each year (D5); 2024 is the El Niño cross-year check.
SCENARIOS: dict[str, tuple[date, date]] = {
    "2025 (normal)": (date(2025, 10, 1), date(2025, 12, 31)),
    "2024 (El Niño, precios altos)": (date(2024, 10, 1), date(2024, 12, 31)),
}

WEEK_HOURS = 168


@dataclass(frozen=True)
class PolicyRun:
    """One policy over the period, averaged over the shift calendars.

    ``policy_used`` is the policy that actually ran (a missing Q-table falls back to
    ``shift_schedule``). ``hourly`` is the first calendar's trace.
    """

    policy_id: str
    policy_used: str
    savings_cop: float
    savings_min_cop: float
    savings_max_cop: float
    cycles: float
    cum_savings: pd.Series
    hourly: pd.DataFrame


@dataclass(frozen=True)
class Kpis:
    """Headline numbers: Q-learning against the best hand-written rule."""

    q_savings_cop: float
    best_rule_id: str
    best_rule_savings_cop: float
    q_vs_best_rule_pct: float | None
    q_cycles: float


def run_policies(
    featured: pd.DataFrame,
    preset_id: str,
    start: date,
    end: date,
    mix: str = "default",
    policy_ids: tuple[str, ...] = APP_POLICIES,
    seeds: tuple[int, ...] = TEST_SHIFT_SEEDS,
    policies_dir: Path | str = config.POLICIES_DIR,
) -> dict[str, PolicyRun]:
    """Simulate each policy on each shift calendar and aggregate.

    Args:
        featured: Hourly prices with `rel_price`.
        preset_id: Battery preset id.
        start: First day.
        end: Last day (same calendar year as ``start``: the Q-tables are per year).
        mix: Shift mix id (D15).
        policy_ids: Policies to run; the result keeps this order.
        seeds: Shift-calendar seeds to average over.
        policies_dir: Where the saved policies live.

    Returns:
        ``policy_id -> PolicyRun``; savings and cycles are means over ``seeds``,
        ``savings_min_cop`` / ``savings_max_cop`` their extremes.
    """
    # Shifts once per seed and policies once each, shared across the loop.
    frames = [add_shifts(featured, mix, seed) for seed in seeds]
    out: dict[str, PolicyRun] = {}
    for pid in policy_ids:
        policy = resolve_policy(pid, preset_id, start.year, policies_dir)
        results = [simulate(policy, start, end, preset_id, prices=f) for f in frames]
        savings = [r.summary.savings_cop for r in results]
        cum = np.mean([r.hourly["cum_savings_cop"].to_numpy() for r in results], axis=0)
        first = results[0]
        out[pid] = PolicyRun(
            policy_id=pid,
            policy_used=first.summary.policy,
            savings_cop=float(np.mean(savings)),
            savings_min_cop=float(min(savings)),
            savings_max_cop=float(max(savings)),
            cycles=float(np.mean([r.summary.cycles for r in results])),
            cum_savings=pd.Series(cum, index=pd.DatetimeIndex(first.hourly["timestamp"])),
            hourly=first.hourly,
        )
    return out


def size_comparison(
    featured: pd.DataFrame,
    start: date,
    end: date,
    mix: str = "default",
    presets: tuple[str, ...] = tuple(PRESETS),
    policy_ids: tuple[str, ...] = APP_POLICIES,
    seeds: tuple[int, ...] = TEST_SHIFT_SEEDS,
    policies_dir: Path | str = config.POLICIES_DIR,
) -> pd.DataFrame:
    """Mean savings of each policy for each battery size.

    Returns:
        Long table with columns ``preset_id``, ``policy_id``, ``savings_cop``.
    """
    rows = []
    for preset_id in presets:
        runs = run_policies(featured, preset_id, start, end, mix, policy_ids, seeds, policies_dir)
        rows += [(preset_id, pid, run.savings_cop) for pid, run in runs.items()]
    return pd.DataFrame(rows, columns=["preset_id", "policy_id", "savings_cop"])


def compute_kpis(runs: dict[str, PolicyRun]) -> Kpis:
    """Q-learning against the best rule among the other policies present.

    Ties go to the first rule in `APP_POLICIES` order. The percentage uses the same
    formula as ``vs_best_rule`` in the agent tool, and is None when the best rule does
    not save money (the ratio would be meaningless).

    Raises:
        KeyError: If ``runs`` has no ``q_learning``.
        ValueError: If there is no rule to compare with.
    """
    q = runs["q_learning"]
    rules = [pid for pid in APP_POLICIES if pid != "q_learning" and pid in runs]
    if not rules:
        raise ValueError("compute_kpis needs at least one rule besides q_learning")
    best_id = max(rules, key=lambda pid: runs[pid].savings_cop)  # max keeps the first tie
    best = runs[best_id].savings_cop
    pct = (q.savings_cop - best) / abs(best) * 100 if best > 0 else None
    return Kpis(q.savings_cop, best_id, best, pct, q.cycles)


def n_weeks(hourly: pd.DataFrame) -> int:
    """Number of 168-hour windows needed to cover ``hourly`` (the last may be partial)."""
    return -(-len(hourly) // WEEK_HOURS)


def week_slice(hourly: pd.DataFrame, week: int) -> pd.DataFrame:
    """Rows of the 0-based ``week``-th 168-hour window.

    Raises:
        ValueError: If ``week`` is outside ``0..n_weeks-1``.
    """
    if not 0 <= week < n_weeks(hourly):
        raise ValueError(f"week {week} out of range 0..{n_weeks(hourly) - 1}")
    return hourly.iloc[week * WEEK_HOURS : (week + 1) * WEEK_HOURS]


_MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")


def _is_month_end(day: date) -> bool:
    return (pd.Timestamp(day) + pd.Timedelta(days=1)).day == 1


def period_label(start: date, end: date) -> str:
    """Short Spanish label of a period: ``oct–dic 2025``, ``oct 2025`` or ``6 oct – 10 oct 2025``.

    Whole months collapse to month names; anything else shows the days.
    """
    whole = start.day == 1 and _is_month_end(end)
    if whole and start.month == end.month and start.year == end.year:
        return f"{_MONTHS[start.month - 1]} {start.year}"
    if whole and start.year == end.year:
        return f"{_MONTHS[start.month - 1]}–{_MONTHS[end.month - 1]} {start.year}"
    left = f"{start.day} {_MONTHS[start.month - 1]}"
    right = f"{end.day} {_MONTHS[end.month - 1]} {end.year}"
    if start.year != end.year:
        left += f" {start.year}"
    return f"{left} – {right}"


def status_line(preset_id: str, start: date, end: date, mix: str) -> str:
    """Caption such as ``Batería de 8 h · oct–dic 2025 · Operación: Mixto``."""
    hours = PRESETS[preset_id].label
    return f"Batería de {hours} · {period_label(start, end)} · Operación: {MIX_LABELS[mix]}"
