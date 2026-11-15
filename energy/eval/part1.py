"""Part 1 results tables: saved policies vs baselines on the test quarters (D14, D15).

Run: ``uv run python -m energy.eval.part1 [--presets 4h 8h 12h] [--years 2024 2025]
[--seeds 5] [--mixes default mostly_double mostly_single] [--prices CSV] [--policies DIR]
[--out results/part1]``.

It evaluates the committed Q-tables and tuned price thresholds in ``artifacts/policies/``
without retraining, against the rule baselines and perfect foresight, on October-December of
each test year, averaged over the first ``--seeds`` test shift calendars (A5). ``--years``
are both the training years and the test years, so the cross-year check (D15) comes for free.
Outputs in ``--out``: ``runs.csv`` (one row per contender and seed), ``summary.csv`` (UTF-8
with BOM so Excel shows accents) and ``summary.md`` (Spanish tables for the report).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from energy import config
from energy.app.model import MIX_LABELS  # pure module, no streamlit
from energy.data.io import write_bytes_atomic
from energy.data.prices import load_prices
from energy.rl.battery import PRESETS
from energy.rl.factory import MIXES, TEST_SHIFT_SEEDS, add_shifts
from energy.rl.features import add_relative_price
from energy.rl.policies import (
    NightDay,
    PerfectForesight,
    Policy,
    PriceThreshold,
    ShiftSchedule,
    threshold_path,
)
from energy.rl.qlearning import TabularPolicy, policy_stem
from energy.rl.simulate import simulate

logger = logging.getLogger(__name__)

# Same order as energy.rl.train.RULE_NAMES: ties go to the first.
RULE_IDS = ("shift_schedule", "price_threshold", "night_day")
Q_ID = "q_learning"
PERFECT_ID = "perfect_foresight"


@dataclass(frozen=True)
class Contender:
    """One policy to score; ``trained_on`` is the cutoff/training year (None for fixed rules)."""

    policy_id: str
    trained_on: int | None
    label: str
    policy: Policy


def _label(policy_id: str, trained_on: int | None) -> str:
    return policy_id if trained_on is None else f"{policy_id} ({trained_on})"


def load_contenders(
    preset_id: str, years: list[int] | tuple[int, ...], policies_dir: Path | str
) -> list[Contender]:
    """Fixed rules, perfect foresight and the saved Q-tables and thresholds of each year.

    A missing or unreadable file logs a warning and is skipped. ``simulate.resolve_policy``
    is avoided on purpose: it would substitute ``shift_schedule`` under the wrong label.
    """
    out = [
        Contender("shift_schedule", None, "shift_schedule", ShiftSchedule()),
        Contender("night_day", None, "night_day", NightDay()),
        Contender(PERFECT_ID, None, PERFECT_ID, PerfectForesight()),
    ]
    for year in years:
        loaders = (
            (Q_ID, policy_stem(policies_dir, preset_id, year), TabularPolicy.load),
            ("price_threshold", threshold_path(policies_dir, preset_id, year), PriceThreshold.load),
        )
        for policy_id, path, load in loaders:
            try:
                out.append(Contender(policy_id, year, _label(policy_id, year), load(path)))
            except (OSError, KeyError, ValueError) as exc:  # JSON errors are ValueErrors
                logger.warning("Skipping %s for %s %s: %s (%s)", policy_id, preset_id, year, path, exc)
    return out


def evaluate(
    featured: pd.DataFrame,
    preset_id: str,
    test_year: int,
    contenders: list[Contender],
    seeds: tuple[int, ...] = TEST_SHIFT_SEEDS,
    mix: str = "default",
) -> pd.DataFrame:
    """Simulate every contender on October-December of ``test_year`` for each shift seed.

    Args:
        featured: Prices with ``rel_price`` over the whole series.
        seeds: Shift calendar seeds; the calendar is built once per seed over the whole
            series, as in training and in the app (A5).

    Returns:
        One row per (seed, contender).

    Raises:
        ValueError: If the quarter is outside the price data.
    """
    start, end = date(test_year, 10, 1), date(test_year, 12, 31)
    rows = []
    for seed in seeds:
        shifted = add_shifts(featured, mix, seed)
        for c in contenders:
            s = simulate(c.policy, start, end, preset_id, prices=shifted).summary
            rows.append(
                {
                    "preset_id": preset_id, "test_year": test_year, "mix": mix,
                    "policy_id": c.policy_id, "trained_on": c.trained_on, "label": c.label,
                    "seed": seed, "savings_cop": s.savings_cop, "cycles": s.cycles,
                    "invalid_actions": s.invalid_actions, "start": start, "end": end,
                }
            )
    return pd.DataFrame(rows)


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


def summarize(runs: pd.DataFrame) -> pd.DataFrame:
    """One row per (preset, test year, [mix], Q-learning training year).

    The best rule is the rule row (any cutoff year) with the highest mean savings in the
    group; ties go to ``RULE_IDS`` order, then to the earlier ``trained_on``.
    """
    keys = ["preset_id", "test_year"] + (["mix"] if "mix" in runs.columns else [])
    means = (
        runs.groupby(keys + ["policy_id", "trained_on", "label"], dropna=False)["savings_cop"]
        .mean()
        .reset_index()
    )
    rows = []
    for group_key, group in means.groupby(keys):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        rules = group[group["policy_id"].isin(RULE_IDS)].copy()
        rules["order"] = rules["policy_id"].map(RULE_IDS.index)
        rules = rules.sort_values(["order", "trained_on"], na_position="first")
        best = rules.loc[rules["savings_cop"].idxmax()] if not rules.empty else None
        best_cop = float(best["savings_cop"]) if best is not None else float("nan")
        perfect = group.loc[group["policy_id"] == PERFECT_ID, "savings_cop"]
        perfect_cop = float(perfect.iloc[0]) if not perfect.empty else float("nan")
        mask = runs["policy_id"] == Q_ID
        for col, val in zip(keys, group_key):
            mask &= runs[col] == val
        for trained_on, q_runs in runs[mask].groupby("trained_on"):
            q = q_runs.groupby("seed")["savings_cop"].sum()
            q_mean = float(q.mean())
            rows.append(
                {
                    **dict(zip(keys, group_key)),
                    "trained_on": int(trained_on),
                    "q_savings_cop": q_mean,
                    "q_min_cop": float(q.min()),
                    "q_max_cop": float(q.max()),
                    "best_rule": None if best is None else best["label"],
                    "best_rule_savings_cop": best_cop,
                    "perfect_savings_cop": perfect_cop,
                    "q_vs_best_rule": _ratio(q_mean, best_cop),
                    "q_vs_perfect": _ratio(q_mean, perfect_cop),
                    "n_seeds": int(q.size),
                }
            )
    return _by_battery_size(pd.DataFrame(rows), keys + ["trained_on"])


def _by_battery_size(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Sort by ``columns`` with batteries in size order (4h, 8h, 12h), not as text."""
    if frame.empty:
        return frame
    hours = lambda col: col.map(lambda p: PRESETS[p].duration_h if p in PRESETS else 0) if col.name == "preset_id" else col  # noqa: E731
    return frame.sort_values(columns, key=hours).reset_index(drop=True)


def _millions(cop: float) -> str:
    return f"{cop / 1e6:,.2f}"


def _ratio_text(value: float) -> str:
    return "n/d" if pd.isna(value) else f"{value:.2f}"


def build_markdown(summary: pd.DataFrame, n_seeds: int) -> str:
    """Spanish tables for the report: main, cross-year, shift mixes and a method note."""
    has_mix = "mix" in summary.columns
    default = summary[summary["mix"] == "default"] if has_mix else summary
    out = ["# Parte 1: resultados de la política Q-learning", ""]

    main = default[default["trained_on"] == default["test_year"]]
    out += [
        "## Resultados principales (entrenado y probado en el mismo año, operación mixta)", "",
        "| Batería | Año de prueba | Ahorro Q-learning (millones COP, mín-máx) | Mejor regla | "
        "Ahorro mejor regla (millones COP) | Q ÷ mejor regla | Q ÷ previsión perfecta |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in main.itertuples():
        out.append(
            f"| {r.preset_id} | {r.test_year} | {_millions(r.q_savings_cop)} "
            f"({_millions(r.q_min_cop)}-{_millions(r.q_max_cop)}) | {r.best_rule} | "
            f"{_millions(r.best_rule_savings_cop)} | {_ratio_text(r.q_vs_best_rule)} | "
            f"{_ratio_text(r.q_vs_perfect)} |"
        )
    out += [
        "", "## Verificación entre años (Q ÷ mejor regla, operación mixta)", "",
        "| Batería | Entrenado en | Probado en | Q ÷ mejor regla |", "|---|---|---|---|",
    ]
    for r in _by_battery_size(default, ["preset_id", "trained_on", "test_year"]).itertuples():
        out.append(f"| {r.preset_id} | {r.trained_on} | {r.test_year} | {_ratio_text(r.q_vs_best_rule)} |")

    if has_mix and summary["mix"].nunique() > 1:
        same_year = summary[summary["trained_on"] == summary["test_year"]]
        out += [
            "", "## Tipos de operación (Q ÷ mejor regla, entrenado y probado en el mismo año)", "",
            "| Batería | Año de prueba | Operación | Q ÷ mejor regla |", "|---|---|---|---|",
        ]
        for r in _by_battery_size(same_year, ["preset_id", "test_year", "mix"]).itertuples():
            out.append(
                f"| {r.preset_id} | {r.test_year} | {MIX_LABELS.get(r.mix, r.mix)} | {_ratio_text(r.q_vs_best_rule)} |"
            )

    out += [
        "", "## Método", "",
        f"- Promedio sobre {n_seeds} calendarios de turnos de prueba; periodos de prueba octubre-diciembre.",
        "- Los ahorros no incluyen el costo ni el desgaste de la batería (A3).",
        "- La batería conoce el precio de la hora actual (A4).",
        "- Los turnos de la fábrica son sintéticos (A5).",
        "- La mejor regla es la de mayor ahorro medio entre shift_schedule, price_threshold "
        "(con cortes de cada año de entrenamiento) y night_day.",
        "",
    ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: evaluate, write ``runs.csv``, ``summary.csv`` and ``summary.md``."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--presets", nargs="+", default=["4h", "8h", "12h"], choices=list(PRESETS))
    parser.add_argument("--years", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--seeds", type=int, default=len(TEST_SHIFT_SEEDS))
    parser.add_argument("--mixes", nargs="+", default=["default"], choices=list(MIXES))
    parser.add_argument("--prices", type=Path, default=config.PRICES_CSV)
    parser.add_argument("--policies", type=Path, default=config.POLICIES_DIR)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "part1")
    args = parser.parse_args(argv)
    if not 1 <= args.seeds <= len(TEST_SHIFT_SEEDS):
        parser.error(f"--seeds must be between 1 and {len(TEST_SHIFT_SEEDS)}")
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles garble Spanish accents
        sys.stdout.reconfigure(encoding="utf-8")

    started = time.perf_counter()
    seeds = TEST_SHIFT_SEEDS[: args.seeds]
    featured = add_relative_price(load_prices(args.prices))  # whole series, as in training (D14)
    frames = []
    for preset_id in args.presets:
        contenders = load_contenders(preset_id, args.years, args.policies)
        for year in args.years:
            for mix in args.mixes:
                frames.append(evaluate(featured, preset_id, year, contenders, seeds, mix))
    runs = pd.concat(frames, ignore_index=True)
    summary = summarize(runs)
    text = build_markdown(summary, args.seeds)

    args.out.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(args.out / "runs.csv", runs.to_csv(index=False).encode("utf-8"))
    write_bytes_atomic(args.out / "summary.csv", summary.to_csv(index=False).encode("utf-8-sig"))
    write_bytes_atomic(args.out / "summary.md", text.encode("utf-8"))
    print(text)
    print(f"done in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
