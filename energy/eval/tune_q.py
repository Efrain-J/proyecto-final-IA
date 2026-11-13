"""Hyper-parameter search for the Q-learning policy (D15).

Run: ``uv run python -m energy.eval.tune_q [--preset 8h] [--year 2025] [--seeds 4]
[--workers N] [--stage {1,2,3,all}] [--out results/tuning]``.

Method: fit on January-June, score candidates on July-September over the fixed
``VAL_SHIFT_SEEDS`` patterns, and touch the October-December test months only in Stage 3
(default vs the best validation configuration, both retrained on January-September, scored
on ``TEST_SHIFT_SEEDS``). The score is the mean Q-learning savings divided by the mean
savings of a ``PriceThreshold`` tuned on the same training frames, on the same patterns.

* Stage 1: one parameter at a time around the defaults (12 configs).
* Stage 2: grid of {default, best} over the (at most 3) parameters whose best value beats
  the default by at least ``MIN_IMPROVEMENT``; skipped if none does.
* Stage 3: test comparison and the adoption recommendation (adopt only if the best
  configuration is >= 1% better on validation AND not worse on test).

Outputs in ``--out``: ``q_<preset>_<year>_stage1.csv``, ``_stage2.csv``, ``_test.csv``
(one row per config and seed) and ``_summary.md``. Every run seed is
``job_seed(base, preset, year) + 10_000 * seed_index`` and the training shift seeds derive
from it, so results do not depend on ``--workers``. It never changes ``QConfig`` defaults
or saved artifacts.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from energy import config
from energy.data.io import write_bytes_atomic
from energy.data.prices import DEFAULT_PRICES_PATH, load_prices, scenario_split
from energy.rl.battery import PRESETS, get_preset
from energy.rl.factory import TEST_SHIFT_SEEDS, ShiftMix, add_shifts, get_mix, train_shift_seeds
from energy.rl.features import add_relative_price
from energy.rl.policies import NightDay, PerfectForesight, PriceThreshold, ShiftSchedule
from energy.rl.qlearning import QConfig, TabularPolicy, train_q_learning
from energy.rl.simulate import simulate
from energy.rl.train import DEFAULT_SHIFT_DRAWS, RULE_NAMES, job_seed

logger = logging.getLogger(__name__)

# Validation patterns: disjoint from TEST_SHIFT_SEEDS and from the training ones, which
# are job_seed + 1.. (D15).
VAL_SHIFT_SEEDS = (2000, 2001, 2002, 2003, 2004)
FIT_LAST_MONTH = 6  # fit Jan-Jun, validate Jul-Sep, test Oct-Dec
SEED_STRIDE = 10_000
MIN_IMPROVEMENT = 0.01  # Stage 2 entry and adoption threshold on validation
MAX_STAGE2_PARAMS = 3

PARAM_NAMES = ("alpha", "alpha_end", "gamma", "epsilon_end", "max_epochs", "shift_draws")


@dataclass(frozen=True)
class TuneConfig:
    """The tuned parameters; the defaults are the D15 ones."""

    alpha: float = QConfig.alpha
    alpha_end: float = QConfig.alpha_end
    gamma: float = QConfig.gamma
    epsilon_end: float = QConfig.epsilon_end
    max_epochs: int = QConfig.max_epochs
    shift_draws: int = DEFAULT_SHIFT_DRAWS

    def to_qconfig(self, seed: int) -> QConfig:
        """Build the ``QConfig`` for one run (``shift_draws`` is not a ``QConfig`` field)."""
        return dataclasses.replace(
            QConfig(),
            alpha=self.alpha,
            alpha_end=self.alpha_end,
            gamma=self.gamma,
            epsilon_end=self.epsilon_end,
            max_epochs=self.max_epochs,
            seed=seed,
        )

    @property
    def label(self) -> str:
        """Short name listing the parameters that differ from the defaults."""
        base = TuneConfig()
        diffs = [
            f"{n}={getattr(self, n)}" for n in PARAM_NAMES if getattr(self, n) != getattr(base, n)
        ]
        return ", ".join(diffs) or "default"


# Values tried per parameter, besides the default (D15 plan).
STAGE1_VALUES: dict[str, tuple[float | int, ...]] = {
    "alpha": (0.05, 0.2),
    "alpha_end": (0.005, 0.05),
    "gamma": (0.95, 0.995),
    "epsilon_end": (0.01, 0.15),
    "max_epochs": (600,),
    "shift_draws": (10,),
}


@dataclass(frozen=True)
class RunSpec:
    """One (config, seed) run; ``mode`` is ``"val"`` (fit/validation) or ``"test"``."""

    config: TuneConfig
    seed_index: int
    base_seed: int
    preset_id: str
    year: int
    mode: str
    featured: pd.DataFrame = field(repr=False, compare=False)
    mix: str | ShiftMix = "default"


def validation_split(
    featured: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut one year into fit (Jan-Jun), validation (Jul-Sep) and test (Oct-Dec) frames.

    The test part comes from ``scenario_split`` so it always matches D5. Each part has a
    reset index.

    Raises:
        ValueError: If ``year`` has no rows.
    """
    split = scenario_split(featured, year)  # raises for an unknown year
    month = split.train["timestamp"].dt.month
    fit = split.train[month <= FIT_LAST_MONTH].reset_index(drop=True)
    val = split.train[month > FIT_LAST_MONTH].reset_index(drop=True)
    return fit, val, split.test


def _mean_savings(
    policies: list,
    featured: pd.DataFrame,
    mix: ShiftMix,
    seeds: tuple[int, ...],
    period: pd.DataFrame,
    preset,
) -> tuple[dict[str, float], list[float]]:
    """Mean savings (COP) per policy over the patterns, plus Q-learning's per pattern."""
    first = period["timestamp"].iloc[0].date()
    last = period["timestamp"].iloc[-1].date()
    per_pattern: dict[str, list[float]] = {p.name: [] for p in policies}
    for seed in seeds:
        shifted = add_shifts(featured, mix, seed)  # over the whole series, as in train.py
        for p in policies:
            result = simulate(p, first, last, preset, prices=shifted)
            per_pattern[p.name].append(result.summary.savings_cop)
    means = {name: float(np.mean(values)) for name, values in per_pattern.items()}
    return means, per_pattern["q_learning"]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else float("nan")


def run_config(spec: RunSpec) -> dict:
    """Train one Q-table, tune the rule on the same frames and evaluate (top level for spawn).

    In ``"val"`` mode it trains on Jan-Jun and scores Jul-Sep on ``VAL_SHIFT_SEEDS``
    (Q-learning vs the tuned ``PriceThreshold``). In ``"test"`` mode it trains on Jan-Sep
    and scores Oct-Dec on ``TEST_SHIFT_SEEDS`` against every rule and perfect foresight.

    Returns:
        A flat dict: config fields, ``seed_index``, ``run_seed``, savings in COP, ``score``
        (Q / threshold), ``epochs_run``, ``seconds`` and, in test mode, the extra ratios.

    Raises:
        ValueError: If ``mode`` is unknown.
    """
    if spec.mode not in ("val", "test"):
        raise ValueError(f"mode must be 'val' or 'test', got {spec.mode!r}")
    started = time.perf_counter()
    preset = get_preset(spec.preset_id)
    mix = get_mix(spec.mix)
    run_seed = job_seed(spec.base_seed, spec.preset_id, spec.year) + SEED_STRIDE * spec.seed_index
    cfg = spec.config.to_qconfig(run_seed)

    fit, val, test = validation_split(spec.featured, spec.year)
    if spec.mode == "val":
        train, period, seeds = fit, val, VAL_SHIFT_SEEDS
    else:
        train, period, seeds = pd.concat([fit, val], ignore_index=True), test, TEST_SHIFT_SEEDS

    train_frames = [
        add_shifts(train, mix, s) for s in train_shift_seeds(run_seed, spec.config.shift_draws)
    ]
    result = train_q_learning(train_frames, preset, cfg)
    q_policy = TabularPolicy(result.q, result.discretizer, preset.id)
    threshold = PriceThreshold.tune(train_frames, preset)

    policies = [q_policy, threshold]
    if spec.mode == "test":
        policies += [ShiftSchedule(), NightDay(), PerfectForesight()]
    means, q_per_pattern = _mean_savings(policies, spec.featured, mix, seeds, period, preset)

    row = {
        **dataclasses.asdict(spec.config),
        "label": spec.config.label,
        "mode": spec.mode,
        "seed_index": spec.seed_index,
        "run_seed": run_seed,
        "savings_cop": means["q_learning"],
        "threshold_savings_cop": means["price_threshold"],
        "score": _ratio(means["q_learning"], means["price_threshold"]),
        "q_per_pattern_cop": ";".join(f"{v:.0f}" for v in q_per_pattern),
        "epochs_run": result.epochs_run,
    }
    if spec.mode == "test":
        rules = {name: means[name] for name in RULE_NAMES}
        best_rule = max(rules, key=lambda n: rules[n])  # first maximum wins ties, as train.py
        row.update(
            shift_schedule_savings_cop=means["shift_schedule"],
            night_day_savings_cop=means["night_day"],
            perfect_savings_cop=means["perfect_foresight"],
            best_rule=best_rule,
            q_vs_best_rule=_ratio(means["q_learning"], rules[best_rule]),
            q_vs_perfect=_ratio(means["q_learning"], means["perfect_foresight"]),
        )
    row["seconds"] = time.perf_counter() - started
    return row


def run_specs(specs: list[RunSpec], workers: int) -> list[dict]:
    """Run the specs (in a process pool when ``workers > 1``); results keep the input order."""
    if workers > 1 and len(specs) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(specs))) as pool:
                return list(pool.map(run_config, specs))
        except (BrokenProcessPool, OSError) as exc:
            logger.warning("Process pool failed (%s); running sequentially", exc)
    return [run_config(spec) for spec in specs]


def stage1_configs() -> list[TuneConfig]:
    """The default followed by every one-at-a-time variant (12 configs)."""
    configs = [TuneConfig()]
    for name, values in STAGE1_VALUES.items():
        configs += [dataclasses.replace(TuneConfig(), **{name: v}) for v in values]
    return configs


def config_table(rows: pd.DataFrame) -> pd.DataFrame:
    """Per-config mean and worst-seed score, in first-seen order, with delta vs default."""
    table = rows.groupby("label", sort=False).agg(
        score_mean=("score", "mean"),
        score_worst=("score", "min"),
        epochs_mean=("epochs_run", "mean"),
        n_seeds=("score", "size"),
    )
    default = table.loc["default", "score_mean"] if "default" in table.index else np.nan
    table["delta_vs_default_pct"] = 100 * (table["score_mean"] / default - 1)
    return table


def _row_config(row: pd.Series) -> TuneConfig:
    """Rebuild a ``TuneConfig`` from a CSV row (CSV floats are cast back to int fields)."""
    base = TuneConfig()
    return TuneConfig(**{n: type(getattr(base, n))(row[n]) for n in PARAM_NAMES})


def stage2_configs(stage1: pd.DataFrame) -> tuple[list[TuneConfig], list[str]]:
    """Choose the Stage 2 grid from the Stage 1 rows.

    Returns:
        The configs ({default, best} crossed over the qualifying parameters, at most
        ``MAX_STAGE2_PARAMS`` of them) and the names of those parameters; both empty if no
        parameter's best value beats the default mean score by ``MIN_IMPROVEMENT``.
    """
    table = config_table(stage1)
    default_score = table.loc["default", "score_mean"]
    base = TuneConfig()
    # Best variant per parameter. Stage 1 variants differ from the default in one field.
    best: dict[str, tuple[float, float | int]] = {}
    for label, row in stage1.drop_duplicates("label").set_index("label").iterrows():
        variant = _row_config(row)
        changed = [n for n in PARAM_NAMES if getattr(variant, n) != getattr(base, n)]
        if len(changed) != 1:
            continue
        gain = table.loc[label, "score_mean"] / default_score - 1
        name = changed[0]
        if name not in best or gain > best[name][0]:
            best[name] = (gain, getattr(variant, name))
    qualifying = sorted(
        ((gain, name, value) for name, (gain, value) in best.items() if gain >= MIN_IMPROVEMENT),
        reverse=True,
    )[:MAX_STAGE2_PARAMS]
    if not qualifying:
        return [], []
    names = [name for _, name, _ in qualifying]
    options = [(getattr(base, name), value) for _, name, value in qualifying]
    configs = [
        dataclasses.replace(base, **dict(zip(names, combo)))
        for combo in itertools.product(*options)
    ]
    return configs, names


def _winner(stage1: pd.DataFrame, stage2: pd.DataFrame | None) -> tuple[TuneConfig, str]:
    """Config with the best mean validation score: Stage 2's if it ran, else Stage 1's."""
    rows = stage2 if stage2 is not None else stage1
    label = str(config_table(rows)["score_mean"].idxmax())  # first wins ties
    return _row_config(rows[rows["label"] == label].iloc[0]), label


def _paths(out: Path, preset_id: str, year: int) -> dict[str, Path]:
    stem = f"q_{preset_id}_{year}"
    return {
        "stage1": out / f"{stem}_stage1.csv",
        "stage2": out / f"{stem}_stage2.csv",
        "test": out / f"{stem}_test.csv",
        "summary": out / f"{stem}_summary.md",
    }


def _save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(path, frame.to_csv(index=False).encode("utf-8"))


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def _fmt_table(table: pd.DataFrame) -> str:
    lines = [
        "| config | mean score | worst seed | delta vs default | epochs run (mean) |",
        "|---|---|---|---|---|",
    ]
    for label, r in table.iterrows():
        lines.append(
            f"| {label} | {r.score_mean:.4f} | {r.score_worst:.4f} | "
            f"{r.delta_vs_default_pct:+.2f}% | {r.epochs_mean:.0f} |"
        )
    return "\n".join(lines)


def build_summary(
    preset_id: str,
    year: int,
    stage1: pd.DataFrame | None,
    stage2: pd.DataFrame | None,
    test: pd.DataFrame | None,
) -> str:
    """Markdown summary of the stages that have results, with the adoption recommendation."""
    out = [
        f"# Q-learning tuning, preset {preset_id}, year {year}",
        "",
        "Fit Jan-Jun, validation Jul-Sep (patterns "
        f"{VAL_SHIFT_SEEDS[0]}..{VAL_SHIFT_SEEDS[-1]}), test Oct-Dec only in Stage 3 "
        "(retrained on Jan-Sep). Score = Q-learning mean savings / tuned PriceThreshold "
        "mean savings on the same patterns.",
        "",
    ]
    val_gain = None
    if stage1 is not None:
        out += ["## Stage 1 (one at a time, validation)", "", _fmt_table(config_table(stage1)), ""]
        if stage2 is not None:
            out += ["## Stage 2 (default/best grid, validation)", "",
                    _fmt_table(config_table(stage2)), ""]
        else:
            out += ["## Stage 2", "",
                    "Skipped: no parameter improved the validation score by >= 1%.", ""]
        _, label = _winner(stage1, stage2)
        table = config_table(stage2 if stage2 is not None else stage1)
        default_score = config_table(stage1).loc["default", "score_mean"]
        val_gain = table.loc[label, "score_mean"] / default_score - 1
        out += [
            "## Chosen configuration",
            "",
            f"`{label}`: mean validation score {table.loc[label, 'score_mean']:.4f}, "
            f"{100 * val_gain:+.2f}% vs the default.",
            "",
        ]
    if test is not None:
        out += [
            "## Stage 3 (test, Oct-Dec)",
            "",
            "| role | config | Q / best rule (mean) | Q / best rule per seed | Q / perfect | "
            "Q savings COP | shift schedule, threshold, night/day COP | perfect COP |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for role, grp in test.groupby("role", sort=False):
            per_seed = ", ".join(f"{v:.3f}" for v in grp["q_vs_best_rule"])
            out.append(
                f"| {role} | {grp['label'].iloc[0]} | {grp['q_vs_best_rule'].mean():.4f} | "
                f"{per_seed} | {100 * grp['q_vs_perfect'].mean():.1f}% | "
                f"{grp['savings_cop'].mean():,.0f} | "
                f"{grp['shift_schedule_savings_cop'].mean():,.0f}, "
                f"{grp['threshold_savings_cop'].mean():,.0f}, "
                f"{grp['night_day_savings_cop'].mean():,.0f} | "
                f"{grp['perfect_savings_cop'].mean():,.0f} |"
            )
        out.append("")
        roles = test.groupby("role")["q_vs_best_rule"].mean()
        if val_gain is not None and {"default", "tuned"} <= set(roles.index):
            test_ok = bool(roles["tuned"] >= roles["default"])
            adopt = val_gain >= MIN_IMPROVEMENT and test_ok
            out += [
                "## Recommendation",
                "",
                f"Validation gain {100 * val_gain:+.2f}% (needs >= +1%); test Q / best rule: "
                f"tuned {roles['tuned']:.4f} vs default {roles['default']:.4f} "
                f"({'not worse' if test_ok else 'worse'}). "
                + ("ADOPT the tuned configuration." if adopt else "KEEP the defaults."),
                "",
            ]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the requested stages, write the outputs, print the summary."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--preset", default="8h", choices=list(PRESETS))
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--stage", choices=["1", "2", "3", "all"], default="all")
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "tuning")
    args = parser.parse_args(argv)

    workers = args.workers or os.cpu_count() or 1
    paths = _paths(args.out, args.preset, args.year)
    # Relative price over the whole series, before any split (D14).
    featured = add_relative_price(load_prices(args.prices))
    stages = ["1", "2", "3"] if args.stage == "all" else [args.stage]

    def run(configs: list[TuneConfig], mode: str, roles: list[str] | None = None) -> pd.DataFrame:
        specs = [
            RunSpec(c, k, QConfig.seed, args.preset, args.year, mode, featured)
            for c in configs
            for k in range(args.seeds)
        ]
        started = time.perf_counter()
        frame = pd.DataFrame(run_specs(specs, workers))
        if roles:
            frame.insert(0, "role", [r for r in roles for _ in range(args.seeds)])
        print(
            f"{mode}: {len(specs)} runs in {time.perf_counter() - started:.1f}s "
            f"with {workers} worker(s)"
        )
        return frame

    if "1" in stages:
        _save_csv(run(stage1_configs(), "val"), paths["stage1"])
    stage1 = _read(paths["stage1"])
    if stage1 is None:
        parser.error("Stage 1 results are missing; run --stage 1 first")

    if "2" in stages:
        configs, names = stage2_configs(stage1)
        if configs:
            print(f"stage 2 parameters: {names}")
            _save_csv(run(configs, "val"), paths["stage2"])
        else:
            paths["stage2"].unlink(missing_ok=True)  # a stale file would be picked as winner
            print("stage 2 skipped: no parameter improved validation by >= 1%")
    stage2 = _read(paths["stage2"])

    if "3" in stages:
        winner, _ = _winner(stage1, stage2)
        _save_csv(run([TuneConfig(), winner], "test", ["default", "tuned"]), paths["test"])

    text = build_summary(args.preset, args.year, stage1, stage2, _read(paths["test"]))
    paths["summary"].parent.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(paths["summary"], text.encode("utf-8"))
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
