"""Train the shift-aware Q-learning policies, one job per (battery preset, year), in parallel.

Run: ``uv run python -m energy.rl.train --scenarios 2025`` (D15, A5). Each job trains on
the January-September prices repeated under ``--shift-draws`` random factory shift
patterns, tunes the ``PriceThreshold`` rule on the same frames, and writes
``artifacts/policies/q_<preset>_<year>.npz/.json`` and ``threshold_<preset>_<year>.json``.
It then prints, per job, the October-December savings (COP, averaged over 5 held-out
shift patterns) of Q-learning, each rule and PerfectForesight, plus Q-learning divided by
the best rule.

Flags: ``--scenarios`` years (default 2025), ``--presets`` (default all), ``--workers``
(default min(jobs, cpu_count); 1 = sequential), ``--max-epochs``, ``--seed``,
``--shift-draws`` (default 5), ``--mix`` (default ``default``), ``--prices`` CSV, ``--out``
directory. Jobs are independent and each seed depends only on (base seed, preset, year),
so results do not depend on workers or job order.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from energy.data.prices import DEFAULT_PRICES_PATH, load_prices, scenario_split
from energy.rl.battery import PRESETS, BatteryPreset, get_preset
from energy.rl.factory import (
    MIXES,
    TEST_SHIFT_SEEDS,
    ShiftMix,
    add_shifts,
    get_mix,
    train_shift_seeds,
)
from energy.rl.features import add_relative_price
from energy.rl.policies import (
    NightDay,
    PerfectForesight,
    Policy,
    PriceThreshold,
    ShiftSchedule,
    threshold_path,
)
from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem, train_q_learning
from energy.rl.simulate import DEFAULT_POLICIES_DIR, simulate

logger = logging.getLogger(__name__)

DEFAULT_SHIFT_DRAWS = 5
# Rules Q-learning is compared with, in tie-break order (D15).
RULE_NAMES = ("shift_schedule", "price_threshold", "night_day")


@dataclass(frozen=True)
class TrainJob:
    """One Q-table to train."""

    preset_id: str
    year: int


@dataclass(frozen=True)
class JobReport:
    """Result of a job: training effort and test-period savings in COP.

    Savings are means over the held-out test shift patterns (D15). ``best_rule`` is the
    rule with the highest savings and ``q_vs_best_rule`` is Q-learning divided by it.
    """

    preset_id: str
    year: int
    epochs_run: int
    seconds: float
    q_savings_cop: float
    shift_schedule_savings_cop: float
    threshold_savings_cop: float
    night_day_savings_cop: float
    perfect_savings_cop: float
    best_rule: str
    q_vs_best_rule: float
    thresholds: tuple[float, float]  # tuned (cheap, expensive) relative-price cut-offs
    stem: Path


def job_seed(base_seed: int, preset_id: str, year: int) -> int:
    """Deterministic per-job seed, independent of job order and worker count."""
    return base_seed + zlib.crc32(f"{preset_id}:{year}".encode()) % 2**31


def _mean_test_savings(
    policies: list[Policy],
    featured: pd.DataFrame,
    mix: ShiftMix,
    first: date,
    last: date,
    preset: BatteryPreset,
) -> dict[str, float]:
    """Mean savings per policy over ``TEST_SHIFT_SEEDS`` (A5)."""
    totals = {p.name: 0.0 for p in policies}
    for seed in TEST_SHIFT_SEEDS:
        # Shifts go over the whole series, once per pattern and shared by all policies.
        shifted = add_shifts(featured, mix, seed)
        for p in policies:
            totals[p.name] += simulate(p, first, last, preset, prices=shifted).summary.savings_cop
    return {name: total / len(TEST_SHIFT_SEEDS) for name, total in totals.items()}


def _run_job(
    args: tuple[TrainJob, pd.DataFrame, QConfig, Path, int, ShiftMix],
) -> JobReport:
    """Train, tune, save and evaluate one job (top level so Windows spawn can pickle it)."""
    job, featured, config, policies_dir, shift_draws, mix = args
    started = time.perf_counter()
    preset = get_preset(job.preset_id)
    split = scenario_split(featured, job.year)
    cfg = dataclasses.replace(config, seed=job_seed(config.seed, job.preset_id, job.year))

    train_frames = [
        add_shifts(split.train, mix, s) for s in train_shift_seeds(cfg.seed, shift_draws)
    ]
    result = train_q_learning(train_frames, preset, cfg)
    q_policy = TabularPolicy(
        result.q,
        result.discretizer,
        preset.id,
        metadata={
            "year": job.year,
            "epochs_run": result.epochs_run,
            "seed": cfg.seed,
            "shift_mix": mix.id,
            "shift_draws": shift_draws,
        },
    )
    stem = policy_stem(policies_dir, preset.id, job.year)
    q_policy.save(stem)
    threshold = PriceThreshold.tune(train_frames, preset)
    threshold.save(threshold_path(policies_dir, preset.id, job.year))

    # Test on the held-out months; the full featured frame gives rel_price its history.
    first = split.test["timestamp"].iloc[0].date()
    last = split.test["timestamp"].iloc[-1].date()
    savings = _mean_test_savings(
        [q_policy, ShiftSchedule(), threshold, NightDay(), PerfectForesight()],
        featured,
        mix,
        first,
        last,
        preset,
    )
    rules = {name: savings[name] for name in RULE_NAMES}
    best_rule = max(rules, key=lambda name: rules[name])  # first maximum wins ties
    best = rules[best_rule]
    return JobReport(
        preset_id=job.preset_id,
        year=job.year,
        epochs_run=result.epochs_run,
        seconds=time.perf_counter() - started,
        q_savings_cop=savings["q_learning"],
        shift_schedule_savings_cop=savings["shift_schedule"],
        threshold_savings_cop=savings["price_threshold"],
        night_day_savings_cop=savings["night_day"],
        perfect_savings_cop=savings["perfect_foresight"],
        best_rule=best_rule,
        q_vs_best_rule=savings["q_learning"] / best if best > 0 else float("nan"),
        thresholds=(threshold.cheap, threshold.expensive),
        stem=stem,
    )


def run_jobs(
    jobs: list[TrainJob],
    prices: pd.DataFrame,
    config: QConfig,
    policies_dir: Path | str,
    workers: int,
    shift_draws: int = DEFAULT_SHIFT_DRAWS,
    mix: str | ShiftMix = "default",
) -> list[JobReport]:
    """Run the jobs (in a process pool when ``workers > 1``) and return reports in order.

    Args:
        jobs: Jobs to run.
        prices: Raw hourly prices from ``load_prices``; the relative price is added here
            over the whole series, before splitting.
        config: Base config; the seed is derived per job with ``job_seed``.
        policies_dir: Output directory for the policy and threshold files.
        workers: Process count; 1 runs sequentially.
        shift_draws: Number of random shift patterns in each job's training data.
        mix: Shift mix (id or object) for training and testing (A5).

    Returns:
        One report per job, in the order of ``jobs``.

    Raises:
        ValueError: If the mix is unknown.
    """
    shift_mix = get_mix(mix)  # fail before any process starts
    featured = add_relative_price(prices)
    tasks = [(job, featured, config, Path(policies_dir), shift_draws, shift_mix) for job in jobs]
    if workers > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                return list(pool.map(_run_job, tasks))
        except (BrokenProcessPool, OSError) as exc:
            logger.warning("Process pool failed (%s); running sequentially", exc)
    return [_run_job(task) for task in tasks]


def _format_report(r: JobReport) -> str:
    """One line per job: savings (COP) of Q-learning, each rule and the optimum."""
    pct = 100 * r.q_savings_cop / r.perfect_savings_cop if r.perfect_savings_cop else float("nan")
    return (
        f"{r.preset_id} {r.year}: Q={r.q_savings_cop:,.0f} "
        f"ShiftSchedule={r.shift_schedule_savings_cop:,.0f} "
        f"PriceThreshold={r.threshold_savings_cop:,.0f} "
        f"NightDay={r.night_day_savings_cop:,.0f} Perfect={r.perfect_savings_cop:,.0f} | "
        f"best rule={r.best_rule} Q/best={r.q_vs_best_rule:.3f} Q/Perfect={pct:.1f}% | "
        f"epochs={r.epochs_run} time={r.seconds:.1f}s -> {r.stem}"
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints one line per job and the total time."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2025])
    parser.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=QConfig.max_epochs)
    parser.add_argument("--seed", type=int, default=QConfig.seed)
    parser.add_argument("--shift-draws", type=int, default=DEFAULT_SHIFT_DRAWS)
    parser.add_argument("--mix", default="default", choices=list(MIXES))
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_POLICIES_DIR)
    args = parser.parse_args(argv)

    jobs = [TrainJob(p, y) for y in args.scenarios for p in args.presets]
    workers = args.workers or min(len(jobs), os.cpu_count() or 1)
    config = QConfig(max_epochs=args.max_epochs, seed=args.seed)

    started = time.perf_counter()
    reports = run_jobs(
        jobs, load_prices(args.prices), config, args.out, workers, args.shift_draws, args.mix
    )
    total = time.perf_counter() - started

    for r in reports:
        print(_format_report(r))
    print(f"total {total:.1f}s with {workers} worker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
