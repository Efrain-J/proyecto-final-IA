"""Train the Q-learning policies, one job per (battery preset, year), in parallel.

Run: ``uv run python -m energy.rl.train --scenarios 2025`` (writes
``artifacts/policies/q_<preset>_<year>.npz/.json`` and prints a table comparing the
test-period savings of Q-learning, NightDay and PerfectForesight).

Flags: ``--scenarios`` years (default 2025), ``--presets`` (default all), ``--workers``
(default min(jobs, cpu_count); 1 = sequential), ``--max-epochs``, ``--seed``,
``--prices`` CSV, ``--out`` directory. Jobs are independent and each seed depends only
on (base seed, preset, year), so results do not depend on workers or job order.
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
from pathlib import Path

import pandas as pd

from energy.data.prices import DEFAULT_PRICES_PATH, load_prices, scenario_split
from energy.rl.battery import PRESETS, get_preset
from energy.rl.features import add_relative_price
from energy.rl.policies import NightDay, PerfectForesight
from energy.rl.qlearning import QConfig, TabularPolicy, policy_stem, train_q_learning
from energy.rl.simulate import DEFAULT_POLICIES_DIR, simulate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainJob:
    """One Q-table to train."""

    preset_id: str
    year: int


@dataclass(frozen=True)
class JobReport:
    """Result of a job: training effort and test-period savings in COP."""

    preset_id: str
    year: int
    epochs_run: int
    seconds: float
    test_savings_cop: float
    night_day_savings_cop: float
    perfect_savings_cop: float
    stem: Path


def job_seed(base_seed: int, preset_id: str, year: int) -> int:
    """Deterministic per-job seed, independent of job order and worker count."""
    return base_seed + zlib.crc32(f"{preset_id}:{year}".encode()) % 2**31


def _run_job(args: tuple[TrainJob, pd.DataFrame, QConfig, Path]) -> JobReport:
    """Train, save and evaluate one job (top level so Windows spawn can pickle it)."""
    job, featured, config, policies_dir = args
    started = time.perf_counter()
    preset = get_preset(job.preset_id)
    split = scenario_split(featured, job.year)
    cfg = dataclasses.replace(config, seed=job_seed(config.seed, job.preset_id, job.year))
    result = train_q_learning(split.train, preset, cfg)
    policy = TabularPolicy(
        result.q,
        result.discretizer,
        preset.id,
        metadata={"year": job.year, "epochs_run": result.epochs_run, "seed": cfg.seed},
    )
    stem = policy_stem(policies_dir, preset.id, job.year)
    policy.save(stem)

    # Test on the held-out months; the full featured frame gives rel_price its history.
    first = split.test["timestamp"].iloc[0].date()
    last = split.test["timestamp"].iloc[-1].date()
    savings = {
        p.name: simulate(p, first, last, preset, prices=featured).summary.savings_cop
        for p in (policy, NightDay(), PerfectForesight())
    }
    return JobReport(
        preset_id=job.preset_id,
        year=job.year,
        epochs_run=result.epochs_run,
        seconds=time.perf_counter() - started,
        test_savings_cop=savings["q_learning"],
        night_day_savings_cop=savings["night_day"],
        perfect_savings_cop=savings["perfect_foresight"],
        stem=stem,
    )


def run_jobs(
    jobs: list[TrainJob],
    prices: pd.DataFrame,
    config: QConfig,
    policies_dir: Path | str,
    workers: int,
) -> list[JobReport]:
    """Run the jobs (in a process pool when ``workers > 1``) and return reports in order.

    Args:
        jobs: Jobs to run.
        prices: Raw hourly prices from ``load_prices``; the relative price is added here
            over the whole series, before splitting.
        config: Base config; the seed is derived per job with ``job_seed``.
        policies_dir: Output directory for the policy files.
        workers: Process count; 1 runs sequentially.

    Returns:
        One report per job, in the order of ``jobs``.
    """
    featured = add_relative_price(prices)
    tasks = [(job, featured, config, Path(policies_dir)) for job in jobs]
    if workers > 1 and len(tasks) > 1:
        try:
            with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
                return list(pool.map(_run_job, tasks))
        except (BrokenProcessPool, OSError) as exc:
            logger.warning("Process pool failed (%s); running sequentially", exc)
    return [_run_job(task) for task in tasks]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; prints one line per job and the total time."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--scenarios", type=int, nargs="+", default=[2025])
    parser.add_argument("--presets", nargs="+", default=list(PRESETS), choices=list(PRESETS))
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--max-epochs", type=int, default=QConfig.max_epochs)
    parser.add_argument("--seed", type=int, default=QConfig.seed)
    parser.add_argument("--prices", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--out", type=Path, default=DEFAULT_POLICIES_DIR)
    args = parser.parse_args(argv)

    jobs = [TrainJob(p, y) for y in args.scenarios for p in args.presets]
    workers = args.workers or min(len(jobs), os.cpu_count() or 1)
    config = QConfig(max_epochs=args.max_epochs, seed=args.seed)

    started = time.perf_counter()
    reports = run_jobs(jobs, load_prices(args.prices), config, args.out, workers)
    total = time.perf_counter() - started

    for r in reports:
        pct = 100 * r.test_savings_cop / r.perfect_savings_cop if r.perfect_savings_cop else float("nan")
        print(
            f"{r.preset_id} {r.year}: epochs={r.epochs_run} time={r.seconds:.1f}s "
            f"test savings COP: Q={r.test_savings_cop:,.0f} NightDay={r.night_day_savings_cop:,.0f} "
            f"Perfect={r.perfect_savings_cop:,.0f} ({pct:.1f}% of optimum) -> {r.stem}"
        )
    print(f"total {total:.1f}s with {workers} worker(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
