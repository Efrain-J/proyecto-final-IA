"""Tabular Q-learning for battery dispatch (D14, D15) and the saved-policy wrapper.

State = (hour block, charge level, price level, today's shift type); actions
idle/charge/discharge, with discharge masked while the factory is off (A5); relative
reward ``rel_price * -grid_kwh`` (option A); weekly episodes starting empty. It trains on
several frames at once (the same prices under different random shift patterns). The hot
loop runs over plain Python lists and lookup tables (see the speed measures in the
plan): the shift type is folded into the per-hour state index and the valid actions are
precomputed per (level, factory on), so one table trains in tens of seconds. Library
only; ``energy.rl.train`` is the CLI: ``uv run python -m energy.rl.train``.

Policies are stored as ``<stem>.npz`` (``q`` float32 and ``edges``) plus ``<stem>.json``
(preset, name, edges, metadata incl. ``shift_mix`` and ``shift_draws``): no pickle, a few
KB.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from energy.data.io import write_bytes_atomic, write_json_atomic
from energy.rl.battery import (
    N_ACTIONS,
    BatteryPreset,
    get_preset,
    transition_tables,
    valid_mask,
)
from energy.rl.factory import N_SHIFT_TYPES
from energy.rl.features import N_HOUR_BLOCKS, N_PRICE_LEVELS, Discretizer, hour_block
from energy.rl.policies import factory_columns

REQUIRED_COLUMNS = ("rel_price", "shift_type", "factory_on")


@dataclass(frozen=True)
class QConfig:
    """Hyper-parameters (defaults from D14).

    ``alpha_end`` is an addition to D14: the learning rate decays linearly from ``alpha``
    to ``alpha_end`` over ``max_epochs``, which damps the constant-alpha jitter that
    otherwise keeps flipping near-tied greedy actions and defeats early stopping.
    """

    max_epochs: int = 300
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    patience: int = 20
    episode_hours: int = 168
    seed: int = 0
    alpha_end: float = 0.02  # learning rate decays linearly to this over max_epochs


@dataclass
class TrainResult:
    """Output of ``train_q_learning``."""

    q: np.ndarray  # float32, (12, n_levels, 4, N_SHIFT_TYPES, 3)
    discretizer: Discretizer
    epochs_run: int
    epoch_returns: list[float]  # summed (relative) reward per epoch, exploring


def _greedy_table(q: np.ndarray, preset: BatteryPreset) -> np.ndarray:
    """Argmax over valid actions: ``[factory_on, hour_block, level, price_level, shift]``.

    ``factory_on`` is 0 or 1; ties go to idle (the first action).
    """
    tables = []
    for factory_on in (False, True):
        mask = valid_mask(preset, factory_on)
        masked = np.where(mask[None, :, None, None, :], q, -np.inf)
        tables.append(masked.argmax(axis=-1).astype(np.int8))
    return np.stack(tables)


def train_q_learning(
    train_frames: pd.DataFrame | list[pd.DataFrame],
    preset: BatteryPreset | str,
    config: QConfig = QConfig(),
) -> TrainResult:
    """Learn a Q-table on one frame or on several (e.g. one per shift pattern).

    Episodes are consecutive ``episode_hours`` blocks inside each frame (the last may be
    shorter), shuffled every epoch, each starting empty; the last step of an episode does
    not bootstrap. Epsilon decays linearly over ``max_epochs``. Training stops after
    ``patience`` consecutive epochs with an unchanged greedy policy. The discretizer is
    fitted on the relative prices of all frames together.

    Args:
        train_frames: Hourly frame(s) with ``timestamp``, ``rel_price``, ``shift_type``
            and ``factory_on``. A single frame equals a list with only that frame.
        preset: Battery (or its id).
        config: Hyper-parameters, including the seed.

    Returns:
        The float32 Q-table, the fitted discretizer and per-epoch returns.

    Raises:
        ValueError: If there are no frames or any of them lacks a required column.
    """
    frames = [train_frames] if isinstance(train_frames, pd.DataFrame) else list(train_frames)
    if not frames:
        raise ValueError("train_frames is empty")
    for frame in frames:
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"training frame is missing {missing} (see add_relative_price, add_shifts)")
    preset = get_preset(preset)
    n_levels = preset.n_levels
    next_level, grid_kwh = transition_tables(preset)

    rel = np.concatenate([f["rel_price"].to_numpy(dtype=float) for f in frames])
    shift = np.concatenate([f["shift_type"].to_numpy() for f in frames]).astype(np.int64)
    blocks = np.concatenate([hour_block(f["timestamp"]) for f in frames]).astype(np.int64)
    disc = Discretizer.fit(rel)
    # Per-hour part of the state index; the level adds ``level * level_stride``.
    level_stride = N_PRICE_LEVELS * N_SHIFT_TYPES
    base = (
        blocks * n_levels * level_stride
        + disc.transform(rel).astype(np.int64) * N_SHIFT_TYPES
        + shift
    ).tolist()
    on = np.concatenate([f["factory_on"].to_numpy() for f in frames]).astype(np.int64).tolist()
    rel_list = rel.tolist()
    nxt = next_level.tolist()
    grid = grid_kwh.tolist()
    masks = [valid_mask(preset, factory_on) for factory_on in (False, True)]
    # valid[level][factory_on] -> allowed actions
    valid = [
        [[a for a in range(N_ACTIONS) if masks[o][lv, a]] for o in (0, 1)]
        for lv in range(n_levels)
    ]

    bounds, offset = [], 0
    for frame in frames:  # episodes never span two frames
        m = len(frame)
        bounds += [
            (offset + s, offset + min(s + config.episode_hours, m))
            for s in range(0, m, config.episode_hours)
        ]
        offset += m
    n = offset
    shape = (N_HOUR_BLOCKS, n_levels, N_PRICE_LEVELS, N_SHIFT_TYPES, N_ACTIONS)
    q = [0.0] * int(np.prod(shape))

    rng = np.random.default_rng(config.seed)
    gamma = config.gamma
    prev_greedy: np.ndarray | None = None
    unchanged, returns = 0, []

    for epoch in range(config.max_epochs):
        eps = config.epsilon_start + (config.epsilon_end - config.epsilon_start) * (
            epoch / config.max_epochs
        )
        alpha = config.alpha + (config.alpha_end - config.alpha) * (epoch / config.max_epochs)
        explore = (rng.random(n) < eps).tolist()  # pre-drawn once per epoch
        pick = rng.random(n).tolist()
        total = 0.0
        for e in rng.permutation(len(bounds)).tolist():
            start, end = bounds[e]
            level = 0
            for t in range(start, end):
                s = (base[t] + level * level_stride) * N_ACTIONS
                acts = valid[level][on[t]]
                if explore[t]:
                    a = acts[int(pick[t] * len(acts))]
                else:
                    a = acts[0]
                    best = q[s + a]
                    for c in acts[1:]:
                        if q[s + c] > best:
                            a, best = c, q[s + c]
                nl = nxt[level][a]
                r = rel_list[t] * -grid[level][a]
                total += r
                if t + 1 < end:
                    s2 = (base[t + 1] + nl * level_stride) * N_ACTIONS
                    target = r + gamma * max(q[s2 + c] for c in valid[nl][on[t + 1]])
                else:
                    target = r  # episode end: no bootstrap
                q[s + a] += alpha * (target - q[s + a])
                level = nl
        returns.append(total)

        greedy = _greedy_table(np.asarray(q).reshape(shape), preset)
        same = prev_greedy is not None and np.array_equal(greedy, prev_greedy)
        unchanged = unchanged + 1 if same else 0
        prev_greedy = greedy
        if unchanged >= config.patience:
            break

    return TrainResult(
        q=np.asarray(q, dtype=np.float32).reshape(shape),
        discretizer=disc,
        epochs_run=len(returns),
        epoch_returns=returns,
    )


def policy_stem(policies_dir: Path | str, preset_id: str, year: int) -> Path:
    """Path stem (no suffix) of a saved policy: ``<dir>/q_<preset>_<year>``."""
    return Path(policies_dir) / f"q_{preset_id}_{year}"


class TabularPolicy:
    """Greedy policy over a Q-table, never choosing an impossible action."""

    def __init__(
        self,
        q: np.ndarray,
        discretizer: Discretizer,
        preset_id: str,
        name: str = "q_learning",
        metadata: dict | None = None,
    ) -> None:
        self.q = np.asarray(q, dtype=np.float32)
        if self.q.ndim != 5:  # e.g. an old D14 table without the shift dimension
            raise ValueError(f"Q-table must have 5 dimensions (D15), got shape {self.q.shape}")
        self.discretizer = discretizer
        self.preset_id = preset_id
        self.name = name
        self.metadata = dict(metadata or {})

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        """Greedy actions for ``frame`` (needs ``rel_price``), starting empty.

        Missing shift columns mean the factory always runs (``factory_columns``); the
        policy never discharges in an hour the factory is off.

        Raises:
            ValueError: If ``preset`` is not the one this table was trained for.
        """
        preset = get_preset(preset)
        if preset.id != self.preset_id:
            raise ValueError(f"policy trained for {self.preset_id!r}, got {preset.id!r}")
        greedy = _greedy_table(self.q, preset)
        shift_type, factory_on = factory_columns(frame)
        blocks = hour_block(frame["timestamp"])
        levels_p = self.discretizer.transform(frame["rel_price"].to_numpy(dtype=float))
        next_level, _ = transition_tables(preset)
        nxt = next_level.tolist()
        out = np.zeros(len(frame), dtype=np.int8)
        level = 0
        for t, (o, b, p, st) in enumerate(
            zip(factory_on.astype(np.int64).tolist(), blocks.tolist(), levels_p.tolist(), shift_type.tolist())
        ):
            a = int(greedy[o, b, level, p, st])
            out[t] = a
            level = nxt[level][a]
        return out

    def save(self, stem: Path | str) -> None:
        """Write ``<stem>.npz`` and ``<stem>.json`` atomically."""
        stem = Path(stem)
        edges = np.asarray(self.discretizer.edges, dtype=np.float64)
        buffer = io.BytesIO()
        np.savez_compressed(buffer, q=self.q.astype(np.float32), edges=edges)
        write_bytes_atomic(stem.with_suffix(".npz"), buffer.getvalue())
        write_json_atomic(
            stem.with_suffix(".json"),
            {
                "preset_id": self.preset_id,
                "name": self.name,
                "edges": edges.tolist(),
                "metadata": self.metadata,
            },
        )

    @classmethod
    def load(cls, stem: Path | str) -> TabularPolicy:
        """Read a policy written by ``save``."""
        stem = Path(stem)
        meta = json.loads(stem.with_suffix(".json").read_text(encoding="utf-8"))
        with np.load(stem.with_suffix(".npz"), allow_pickle=False) as data:
            q, edges = data["q"], data["edges"]
        return cls(
            q,
            Discretizer(edges=tuple(float(x) for x in edges)),  # type: ignore[arg-type]
            meta["preset_id"],
            name=meta.get("name", "q_learning"),
            metadata=meta.get("metadata"),
        )
