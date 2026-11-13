"""Tabular Q-learning for battery dispatch (D14) and the saved-policy wrapper.

State = (hour block, charge level, price level); actions idle/charge/discharge; relative
reward ``rel_price * -grid_kwh`` (option A); weekly episodes starting empty. The hot
loop runs over plain Python lists and lookup tables (see the speed measures in the
plan), so one table trains in seconds. Library only; ``energy.rl.train`` is the CLI:
``uv run python -m energy.rl.train``.

Policies are stored as ``<stem>.npz`` (``q`` float32 and ``edges``) plus ``<stem>.json``
(preset, name, edges, metadata): no pickle, a few KB.
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
from energy.rl.features import N_HOUR_BLOCKS, N_PRICE_LEVELS, Discretizer, hour_block


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

    q: np.ndarray  # float32, (12, n_levels, 4, 3)
    discretizer: Discretizer
    epochs_run: int
    epoch_returns: list[float]  # summed (relative) reward per epoch, exploring


def _greedy_table(q: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Argmax over valid actions: ``[hour_block, level, price_level]``; ties go to idle."""
    masked = np.where(mask[None, :, None, :], q, -np.inf)
    return masked.argmax(axis=-1).astype(np.int8)


def train_q_learning(
    train_frame: pd.DataFrame, preset: BatteryPreset | str, config: QConfig = QConfig()
) -> TrainResult:
    """Learn a Q-table on ``train_frame``.

    Episodes are consecutive ``episode_hours`` blocks (the last may be shorter),
    shuffled every epoch, each starting empty; the last step of an episode does not
    bootstrap. Epsilon decays linearly over ``max_epochs``. Training stops after
    ``patience`` consecutive epochs with an unchanged greedy policy.

    Args:
        train_frame: Hourly frame with ``timestamp``, ``price_cop_kwh`` and ``rel_price``.
        preset: Battery (or its id).
        config: Hyper-parameters, including the seed.

    Returns:
        The float32 Q-table, the fitted discretizer and per-epoch returns.

    Raises:
        ValueError: If ``rel_price`` is missing.
    """
    if "rel_price" not in train_frame.columns:
        raise ValueError("train_frame needs a 'rel_price' column (see add_relative_price)")
    preset = get_preset(preset)
    n_levels = preset.n_levels
    next_level, grid_kwh = transition_tables(preset)
    mask = valid_mask(preset)

    rel = train_frame["rel_price"].to_numpy(dtype=float)
    disc = Discretizer.fit(rel)
    # Per-hour part of the state index; the level adds ``level * N_PRICE_LEVELS``.
    base = (
        hour_block(train_frame["timestamp"]).astype(np.int64) * n_levels * N_PRICE_LEVELS
        + disc.transform(rel)
    ).tolist()
    rel_list = rel.tolist()
    nxt = next_level.tolist()
    grid = grid_kwh.tolist()
    valid = [[a for a in range(N_ACTIONS) if mask[lv, a]] for lv in range(n_levels)]

    n = len(rel_list)
    bounds = [(s, min(s + config.episode_hours, n)) for s in range(0, n, config.episode_hours)]
    q = [0.0] * (N_HOUR_BLOCKS * n_levels * N_PRICE_LEVELS * N_ACTIONS)
    shape = (N_HOUR_BLOCKS, n_levels, N_PRICE_LEVELS, N_ACTIONS)

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
                s = (base[t] + level * N_PRICE_LEVELS) * N_ACTIONS
                acts = valid[level]
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
                    s2 = (base[t + 1] + nl * N_PRICE_LEVELS) * N_ACTIONS
                    target = r + gamma * max(q[s2 + c] for c in valid[nl])
                else:
                    target = r  # episode end: no bootstrap
                q[s + a] += alpha * (target - q[s + a])
                level = nl
        returns.append(total)

        greedy = _greedy_table(np.asarray(q).reshape(shape), mask)
        unchanged = unchanged + 1 if prev_greedy is not None and np.array_equal(greedy, prev_greedy) else 0
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
        self.discretizer = discretizer
        self.preset_id = preset_id
        self.name = name
        self.metadata = dict(metadata or {})

    def actions(self, frame: pd.DataFrame, preset: BatteryPreset) -> np.ndarray:
        """Greedy actions for ``frame`` (needs ``rel_price``), starting empty.

        Raises:
            ValueError: If ``preset`` is not the one this table was trained for.
        """
        preset = get_preset(preset)
        if preset.id != self.preset_id:
            raise ValueError(f"policy trained for {self.preset_id!r}, got {preset.id!r}")
        greedy = _greedy_table(self.q, valid_mask(preset))
        blocks = hour_block(frame["timestamp"])
        levels_p = self.discretizer.transform(frame["rel_price"].to_numpy(dtype=float))
        next_level, _ = transition_tables(preset)
        nxt = next_level.tolist()
        out = np.zeros(len(frame), dtype=np.int8)
        level = 0
        for t, (b, p) in enumerate(zip(blocks.tolist(), levels_p.tolist())):
            a = int(greedy[b, level, p])
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
