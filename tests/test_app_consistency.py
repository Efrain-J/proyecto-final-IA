"""The dashboard and the chat agent must report the same simulation numbers.

Both average `simulate()` over the 5 test shift calendars (D15, A5), but they reach it
through two code paths: `energy.app.model.run_policies` (indicators) and the agent tool
`simulate_battery` (chat). Nothing else enforces that they agree, and a demo showing one
figure in the indicators and another in the chat would look broken. Offline; uses the
committed prices and Q-tables on a short period. Run with
``uv run pytest tests/test_app_consistency.py``.
"""
from __future__ import annotations

import json
from datetime import date

import pytest

from energy.agent.tools import build_registry, run_tool
from energy.app.model import run_policies
from energy.data.prices import load_prices
from energy.rag.retrieve import Hit
from energy.rl.features import add_relative_price

START, END = date(2025, 10, 1), date(2025, 10, 14)
PRESET = "8h"


class _NoSearch:
    """Retriever stub: the simulation tool never searches."""

    name = "stub"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return []


@pytest.fixture(scope="module")
def prices():
    return load_prices()


@pytest.fixture(scope="module")
def tool_result(prices) -> dict:
    registry = build_registry(_NoSearch(), prices=prices)
    args = {
        "policy": "q_learning", "preset": PRESET,
        "start_date": START.isoformat(), "end_date": END.isoformat(),
    }
    _, parsed = run_tool(registry, "simulate_battery", json.dumps(args))
    assert "error" not in parsed, parsed
    return parsed


@pytest.fixture(scope="module")
def runs(prices):
    return run_policies(add_relative_price(prices), PRESET, START, END, mix="default")


def test_q_learning_savings_match(tool_result, runs):
    q = runs["q_learning"]
    got = tool_result["savings_cop"]  # rounded to 2 decimals by the tool
    assert got["mean"] == pytest.approx(q.savings_cop, abs=1)
    assert got["min"] == pytest.approx(q.savings_min_cop, abs=1)
    assert got["max"] == pytest.approx(q.savings_max_cop, abs=1)


@pytest.mark.parametrize("rule", ["price_threshold", "shift_schedule", "night_day"])
def test_baselines_match(tool_result, runs, rule):
    # The tool rounds baselines to whole COP.
    assert tool_result["baselines"][rule] == pytest.approx(runs[rule].savings_cop, abs=1)
