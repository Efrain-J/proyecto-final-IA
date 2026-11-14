"""Tests for `energy.app.charts`: pure Plotly figure builders for the dashboard.

Offline and simulation-free: hand-made hourly frames with known values.
Run with `uv run pytest tests/test_app_charts.py`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from energy.agent.render import POLICY_NAMES
from energy.app.charts import POLICY_COLORS, cumulative_figure, operation_figure, size_figure
from energy.app.model import APP_POLICIES, CalendarRun, PolicyRun
from energy.rl.battery import get_preset

PRESET_ID = "4h"
N = 48
# Factory off at hours [5, 9), [20, 22) and the last hour, so it runs in [0, 5), [9, 20),
# [22, 47): three contiguous on-runs.
OFF_RUNS = [(5, 9), (20, 22), (47, 48)]
ON_RUNS = [(0, 5), (9, 20), (22, 47)]


@pytest.fixture
def week() -> pd.DataFrame:
    ts = pd.date_range("2025-10-06", periods=N, freq="h")
    action = np.zeros(N, dtype=int)
    action[[2, 3, 30]] = 1
    action[[10, 11, 40]] = 2
    factory_on = np.ones(N, dtype=bool)
    for a, b in OFF_RUNS:
        factory_on[a:b] = False
    return pd.DataFrame({
        "timestamp": ts,
        "price_cop_kwh": 200.0 + np.arange(N),
        "action": action,
        "soc_level": np.linspace(0, 4, N),
        "factory_on": factory_on,
    })


def make_run(pid: str, values: list[float]) -> PolicyRun:
    idx = pd.date_range("2025-10-06", periods=len(values), freq="h")
    return PolicyRun(
        policy_id=pid, policy_used=pid, savings_cop=values[-1], savings_min_cop=values[-1],
        savings_max_cop=values[-1], cycles=1.0, cum_savings=pd.Series(values, index=idx),
        calendars=(
            CalendarRun(
                seed=1000, savings_cop=values[-1], cycles=1.0,
                hourly=pd.DataFrame({"timestamp": idx}),
            ),
        ),
    )


def _by_prefix(fig: go.Figure, prefix: str) -> list:
    return [t for t in fig.data if t.name.startswith(prefix)]


def _soc_traces(fig: go.Figure) -> list:
    return [
        t for t in fig.data
        if "carga" in t.name.lower() and not t.name.startswith(("Carga", "Descarga"))
    ]


# --- operation_figure ---


def test_operation_trace_names(week):
    fig = operation_figure(week, PRESET_ID, "Q-learning")
    assert isinstance(fig, go.Figure)
    assert len([t for t in fig.data if "Precio" in t.name]) == 1
    assert len(_by_prefix(fig, "Carga")) == 1
    assert len(_by_prefix(fig, "Descarga")) == 1
    assert len(_soc_traces(fig)) == 1


def test_operation_markers_at_action_hours(week):
    fig = operation_figure(week, PRESET_ID, "Q-learning")
    charge = _by_prefix(fig, "Carga")[0]
    discharge = _by_prefix(fig, "Descarga")[0]
    assert list(pd.to_datetime(charge.x)) == list(week.loc[week["action"] == 1, "timestamp"])
    assert list(pd.to_datetime(discharge.x)) == list(week.loc[week["action"] == 2, "timestamp"])
    assert charge.mode == "markers" and discharge.mode == "markers"


def test_operation_price_trace_is_full_series(week):
    fig = operation_figure(week, PRESET_ID, "Q-learning")
    price = next(t for t in fig.data if "Precio" in t.name)
    assert len(price.x) == N
    assert np.allclose(price.y, week["price_cop_kwh"])


def test_operation_shapes_one_per_factory_on_run(week):
    fig = operation_figure(week, PRESET_ID, "Q-learning")
    assert len(fig.layout.shapes) == len(ON_RUNS)
    assert all(s.type == "rect" for s in fig.layout.shapes)
    starts = [pd.Timestamp(s.x0) for s in fig.layout.shapes]
    assert starts == [week["timestamp"].iloc[a] for a, _ in ON_RUNS]


def test_operation_shapes_span_full_height(week):
    for s in operation_figure(week, PRESET_ID, "Q-learning").layout.shapes:
        assert s.yref == "paper" and s.y0 == 0 and s.y1 == 1


def test_operation_single_shape_when_factory_always_on(week):
    week["factory_on"] = True
    assert len(operation_figure(week, PRESET_ID, "x").layout.shapes) == 1


def test_operation_no_shapes_when_always_off(week):
    week["factory_on"] = False
    assert len(operation_figure(week, PRESET_ID, "x").layout.shapes) == 0


def test_operation_legend_entry_for_factory_running(week):
    fig = operation_figure(week, PRESET_ID, "Q-learning")
    assert len(_by_prefix(fig, "Fábrica funcionando")) == 1


def test_operation_soc_percent(week):
    preset = get_preset(PRESET_ID)
    soc = _soc_traces(operation_figure(week, PRESET_ID, "Q-learning"))[0]
    y = np.asarray(soc.y, dtype=float)
    assert np.allclose(y, week["soc_level"] / preset.duration_h * 100)
    assert y.min() >= 0 and y.max() <= 100


@pytest.mark.parametrize("as_object", [False, True], ids=["id", "object"])
def test_operation_accepts_preset_id_or_object(week, as_object):
    preset = get_preset(PRESET_ID) if as_object else PRESET_ID
    fig = operation_figure(week, preset, "Q-learning")
    assert len(fig.data) == 5


def test_operation_two_rows_share_x(week):
    fig = operation_figure(week, PRESET_ID, "Q-learning")
    assert fig.layout.xaxis2 is not None
    assert fig.layout.xaxis.matches == "x2" or fig.layout.xaxis2.matches == "x"


# --- cumulative_figure ---


def test_cumulative_one_trace_per_run_with_spanish_names():
    runs = {
        "q_learning": make_run("q_learning", [0.0, 5.0, 12.0]),
        "night_day": make_run("night_day", [0.0, 3.0, 4.0]),
    }
    fig = cumulative_figure(runs)
    assert [t.name for t in fig.data] == [POLICY_NAMES["q_learning"], POLICY_NAMES["night_day"]]
    assert np.allclose(fig.data[0].y, runs["q_learning"].cum_savings.values)
    assert np.allclose(fig.data[1].y, runs["night_day"].cum_savings.values)


# --- size_figure ---


@pytest.fixture
def table() -> pd.DataFrame:
    rows = [
        (preset, pid, 100.0 * (i + 1) + 10.0 * j)
        for i, preset in enumerate(["4h", "8h", "12h"])
        for j, pid in enumerate(APP_POLICIES)
    ]
    return pd.DataFrame(rows, columns=["preset_id", "policy_id", "savings_cop"])


def test_size_one_bar_trace_per_policy(table):
    fig = size_figure(table)
    assert len(fig.data) == len(APP_POLICIES)
    assert all(t.type == "bar" for t in fig.data)
    assert {t.name for t in fig.data} == {POLICY_NAMES[p] for p in APP_POLICIES}


def test_size_x_labels_in_hours_and_values(table):
    fig = size_figure(table)
    q = next(t for t in fig.data if t.name == POLICY_NAMES["q_learning"])
    assert len(q.x) == 3
    assert all(" h" in str(x) for x in q.x)
    expected = table.loc[table["policy_id"] == "q_learning", "savings_cop"].tolist()
    assert np.allclose(q.y, expected)


# --- colors ---


def test_policy_colors_cover_app_policies_and_are_distinct():
    assert set(APP_POLICIES) <= set(POLICY_COLORS)
    colors = [POLICY_COLORS[p] for p in APP_POLICIES]
    assert len(set(colors)) == len(colors)


def test_title_sits_above_the_legend_band(week, table):
    # The title is pinned to the top of the figure and the top margin leaves a band for
    # the horizontal legend, so the two never overlap (seen in the first dashboard run).
    run = make_run("q_learning", [0.0, 1.0])
    for fig in (
        operation_figure(week, PRESET_ID, "Q-learning"),
        cumulative_figure({"q_learning": run}),
        size_figure(table),
    ):
        assert fig.layout.title.yref == "container"
        assert fig.layout.title.yanchor == "top"
        assert fig.layout.legend.yanchor == "bottom"
        assert fig.layout.margin.t >= 120
