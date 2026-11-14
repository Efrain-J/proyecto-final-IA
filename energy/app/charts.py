"""Plotly figure builders for the dashboard (pure, no streamlit).

Inputs: hourly traces and `PolicyRun`s from `energy.app.model`. Output: `go.Figure`s
that `app.py` shows with `st.plotly_chart`. Backgrounds are transparent and text uses
the host theme, so they read in light and dark mode. Q-learning gets the one saturated
color; the hand-written rules are muted so the comparison is not a rainbow.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from energy.agent.render import POLICY_NAMES
from energy.app.model import APP_POLICIES, PolicyRun
from energy.rl.battery import BatteryPreset, get_preset

# Mid-tone hues: legible on both light and dark surfaces. Q-learning is the saturated one.
POLICY_COLORS: dict[str, str] = {
    "q_learning": "#2F7BEF",
    "price_threshold": "#C58B2A",
    "shift_schedule": "#7B8A99",
    "night_day": "#A06FB0",
}
CHARGE_COLOR = "#2E9E6B"
DISCHARGE_COLOR = "#D9534F"
PRICE_COLOR = "#8A94A0"
_GRID = "rgba(128,128,128,0.2)"
# Purple = factory running (A5, D15). Translucent so the grid and lines stay readable; the
# hue differs from the blue of Q-learning and from the muted purple of night/day.
_FACTORY_ON = "rgba(124, 77, 255, 0.12)"
_FACTORY_ON_SWATCH = "rgba(124, 77, 255, 0.45)"


def _layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        # Title pinned to the top of the figure; the 120 px top margin leaves a band
        # between it and the plot for the horizontal legend, so they don't crowd.
        title=dict(text=title, x=0, xanchor="left", y=0.98, yref="container", yanchor="top"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(t=120, r=20, l=60, b=40),
        hovermode="x unified",
        separators=",.",  # es-CO: decimal comma, thousands dot, as render.cop() in the chat
    )
    fig.update_xaxes(gridcolor=_GRID, showline=False)
    fig.update_yaxes(gridcolor=_GRID, zeroline=False)
    return fig


def _on_runs(factory_on: pd.Series) -> list[tuple[int, int]]:
    """Half-open index ranges ``[a, b)`` of contiguous hours with the factory running."""
    on = factory_on.to_numpy(dtype=bool).astype(int)
    edges = pd.Series([0, *on, 0]).diff().to_numpy()[1:]
    starts = (edges == 1).nonzero()[0]
    ends = (edges == -1).nonzero()[0]
    return list(zip(starts.tolist(), ends.tolist()))


def operation_figure(
    week: pd.DataFrame, preset: str | BatteryPreset, policy_name: str
) -> go.Figure:
    """Price with charge/discharge markers on top, state of charge (%) below.

    Hours with the factory running (A5, D15) are shaded purple across both panels, since
    the battery can only discharge then; a legend entry explains the shading.

    Args:
        week: Hourly trace slice (timestamp, price_cop_kwh, action, soc_level, factory_on).
        preset: Battery preset or its id; its duration converts the level to percent.
        policy_name: Spanish policy name shown in the title.
    """
    preset = get_preset(preset)
    ts = week["timestamp"]
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.06
    )
    fig.add_trace(
        go.Scatter(
            x=ts, y=week["price_cop_kwh"], mode="lines", name="Precio (COP/kWh)",
            line=dict(color=PRICE_COLOR, width=2),
            hovertemplate="%{y:,.0f} COP/kWh<extra>Precio</extra>",
        ),
        row=1, col=1,
    )
    for code, name, color, symbol in (
        (1, "Carga (compra)", CHARGE_COLOR, "triangle-up"),
        (2, "Descarga (ahorro)", DISCHARGE_COLOR, "triangle-down"),
    ):
        sel = week[week["action"] == code]
        fig.add_trace(
            go.Scatter(
                x=sel["timestamp"], y=sel["price_cop_kwh"], mode="markers", name=name,
                marker=dict(
                    color=color, symbol=symbol, size=10,
                    line=dict(width=2, color="rgba(255,255,255,0.8)"),
                ),
                hovertemplate="%{y:,.0f} COP/kWh<extra>" + name + "</extra>",
            ),
            row=1, col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=ts, y=week["soc_level"] / preset.duration_h * 100, mode="lines",
            name="Nivel de carga (%)", line=dict(color=POLICY_COLORS["q_learning"], width=2),
            line_shape="hv", hovertemplate="%{y:.0f}%<extra>Nivel de carga</extra>",
        ),
        row=2, col=1,
    )
    ts = ts.reset_index(drop=True)
    # One shape per run, in paper coordinates: it spans both subplots (shared x axis).
    for a, b in _on_runs(week["factory_on"].reset_index(drop=True)):
        fig.add_shape(
            type="rect", xref="x", yref="paper", y0=0, y1=1,
            x0=ts[a], x1=ts[b - 1] + pd.Timedelta(hours=1),
            fillcolor=_FACTORY_ON, line_width=0, layer="below",
        )
    # Legend-only trace: shapes have no legend entry.
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers", name="Fábrica funcionando",
            marker=dict(color=_FACTORY_ON_SWATCH, symbol="square", size=12),
            hoverinfo="skip",
        ),
        row=1, col=1,
    )
    fig.update_yaxes(title_text="Precio (COP/kWh)", row=1, col=1)
    fig.update_yaxes(title_text="Carga (%)", range=[0, 100], row=2, col=1)
    return _layout(
        fig,
        f"Operación de la batería ({preset.label}) con {policy_name}",
    )


def cumulative_figure(runs: dict[str, PolicyRun]) -> go.Figure:
    """Cumulative savings (COP) over the period, one line per policy."""
    fig = go.Figure()
    for pid, run in runs.items():
        fig.add_trace(
            go.Scatter(
                x=run.cum_savings.index, y=run.cum_savings.values, mode="lines",
                name=POLICY_NAMES[pid],
                line=dict(color=POLICY_COLORS[pid], width=3 if pid == "q_learning" else 2),
                hovertemplate="%{y:,.0f} COP<extra>" + POLICY_NAMES[pid] + "</extra>",
            )
        )
    fig.update_yaxes(title_text="Ahorro acumulado (COP)")
    return _layout(fig, "Ahorro acumulado en el periodo (promedio de 5 calendarios de turnos)")


def size_figure(table: pd.DataFrame) -> go.Figure:
    """Grouped bars: savings (COP) per battery size, one bar per policy."""
    fig = go.Figure()
    for pid in APP_POLICIES:
        sub = table[table["policy_id"] == pid]
        if sub.empty:
            continue
        fig.add_trace(
            go.Bar(
                x=[get_preset(p).label for p in sub["preset_id"]], y=sub["savings_cop"],
                name=POLICY_NAMES[pid], marker_color=POLICY_COLORS[pid],
                hovertemplate="%{y:,.0f} COP<extra>" + POLICY_NAMES[pid] + "</extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text="Ahorro en el periodo (COP)")
    fig.update_xaxes(title_text="Tamaño de la batería")
    return _layout(fig, "Ahorro según el tamaño de la batería")
