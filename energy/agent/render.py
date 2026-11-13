"""Deterministic Spanish rendering of ``simulate_battery`` results (D16).

The model used to copy the simulation numbers into its answer by hand and sometimes changed
a digit or flipped the sign of a comparison. These pure functions (no I/O) write every
figure, table, assumption and source line from the tool output, so the loop prepends the
block to the answer and the model writes only the explanation.

Inputs are the successful ``simulate_battery`` result dicts; outputs are Markdown strings.
Library only; see ``energy.agent.loop``.
"""

from __future__ import annotations

import calendar
from datetime import date
from typing import Any

MAX_COMPARISON_ROWS = 6
_MONTHS = ("ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic")

POLICY_NAMES: dict[str, str] = {
    "q_learning": "Q-learning",
    "price_threshold": "Regla de umbral de precio",
    "shift_schedule": "Horario por turnos",
    "night_day": "Carga nocturna / descarga diurna",
    "perfect_foresight": "Óptimo con previsión perfecta",
    "no_battery": "Sin batería",
}


def _plain(value: float) -> str:
    """Integer with ``.`` as thousands separator (es-CO)."""
    return f"{round(value):,}".replace(",", ".")


def cop(value: float) -> str:
    """Pesos as ``3.131.506 COP`` (thousands with ``.``, no decimals)."""
    return f"{_plain(value)} COP"


def pct(value: float) -> str:
    """Percentage as ``5,3 %`` (one decimal, comma)."""
    return f"{value:.1f}".replace(".", ",") + " %"


def _name(policy_id: str) -> str:
    return POLICY_NAMES.get(policy_id, policy_id)


def _year(result: dict[str, Any]) -> str:
    return result["period"]["start"][:4]


def _comparison_sentence(policy: str, mean: float, best_rule: str, best: float, optimum: float) -> str:
    """Sentence comparing ``policy`` with the best rule and the optimum.

    The sign is computed here from the numbers, never taken from the tool's own
    ``vs_best_rule`` (a precomputed value was once copied with the wrong sign).
    """
    parts = []
    if best:
        diff = (mean - best) / abs(best) * 100
        word = "más" if diff >= 0 else "menos"
        parts.append(f"{policy} ahorra {pct(abs(diff))} {word} que la mejor regla fija ({best_rule})")
    if optimum:
        parts.append(f"alcanza el {pct(mean / optimum * 100)} del óptimo con previsión perfecta")
    return " y ".join(parts) + "." if parts else ""


def _notes(result: dict[str, Any]) -> list[str]:
    """Fallback and in-sample paragraphs, in that order (empty when neither applies)."""
    notes = []
    if result["policy_used"] != result["policy"]:
        notes.append(
            f"Nota: no se encontró la tabla de {_name(result['policy'])}; "
            f"se usó {_name(result['policy_used'])}."
        )
    if result.get("in_sample"):
        notes.append("Nota: el periodo incluye meses usados para el entrenamiento, así que el ahorro es optimista.")
    return notes


def render_simulation(result: dict[str, Any]) -> str:
    """Render one simulation: savings line, table, comparison, notes, assumptions, source."""
    mean = result["savings_cop"]["mean"]
    low, high = result["savings_cop"]["min"], result["savings_cop"]["max"]
    base = result["baselines"]
    best_rule = result["best_rule"]
    best, optimum = base[best_rule], base["perfect_foresight"]
    head = (
        f"**Ahorro promedio: {cop(mean)} (rango mín–máx: {_plain(low)} a {cop(high)}) "
        f"según el calendario de turnos ({result['shift_patterns']} calendarios simulados)**"
    )
    table = "\n".join(
        [
            "| Política | Ahorro promedio (COP) |",
            "|---|---|",
            f"| {_name(result['policy'])} | {_plain(mean)} |",
            f"| {_name(best_rule)} | {_plain(best)} |",
            f"| {_name('perfect_foresight')} | {_plain(optimum)} |",
        ]
    )
    sentence = _comparison_sentence(_name(result["policy"]), mean, _name(best_rule), best, optimum)
    blocks = [head, table, sentence, *_notes(result), f"**Supuestos:** {result['notes']}"]
    blocks.append(f"*Fuente: simulación propia con precios de bolsa de XM, {_year(result)}.*")
    return "\n\n".join(b for b in blocks if b)


def _period_label(period: dict[str, str]) -> str:
    start, end = date.fromisoformat(period["start"]), date.fromisoformat(period["end"])
    whole_months = start.day == 1 and end.day == calendar.monthrange(end.year, end.month)[1]
    if whole_months and start.year == end.year:
        first, last = _MONTHS[start.month - 1], _MONTHS[end.month - 1]
        span = first if first == last else f"{first}–{last}"
        return f"{span} {start.year}"
    return f"{period['start']} a {period['end']}"


def _labels(results: list[dict[str, Any]]) -> list[str]:
    """Row labels built only from the fields that differ between the runs."""
    keys = {
        "preset": [r["preset"] for r in results],
        "period": [(r["period"]["start"], r["period"]["end"]) for r in results],
        "policy": [r["policy"] for r in results],
    }
    varying = {k for k, values in keys.items() if len(set(values)) > 1} or {"preset"}
    labels = []
    for r in results:
        parts = []
        if "preset" in varying:
            parts.append(f"Batería {r['preset'].replace('h', ' h')}")
        if "policy" in varying:
            parts.append(_name(r["policy"]))
        if "period" in varying:
            parts.append(_period_label(r["period"]))
        labels.append(" · ".join(parts))
    return labels


def render_comparison(results: list[dict[str, Any]]) -> str:
    """Render two or more simulations as one table (at most 6 rows), plus notes and source."""
    shown = results[:MAX_COMPARISON_ROWS]
    rows = [
        "| Escenario | Ahorro promedio (COP) | Mejor regla (COP) | % del óptimo |",
        "|---|---|---|---|",
    ]
    for label, r in zip(_labels(shown), shown):
        mean, base = r["savings_cop"]["mean"], r["baselines"]
        optimum = base["perfect_foresight"]
        share = pct(mean / optimum * 100) if optimum else "-"
        rows.append(f"| {label} | {_plain(mean)} | {_plain(base[r['best_rule']])} | {share} |")
    blocks = ["\n".join(rows)]
    if len(results) > len(shown):
        blocks.append(f"Se muestran {len(shown)} de {len(results)} escenarios.")
    for note in dict.fromkeys(n for r in shown for n in _notes(r)):
        blocks.append(note)
    blocks.append(f"**Supuestos:** {shown[0]['notes']}")
    years = sorted({_year(r) for r in shown})
    blocks.append(f"*Fuente: simulación propia con precios de bolsa de XM, {' y '.join(years)}.*")
    return "\n\n".join(blocks)


def render_results(results: list[dict[str, Any]]) -> str:
    """Render all successful simulations of a run; ``""`` when there are none.

    Duplicates (same policy, battery and period) collapse into one.
    """
    unique: dict[tuple, dict[str, Any]] = {}
    for r in results:
        unique.setdefault((r["policy"], r["preset"], r["period"]["start"], r["period"]["end"]), r)
    distinct = list(unique.values())
    if not distinct:
        return ""
    return render_simulation(distinct[0]) if len(distinct) == 1 else render_comparison(distinct)
