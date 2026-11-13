"""Tests for `energy.agent.render`: the deterministic simulation results block (D16).

Pure functions, so the inputs are synthetic dicts shaped like the ``simulate_battery``
output (minus ``comparison``, which the renderer now computes). No simulation runs.
Run with ``uv run pytest tests/test_agent_render.py``.
"""
from __future__ import annotations

import copy

import pytest

from energy.agent.render import (
    POLICY_NAMES,
    cop,
    pct,
    render_comparison,
    render_results,
    render_simulation,
)
from energy.rl.simulate import POLICY_IDS

NOTE = "Nota de prueba sobre los supuestos."
HEADER = "| Política | Ahorro promedio (COP) |"
COMPARISON_HEADER = "| Escenario | Ahorro promedio (COP) | Mejor regla (COP) | % del óptimo |"
SOURCE_2025 = "*Fuente: simulación propia con precios de bolsa de XM, 2025.*"


def make_result(**overrides) -> dict:
    """A successful ``simulate_battery`` result (fixed numbers, no ``comparison`` key)."""
    result = {
        "policy": "q_learning",
        "policy_used": "q_learning",
        "preset": "8h",
        "period": {"start": "2025-10-06", "end": "2025-10-19"},
        "shift_patterns": 5,
        "mix": "default",
        "savings_cop": {"mean": 3131505.6, "min": 2900000.0, "max": 3300000.4},
        "baselines": {
            "no_battery": 0,
            "night_day": 2500000,
            "shift_schedule": 2800000,
            "price_threshold": 3000000,
            "perfect_foresight": 4000000,
        },
        "best_rule": "price_threshold",
        "vs_best_rule": 4.4,
        "vs_perfect_foresight": 78.3,
        "energy": {"bought_kwh": 1000, "delivered_kwh": 800, "cycles": 12.3},
        "by_hour_block": [],
        "by_price_level": [],
        "in_sample": False,
        "notes": NOTE,
    }
    result.update(overrides)
    return result


def table_rows(text: str) -> list[str]:
    """Data rows of the Markdown tables in ``text`` (no header, no separator)."""
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    return [ln for ln in lines if "---" not in ln and "Ahorro promedio (COP)" not in ln]


# --- number formats --------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (3131505.6, "3.131.506 COP"),
        (0, "0 COP"),
        (999, "999 COP"),
        (1000, "1.000 COP"),
        (-1234567.4, "-1.234.567 COP"),
    ],
)
def test_cop_uses_dot_thousands_and_no_decimals(value, expected):
    assert cop(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(5.27, "5,3 %"), (0, "0,0 %"), (100, "100,0 %"), (-16.74, "-16,7 %"), (78.26, "78,3 %")],
)
def test_pct_uses_one_decimal_with_comma(value, expected):
    assert pct(value) == expected


def test_policy_names_cover_every_policy_id():
    assert set(POLICY_NAMES) >= set(POLICY_IDS)
    assert POLICY_NAMES == {
        "q_learning": "Q-learning",
        "price_threshold": "Regla de umbral de precio",
        "shift_schedule": "Horario por turnos",
        "night_day": "Carga nocturna / descarga diurna",
        "perfect_foresight": "Óptimo con previsión perfecta",
        "no_battery": "Sin batería",
    }


# --- render_simulation -----------------------------------------------------------

# Golden layout (blocks separated by one blank line, in this order): bold line, table,
# comparison sentence, "**Supuestos:** <notes>", source line. Fallback and in-sample notes,
# when present, are their own paragraphs between the sentence and the Supuestos.
GOLDEN = "\n".join(
    [
        "**Ahorro promedio: 3.131.506 COP (rango mín–máx: 2.900.000 a 3.300.000 COP) "
        "según el calendario de turnos (5 calendarios simulados)**",
        "",
        "| Política | Ahorro promedio (COP) |",
        "|---|---|",
        "| Q-learning | 3.131.506 |",
        "| Regla de umbral de precio | 3.000.000 |",
        "| Óptimo con previsión perfecta | 4.000.000 |",
        "",
        "Q-learning ahorra 4,4 % más que la mejor regla fija (Regla de umbral de precio) "
        "y alcanza el 78,3 % del óptimo con previsión perfecta.",
        "",
        f"**Supuestos:** {NOTE}",
        "",
        SOURCE_2025,
    ]
)


def test_render_simulation_golden_string():
    assert render_simulation(make_result()) == GOLDEN


def test_first_line_is_the_bold_savings_line():
    first = render_simulation(make_result()).splitlines()[0]
    assert first == (
        "**Ahorro promedio: 3.131.506 COP (rango mín–máx: 2.900.000 a 3.300.000 COP) "
        "según el calendario de turnos (5 calendarios simulados)**"
    )


def test_table_has_exactly_three_rows_with_cop_values():
    out = render_simulation(make_result())
    assert HEADER in out
    assert table_rows(out) == [
        "| Q-learning | 3.131.506 |",
        "| Regla de umbral de precio | 3.000.000 |",
        "| Óptimo con previsión perfecta | 4.000.000 |",
    ]


def test_table_names_the_requested_policy_and_best_rule():
    out = render_simulation(
        make_result(policy="night_day", policy_used="night_day", best_rule="shift_schedule")
    )
    rows = table_rows(out)
    assert rows[0].startswith("| Carga nocturna / descarga diurna |")
    assert rows[1] == "| Horario por turnos | 2.800.000 |"


def test_sentence_says_mas_when_above_the_best_rule():
    out = render_simulation(make_result())
    assert "4,4 % más que la mejor regla fija (Regla de umbral de precio)" in out
    assert "menos que la mejor regla" not in out


def test_sentence_says_menos_when_below_the_best_rule():
    result = make_result(
        savings_cop={"mean": 2500000.0, "min": 2400000.0, "max": 2600000.0},
        vs_best_rule=-16.7,
        vs_perfect_foresight=62.5,
    )
    out = render_simulation(result)
    assert "16,7 % menos que la mejor regla fija (Regla de umbral de precio)" in out
    assert "más que la mejor regla" not in out
    assert "62,5 % del óptimo" in out


def test_sign_comes_from_the_numbers_not_from_vs_best_rule():
    """A wrong precomputed ``vs_best_rule`` must not flip the sentence (seen live)."""
    result = make_result(
        savings_cop={"mean": 2500000.0, "min": 2400000.0, "max": 2600000.0},
        vs_best_rule=+16.7,
    )
    out = render_simulation(result)
    assert "menos que la mejor regla" in out


def test_fallback_note_when_policy_used_differs():
    result = make_result(policy="q_learning", policy_used="price_threshold")
    out = render_simulation(result)
    assert "no se encontró la tabla de Q-learning; se usó Regla de umbral de precio" in out


def test_no_fallback_note_when_policy_used_matches():
    assert "no se encontró" not in render_simulation(make_result())


def test_in_sample_note_only_when_in_sample():
    plain = render_simulation(make_result(in_sample=False))
    flagged = render_simulation(make_result(in_sample=True))
    assert "entren" not in plain.lower()
    assert "entren" in flagged.lower()  # mentions that the months were used for training
    assert flagged.index("entren") < flagged.index("**Supuestos:**")


def test_supuestos_line_carries_the_notes():
    assert f"**Supuestos:** {NOTE}" in render_simulation(make_result())


@pytest.mark.parametrize("start", ["2024-03-01", "2025-10-06"])
def test_source_line_takes_the_year_from_the_period_start(start):
    out = render_simulation(make_result(period={"start": start, "end": start[:5] + "12-01"}))
    assert out.splitlines()[-1] == (
        f"*Fuente: simulación propia con precios de bolsa de XM, {start[:4]}.*"
    )


def test_render_simulation_does_not_mutate_its_input():
    result = make_result()
    snapshot = copy.deepcopy(result)
    render_simulation(result)
    assert result == snapshot


# --- render_comparison -----------------------------------------------------------


def preset_results() -> list[dict]:
    return [
        make_result(preset="4h", savings_cop={"mean": 2000000.0, "min": 1.0, "max": 3.0}),
        make_result(preset="8h"),
    ]


def test_comparison_header_and_preset_only_labels():
    out = render_comparison(preset_results())
    assert COMPARISON_HEADER in out
    rows = table_rows(out)
    assert len(rows) == 2
    assert rows[0].startswith("| Batería 4 h |")
    assert rows[1].startswith("| Batería 8 h |")
    assert "2025" not in rows[0].split("|")[1]  # period does not vary, so it is not shown
    assert "Q-learning" not in rows[0].split("|")[1]


def test_comparison_row_values():
    rows = table_rows(render_comparison(preset_results()))
    cells = [c.strip() for c in rows[1].strip("|").split("|")]
    assert cells[1:] == ["3.131.506", "3.000.000", "78,3 %"]
    cells4 = [c.strip() for c in rows[0].strip("|").split("|")]
    assert cells4[1:] == ["2.000.000", "3.000.000", "50,0 %"]


def test_comparison_labels_include_the_period_when_it_varies():
    results = [
        make_result(period={"start": "2025-10-01", "end": "2025-10-31"}),
        make_result(period={"start": "2025-11-01", "end": "2025-11-30"}),
    ]
    labels = [r.strip("|").split("|")[0].strip() for r in table_rows(render_comparison(results))]
    assert len(set(labels)) == 2
    assert all("2025" in label for label in labels)
    assert not any("Batería" in label or "Q-learning" in label for label in labels)


def test_comparison_labels_include_the_policy_when_it_varies():
    results = [
        make_result(policy="q_learning", policy_used="q_learning"),
        make_result(policy="price_threshold", policy_used="price_threshold"),
    ]
    labels = [r.strip("|").split("|")[0].strip() for r in table_rows(render_comparison(results))]
    assert "Q-learning" in labels[0]
    assert "Regla de umbral de precio" in labels[1]
    assert not any("Batería" in label for label in labels)


def test_comparison_labels_combine_every_varying_field():
    results = [
        make_result(preset="4h", period={"start": "2025-10-01", "end": "2025-10-31"}),
        make_result(preset="8h", period={"start": "2025-11-01", "end": "2025-11-30"}),
    ]
    labels = [r.strip("|").split("|")[0].strip() for r in table_rows(render_comparison(results))]
    assert "Batería 4 h" in labels[0] and "2025" in labels[0]
    assert "Batería 8 h" in labels[1] and "2025" in labels[1]


def test_comparison_caps_at_six_rows_and_says_so():
    results = [
        make_result(period={"start": f"2025-{m:02d}-01", "end": f"2025-{m:02d}-20"}) for m in range(1, 9)
    ]
    out = render_comparison(results)
    assert len(table_rows(out)) == 6
    note = [ln for ln in out.splitlines() if "6 de 8" in ln and not ln.startswith("|")]
    assert note, out


def test_comparison_no_truncation_note_at_the_cap():
    results = [
        make_result(period={"start": f"2025-{m:02d}-01", "end": f"2025-{m:02d}-20"}) for m in range(1, 7)
    ]
    out = render_comparison(results)
    assert len(table_rows(out)) == 6
    assert " de 6" not in out


def test_comparison_has_supuestos_and_fuente_once():
    out = render_comparison(preset_results())
    assert out.count("**Supuestos:**") == 1
    assert out.count("*Fuente:") == 1
    assert SOURCE_2025 in out


def test_comparison_source_lists_both_years_when_they_differ():
    results = [
        make_result(period={"start": "2024-10-01", "end": "2024-12-31"}),
        make_result(period={"start": "2025-10-01", "end": "2025-12-31"}),
    ]
    (source,) = [ln for ln in render_comparison(results).splitlines() if ln.startswith("*Fuente:")]
    assert "2024" in source and "2025" in source


# --- render_results --------------------------------------------------------------


def test_render_results_empty_is_empty_string():
    assert render_results([]) == ""


def test_render_results_single_equals_render_simulation():
    result = make_result()
    assert render_results([result]) == render_simulation(result)


def test_render_results_collapses_duplicates():
    result = make_result()
    assert render_results([result, copy.deepcopy(result)]) == render_simulation(result)


def test_render_results_two_distinct_use_the_comparison():
    results = preset_results()
    out = render_results(results)
    assert out == render_comparison(results)
    assert COMPARISON_HEADER in out


def test_render_results_duplicates_are_by_policy_preset_and_period():
    a, b = make_result(preset="4h"), make_result(preset="8h")
    out = render_results([a, copy.deepcopy(a), b, copy.deepcopy(b)])
    assert len(table_rows(out)) == 2
