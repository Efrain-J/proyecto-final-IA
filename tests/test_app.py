"""Smoke tests for `app.py` with `streamlit.testing.v1.AppTest`.

Offline: the real keys are never loaded (`load_env` is replaced and the key variables are
removed), so the assistant runs retrieval-only over BM25 and Groq is never called.
"""
from __future__ import annotations

from datetime import date

import pytest
from streamlit.testing.v1 import AppTest

from energy import config


@pytest.fixture
def at(monkeypatch: pytest.MonkeyPatch) -> AppTest:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setattr(config, "load_env", lambda *a, **k: False)
    return AppTest.from_file(str(config.ROOT / "app.py"), default_timeout=120)


def _captions(at: AppTest) -> str:
    return " | ".join(c.value for c in at.caption)


def test_default_run(at: AppTest) -> None:
    at.run()
    assert not at.exception
    assert len(at.metric) == 6  # 3 averaged + 3 for shift calendar 1
    headings = " | ".join(m.value for m in at.markdown)
    assert "Promedio de 5 calendarios" in headings
    assert "Calendario de turnos 1 de 5" in headings
    assert len(at.tabs) == 4
    text = _captions(at)
    assert "8 h" in text and "2025" in text and "Mixto" in text


def _calendar_values(at: AppTest) -> list[str]:
    """Values of the metrics under the calendar heading (the last 3, or 4 for a rule)."""
    return [m.value for m in at.metric][3:]


def test_calendar_selector_defaults_to_first(at: AppTest) -> None:
    at.run()
    assert not at.exception
    selector = at.get("segmented_control")
    assert [c.proto.label for c in selector] == ["Calendario de turnos"]


def test_selecting_another_calendar_changes_heading_and_metrics(at: AppTest) -> None:
    at.run()
    first = _calendar_values(at)
    averaged = [m.value for m in at.metric][:3]
    at.get("segmented_control")[0].set_value(3).run()
    assert not at.exception
    assert len(at.metric) == 6
    headings = " | ".join(m.value for m in at.markdown)
    assert "Calendario de turnos 3 de 5" in headings
    assert "Calendario de turnos 1 de 5" not in headings
    assert _calendar_values(at) != first
    assert [m.value for m in at.metric][:3] == averaged  # the top row stays averaged


def test_calendar_with_another_policy_keeps_seven_metrics(at: AppTest) -> None:
    at.run()
    at.selectbox(key="politica").set_value("night_day").run()
    at.get("segmented_control")[0].set_value(2).run()
    assert not at.exception
    assert len(at.metric) == 7
    assert "Calendario de turnos 2 de 5" in " | ".join(m.value for m in at.markdown)


def test_sidebar_changes_update_status_line(at: AppTest) -> None:
    at.run()
    at.selectbox(key="bateria").set_value("4h")
    at.selectbox(key="operacion").set_value("mostly_single")
    at.radio(key="escenario").set_value("Oct-Dic 2024 (El Niño)")
    at.run()
    assert not at.exception
    text = _captions(at)
    assert "4 h" in text and "2024" in text and "Principalmente un turno" in text


def test_non_q_policy_adds_a_calendar_metric(at: AppTest) -> None:
    at.run()
    at.selectbox(key="politica").set_value("night_day").run()
    assert not at.exception
    assert len(at.metric) == 7


def test_custom_period_shows_date_input_directly(at: AppTest) -> None:
    at.run()
    assert not [d for d in at.date_input if d.key == "rango"]  # fixed scenario: no picker
    at.radio(key="escenario").set_value("Personalizado").run()
    assert not at.exception
    assert at.date_input(key="rango").value == (date(2025, 10, 1), date(2025, 12, 31))
    at.radio(key="escenario").set_value("Oct-Dic 2025").run()
    assert not [d for d in at.date_input if d.key == "rango"]


def test_short_custom_period_has_no_week_slider(at: AppTest) -> None:
    at.run()
    at.radio(key="escenario").set_value("Personalizado").run()
    at.date_input(key="rango").set_value((date(2025, 10, 6), date(2025, 10, 10))).run()
    assert not at.exception
    assert not [s for s in at.slider if s.label == "Semana"]


def test_week_is_clamped_when_period_shrinks(at: AppTest) -> None:
    at.run()
    slider = at.slider(key="semana")
    slider.set_value(slider.max).run()
    assert not at.exception
    at.radio(key="escenario").set_value("Personalizado").run()
    at.date_input(key="rango").set_value((date(2025, 10, 6), date(2025, 10, 10))).run()
    assert not at.exception


def test_chat_roundtrip_and_reset(at: AppTest) -> None:
    at.run()
    at.chat_input[0].set_value("¿Qué dice la CREG sobre almacenamiento?").run()
    assert not at.exception
    roles = [m.name for m in at.chat_message]
    assert "user" in roles and "assistant" in roles
    at.button(key="nueva").click().run()
    assert not at.exception
    assert len(at.chat_message) == 0


def _infos(at: AppTest) -> str:
    return " | ".join(i.value for i in at.info)


def test_default_scenario_has_no_in_sample_warning(at: AppTest) -> None:
    at.run()
    assert not at.exception
    assert "entrenar" not in _infos(at)


def test_custom_training_period_shows_in_sample_warning(at: AppTest) -> None:
    at.run()
    at.radio(key="escenario").set_value("Personalizado").run()
    at.date_input(key="rango").set_value((date(2025, 3, 1), date(2025, 3, 31))).run()
    assert not at.exception
    assert "entrenar" in _infos(at)
