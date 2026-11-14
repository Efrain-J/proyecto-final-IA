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
    assert len(at.metric) == 3
    assert len(at.tabs) == 4
    text = _captions(at)
    assert "8 h" in text and "2025" in text and "Mixto" in text


def test_sidebar_changes_update_status_line(at: AppTest) -> None:
    at.run()
    at.selectbox(key="bateria").set_value("4h")
    at.selectbox(key="operacion").set_value("mostly_single")
    at.radio(key="escenario").set_value("2024 (El Niño, precios altos)")
    at.run()
    assert not at.exception
    text = _captions(at)
    assert "4 h" in text and "2024" in text and "Principalmente un turno" in text


def test_short_custom_period_has_no_week_slider(at: AppTest) -> None:
    at.run()
    at.checkbox(key="fechas_custom").set_value(True).run()
    at.date_input(key="rango").set_value((date(2025, 10, 6), date(2025, 10, 10))).run()
    assert not at.exception
    assert not [s for s in at.slider if s.label == "Semana"]


def test_week_is_clamped_when_period_shrinks(at: AppTest) -> None:
    at.run()
    slider = at.slider(key="semana")
    slider.set_value(slider.max).run()
    assert not at.exception
    at.checkbox(key="fechas_custom").set_value(True).run()
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
