"""Tests for `energy.agent.prompts`: the Spanish system prompt (step 5, D16).

Only the agreed anchors are asserted (refusal phrase, citation format, data range,
rules, the no-tables rule), so the rest of the wording stays free. Run with
``uv run pytest tests/test_agent_prompts.py``.
"""
from __future__ import annotations

from datetime import date

import pytest

from energy.agent.prompts import REFUSAL, system_prompt
from energy.agent.settings import UserSettings


def test_refusal_phrase_is_exact():
    assert REFUSAL == (
        "No encuentro información suficiente en los documentos disponibles para responder esa pregunta."
    )


@pytest.fixture(scope="module")
def prompt() -> str:
    return system_prompt()


@pytest.mark.parametrize(
    "anchor",
    [
        REFUSAL,
        "[cita]",
        "2024-01-01",
        "2025-12-31",
        "detrás del medidor",
        "asesoría legal",
        "No escribas tablas",
        "el sistema agrega automáticamente",
        "search_regulation",
        "simulate_battery",
    ],
)
def test_prompt_contains_agreed_anchor(prompt, anchor):
    assert anchor in prompt


def test_prompt_states_the_superseded_rule(prompt):
    assert "superseded_by" in prompt or "sustituid" in prompt.lower()
    assert "in_force" in prompt


def test_prompt_is_written_in_spanish(prompt):
    lowered = prompt.lower()
    assert sum(w in lowered for w in ("ahorro", "regulación", "responde", "pregunta")) >= 3


def test_system_prompt_is_stable_between_calls():
    assert system_prompt() == system_prompt()


def test_prompt_states_the_user_settings():
    text = system_prompt(UserSettings("12h", date(2024, 10, 1), date(2024, 12, 31)))
    assert "12 h" in text
    assert "2024-10-01" in text
    assert "2024-12-31" in text


def test_default_prompt_states_the_default_settings(prompt):
    assert "8 h" in prompt
    assert "2025-10-01" in prompt


def test_old_default_sentence_is_gone(prompt):
    assert "Si el usuario no da fechas" not in prompt


def test_none_settings_equal_no_argument():
    assert system_prompt(None) == system_prompt()
