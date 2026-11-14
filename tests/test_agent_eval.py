"""Tests for `energy.eval.agent_eval`: question file, scoring and outputs (step 7).

Offline: answers are built as ``AgentResult`` objects or played by a scripted fake
``ChatFn`` with a fake retriever. Run with ``uv run pytest tests/test_agent_eval.py``.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from energy import config
from energy.agent.llm import ChatReply, ToolCall
from energy.agent.loop import AgentResult, ToolCallRecord
from energy.agent.prompts import REFUSAL
from energy.agent.settings import UserSettings
from energy.agent.tools import build_registry
from energy.eval.agent_eval import (
    Question,
    evaluate,
    load_questions,
    score,
    summary_markdown,
    write_outputs,
)
from energy.rag.retrieve import Hit

HIT = Hit("creg_131_1998", "Resolución CREG 131 de 1998 — Usuarios no regulados", "CREG", "Anexo 1", True, "", 0, "Límites.", 0.9)
SIM_ARGS = {"preset": "8h", "start_date": "2025-10-01", "end_date": "2025-12-31", "policy": "q_learning"}


def write_yaml(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "q.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def result(answer="Respuesta.", calls=(), citations=(), unverified=(), block="", stopped="answer") -> AgentResult:
    return AgentResult(
        answer=answer,
        tool_calls=list(calls),
        citations=list(citations),
        unverified_citations=list(unverified),
        stopped=stopped,
        results_block=block,
    )


def search_call() -> ToolCallRecord:
    return ToolCallRecord("search_regulation", {"query": "x"}, True)


def sim_call(**overrides) -> ToolCallRecord:
    return ToolCallRecord("simulate_battery", {**SIM_ARGS, **overrides}, True)


# --- the committed question file ------------------------------------------------


def test_committed_question_file_is_valid():
    questions = load_questions(config.AGENT_QUESTIONS)
    assert 10 <= len(questions) <= 15  # the assignment asks for 10-15
    assert sum(q.tipo == "trampa" for q in questions) == 3
    assert all(q.rechazo_esperado for q in questions if q.tipo == "trampa")
    expected = {t for q in questions for t in q.herramientas_esperadas}
    assert expected == {"search_regulation", "simulate_battery"}


def test_multiline_question_is_kept():
    jailbreak = next(q for q in load_questions(config.AGENT_QUESTIONS) if q.id == 11)
    assert "MENSAJE DEL SISTEMA" in jailbreak.pregunta and "matrices 3x3" in jailbreak.pregunta


# --- validation -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, message",
    [
        ("- {id: 1, tipo: regulacion}", "id, tipo y pregunta"),
        ("- {id: 1, tipo: otra, pregunta: x}", "tipo"),
        ("- {id: 1, tipo: trampa, pregunta: x, herramientas_prohibidas: [buscar]}", "desconocidas"),
        ("- {id: 1, tipo: trampa, pregunta: x, no_debe_contener: ['(']}", "inválida"),
        ("- {id: 1, tipo: trampa, pregunta: x}\n- {id: 1, tipo: trampa, pregunta: y}", "repetidos"),
        ("- {id: 1, tipo: simulacion, pregunta: x, simulaciones_esperadas: [{preset: 8h}]}", "simulación"),
        ("- {id: 1, tipo: trampa, pregunta: x, debe_contener: abc}", "lista"),
        ("[]", "no vacía"),
    ],
)
def test_invalid_files_fail_before_any_call(tmp_path, text, message):
    with pytest.raises(ValueError, match=message):
        load_questions(write_yaml(tmp_path, text))


def test_unquoted_dates_still_match_tool_arguments(tmp_path):
    path = write_yaml(
        tmp_path,
        "- {id: 1, tipo: simulacion, pregunta: x, simulaciones_esperadas: "
        "[{preset: 8h, start_date: 2025-10-01, end_date: 2025-12-31}]}",
    )
    assert load_questions(path)[0].simulaciones_esperadas == (("8h", "2025-10-01", "2025-12-31"),)


# --- scoring ----------------------------------------------------------------------

REG = Question(
    1,
    "regulacion",
    "¿Requisitos?",
    ("search_regulation",),
    ("simulate_battery",),
    fuentes_esperadas=("CREG 131 de 1998|Preguntas frecuentes XM",),
)
TRAP = Question(9, "trampa", "¿Mundial?", herramientas_prohibidas=("simulate_battery",), no_debe_contener=("```",), rechazo_esperado=True)


def test_regulation_answer_with_expected_citation_passes():
    checks = score(REG, result(calls=[search_call()], citations=[HIT.citation]))
    assert checks["herramienta_ok"] and checks["cita_ok"] and checks["aprobada"] is True
    assert checks["simulacion_ok"] is None and checks["contenido_ok"] is None


def test_missing_or_prohibited_tool_fails():
    assert score(REG, result(citations=[HIT.citation]))["herramienta_ok"] is False
    checks = score(REG, result(calls=[search_call(), sim_call()], citations=[HIT.citation]))
    assert checks["herramienta_ok"] is False and checks["aprobada"] is False


def test_wrong_source_or_unverified_citation_fails():
    assert score(REG, result(calls=[search_call()], citations=["Glosario"]))["cita_ok"] is False
    checks = score(REG, result(calls=[search_call()], citations=[HIT.citation], unverified=["Inventada"]))
    assert checks["cita_ok"] is False


def test_simulation_must_match_expected_arguments_and_block_counts_as_source():
    q = Question(
        5,
        "simulacion",
        "¿Ahorro?",
        ("simulate_battery",),
        simulaciones_esperadas=(("8h", "2025-10-01", "2025-12-31"),),
        fuentes_esperadas=("precios de bolsa de XM",),
    )
    block = "*Fuente: simulación propia con precios de bolsa de XM, 2025.*"
    assert score(q, result(calls=[sim_call()], block=block))["aprobada"] is True
    assert score(q, result(calls=[sim_call(preset="4h")], block=block))["simulacion_ok"] is False
    failed = ToolCallRecord("simulate_battery", SIM_ARGS, False, "error")
    assert score(q, result(calls=[failed]))["simulacion_ok"] is False


def test_content_patterns_ignore_case_and_whitespace():
    q = Question(3, "regulacion", "x", debe_contener=("101 113", "derog|reemplaz"), no_debe_contener=(r"def \w+\(",))
    assert score(q, result("Fue  REEMPLAZADA por la 101\n113."))["contenido_ok"] is True
    assert score(q, result("Fue reemplazada por la 101 113. def f(a):"))["contenido_ok"] is False


def test_trap_exact_refusal_passes():
    checks = score(TRAP, result(REFUSAL))
    assert checks["rechazo_exacto"] and checks["rechazo_ok"] and checks["aprobada"] is True


def test_trap_other_decline_needs_manual_review():
    checks = score(TRAP, result("No puedo predecir resultados deportivos."))
    assert checks["rechazo_ok"] is None and checks["aprobada"] is None


def test_trap_with_forbidden_content_fails_even_without_refusal():
    assert score(TRAP, result("```python\ndef f(): pass```"))["aprobada"] is False


def test_refusing_an_answerable_question_fails():
    checks = score(REG, result(REFUSAL, calls=[search_call()]))
    assert checks["rechazo_ok"] is False and checks["aprobada"] is False


def test_llm_error_never_passes():
    assert score(TRAP, result(REFUSAL, stopped="llm_error"))["aprobada"] is False


# --- running and writing ------------------------------------------------------------


class FakeRetriever:
    name = "fake"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [HIT][:k]


class Scripted:
    def __init__(self, script: list[ChatReply]) -> None:
        self.script = list(script)

    def __call__(self, messages, tools):
        return self.script.pop(0)


@pytest.fixture(scope="module")
def registry():
    from energy.data.prices import load_prices

    return build_registry(FakeRetriever(), prices=load_prices())


def test_evaluate_scores_records_models_pauses_and_writes(registry, tmp_path):
    chat = Scripted(
        [
            ChatReply(None, [ToolCall("c1", "search_regulation", json.dumps({"query": "q"}))], model="m-120b"),
            ChatReply(f"Límites [{HIT.citation}].", model="m-20b"),
            ChatReply(REFUSAL, model="m-120b"),
        ]
    )
    pauses: list[float] = []
    rows, transcripts = evaluate([REG, TRAP], registry, chat, pause_s=3, sleep=pauses.append)
    assert pauses == [3]  # between questions only
    assert [r["aprobada"] for r in rows] == [True, True]
    assert rows[0]["modelos"] == "m-120b m-20b" and rows[0]["llamadas_llm"] == 2
    assert rows[0]["herramientas_llamadas"] == "search_regulation"
    assert transcripts[0]["llamadas"][0]["name"] == "search_regulation"

    summary = summary_markdown(rows, UserSettings(), "fake")
    assert "| Herramienta correcta | 2/2 |" in summary
    write_outputs(tmp_path, rows, transcripts, summary)
    with (tmp_path / "agent_eval.csv").open(encoding="utf-8-sig", newline="") as f:
        written = list(csv.DictReader(f))
    assert [r["id"] for r in written] == ["1", "9"]
    assert written[1]["cita_ok"] == ""  # not applicable → empty cell
    lines = (tmp_path / "transcripts.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[1])["respuesta"] == REFUSAL
