"""Tests for `energy.agent.loop`: the hand-written agent loop (step 5, D16).

Offline: a scripted fake ``ChatFn`` plays the model and a fake retriever supplies
the regulation hits; the registry is the real one from `energy.agent.tools`. The
single ``network`` test calls Groq for real and is skipped without ``GROQ_API_KEY``
(run with ``uv run pytest -m network``). Run the rest with
``uv run pytest tests/test_agent_loop.py``.
"""
from __future__ import annotations

import copy
import json
import os
from datetime import date

import pytest

from energy import config
from energy.agent.llm import ChatReply, LLMError, ToolCall
from energy.agent.loop import AgentResult, ToolCallRecord, open_agent, run_agent
from energy.agent.prompts import REFUSAL, system_prompt
from energy.agent.render import cop
from energy.agent.settings import UserSettings
from energy.agent.tools import build_registry
from energy.data.prices import load_prices
from energy.rag.retrieve import Hit

HIT = Hit(
    doc_id="creg101113",
    title="Resolución CREG 101 113 de 2025",
    publisher="CREG",
    article="5",
    in_force=True,
    superseded_by="",
    chunk_index=0,
    text="Los sistemas de almacenamiento deben cumplir con la eficiencia mínima.",
    score=0.9,
)
CITA = HIT.citation


class FakeRetriever:
    name = "fake"

    def search(self, query: str, k: int = 5) -> list[Hit]:
        return [HIT][:k]


class ScriptedChat:
    """A ``ChatFn`` that replays replies (or raises exceptions) and records each call."""

    def __init__(self, script: list) -> None:
        self.script = list(script)
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def __call__(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), tools))
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class AlwaysToolCalls:
    """Asks for a tool on every call that allows tools; answers when tools are off."""

    def __init__(self) -> None:
        self.calls: list[tuple[list[dict], list[dict] | None]] = []

    def __call__(self, messages, tools):
        self.calls.append((copy.deepcopy(messages), tools))
        if tools is None:
            return ChatReply("Respuesta final.", [])
        n = len(self.calls)
        return ChatReply(None, [ToolCall(f"call{n}", "search_regulation", json.dumps({"query": "q"}))])


def say(text: str) -> ChatReply:
    return ChatReply(text, [])


START, END = "2025-10-06", "2025-10-19"  # two weeks of the test period
HEADER = "| Política | Ahorro promedio (COP) |"
COMPARISON_HEADER = "| Escenario | Ahorro promedio (COP) | Mejor regla (COP) | % del óptimo |"
SOURCE_LINE = "*Fuente: simulación propia con precios de bolsa de XM, 2025.*"


def ask_sim(id_: str = "s1", preset: str = "8h", policy: str = "q_learning") -> ToolCall:
    args = {"start_date": START, "end_date": END, "preset": preset, "policy": policy}
    return ToolCall(id_, "simulate_battery", json.dumps(args))


def ask_search(id_: str = "c1", query: str = "eficiencia") -> ToolCall:
    return ToolCall(id_, "search_regulation", json.dumps({"query": query}))


@pytest.fixture(scope="module")
def registry():
    return build_registry(FakeRetriever(), prices=load_prices())


def norm(text: str) -> str:
    return " ".join(text.lower().split())


# --- basic paths ---------------------------------------------------------------


def test_direct_answer_without_tools(registry):
    chat = ScriptedChat([say("Hola, ¿en qué te ayudo?")])
    result = run_agent("hola", [], registry, chat)
    assert isinstance(result, AgentResult)
    assert result.answer == "Hola, ¿en qué te ayudo?"
    assert result.stopped == "answer"
    assert result.tool_calls == []
    assert result.citations == []
    assert len(chat.calls) == 1
    assert chat.calls[0][1]  # tool schemas were offered


def test_first_call_has_system_prompt_then_question(registry):
    chat = ScriptedChat([say("ok")])
    run_agent("¿Qué es un SAEB?", [], registry, chat)
    messages = chat.calls[0][0]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == system_prompt()
    assert messages[-1] == {"role": "user", "content": "¿Qué es un SAEB?"}


def test_one_tool_round_then_answer_uses_openai_message_shapes(registry):
    chat = ScriptedChat([ChatReply(None, [ask_search("abc")]), say(f"Eficiencia mínima [{CITA}].")])
    result = run_agent("eficiencia", [], registry, chat)
    assert result.stopped == "answer"
    assert result.retriever == "fake"
    assert len(chat.calls) == 2

    second = chat.calls[1][0]
    (assistant,) = [m for m in second if m["role"] == "assistant"]
    (call,) = assistant["tool_calls"]
    assert call["id"] == "abc"
    assert call["type"] == "function"
    assert call["function"]["name"] == "search_regulation"
    assert json.loads(call["function"]["arguments"]) == {"query": "eficiencia"}
    (tool_msg,) = [m for m in second if m["role"] == "tool"]
    assert tool_msg["tool_call_id"] == "abc"
    observation = json.loads(tool_msg["content"])
    assert observation["hits"][0]["cita"] == CITA
    assert second.index(assistant) < second.index(tool_msg)

    (record,) = result.tool_calls
    assert isinstance(record, ToolCallRecord)
    assert (record.name, record.arguments, record.ok, record.error) == (
        "search_regulation", {"query": "eficiencia"}, True, None,
    )


def test_two_tool_calls_in_one_reply_are_paired_by_id(registry):
    calls = [ask_search("a", "uno"), ask_search("b", "dos")]
    chat = ScriptedChat([ChatReply(None, calls), say(f"listo [{CITA}]")])
    result = run_agent("q", [], registry, chat)
    second = chat.calls[1][0]
    (assistant,) = [m for m in second if m["role"] == "assistant"]
    assert [c["id"] for c in assistant["tool_calls"]] == ["a", "b"]
    assert [m["tool_call_id"] for m in second if m["role"] == "tool"] == ["a", "b"]
    assert [r.arguments["query"] for r in result.tool_calls] == ["uno", "dos"]
    assert len(chat.calls) == 2


# --- the deterministic results block (D16) -------------------------------------


def test_simulation_block_comes_first_then_the_model_text(registry):
    text = "La batería carga de noche y descarga de día."
    chat = ScriptedChat([ChatReply(None, [ask_sim()]), say(text)])
    result = run_agent("simula", [], registry, chat)
    assert result.stopped == "answer"
    assert len(result.simulations) == 1
    assert result.model_text == text
    assert result.results_block.startswith("Escenario: ")
    assert "**Ahorro promedio: " in result.results_block
    assert result.answer == result.results_block + "\n\n" + text
    assert result.answer.startswith("Escenario:")


def test_block_numbers_equal_the_tool_output_exactly(registry):
    chat = ScriptedChat([ChatReply(None, [ask_sim()]), say("Explicación.")])
    result = run_agent("simula", [], registry, chat)
    mean = result.simulations[0]["savings_cop"]["mean"]
    assert f"**Ahorro promedio: {cop(mean)} (" in result.results_block
    assert HEADER in result.results_block
    assert f"| Q-learning | {cop(mean).removesuffix(' COP')} |" in result.results_block
    assert SOURCE_LINE in result.results_block


def test_table_lines_are_stripped_from_the_model_text_when_a_block_exists(registry):
    text = "Explicación inicial.\n| a | b |\n|---|---|\n| 1 | 2 |\nExplicación final."
    chat = ScriptedChat([ChatReply(None, [ask_sim()]), say(text)])
    result = run_agent("simula", [], registry, chat)
    assert not any(ln.lstrip().startswith("|") for ln in result.model_text.splitlines())
    assert "Explicación inicial." in result.model_text and "Explicación final." in result.model_text
    assert result.answer.count(HEADER) == 1  # the block's table is the only one
    assert "| a | b |" not in result.answer


def test_without_a_simulation_there_is_no_block_and_answer_is_the_model_text(registry):
    text = "Una tabla del modelo:\n| a | b |\n|---|---|"
    chat = ScriptedChat([say(text)])
    result = run_agent("hola", [], registry, chat)
    assert result.results_block == ""
    assert result.simulations == []
    assert result.answer == result.model_text == text  # nothing is stripped without a block


def test_failed_simulation_adds_no_block(registry):
    bad = ToolCall("bad", "simulate_battery", json.dumps({"start_date": "2025-10-06"}))
    chat = ScriptedChat([ChatReply(None, [bad]), say("No pude simular.")])
    result = run_agent("simula", [], registry, chat)
    assert result.simulations == []
    assert result.results_block == ""
    assert result.answer == "No pude simular."


def test_two_different_simulations_give_the_comparison_table(registry):
    chat = ScriptedChat([ChatReply(None, [ask_sim("a", "4h"), ask_sim("b", "8h")]), say("Explicación.")])
    result = run_agent("compara", [], registry, chat)
    assert [s["preset"] for s in result.simulations] == ["4h", "8h"]
    assert COMPARISON_HEADER in result.results_block
    assert HEADER not in result.results_block
    assert result.answer.startswith(result.results_block)
    assert result.answer.endswith("Explicación.")


def test_the_same_simulation_twice_gives_a_single_block(registry):
    chat = ScriptedChat([ChatReply(None, [ask_sim("a")]), ChatReply(None, [ask_sim("b")]), say("Listo.")])
    result = run_agent("simula", [], registry, chat)
    assert len(result.simulations) == 2  # every successful call is kept, in order
    assert result.results_block.count("**Ahorro promedio:") == 1
    assert HEADER in result.results_block
    assert COMPARISON_HEADER not in result.results_block


def test_llm_error_after_a_simulation_still_shows_the_block(registry):
    chat = ScriptedChat([ChatReply(None, [ask_sim()]), LLMError("caído")])
    result = run_agent("simula eficiencia", [], registry, chat)
    assert result.stopped == "llm_error"
    assert result.results_block.startswith("Escenario:")
    assert "**Ahorro promedio:" in result.results_block
    assert result.answer.startswith(result.results_block)
    assert "caído" in result.model_text or "No pude consultar" in result.model_text


def test_citations_are_checked_on_the_model_text_with_a_block_present(registry):
    text = f"Explicación. Eficiencia mínima [{CITA}]."
    chat = ScriptedChat([ChatReply(None, [ask_search(), ask_sim()]), say(text)])
    result = run_agent("q", [], registry, chat)
    assert [norm(c) for c in result.citations] == [norm(CITA)]
    assert result.unverified_citations == []
    assert len(chat.calls) == 2  # no repair needed


def test_repair_round_rewrites_only_the_model_text(registry):
    fixed = f"Explicación con cita [{CITA}]."
    chat = ScriptedChat([ChatReply(None, [ask_search(), ask_sim()]), say("Sin citas."), say(fixed)])
    result = run_agent("q", [], registry, chat)
    assert len(chat.calls) == 3
    assert result.model_text == fixed
    assert result.answer == result.results_block + "\n\n" + fixed
    assert [norm(c) for c in result.citations] == [norm(CITA)]


def test_t_max_answer_keeps_the_block(registry):
    class SimThenLoop:
        def __init__(self) -> None:
            self.n = 0

        def __call__(self, messages, tools):
            self.n += 1
            if tools is None:
                return ChatReply("Final.", [])
            return ChatReply(None, [ask_sim(f"s{self.n}")])

    result = run_agent("simula", [], registry, SimThenLoop(), t_max=1)
    assert result.stopped == "t_max"
    assert result.answer == result.results_block + "\n\n" + "Final."


# --- errors and the T_max cap --------------------------------------------------


def test_tool_error_goes_back_as_observation_and_loop_continues(registry):
    bad = ToolCall("bad", "simulate_battery", json.dumps({"start_date": "2025-10-06"}))  # no end_date
    chat = ScriptedChat([ChatReply(None, [bad]), say("No pude simular; faltaba una fecha.")])
    result = run_agent("simula", [], registry, chat)
    assert result.stopped == "answer"
    (tool_msg,) = [m for m in chat.calls[1][0] if m["role"] == "tool"]
    assert "error" in json.loads(tool_msg["content"])
    (record,) = result.tool_calls
    assert record.ok is False
    assert record.error


def test_unknown_tool_name_does_not_crash_the_loop(registry):
    chat = ScriptedChat([ChatReply(None, [ToolCall("x", "hack_the_planet", "{}")]), say("ok")])
    result = run_agent("q", [], registry, chat)
    assert result.stopped == "answer"
    assert result.tool_calls[0].ok is False


@pytest.mark.parametrize("t_max", [1, 3])
def test_endless_tool_calls_stop_at_t_max_plus_one_llm_calls(registry, t_max):
    chat = AlwaysToolCalls()
    result = run_agent("q", [], registry, chat, t_max=t_max)
    assert len(chat.calls) == t_max + 1
    assert all(tools for _, tools in chat.calls[:t_max])
    assert chat.calls[-1][1] is None  # last call has tools switched off
    assert result.stopped == "t_max"
    assert result.answer == "Respuesta final."
    assert len(result.tool_calls) == t_max


def test_default_t_max_comes_from_config(registry):
    chat = AlwaysToolCalls()
    run_agent("q", [], registry, chat)
    assert len(chat.calls) == config.T_MAX + 1


# --- LLM failure ---------------------------------------------------------------


def test_llm_error_on_first_call_gives_retrieval_only_answer(registry):
    chat = ScriptedChat([LLMError("GROQ_API_KEY no está configurada")])
    result = run_agent("eficiencia mínima", [], registry, chat)
    assert result.stopped == "llm_error"
    assert f"[{CITA}]" in result.answer
    assert "GROQ_API_KEY" in result.answer or "error" in result.answer.lower()


def test_llm_error_after_tool_round_still_does_not_raise(registry):
    chat = ScriptedChat([ChatReply(None, [ask_search()]), LLMError("caído")])
    result = run_agent("q", [], registry, chat)
    assert result.stopped == "llm_error"
    assert f"[{CITA}]" in result.answer


# --- citations -----------------------------------------------------------------


def test_citations_are_split_into_verified_and_unverified(registry):
    sloppy = f"  {CITA.upper()}  ".replace(" ", "  ")  # other case, extra whitespace
    answer = f"Dice A [{sloppy}] y también B [Ley Inventada 999, Art. 1]."
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say(answer)])
    result = run_agent("q", [], registry, chat)
    assert [norm(c) for c in result.citations] == [norm(CITA)]
    assert [norm(c) for c in result.unverified_citations] == [norm("Ley Inventada 999, Art. 1")]


def test_citation_not_retrieved_in_this_run_is_unverified(registry):
    chat = ScriptedChat([say(f"Sin buscar [{CITA}].")])
    result = run_agent("q", [], registry, chat)
    assert result.citations == []
    assert len(result.unverified_citations) == 1


@pytest.mark.parametrize(
    "fragment",
    ["[1]", "[2, 3]", "[x]", "[enlace](https://www.creg.gov.co)", "[ ]"],
    ids=["footnote-number", "number-list", "checkbox", "markdown-link", "empty"],
)
def test_non_citation_brackets_are_ignored(registry, fragment):
    """Only bracket text with at least 3 letters that is not a Markdown link counts."""
    answer = f"Texto {fragment} y una cita real [{CITA}]."
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say(answer)])
    result = run_agent("q", [], registry, chat)
    assert [norm(c) for c in result.citations] == [norm(CITA)]
    assert result.unverified_citations == []


def test_answer_without_citations_after_search_gets_one_repair_call(registry):
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say("Respuesta sin citas."), say("Sigue sin citas.")])
    result = run_agent("q", [], registry, chat)
    assert len(chat.calls) == 3
    repair_messages, repair_tools = chat.calls[2]
    assert repair_tools is None
    assert CITA in repair_messages[-1]["content"]  # the repair lists the retrieved citations
    assert result.answer == "Respuesta sin citas."  # a failed repair keeps the original
    assert result.citations == [] and result.unverified_citations == []


def test_successful_repair_replaces_the_answer(registry):
    fixed = f"Respuesta con cita [{CITA}]."
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say("Respuesta 【cita】."), say(fixed)])
    result = run_agent("q", [], registry, chat)
    assert result.answer == fixed
    assert [norm(c) for c in result.citations] == [norm(CITA)]


def test_no_repair_without_a_search_or_for_the_refusal(registry):
    chat = ScriptedChat([say("Respuesta sin buscar.")])
    run_agent("q", [], registry, chat)
    assert len(chat.calls) == 1
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say(REFUSAL)])
    result = run_agent("q", [], registry, chat)
    assert len(chat.calls) == 2 and result.answer == REFUSAL


def test_native_lenticular_brackets_and_cita_prefix_count_as_citations(registry):
    answer = f"Dice A 【{CITA}】 y B [cita: {CITA}]."
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say(answer)])
    result = run_agent("q", [], registry, chat)
    assert [norm(c) for c in result.citations] == [norm(CITA)]
    assert "【" not in result.answer and "cita:" not in result.answer


# --- history -------------------------------------------------------------------


def test_history_is_trimmed_to_the_last_n_messages(registry):
    history = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"mensaje {i}"}
        for i in range(config.HISTORY_MESSAGES + 4)
    ]
    chat = ScriptedChat([say("ok")])
    run_agent("pregunta nueva", history, registry, chat)
    messages = chat.calls[0][0]
    middle = messages[1:-1]
    assert [m["content"] for m in middle] == [m["content"] for m in history[-config.HISTORY_MESSAGES:]]
    assert messages[0]["role"] == "system"
    assert messages[-1]["content"] == "pregunta nueva"


def test_short_history_is_kept_whole(registry):
    history = [{"role": "user", "content": "antes"}, {"role": "assistant", "content": "respuesta"}]
    chat = ScriptedChat([say("ok")])
    run_agent("ahora", history, registry, chat)
    assert [m["content"] for m in chat.calls[0][0][1:-1]] == ["antes", "respuesta"]


# --- network -------------------------------------------------------------------


@pytest.mark.network
def test_real_groq_regulation_question_answers_with_a_valid_citation():
    config.load_env()
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set")
    registry, chat = open_agent()
    result = run_agent("¿Qué eficiencia mínima exige la CREG a un SAEB?", [], registry, chat)
    assert result.stopped == "answer"
    assert len(result.citations) >= 1
    assert "|---" not in result.answer.replace(" ", "")  # no tables in regulation answers


@pytest.mark.network
def test_real_groq_simulation_question_answers_with_the_results_block():
    config.load_env()
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set")
    registry, chat = open_agent()
    result = run_agent(
        "¿Cuánto ahorro con una batería de 8 h entre el 1 de octubre y el 31 de diciembre de 2025?",
        [], registry, chat,
    )
    assert result.stopped == "answer"
    assert any(call.name == "simulate_battery" and call.ok for call in result.tool_calls)
    assert result.answer.startswith("Escenario:")
    assert "**Ahorro promedio:" in result.answer
    assert HEADER in result.answer
    assert SOURCE_LINE.split(",")[0] in result.answer  # year depends on the question
    assert cop(result.simulations[0]["savings_cop"]["mean"]) in result.answer


@pytest.mark.network
def test_real_groq_uses_the_user_settings_when_the_question_names_none():
    config.load_env()
    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set")
    registry, chat = open_agent()
    settings = UserSettings("12h", date(2024, 10, 1), date(2024, 12, 31))
    result = run_agent("¿Cuánto ahorro?", [], registry, chat, settings=settings)
    assert result.stopped == "answer"
    sims = [c for c in result.tool_calls if c.name == "simulate_battery" and c.ok]
    assert sims
    # The outcome, not how it got there: a model may omit the arguments (filled from the
    # settings) or copy them from the prompt; either way it must simulate the user's battery.
    assert sims[0].arguments["preset"] == "12h"
    assert (sims[0].arguments["start_date"], sims[0].arguments["end_date"]) == ("2024-10-01", "2024-12-31")
    assert "12 h" in result.results_block


# --- the user's settings drive simulate_battery (D16, option A) ------------------

SETTINGS = UserSettings("4h", date(2025, 10, 6), date(2025, 10, 19))


def ask_bare(id_: str = "s1", **args) -> ToolCall:
    """A simulate_battery call with only the given arguments (the model omitted the rest)."""
    return ToolCall(id_, "simulate_battery", json.dumps(args))


def test_omitted_arguments_are_filled_from_the_settings(registry):
    chat = ScriptedChat([ChatReply(None, [ask_bare()]), say("Explicación.")])
    result = run_agent("¿Cuánto ahorro?", [], registry, chat, settings=SETTINGS)
    (record,) = result.tool_calls
    assert record.ok
    assert record.arguments["preset"] == "4h"
    assert record.arguments["start_date"] == "2025-10-06"
    assert record.arguments["end_date"] == "2025-10-19"
    assert record.from_settings == ("preset", "start_date", "end_date")
    (sim,) = result.simulations
    assert sim["preset"] == "4h"
    assert sim["period"] == {"start": "2025-10-06", "end": "2025-10-19"}
    assert "batería de 4 h" in result.results_block


def test_explicit_preset_wins_but_dates_are_still_filled(registry):
    chat = ScriptedChat([ChatReply(None, [ask_bare(preset="12h")]), say("Explicación.")])
    settings = UserSettings("8h", date(2025, 10, 6), date(2025, 10, 19))
    result = run_agent("¿y con 12 h?", [], registry, chat, settings=settings)
    (record,) = result.tool_calls
    assert record.ok
    assert record.arguments["preset"] == "12h"
    assert "preset" not in record.from_settings
    assert record.from_settings == ("start_date", "end_date")
    assert result.simulations[0]["preset"] == "12h"


def test_explicit_dates_are_not_overridden(registry):
    chat = ScriptedChat(
        [ChatReply(None, [ask_bare(start_date="2025-11-03", end_date="2025-11-16")]), say("Listo.")]
    )
    result = run_agent("noviembre", [], registry, chat, settings=SETTINGS)
    (record,) = result.tool_calls
    assert record.from_settings == ("preset",)
    assert result.simulations[0]["period"] == {"start": "2025-11-03", "end": "2025-11-16"}
    assert result.simulations[0]["preset"] == "4h"


def test_only_one_date_is_not_filled_and_the_error_goes_back(registry):
    chat = ScriptedChat([ChatReply(None, [ask_bare(start_date="2025-10-06")]), say("Faltó una fecha.")])
    result = run_agent("simula", [], registry, chat, settings=SETTINGS)
    assert result.stopped == "answer"
    (record,) = result.tool_calls
    assert record.ok is False
    assert "start_date" not in record.from_settings and "end_date" not in record.from_settings
    assert "end_date" not in record.arguments
    (tool_msg,) = [m for m in chat.calls[1][0] if m["role"] == "tool"]
    assert "error" in json.loads(tool_msg["content"])
    assert result.simulations == []


def test_without_settings_the_config_defaults_are_used(registry):
    chat = ScriptedChat([ChatReply(None, [ask_bare()]), say("Explicación.")])
    result = run_agent("¿Cuánto ahorro?", [], registry, chat)  # one 3-month simulation
    (record,) = result.tool_calls
    assert record.ok
    assert record.arguments["preset"] == config.DEFAULT_PRESET
    assert record.arguments["start_date"] == config.DEFAULT_START.isoformat()
    assert record.arguments["end_date"] == config.DEFAULT_END.isoformat()
    assert record.from_settings == ("preset", "start_date", "end_date")


def test_system_message_carries_the_settings(registry):
    chat = ScriptedChat([say("ok")])
    run_agent("hola", [], registry, chat, settings=SETTINGS)
    content = chat.calls[0][0][0]["content"]
    assert content == system_prompt(SETTINGS)
    assert "4 h" in content and "2025-10-06" in content and "2025-10-19" in content


def test_two_explicit_presets_still_give_the_comparison(registry):
    calls = [ask_bare("a", preset="4h"), ask_bare("b", preset="12h")]
    chat = ScriptedChat([ChatReply(None, calls), say("Explicación.")])
    result = run_agent("compara 4 y 12 h", [], registry, chat, settings=SETTINGS)
    assert [s["preset"] for s in result.simulations] == ["4h", "12h"]
    assert COMPARISON_HEADER in result.results_block


def test_search_regulation_calls_take_nothing_from_the_settings(registry):
    chat = ScriptedChat([ChatReply(None, [ask_search()]), say(f"Eficiencia [{CITA}].")])
    result = run_agent("eficiencia", [], registry, chat, settings=SETTINGS)
    (record,) = result.tool_calls
    assert record.from_settings == ()
    assert record.arguments == {"query": "eficiencia"}


def test_model_never_sees_the_figures_shown_in_the_block(registry):
    chat = ScriptedChat([ChatReply(None, [ask_sim()]), say("Explicación.")])
    result = run_agent("simula", [], registry, chat)
    (tool_msg,) = [m for m in chat.calls[1][0] if m["role"] == "tool"]
    observation = json.loads(tool_msg["content"])
    for hidden in ("savings_cop", "baselines", "vs_best_rule", "vs_perfect_foresight", "energy"):
        assert hidden not in observation
    assert "by_hour_block" in observation and "best_rule" in observation
    assert result.simulations[0]["savings_cop"]["mean"] > 0  # the block still has them
