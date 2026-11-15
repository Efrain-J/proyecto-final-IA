"""The hand-written agent loop (D16).

``run_agent`` asks the model, runs the tools it requests and feeds the observations back,
for at most ``t_max`` tool rounds; if the rounds run out it makes one last call with tools
switched off (at most ``t_max + 1`` LLM calls). It never raises: an ``LLMError`` becomes a
retrieval-only answer. Citations in the model's text are checked against what
``search_regulation`` returned during this run.

Simulation figures are not written by the model: the successful ``simulate_battery`` results
are rendered by ``energy.agent.render`` into a block placed before the model's text, on every
path (D16).

``open_agent`` assembles the real retriever, tools and Groq chat for the app and the
evaluation; entry points call ``config.load_env()`` first. Manual check:
``uv run python -m energy.agent "pregunta"``.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from energy import config
from energy.agent.llm import ChatFn, ChatReply, LLMError, MissingKeyError, make_groq_chat
from energy.agent.prompts import REFUSAL, system_prompt
from energy.agent.render import render_results
from energy.agent.settings import UserSettings
from energy.agent.tools import Tool, build_registry, run_tool, tool_schemas
from energy.rag.retrieve import Retriever, open_retriever

logger = logging.getLogger(__name__)

NUDGE = "Responde ahora con la información obtenida hasta aquí, sin llamar más herramientas."
FALLBACK_HITS = 3
# [text] with >= 3 letters, not followed by "(" (Markdown link); skips [1], [x], [ ].
_BRACKET = re.compile(r"\[([^\[\]]+)\](?!\()")


@dataclass(frozen=True)
class ToolCallRecord:
    """One executed tool call, for the evaluation and the app."""

    name: str
    arguments: dict[str, Any]
    ok: bool
    error: str | None = None
    from_settings: tuple[str, ...] = ()  # arguments filled from the user's settings (D16)


@dataclass
class AgentResult:
    """Outcome of ``run_agent``.

    Attributes:
        answer: Final text shown to the user: ``results_block`` then ``model_text``.
        tool_calls: Tool calls in order.
        citations: Cited sources that ``search_regulation`` returned in this run.
        unverified_citations: Cited sources that it did not.
        stopped: ``answer`` (normal), ``t_max`` (round cap hit) or ``llm_error``.
        retriever: Name of the retriever seen in a search result, or "".
        simulations: Successful ``simulate_battery`` results, in call order.
        results_block: Deterministic rendering of ``simulations``, or "".
        model_text: What the model wrote (tables removed when a block exists).
        error: Technical message of the ``LLMError`` when ``stopped == "llm_error"``, else None.
        models: Model id of every chat call in order ("?" when unknown), for "Detalles" and
            the fallback-model notice in the app (D16).
    """

    answer: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    unverified_citations: list[str] = field(default_factory=list)
    stopped: Literal["answer", "t_max", "llm_error"] = "answer"
    retriever: str = ""
    simulations: list[dict[str, Any]] = field(default_factory=list)
    results_block: str = ""
    model_text: str = ""
    error: str | None = None
    models: list[str] = field(default_factory=list)


def _strip_tables(text: str) -> str:
    """Drop Markdown table lines: the results block is the only source of tables (D16)."""
    kept = [ln for ln in text.splitlines() if not ln.lstrip().startswith("|")]
    return "\n".join(kept).strip()


def _compose(block: str, text: str) -> str:
    return f"{block}\n\n{text}" if block and text else block or text


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def _normalize_brackets(answer: str) -> str:
    """Turn gpt-oss's native 【…】 citations into [...] and drop a "cita:" prefix (D16)."""
    answer = answer.replace("【", "[").replace("】", "]")
    return re.sub(r"\[\s*cita\s*:\s*", "[", answer, flags=re.IGNORECASE)


# Figures the results block already shows; the model never sees them, so it can't
# restate them wrongly (it once wrote 3.131.505 for 3.131.506, D16).
_HIDDEN_FROM_MODEL = ("savings_cop", "baselines", "vs_best_rule", "vs_perfect_foresight", "energy")


def _model_view(result: dict) -> str:
    """JSON observation for the model: the simulation without the figures in the block."""
    view = {k: v for k, v in result.items() if k not in _HIDDEN_FROM_MODEL}
    view["resultados"] = (
        "El ahorro, la tabla y la comparación ya se muestran al usuario en un bloque "
        "automático; no escribas cifras de ahorro ni porcentajes de comparación."
    )
    return json.dumps(view, ensure_ascii=False)


def _repair_request(citas: list[str]) -> str:
    listed = "\n".join(f"[{c}]" for c in citas)
    return (
        "Tu respuesta no tiene citas válidas. Reescríbela completa, con el mismo contenido y "
        "formato, pero termina cada afirmación regulatoria con su cita entre corchetes, "
        "copiada exacta de esta lista (no escribas la palabra cita):\n" + listed
    )


def _split_citations(answer: str, retrieved: set[str]) -> tuple[list[str], list[str]]:
    valid: list[str] = []
    unverified: list[str] = []
    for match in _BRACKET.findall(answer):
        text = " ".join(match.split())
        if sum(ch.isalpha() for ch in text) < 3:
            continue
        target = valid if _norm(text) in retrieved else unverified
        if text not in target:
            target.append(text)
    return valid, unverified


NO_KEY_TEXT = (
    "El asistente no tiene configurado el modelo de lenguaje (falta la clave de Groq), "
    "así que no puede redactar una respuesta ni simular."
)
LLM_DOWN_TEXT = "El modelo de lenguaje no está disponible en este momento (límite de uso o conexión)."
FALLBACK_INTRO = "Estos son los pasajes de la normativa más relevantes para tu pregunta:"
FALLBACK_OUTRO = "Para ver el ahorro de tu batería usa las pestañas Operación y Comparación."


def _fallback_answer(
    error: LLMError, registry: dict[str, Tool], question: str
) -> tuple[str, list[str], str]:
    """Retrieval-only answer when the model is unavailable.

    Plain Spanish first; the technical message goes to ``AgentResult.error`` instead (D16).

    Returns:
        ``(text, citas of the passages shown, retriever name)``.
    """
    lead = NO_KEY_TEXT if isinstance(error, MissingKeyError) else LLM_DOWN_TEXT
    lines = [lead, "", FALLBACK_INTRO]
    _, found = run_tool(registry, "search_regulation", json.dumps({"query": question, "k": FALLBACK_HITS}))
    citas: list[str] = []
    for hit in found.get("hits", [])[:FALLBACK_HITS]:
        # Drop markdown heading marks ("### Término:" in the XM glossary): inside a bullet
        # they render as big headings in the app.
        plain = re.sub(r"#{1,6}\s*", "", hit["text"])
        snippet = " ".join(plain.split())[:300]
        lines.append(f"- {snippet}... [{hit['cita']}]")
        citas.append(hit["cita"])
    if not citas:
        lines.append(found.get("error") or "No hay fragmentos relevantes.")
    lines += ["", FALLBACK_OUTRO]
    return "\n".join(lines), citas, found.get("retriever", "")


def _fill_from_settings(raw: str, settings: UserSettings) -> tuple[str, tuple[str, ...]]:
    """Fill the omitted ``simulate_battery`` arguments from the user's settings (D16).

    Dates are filled only as a pair, so one date alone reaches the tool and comes back as
    an error the model can fix. Unparseable arguments are left for ``run_tool`` to reject.

    Returns:
        The (re-serialized) arguments and the names that were filled.
    """
    try:
        args = json.loads(raw) if raw and raw.strip() else {}
    except json.JSONDecodeError:
        return raw, ()
    if not isinstance(args, dict):
        return raw, ()
    filled: list[str] = []
    if "preset" not in args:
        args["preset"] = settings.preset
        filled.append("preset")
    if "start_date" not in args and "end_date" not in args:
        args["start_date"] = settings.start.isoformat()
        args["end_date"] = settings.end.isoformat()
        filled += ["start_date", "end_date"]
    return json.dumps(args), tuple(filled)


def run_agent(
    question: str,
    history: list[dict],
    registry: dict[str, Tool],
    chat: ChatFn,
    t_max: int = config.T_MAX,
    settings: UserSettings | None = None,
) -> AgentResult:
    """Answer one question with tool calling.

    Args:
        question: The user's question.
        history: Earlier chat messages (``role`` and ``content``); only the last
            ``config.HISTORY_MESSAGES`` are sent.
        registry: Tools by name.
        chat: The model.
        t_max: Maximum tool rounds.
        settings: The user's battery and period; None means the config defaults.

    Returns:
        The result; it never raises.
    """
    settings = settings or UserSettings()
    messages: list[dict] = [{"role": "system", "content": system_prompt(settings)}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history[-config.HISTORY_MESSAGES :]]
    messages.append({"role": "user", "content": question})
    schemas = tool_schemas(registry)
    records: list[ToolCallRecord] = []
    retrieved: set[str] = set()
    retrieved_citas: list[str] = []  # original spelling, for the repair request
    retriever = ""
    simulations: list[dict[str, Any]] = []
    answer, stopped = "", "answer"
    models: list[str] = []

    def ask(msgs: list[dict], tools: list[dict] | None) -> ChatReply:
        reply = chat(msgs, tools)
        models.append(reply.model or "?")  # same convention as the evaluation's recorder
        return reply

    try:
        for _ in range(t_max):
            reply = ask(messages, schemas)
            if not reply.tool_calls:
                answer = reply.content or ""
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": reply.content,
                    "tool_calls": [
                        {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                        for c in reply.tool_calls
                    ],
                }
            )
            for call in reply.tool_calls:
                call_args, from_settings = call.arguments, ()
                if call.name == "simulate_battery":
                    call_args, from_settings = _fill_from_settings(call.arguments, settings)
                raw, parsed = run_tool(registry, call.name, call_args)
                try:
                    arguments = json.loads(call_args)
                except json.JSONDecodeError:
                    arguments = {}
                error = parsed.get("error") if isinstance(parsed, dict) else None
                records.append(
                    ToolCallRecord(
                        call.name,
                        arguments if isinstance(arguments, dict) else {},
                        error is None,
                        error,
                        from_settings,
                    )
                )
                if call.name == "simulate_battery" and error is None:
                    simulations.append(parsed)
                    raw = _model_view(parsed)
                if call.name == "search_regulation" and error is None:
                    retriever = parsed.get("retriever", retriever)
                    for hit in parsed.get("hits", []):
                        if _norm(hit["cita"]) not in retrieved:
                            retrieved.add(_norm(hit["cita"]))
                            retrieved_citas.append(hit["cita"])
                messages.append({"role": "tool", "tool_call_id": call.id, "content": raw})
        else:  # rounds used up: one last call with tools off
            stopped = "t_max"
            answer = ask(messages + [{"role": "user", "content": NUDGE}], None).content or ""
    except LLMError as exc:
        logger.warning("LLM failed: %s", exc)
        text, citas, found_retriever = _fallback_answer(exc, registry, question)
        block = render_results(simulations)
        return AgentResult(
            answer=_compose(block, text),
            tool_calls=records,
            citations=citas,
            unverified_citations=[],
            stopped="llm_error",
            retriever=found_retriever or retriever,
            simulations=simulations,
            results_block=block,
            model_text=text,
            error=str(exc),
            models=models,
        )

    block = render_results(simulations)

    def clean(text: str) -> str:
        text = _normalize_brackets(text)
        return _strip_tables(text) if block else text

    answer = clean(answer)
    valid, unverified = _split_citations(answer, retrieved)
    if stopped == "answer" and retrieved and not valid and answer.strip() != REFUSAL:
        # One repair round, outside t_max: the model sometimes drops or garbles the
        # citations (e.g. a literal 【cita】); keep the original if the repair doesn't help.
        try:
            repaired = ask(
                messages
                + [{"role": "assistant", "content": answer}, {"role": "user", "content": _repair_request(retrieved_citas)}],
                None,
            ).content or ""
        except LLMError as exc:
            logger.warning("Citation repair failed: %s", exc)
        else:
            repaired = clean(repaired)
            new_valid, new_unverified = _split_citations(repaired, retrieved)
            if new_valid:
                answer, valid, unverified = repaired, new_valid, new_unverified
    return AgentResult(
        _compose(block, answer), records, valid, unverified, stopped, retriever,  # type: ignore[arg-type]
        simulations, block, answer, None, models,
    )


def open_agent(retriever: Retriever | None = None) -> tuple[dict[str, Tool], ChatFn]:
    """Assemble the real retriever, tools and Groq chat.

    Args:
        retriever: Use this retriever instead of ``open_retriever()`` (e.g. BM25).

    Returns:
        ``(registry, chat)``. Does not call ``load_env``; the entry point does.
    """
    from energy.data.prices import load_prices

    retriever = retriever or open_retriever()
    registry = build_registry(retriever, prices=load_prices())
    return registry, make_groq_chat()
