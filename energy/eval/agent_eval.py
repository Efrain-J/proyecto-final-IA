"""Agent evaluation (step 7): run the fixed question set and score each answer.

Run from the repo root::

    uv run python -m energy.eval.agent_eval [--ids 1,5,9] [--pause 20] [--bm25]
        [--questions energy/eval/agent_questions.yaml] [--out results/agent_eval]

Input: ``config.AGENT_QUESTIONS``, a YAML list of questions with what each one expects
(the fields are documented at the top of that file). Every question runs once, in a fresh
chat with the default ``UserSettings``.

Outputs in ``--out``:

* ``agent_eval.csv``: one row per question with the automatic checks. A check is True,
  False or empty when it doesn't apply or needs a human (a decline that isn't the exact
  refusal phrase). ``aprobada`` is empty when any applicable check is empty, and
  ``revision_manual`` is left blank for the reviewer.
* ``transcripts.jsonl``: the full answer, tool calls, citations and models per question.
* ``summary.md``: the rates for the report.

Needs ``GROQ_API_KEY``. Questions are spaced by ``--pause`` seconds because the free tier
allows 8,000 tokens per minute; a fallback to the 20b model shows in ``modelos`` (D16).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import os
import re
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from energy import config
from energy.agent.llm import ChatFn, ChatReply
from energy.agent.loop import AgentResult, run_agent
from energy.agent.prompts import REFUSAL
from energy.agent.settings import UserSettings
from energy.agent.tools import Tool
from energy.data.io import write_bytes_atomic

logger = logging.getLogger(__name__)

TIPOS = ("regulacion", "simulacion", "mixta", "trampa")
TOOLS = ("search_regulation", "simulate_battery")
DEFAULT_PAUSE_S = 20.0
SIM_KEYS = ("preset", "start_date", "end_date")


@dataclass(frozen=True)
class Question:
    """One evaluation question and what it expects (see the YAML header)."""

    id: int
    tipo: str
    pregunta: str
    herramientas_esperadas: tuple[str, ...] = ()
    herramientas_prohibidas: tuple[str, ...] = ()
    simulaciones_esperadas: tuple[tuple[str, str, str], ...] = ()  # (preset, start, end)
    fuentes_esperadas: tuple[str, ...] = ()
    debe_contener: tuple[str, ...] = ()
    no_debe_contener: tuple[str, ...] = ()
    rechazo_esperado: bool = False
    notas: str = ""


def _strs(entry: dict, key: str) -> tuple[str, ...]:
    value = entry.get(key) or []
    if not isinstance(value, list):
        raise ValueError(f"pregunta {entry.get('id')}: {key} debe ser una lista")
    return tuple(str(v) for v in value)


def load_questions(path: Path = config.AGENT_QUESTIONS) -> list[Question]:
    """Read and validate the question file.

    Raises:
        ValueError: On a missing field, a duplicate id, an unknown tipo or tool, or an
            invalid regular expression, so a typo fails before any API call.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{path} debe ser una lista no vacía de preguntas")
    questions: list[Question] = []
    for entry in raw:
        if not isinstance(entry, dict) or not {"id", "tipo", "pregunta"} <= entry.keys():
            raise ValueError(f"cada pregunta necesita id, tipo y pregunta: {entry!r}")
        qid = int(entry["id"])
        if entry["tipo"] not in TIPOS:
            raise ValueError(f"pregunta {qid}: tipo {entry['tipo']!r} no es uno de {TIPOS}")
        sims = []
        for sim in entry.get("simulaciones_esperadas") or []:
            if not isinstance(sim, dict) or set(sim) != set(SIM_KEYS):
                raise ValueError(f"pregunta {qid}: cada simulación necesita {SIM_KEYS}")
            sims.append(tuple(str(sim[k]) for k in SIM_KEYS))
        q = Question(
            id=qid,
            tipo=entry["tipo"],
            pregunta=str(entry["pregunta"]).strip(),
            herramientas_esperadas=_strs(entry, "herramientas_esperadas"),
            herramientas_prohibidas=_strs(entry, "herramientas_prohibidas"),
            simulaciones_esperadas=tuple(sims),  # type: ignore[arg-type]
            fuentes_esperadas=_strs(entry, "fuentes_esperadas"),
            debe_contener=_strs(entry, "debe_contener"),
            no_debe_contener=_strs(entry, "no_debe_contener"),
            rechazo_esperado=bool(entry.get("rechazo_esperado", False)),
            notas=str(entry.get("notas", "")).strip(),
        )
        unknown = set(q.herramientas_esperadas + q.herramientas_prohibidas) - set(TOOLS)
        if unknown:
            raise ValueError(f"pregunta {qid}: herramientas desconocidas {sorted(unknown)}")
        for pattern in q.fuentes_esperadas + q.debe_contener + q.no_debe_contener:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"pregunta {qid}: expresión inválida {pattern!r}: {exc}") from exc
        questions.append(q)
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("hay ids de pregunta repetidos")
    return questions


def _search(pattern: str, text: str) -> bool:
    return re.search(pattern, " ".join(text.split()), flags=re.IGNORECASE) is not None


def _all_or_none(checks: list[bool]) -> bool | None:
    return all(checks) if checks else None


def score(question: Question, result: AgentResult) -> dict[str, Any]:
    """Automatic checks for one answer.

    Returns:
        The CSV row fields: each check is True, False or None (doesn't apply, or needs a
        human). ``aprobada`` is None when any applicable check is None.
    """
    called = [c.name for c in result.tool_calls]
    herramienta_ok = all(t in called for t in question.herramientas_esperadas) and not any(
        t in called for t in question.herramientas_prohibidas
    )

    ran = {tuple(str(c.arguments.get(k)) for k in SIM_KEYS) for c in result.tool_calls if c.name == "simulate_battery" and c.ok}
    simulacion_ok = _all_or_none([sim in ran for sim in question.simulaciones_esperadas])

    # The results block's "Fuente:" line counts as the simulation's citation.
    sources = list(result.citations) + ([result.results_block] if result.results_block else [])
    cita_ok = None
    if question.fuentes_esperadas:
        cita_ok = not result.unverified_citations and all(
            any(_search(p, s) for s in sources) for p in question.fuentes_esperadas
        )

    contenido = [_search(p, result.answer) for p in question.debe_contener]
    contenido += [not _search(p, result.answer) for p in question.no_debe_contener]
    contenido_ok = _all_or_none(contenido)

    rechazo_exacto = result.answer.strip() == REFUSAL
    rechazo_ok = None
    if question.rechazo_esperado:
        rechazo_ok = True if rechazo_exacto else None  # another decline: a human decides
    elif rechazo_exacto:
        rechazo_ok = False  # refused an answerable question

    checks = [herramienta_ok, simulacion_ok, cita_ok, contenido_ok, rechazo_ok]
    if result.stopped == "llm_error" or any(c is False for c in checks):
        aprobada: bool | None = False
    elif question.rechazo_esperado and rechazo_ok is None:
        aprobada = None  # declined in its own words: a human decides
    else:
        aprobada = True
    return {
        "herramienta_ok": herramienta_ok,
        "simulacion_ok": simulacion_ok,
        "cita_ok": cita_ok,
        "contenido_ok": contenido_ok,
        "rechazo_exacto": rechazo_exacto,
        "rechazo_ok": rechazo_ok,
        "aprobada": aprobada,
    }


@dataclass
class _Recorder:
    """Wraps a ``ChatFn`` to record which model answered each call."""

    chat: ChatFn
    models: list[str] = field(default_factory=list)

    def __call__(self, messages: list[dict], tools: list[dict] | None) -> ChatReply:
        reply = self.chat(messages, tools)
        self.models.append(reply.model or "?")
        return reply


def evaluate(
    questions: list[Question],
    registry: dict[str, Tool],
    chat: ChatFn,
    pause_s: float = DEFAULT_PAUSE_S,
    sleep: Callable[[float], None] = time.sleep,
    settings: UserSettings | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run every question in a fresh chat and score it.

    Returns:
        ``(rows, transcripts)``: the CSV rows and the full records, in question order.
    """
    settings = settings or UserSettings()
    rows: list[dict[str, Any]] = []
    transcripts: list[dict[str, Any]] = []
    for i, q in enumerate(questions):
        if i and pause_s > 0:
            sleep(pause_s)
        recorder = _Recorder(chat)
        started = time.perf_counter()
        result = run_agent(q.pregunta, [], registry, recorder, settings=settings)
        seconds = round(time.perf_counter() - started, 1)
        checks = score(q, result)
        called = [c.name for c in result.tool_calls]
        models = sorted(set(recorder.models))
        rows.append(
            {
                "id": q.id,
                "tipo": q.tipo,
                "pregunta": " ".join(q.pregunta.split())[:120],
                "herramientas_esperadas": " ".join(q.herramientas_esperadas),
                "herramientas_llamadas": " ".join(called),
                **checks,
                "citas": " ; ".join(result.citations),
                "citas_no_verificadas": " ; ".join(result.unverified_citations),
                "detenido": result.stopped,
                "llamadas_llm": len(recorder.models),
                "modelos": " ".join(models),
                "segundos": seconds,
                "revision_manual": "",
                "notas": q.notas,
            }
        )
        transcripts.append(
            {
                "id": q.id,
                "pregunta": q.pregunta,
                "respuesta": result.answer,
                "texto_modelo": result.model_text,
                "bloque_resultados": result.results_block,
                "llamadas": [asdict(c) for c in result.tool_calls],
                "citas": result.citations,
                "citas_no_verificadas": result.unverified_citations,
                "detenido": result.stopped,
                "retriever": result.retriever,
                "modelos_por_llamada": recorder.models,
                "segundos": seconds,
                "checks": checks,
            }
        )
        logger.info("pregunta %s: aprobada=%s herramientas=%s modelos=%s", q.id, checks["aprobada"], called, models)
    return rows, transcripts


def _cell(value: Any) -> Any:
    return "" if value is None else value


def _rate(rows: list[dict[str, Any]], key: str) -> str:
    applicable = [r[key] for r in rows if r[key] is not None]
    if not applicable:
        return "n/a"
    return f"{sum(applicable)}/{len(applicable)}"


def summary_markdown(rows: list[dict[str, Any]], settings: UserSettings, retriever: str) -> str:
    """The rates and the per-question table for the report."""
    pending = [r["id"] for r in rows if r["aprobada"] is None]
    models = sorted({m for r in rows for m in r["modelos"].split()})
    lines = [
        "# Evaluación del agente",
        "",
        f"Fecha: {datetime.now():%Y-%m-%d %H:%M}. Configuración: batería {settings.preset}, "
        f"{settings.start} a {settings.end}. Retriever: {retriever or '-'}. Modelos: {', '.join(models) or '-'}.",
        "",
        "| Métrica | Resultado |",
        "|---|---|",
        f"| Herramienta correcta | {_rate(rows, 'herramienta_ok')} |",
        f"| Simulación con los parámetros esperados | {_rate(rows, 'simulacion_ok')} |",
        f"| Cita de la fuente esperada | {_rate(rows, 'cita_ok')} |",
        f"| Contenido obligatorio / prohibido | {_rate(rows, 'contenido_ok')} |",
        f"| Rechazo con la frase exacta (trampas) | {_rate([r for r in rows if r['tipo'] == 'trampa'], 'rechazo_exacto')} |",
        f"| Aprobadas (automático) | {_rate(rows, 'aprobada')} |",
        "",
    ]
    if pending:
        lines += [f"Pendientes de revisión manual: {', '.join(str(i) for i in pending)}.", ""]
    lines += ["| id | tipo | herramientas | cita | aprobada | modelos |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['tipo']} | {r['herramientas_llamadas'] or '-'} | "
            f"{_cell(r['cita_ok']) if r['cita_ok'] is not None else 'n/a'} | "
            f"{'revisar' if r['aprobada'] is None else r['aprobada']} | {r['modelos']} |"
        )
    return "\n".join(lines) + "\n"


def write_outputs(out: Path, rows: list[dict], transcripts: list[dict], summary: str) -> None:
    """Write the CSV (UTF-8 with BOM, so Excel shows the accents), JSONL and summary."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows({k: _cell(v) for k, v in r.items()} for r in rows)
    write_bytes_atomic(out / "agent_eval.csv", buffer.getvalue().encode("utf-8-sig"))
    jsonl = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in transcripts)
    write_bytes_atomic(out / "transcripts.jsonl", jsonl.encode("utf-8"))
    write_bytes_atomic(out / "summary.md", summary.encode("utf-8"))


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    config.load_env()
    if hasattr(sys.stdout, "reconfigure"):  # Windows consoles garble Spanish accents
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Run the fixed agent evaluation.")
    parser.add_argument("--questions", type=Path, default=config.AGENT_QUESTIONS)
    parser.add_argument("--ids", help="comma-separated ids to run (default: all)")
    parser.add_argument("--pause", type=float, default=DEFAULT_PAUSE_S, help="seconds between questions")
    parser.add_argument("--bm25", action="store_true", help="force the BM25 retriever")
    parser.add_argument("--out", type=Path, default=config.RESULTS_DIR / "agent_eval")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    questions = load_questions(args.questions)
    if args.ids:
        wanted = {int(i) for i in args.ids.split(",")}
        questions = [q for q in questions if q.id in wanted]
        if not questions:
            parser.error(f"ningún id coincide con {args.ids}")
    if not os.environ.get("GROQ_API_KEY"):
        parser.error("GROQ_API_KEY no está configurada; sin el modelo la evaluación no tiene sentido")

    from energy.agent.loop import open_agent  # heavy imports only when running for real
    from energy.rag.chunk import read_chunks
    from energy.rag.retrieve import BM25Retriever

    retriever = BM25Retriever(read_chunks(config.CHUNKS_PATH)) if args.bm25 else None
    registry, chat = open_agent(retriever)
    settings = UserSettings()
    rows, transcripts = evaluate(questions, registry, chat, pause_s=args.pause, settings=settings)
    retriever_name = next((t["retriever"] for t in transcripts if t["retriever"]), "")
    summary = summary_markdown(rows, settings, retriever_name)
    write_outputs(args.out, rows, transcripts, summary)
    print(summary)
    print(f"Escrito en {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
