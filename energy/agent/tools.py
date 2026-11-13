"""The agent's two tools and the registry that runs them (D16).

- ``search_regulation``: regulation passages with an exact ``cita`` string to quote.
- ``simulate_battery``: the Part 1 simulation (D14, D15), averaged over the 5 test shift
  patterns with the ``default`` mix, always next to the baselines, plus decision profiles
  (by hour block and by price level) so the agent can explain *why* the battery acts.

``run_tool`` never raises: every failure becomes ``{"error": "<mensaje en español>"}`` so it
goes back to the model as an observation. Library only; see ``energy.agent.loop``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from energy import config
from energy.data.prices import load_prices
from energy.rag.retrieve import Retriever
from energy.rl.battery import PRESETS
from energy.rl.factory import TEST_SHIFT_SEEDS, add_shifts
from energy.rl.features import add_relative_price
from energy.rl.simulate import POLICY_IDS, resolve_policy, simulate

logger = logging.getLogger(__name__)

MAX_HIT_CHARS = 1500
MAX_DAYS = 366
MIX = "default"
TRAIN_LAST_MONTH = 9  # D5
RULES = ("night_day", "shift_schedule", "price_threshold")
REFERENCE_POLICIES = ("no_battery", *RULES, "perfect_foresight")
NOTES = (
    "Los turnos de la planta son sintéticos (supuesto A5) y la batería conoce el precio de "
    "bolsa de la hora al decidir, lo que hace el ahorro optimista (A4). Ahorro en COP, sin la "
    "carga total de la planta (A2); no incluye el costo de la batería ni su degradación. "
    "Precios en COP/kWh. charge_pct y discharge_pct = % de las horas de ese bloque (o nivel "
    "de precio) en que la batería carga o descarga, no % de la energía total. "
    "vs_best_rule = % de diferencia frente a la mejor regla fija; "
    "vs_perfect_foresight = % del óptimo con previsión perfecta."
)
HOUR_LABELS = [f"{h:02d}-{h + 1:02d}" for h in range(0, 24, 2)]
PRICE_LEVELS = ("Q1 (25% más barato)", "Q2", "Q3", "Q4 (25% más caro)")


def _comparison_sentence(policy: str, mean: float, best_rule: str, best: float, optimum: float) -> str:
    """Spanish sentence comparing ``policy`` with the best rule and the optimum."""
    parts = []
    if best:
        diff = (mean - best) / abs(best) * 100
        word = "más" if diff >= 0 else "menos"
        parts.append(f"{policy} ahorra {abs(diff):.1f} % {word} que la mejor regla fija ({best_rule})")
    if optimum:
        parts.append(f"alcanza el {mean / optimum * 100:.1f} % del óptimo con previsión perfecta")
    return " y ".join(parts) + "." if parts else ""


@dataclass(frozen=True)
class Tool:
    """A callable tool with its JSON-Schema parameters (OpenAI function format)."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., dict[str, Any]]


class ToolInputError(ValueError):
    """Bad arguments; the message is Spanish and goes back to the model."""


# --- argument validation ---------------------------------------------------------


def _validate(schema: dict[str, Any], args: Any) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ToolInputError("Los argumentos deben ser un objeto JSON.")
    props = schema["properties"]
    for name in schema.get("required", []):
        if name not in args:
            raise ToolInputError(f"Falta el argumento obligatorio '{name}'.")
    clean: dict[str, Any] = {}
    for name, value in args.items():
        if name not in props:
            raise ToolInputError(f"Argumento desconocido '{name}'.")
        spec = props[name]
        kind = spec.get("type")
        if kind == "string" and not isinstance(value, str):
            raise ToolInputError(f"'{name}' debe ser texto.")
        if kind == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                raise ToolInputError(f"'{name}' debe ser un entero.")
            if not spec.get("minimum", value) <= value <= spec.get("maximum", value):
                raise ToolInputError(
                    f"'{name}' debe estar entre {spec.get('minimum')} y {spec.get('maximum')}."
                )
        if "enum" in spec and value not in spec["enum"]:
            raise ToolInputError(f"'{name}' debe ser uno de {spec['enum']}, no {value!r}.")
        clean[name] = value
    return clean


def run_tool(registry: dict[str, Tool], name: str, raw_args: str) -> tuple[str, dict[str, Any]]:
    """Run one tool call; never raises.

    Args:
        registry: Tools by name.
        name: Tool name requested by the model.
        raw_args: Arguments as the JSON string the model produced.

    Returns:
        The JSON observation string and the parsed result (``{"error": ...}`` on failure).
    """
    try:
        tool = registry.get(name)
        if tool is None:
            raise ToolInputError(f"La herramienta '{name}' no existe; usa {sorted(registry)}.")
        try:
            args = json.loads(raw_args) if raw_args and raw_args.strip() else {}
        except json.JSONDecodeError as exc:
            raise ToolInputError(f"Los argumentos no son JSON válido ({exc.msg}).") from exc
        result = tool.fn(**_validate(tool.parameters, args))
    except ToolInputError as exc:
        result = {"error": str(exc)}
    except Exception as exc:  # a tool must never take the agent down (observation instead)
        logger.exception("Tool %s failed", name)
        result = {"error": f"La herramienta '{name}' falló: {type(exc).__name__}: {exc}"}
    return json.dumps(result, ensure_ascii=False), result


def tool_schemas(registry: dict[str, Tool]) -> list[dict[str, Any]]:
    """The registry in the OpenAI ``tools`` format."""
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in registry.values()
    ]


# --- search_regulation -----------------------------------------------------------


def _make_search(retriever: Retriever) -> Callable[..., dict[str, Any]]:
    def search_regulation(query: str, k: int = config.RETRIEVAL_K) -> dict[str, Any]:
        hits = retriever.search(query, k=k)
        out: dict[str, Any] = {
            "retriever": retriever.name,
            "hits": [
                {
                    "cita": h.citation,
                    "doc_id": h.doc_id,
                    "in_force": h.in_force,
                    "superseded_by": h.superseded_by,
                    "text": h.text[:MAX_HIT_CHARS],
                }
                for h in hits
            ],
        }
        if not hits:
            out["nota"] = "sin resultados"
        else:  # a reminder next to the data: the model sometimes drops the brackets (D16)
            out["como_citar"] = (
                "Termina cada afirmación regulatoria con su cita entre corchetes, copiando el "
                f"campo cita exacto, por ejemplo [{hits[0].citation}]."
            )
        return out

    return search_regulation


# --- simulate_battery ------------------------------------------------------------


def _parse_date(name: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ToolInputError(f"'{name}' debe tener formato YYYY-MM-DD, no {value!r}.") from exc


def _pct(series: pd.Series) -> float:
    return round(float(series.mean()) * 100, 1)


def _make_simulate(
    prices: pd.DataFrame | None, policies_dir: Path | str
) -> Callable[..., dict[str, Any]]:
    cache: dict[str, Any] = {}

    def seed_frames() -> list[tuple[int, pd.DataFrame]]:
        # Built once, on first use: the relative price over the whole series, then one
        # shifted copy per test seed, so a call does not repeat that work.
        if "frames" not in cache:
            raw = prices if prices is not None else load_prices()
            featured = add_relative_price(raw)
            cache["raw"] = raw
            cache["frames"] = [(s, add_shifts(featured, MIX, s)) for s in TEST_SHIFT_SEEDS]
        return cache["frames"]

    def simulate_battery(
        start_date: str, end_date: str, policy: str = "q_learning", preset: str = "8h"
    ) -> dict[str, Any]:
        start, end = _parse_date("start_date", start_date), _parse_date("end_date", end_date)
        if start > end:
            raise ToolInputError("start_date no puede ser posterior a end_date.")
        if start.year != end.year:
            raise ToolInputError(
                "El rango debe estar dentro de un mismo año calendario (las tablas Q son por año)."
            )
        if (end - start).days + 1 > MAX_DAYS:
            raise ToolInputError(f"El rango no puede superar {MAX_DAYS} días.")
        frames = seed_frames()
        stamps = frames[0][1]["timestamp"]
        first_day, last_day = stamps.min().date(), stamps.max().date()
        if start < first_day or end > last_day:
            raise ToolInputError(f"Hay precios solo entre {first_day} y {last_day}.")

        ids = list(dict.fromkeys([policy, *REFERENCE_POLICIES]))
        resolved = {pid: resolve_policy(pid, preset, start.year, policies_dir) for pid in ids}
        per_policy: dict[str, list[Any]] = {pid: [] for pid in ids}
        for _, frame in frames:
            for pid in ids:
                per_policy[pid].append(
                    simulate(resolved[pid], start, end, preset, prices=frame, policies_dir=policies_dir)
                )

        mean_savings = {
            pid: float(np.mean([r.summary.savings_cop for r in runs])) for pid, runs in per_policy.items()
        }
        own = [r.summary.savings_cop for r in per_policy[policy]]
        mean = mean_savings[policy]
        best_rule = max(RULES, key=lambda r: mean_savings[r])
        best, optimum = mean_savings[best_rule], mean_savings["perfect_foresight"]
        runs = per_policy[policy]

        hourly = [r.hourly for r in runs]
        price = hourly[0]["price_cop_kwh"].to_numpy()
        edges = np.quantile(price, [0.25, 0.5, 0.75])
        level_of = np.digitize(price, edges, right=True)
        block_of = hourly[0]["timestamp"].dt.hour.to_numpy() // 2
        by_hour, by_level = [], []
        for b, label in enumerate(HOUR_LABELS):
            sel = block_of == b
            by_hour.append(
                {
                    "hours": label,
                    "charge_pct": round(float(np.mean([(h["action"][sel] == 1).mean() for h in hourly])) * 100, 1),
                    "discharge_pct": round(float(np.mean([(h["action"][sel] == 2).mean() for h in hourly])) * 100, 1),
                    "mean_price": round(float(price[sel].mean()), 1),
                }
            )
        for q, label in enumerate(PRICE_LEVELS):
            sel = level_of == q
            by_level.append(
                {
                    "level": label,
                    "price_range": f"{price[sel].min():.0f}-{price[sel].max():.0f}" if sel.any() else "",
                    "charge_pct": round(float(np.mean([(h["action"][sel] == 1).mean() for h in hourly])) * 100, 1),
                    "discharge_pct": round(float(np.mean([(h["action"][sel] == 2).mean() for h in hourly])) * 100, 1),
                }
            )

        return {
            "policy": policy,
            "policy_used": resolved[policy].name,
            "preset": preset,
            "period": {"start": start.isoformat(), "end": end.isoformat()},
            "shift_patterns": len(TEST_SHIFT_SEEDS),
            "mix": MIX,
            "savings_cop": {"mean": round(mean, 2), "min": round(min(own), 2), "max": round(max(own), 2)},
            "baselines": {pid: round(mean_savings[pid]) for pid in REFERENCE_POLICIES},
            "best_rule": best_rule,
            "vs_best_rule": round((mean - best) / abs(best) * 100, 1) if best else 0.0,
            "vs_perfect_foresight": round(mean / optimum * 100, 1) if optimum else 0.0,
            # Pre-written so the model can't flip the sign of the comparison (seen live).
            "comparison": _comparison_sentence(policy, mean, best_rule, best, optimum),
            "energy": {
                "bought_kwh": round(float(np.mean([r.summary.bought_kwh for r in runs]))),
                "delivered_kwh": round(float(np.mean([r.summary.delivered_kwh for r in runs]))),
                "cycles": round(float(np.mean([r.summary.cycles for r in runs])), 1),
            },
            "by_hour_block": by_hour,
            "by_price_level": by_level,
            "in_sample": policy == "q_learning" and start.month <= TRAIN_LAST_MONTH,
            "notes": NOTES,
        }

    return simulate_battery


# --- registry ----------------------------------------------------------------------


def build_registry(
    retriever: Retriever,
    prices: pd.DataFrame | None = None,
    policies_dir: Path | str = config.POLICIES_DIR,
) -> dict[str, Tool]:
    """Assemble the two tools.

    Args:
        retriever: Regulation search backend.
        prices: Hourly prices; the committed CSV is loaded on the first simulation if None.
        policies_dir: Where the Q-tables and thresholds live.

    Returns:
        Tools by name.
    """
    search = Tool(
        name="search_regulation",
        description=(
            "Busca en la regulación colombiana de almacenamiento y energía (CREG, MinEnergía, ...). "
            "Devuelve fragmentos con 'cita' (texto exacto para citar), in_force y superseded_by."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Pregunta o términos de búsqueda, en español."},
                "k": {"type": "integer", "minimum": 1, "maximum": 8, "description": "Fragmentos a devolver (por defecto 5)."},
            },
            "required": ["query"],
        },
        fn=_make_search(retriever),
    )
    simulation = Tool(
        name="simulate_battery",
        description=(
            "Simula el despacho de una batería con precios de bolsa reales de XM (2024-01-01 a 2025-12-31), "
            "promediando 5 calendarios de turnos sintéticos, y lo compara con reglas fijas y con el óptimo. "
            "Devuelve ahorro en COP y perfiles de decisión por hora y por nivel de precio. "
            "El rango debe estar en un solo año. Sin fechas del usuario, usa 2025-10-01 a 2025-12-31 (periodo de prueba)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "policy": {"type": "string", "enum": list(POLICY_IDS), "description": "Política a simular (por defecto q_learning)."},
                "preset": {"type": "string", "enum": sorted(PRESETS), "description": "Batería: 4h, 8h o 12h (por defecto 8h)."},
                "start_date": {"type": "string", "description": "Primer día, YYYY-MM-DD."},
                "end_date": {"type": "string", "description": "Último día, YYYY-MM-DD."},
            },
            "required": ["start_date", "end_date"],
        },
        fn=_make_simulate(prices, policies_dir),
    )
    return {t.name: t for t in (search, simulation)}
