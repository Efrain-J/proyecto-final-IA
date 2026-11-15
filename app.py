"""Streamlit dashboard: a battery for a Colombian plant, with a regulation assistant.

Run from the repo root::

    uv run streamlit run app.py

Inputs: the committed XM prices, the saved Q-tables and the RAG index/chunks. The sidebar
(scenario, battery, shift mix) builds one `UserSettings` that feeds both the charts and
the chat agent, so the assistant answers for the scenario on screen (D16). Without
`GROQ_API_KEY` the assistant shows retrieved passages only; without `NVIDIA_API_KEY` the
search falls back to BM25. All figures come from `energy.app.model` (pure) and
`energy.app.charts` (plotly); this file only wires them to widgets (architecture §9).
"""
from __future__ import annotations

import os
from datetime import date
from typing import Any

import pandas as pd
import streamlit as st

from energy import config
from energy.agent.loop import AgentResult, open_agent, run_agent
from energy.agent.render import POLICY_NAMES, cop, pct
from energy.agent.settings import DATA_FIRST_DAY, DATA_LAST_DAY, UserSettings
from energy.app import charts, model
from energy.data.prices import load_prices
from energy.rag.retrieve import open_retriever
from energy.rl.battery import PRESETS
from energy.rl.features import add_relative_price

SECRET_KEYS = ("GROQ_API_KEY", "NVIDIA_API_KEY")
MESSAGES_KEY = "mensajes"
CUSTOM_PERIOD = "Personalizado"
SUGGESTED_KEY = "pregunta_sugerida"
EXAMPLE_QUESTIONS = (
    "¿Cuánto ahorro con mi batería en este periodo?",
    "¿Por qué la batería carga a ciertas horas?",
    "¿Qué dice la CREG sobre almacenamiento con baterías?",
    "¿Necesito un permiso para instalar una batería en mi planta?",
)
FALLBACK_MODEL_NOTICE = (
    "Respondió el modelo de respaldo (más pequeño); la explicación puede ser menos precisa."
)


def error_text(exc: BaseException) -> str:
    """One-line description of an exception for `st.error`."""
    return f"{type(exc).__name__}: {exc}"


# --------------------------------------------------------------------------- cached data


@st.cache_data(show_spinner="Cargando precios…")
def featured_prices() -> pd.DataFrame:
    """Hourly prices with the relative-price feature the policies need."""
    return add_relative_price(load_prices())


@st.cache_data(show_spinner="Simulando la operación…")
def cached_runs(preset: str, start: date, end: date, mix: str) -> dict[str, model.PolicyRun]:
    """Policy runs for the sidebar choice (hashable arguments only)."""
    return model.run_policies(featured_prices(), preset, start, end, mix)


@st.cache_data(show_spinner="Comparando tamaños de batería…")
def cached_sizes(start: date, end: date, mix: str) -> pd.DataFrame:
    """Savings per battery size for the sidebar period and mix."""
    return model.size_comparison(featured_prices(), start, end, mix)


@st.cache_resource(show_spinner="Abriendo el asistente…")
def load_agent() -> tuple[Any, Any, str | None, str | None]:
    """Open the retriever and the agent once per server process.

    Returns:
        ``(registry, chat, retriever_name, None)`` or ``(None, None, None, error)``.
        One shared retriever: Qdrant local mode takes a file lock (architecture §9).
    """
    try:
        retriever = open_retriever()
        registry, chat = open_agent(retriever)
        return registry, chat, retriever.name, None
    except Exception as exc:  # noqa: BLE001 - the app must keep running without the agent
        return None, None, None, error_text(exc)


# --------------------------------------------------------------------------- sidebar


def _period_widgets() -> tuple[date, date]:
    """One radio for the period (two scenarios or custom); returns the dates to simulate."""
    choice = st.sidebar.radio(
        "Periodo", [*model.SCENARIOS, CUSTOM_PERIOD], key="escenario"
    )
    if choice != CUSTOM_PERIOD:
        return model.SCENARIOS[choice]
    default = next(iter(model.SCENARIOS.values()))  # 2025 test quarter
    picked = st.sidebar.date_input(
        "Fechas (mismo año)", value=default, min_value=DATA_FIRST_DAY,
        max_value=DATA_LAST_DAY, key="rango",
    )
    # Mid-selection the widget returns a 1-tuple: keep the default dates meanwhile.
    if isinstance(picked, (tuple, list)) and len(picked) == 2:
        return picked[0], picked[1]
    st.sidebar.info("Elige también la fecha final.")
    return default


def sidebar() -> tuple[UserSettings, str]:
    """Draw the sidebar and return the validated settings and the shift mix id."""
    st.sidebar.header("Parámetros")
    start, end = _period_widgets()
    preset = st.sidebar.selectbox(
        "Batería", list(PRESETS), index=list(PRESETS).index(config.DEFAULT_PRESET),
        format_func=lambda p: PRESETS[p].label, key="bateria",
    )
    mix = st.sidebar.selectbox(
        "Tipo de operación", list(model.MIX_LABELS), format_func=model.mix_option_label,
        key="operacion",
        help="Horas de operación por semana en promedio. La batería solo puede descargar "
        "mientras la fábrica funciona. Los turnos son simulados.",
    )
    try:
        settings = UserSettings(preset, start, end)
    except ValueError as exc:
        st.sidebar.error(str(exc))
        st.stop()

    st.sidebar.subheader("Estado")
    if os.environ.get("GROQ_API_KEY"):
        st.sidebar.caption("Groq: clave configurada")
    else:
        st.sidebar.caption("Groq: sin clave (el asistente solo mostrará pasajes)")
    retriever_name = load_agent()[2]
    if retriever_name == "qdrant-nvidia":
        st.sidebar.caption("Búsqueda: Qdrant + NVIDIA")
    else:
        st.sidebar.caption("Búsqueda: BM25 (sin conexión)")
    return settings, mix


# --------------------------------------------------------------------------- indicators


def _kpi_metrics(kpis: model.Kpis) -> None:
    """The three Q-learning metrics for one `Kpis`."""
    rule = POLICY_NAMES[kpis.best_rule_id]
    c1, c2, c3 = st.columns(3)
    c1.metric("Ahorro con Q-learning", cop(kpis.q_savings_cop))
    c2.metric(
        "Frente a la mejor regla simple",
        pct(kpis.q_vs_best_rule_pct) if kpis.q_vs_best_rule_pct is not None else "—",
        help=f"La mejor regla simple en este caso: {rule} ({cop(kpis.best_rule_savings_cop)}).",
    )
    c3.metric("Ciclos de la batería", f"{kpis.q_cycles:.1f}")


def indicators(runs: dict[str, model.PolicyRun]) -> None:
    """Headline metrics averaged over the 5 shift calendars (D15, A5)."""
    st.markdown("##### Promedio de 5 calendarios de turnos simulados")
    _kpi_metrics(model.compute_kpis(runs))


# --------------------------------------------------------------------------- tabs


def tab_operacion(runs: dict[str, model.PolicyRun], settings: UserSettings) -> None:
    """Price, charge/discharge actions and state of charge for one week."""
    try:
        pid = st.selectbox(
            "Política mostrada", list(model.APP_POLICIES), format_func=POLICY_NAMES.get,
            key="politica",
        )
        run = runs[pid]
        if run.policy_used != pid:
            st.warning(
                "Falta la tabla entrenada de esta política; se muestra el horario por turnos."
            )
        total = len(run.calendars)
        # segmented_control returns None when the user deselects the active option.
        picked = st.segmented_control(
            "Calendario de turnos", list(range(1, total + 1)), selection_mode="single",
            default=1, key="calendario",
        )
        k = picked or 1
        calendar = run.calendars[k - 1]
        # Same simulation as the plot below, so these are real numbers, not averages.
        st.markdown(f"##### Calendario de turnos {k} de {total}")
        _kpi_metrics(model.compute_kpis(runs, k - 1))
        if pid != "q_learning":
            st.metric(f"Ahorro con {POLICY_NAMES[pid]}", cop(calendar.savings_cop))
        st.caption("Ahorro del periodo completo; el gráfico muestra una semana.")
        # All calendars cover the same dates, so the week count does not depend on k.
        n = model.n_weeks(calendar.hourly)
        # Reset before the widget exists: a stale value above the new max would raise.
        if st.session_state.get("semana", 1) > n:
            st.session_state["semana"] = 1
        week = st.slider("Semana", 1, n, key="semana") if n > 1 else 1
        st.plotly_chart(
            charts.operation_figure(
                model.week_slice(calendar.hourly, week - 1), settings.preset, POLICY_NAMES[pid]
            ),
            width="stretch",
        )
        st.caption(
            "Línea gris: precio de la energía. Triángulos verdes: la batería carga (compra); "
            "rojos: descarga (ahorra). Zonas moradas: fábrica funcionando; fuera de ellas la "
            "batería no puede descargar. Abajo: nivel de carga."
        )
    except Exception as exc:  # noqa: BLE001 - no exception may reach the top level
        st.error(error_text(exc))


def tab_comparacion(runs: dict[str, model.PolicyRun], settings: UserSettings, mix: str) -> None:
    """Cumulative savings per strategy and savings per battery size."""
    try:
        st.plotly_chart(charts.cumulative_figure(runs), width="stretch")
        st.caption(
            "Las cuatro estrategias durante todo el periodo. Haz clic en la leyenda para "
            "ocultar líneas."
        )
        sizes = cached_sizes(settings.start, settings.end, mix)
        st.plotly_chart(charts.size_figure(sizes), width="stretch")
        st.caption(
            "¿Qué tamaño conviene? Compara el ahorro con la cotización de tu proveedor; "
            "aquí no hay costos de batería."
        )
    except Exception as exc:  # noqa: BLE001
        st.error(error_text(exc))


def _details(result: AgentResult) -> dict[str, Any]:
    """Serializable summary of one agent run for the "Detalles" expander."""
    return {
        "tool_calls": [
            {"name": c.name, "arguments": c.arguments, "ok": c.ok, "error": c.error}
            for c in result.tool_calls
        ],
        "citations": result.citations,
        "unverified": result.unverified_citations,
        "retriever": result.retriever,
        "stopped": result.stopped,
        "error": result.error,
        "models": result.models,
    }


def _used_fallback_model(models: list[str]) -> bool:
    """True when a call was answered by a model other than the preferred one (D16)."""
    return any(m != "?" and m != config.GROQ_MODELS[0] for m in models)


def _show_details(details: dict[str, Any]) -> None:
    models = ", ".join(dict.fromkeys(details.get("models", []))) or "-"
    with st.expander("Detalles"):
        st.markdown("**Herramientas usadas**")
        if not details["tool_calls"]:
            st.write("Ninguna.")
        for call in details["tool_calls"]:
            status = "ok" if call["ok"] else f"error: {call['error']}"
            st.code(f"{call['name']}({call['arguments']}) -> {status}", language=None)
        st.markdown("**Fuentes citadas:** " + (", ".join(details["citations"]) or "ninguna"))
        if details["unverified"]:
            st.markdown("**Citas sin verificar:** " + ", ".join(details["unverified"]))
        st.markdown(f"**Búsqueda:** {details['retriever'] or '-'}")
        st.markdown(f"**Modelo:** {models}")
        st.markdown(f"**Motivo de parada:** {details['stopped']}")
        if details.get("error"):
            st.markdown(f"**Error técnico:** {details['error']}")


def _show_answer_extras(details: dict[str, Any] | None) -> None:
    """Fallback-model notice and the "Detalles" expander under an answer."""
    if not details:
        return
    if _used_fallback_model(details.get("models", [])):
        st.caption(FALLBACK_MODEL_NOTICE)
    _show_details(details)


def tab_asistente(settings: UserSettings) -> None:
    """Chat with the agent, using the sidebar battery and period."""
    try:
        st.caption(
            "El asistente usa la batería y el periodo de la barra lateral y simula la "
            "operación Mixta."
        )
        messages: list[dict[str, Any]] = st.session_state.setdefault(MESSAGES_KEY, [])
        if st.button("Nueva conversación", key="nueva"):
            st.session_state[MESSAGES_KEY] = []
            st.rerun()
        # The history lives in a container created before the input, so a new answer is
        # drawn above it instead of below the input (D16 chat fixes, 2026-11-14).
        history_box = st.container()
        # Both the typed and the suggested question go through the same flow. The input is
        # read before the history is drawn, so the example buttons never survive a question.
        question = st.chat_input("Pregunta sobre tu batería o la regulación…") or st.session_state.pop(
            SUGGESTED_KEY, None
        )
        history = [{"role": m["role"], "content": m["content"]} for m in messages]
        if question:
            messages.append({"role": "user", "content": question})
        with history_box:
            if not messages:
                st.markdown("**Preguntas de ejemplo**")
                for i, text in enumerate(EXAMPLE_QUESTIONS):
                    st.button(
                        text, key=f"ejemplo_{i}", on_click=st.session_state.__setitem__,
                        args=(SUGGESTED_KEY, text),
                    )
                st.caption("Cada pregunta usa el modelo de lenguaje (cuota diaria de Groq).")
            for msg in messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    _show_answer_extras(msg.get("details"))
            if not question:
                return
            registry, chat, _, load_error = load_agent()
            if load_error:
                st.error(f"No se pudo abrir el asistente: {load_error}")
                return
            with st.chat_message("assistant"):
                with st.spinner("Pensando…"):
                    result = run_agent(question, history, registry, chat, settings=settings)
                st.markdown(result.answer)
                details = _details(result)
                _show_answer_extras(details)
        messages.append({"role": "assistant", "content": result.answer, "details": details})
    except Exception as exc:  # noqa: BLE001
        st.error(error_text(exc))


def tab_acerca() -> None:
    """Data, method and limitations in plain language."""
    st.markdown(
        """
### Datos
- **Precio de bolsa horario** (`PrecBolsNaci`) publicado por XM, 2024-2025. Se guarda una
  copia en el repositorio, así que la aplicación funciona sin conexión.
- **Normativa**: resoluciones de la CREG y documentos de XM sobre almacenamiento con baterías.
  El asistente responde con estos textos y cita la fuente.

### Método
Un programa aprende, por ensayo y error sobre los precios de enero a septiembre, a decidir
cada hora si **cargar**, **descargar** o **esperar** (Q-learning tabular, D14/D15). Se
evalúa en octubre-diciembre, que no vio al entrenar, y se compara con reglas simples
(umbral de precio, horario por turnos, carga nocturna y descarga diurna). El ahorro se mide
frente a no tener batería.

### Limitaciones
- Se usa el precio de bolsa. No se tiene en cuenta una tarifa.
- Los turnos de la fábrica son **simulados**.
- El modelo basado en Q-learning se entrenó con operación mixta.
- No se modela el desgaste de la batería.
- El ahorro es frente a no tener batería y **no incluye costos** de inversión o de la operación de la batería.
"""
    )


# --------------------------------------------------------------------------- main


def load_secrets() -> None:
    """Copy the API keys from ``st.secrets`` into the environment when they are missing.

    Streamlit Cloud provides the keys as secrets (D17). The environment, filled first by
    the shell and then by ``.env``, always wins, so a local run behaves as before.
    """
    for key in SECRET_KEYS:
        if os.environ.get(key):
            continue
        try:
            value = st.secrets.get(key)
        except Exception:  # noqa: BLE001 - no secrets.toml raises, and its type varies by version
            return
        if value:
            os.environ[key] = str(value)


def main() -> None:
    """Page entry point."""
    st.set_page_config(page_title="Batería inteligente", layout="wide")
    config.load_env()  # via the module so tests can replace it (never loads real keys)
    load_secrets()  # Streamlit Cloud secrets (D17)
    st.title("Carga inteligente de batería para tu planta")
    settings, mix = sidebar()
    st.caption(model.status_line(settings.preset, settings.start, settings.end, mix))
    if model.in_sample(settings.start, settings.end):
        st.info(
            "El periodo incluye meses usados para entrenar el modelo (ene–sep): "
            "los resultados de Q-learning son optimistas."
        )
    st.warning(
        "Observa cuánto ahorrarías si usaras una batería inteligente para los turnos de tu fábrica. "
        "Cinco calendarios de turnos son simulados. El ahorro se mide frente a no tener batería."
    )
    try:
        runs = cached_runs(settings.preset, settings.start, settings.end, mix)
        indicators(runs)
    except Exception as exc:  # noqa: BLE001
        st.error(error_text(exc))
        runs = None

    t_op, t_cmp, t_chat, t_about = st.tabs(["Operación", "Comparación", "Asistente", "Acerca de"])
    with t_op:
        if runs is not None:
            tab_operacion(runs, settings)
    with t_cmp:
        if runs is not None:
            tab_comparacion(runs, settings, mix)
    with t_chat:
        tab_asistente(settings)
    with t_about:
        tab_acerca()


if __name__ == "__main__":
    main()
