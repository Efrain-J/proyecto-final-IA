# Batería inteligente para un consumidor industrial en bolsa

Proyecto final de Inteligencia Artificial (SI3003, 2026), escenario 3.3.

Por Efraín Jurado Torres

El usuario al que hiría dirijido este proyecto, es un usuario no regulado que paga la energía al precio de bolsa de Colombia. Supongamos que este usuario tiene una fabrica que requiere de electricidad. El usuario quiere saber cuánto ahorraría con una batería que compre energía en las horas baratas y la use en las caras mientras la fábrica funciona. El proyecto tiene tres partes:

- **Parte 1, la batería (MDP + Q-learning).** Una política Q-learning tabular decide cada hora si cargar, descargar o esperar, según el precio y los turnos de la fábrica. Se compara con unas politicas de reglas fijas y con la previsión perfecta, usando precios reales de XM (2024–2025). Los resultados están en `results/part1/summary.md`.
- **Asistente RAG.** Un agente LLM (Groq) responde preguntas sobre la regulación (CREG, XM) citando sus fuentes, y simula la batería con el modelo de la Parte 1. Tiene dos herramientas: `search_regulation` y `simulate_battery`. Su evaluación realizada con 11 preguntas está en `results/agent_eval/`.
- **Dashboard en Streamlit.** Tiene cuatro pestañas:
  - **Operación:** la operación hora a hora.
  - **Comparación:** la política frente a las reglas fijas y los tamaños de batería.
  - **Asistente:** el chat con el agente.
  - **Acerca de:** información del proyecto.

> Los ahorros presentados no incluyen el costo ni el desgaste de la batería. Las limitaciones están en `docs/decisions.md` («Known limitations»).

## Requisitos

- Python 3.12.
- [uv](https://docs.astral.sh/uv/), recomendado, o `pip`.
- Claves de API, gratuitas: Groq en https://console.groq.com y NVIDIA en https://build.nvidia.com. La aplicación funciona parcialmente sin ellas:

| Clave | Para qué | Sin la clave |
|---|---|---|
| `GROQ_API_KEY` | El modelo de lenguaje del asistente | El chat muestra solo los pasajes de la normativa encontrados; las pestañas de simulación funcionan completas |
| `NVIDIA_API_KEY` | Búsqueda semántica (embeddings + Qdrant) | La búsqueda usa BM25 sobre `data/corpus/chunks.jsonl` |

## Cómo ejecutarlo

Hay dos opciones.

### Opción 1: demo en línea (sin instalar nada)

Abre **<URL de la demo>**. La aplicación está publicada en Streamlit Community Cloud, con las claves guardadas como secretos del servidor, así que el asistente y la búsqueda semántica funcionan completos. Tarda cerca de un minuto en iniciar.

### Opción 2: instalación local

1. Clonar el repositorio:

   ```bash
   git clone https://github.com/Efrain-J/proyecto-final-IA.git
   cd proyecto-final-IA
   ```

2. Instalar las dependencias:

   ```bash
   uv sync
   ```

   Sin uv, crea un entorno virtual e instala `requirements.txt`. En los comandos siguientes, cambia `uv run` por `python -m` cuando el comando sea `streamlit` o `pytest`, y por `python` cuando sea `python`:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows; en Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Opcional: configurar las claves. El índice de búsqueda semántica (`data/qdrant/`) ya está en el repositorio, así que con `NVIDIA_API_KEY` basta. Solo hay que reconstruirlo (`uv run python -m energy.rag.index`) si cambia el corpus.

   ```bash
   cp .env.example .env          # luego escribe tus claves en .env
   ```

4. Abrir el dashboard. Queda en http://localhost:8501.

   ```bash
   uv run streamlit run app.py
   ```

5. Correr las pruebas. No necesitan internet ni claves.

   ```bash
   uv run pytest
   ```

### Publicar una copia propia en Streamlit Cloud

1. Entra a https://share.streamlit.io con GitHub y crea una app desde este repositorio: rama `main`, archivo `app.py`.
2. En **Advanced settings**, elige Python 3.12 y escribe los secretos:

   ```toml
   GROQ_API_KEY = "..."
   NVIDIA_API_KEY = "..."
   ```

   Las claves nunca quedan en el repositorio. Localmente, `.env` sigue funcionando igual y tiene prioridad sobre los secretos.

### Reproducir los resultados (opcional)

Los datos, las políticas entrenadas y los resultados ya están en el repositorio. Estos comandos los regeneran:

```bash
# Tablas de la Parte 1 (sin reentrenar, ~6 s) -> results/part1/
uv run python -m energy.eval.part1 --mixes default mostly_double mostly_single

# Reentrenar las políticas (~1 min; sobrescribe artifacts/policies/)
uv run python -m energy.rl.train --scenarios 2024 2025

# Evaluación del agente: 11 preguntas fijas (necesita GROQ_API_KEY, ~45k tokens) -> results/agent_eval/
uv run python -m energy.eval.agent_eval

# Probar el agente desde la terminal
uv run python -m energy.agent "¿Cuánto ahorraría con una batería de 8 h?"
```

Para volver a descargar los datos:

```bash
uv run python -m energy.data.xm_prices     # precios de bolsa de XM -> data/processed/prices_hourly.csv
uv run python -m energy.data.corpus        # documentos de data/corpus/sources.csv -> data/corpus/raw/
uv run python -m energy.rag.clean          # -> data/corpus/clean/
uv run python -m energy.rag.chunk          # -> data/corpus/chunks.jsonl
uv run python -m energy.rag.index          # -> data/qdrant/ (necesita NVIDIA_API_KEY; luego haz commit del índice)
```

## Organización del repositorio

```
app.py                  Punto de entrada del dashboard (Streamlit)
energy/
  config.py             Rutas y parámetros compartidos
  data/                 Descarga de precios de XM y del corpus; carga de precios
  rag/                  Limpieza, fragmentación, índice Qdrant y búsqueda (con BM25 de respaldo)
  rl/                   Parte 1: batería, turnos de la fábrica, políticas, Q-learning, simulate()
  agent/                Agente: cliente Groq, herramientas, prompt, ciclo con T_max y verificación de citas
  app/                  Lógica y gráficas del dashboard (sin Streamlit)
  eval/                 Resultados de la Parte 1, ajuste de Q-learning y evaluación del agente
data/
  processed/            Precios horarios de bolsa 2024–2025
  raw/xm/               Respuestas crudas de la API de XM
  corpus/               Documentos de regulación: sources.csv, raw/, clean/, chunks.jsonl
artifacts/policies/     Tablas Q y umbrales entrenados (por batería y año)
results/                part1/, agent_eval/, tuning/
tests/                  Pruebas con pytest (sin red)
docs/                   decisions.md (decisiones y limitaciones), architecture.md, planned_steps.md
```

## Fuentes de datos

- **Precios de bolsa:** precio horario de la energía en bolsa de XM, 2024-01-01 a 2025-12-31, obtenido con la API pública de XM (`energy/data/xm_prices.py`).
- **Normativa y documentos del mercado:**
  - Resoluciones CREG 101 113 de 2026, 098 de 2019, 174 de 2021 y 131 de 1998.
  - Glosario y preguntas frecuentes de XM.
  - XM Paper N°52.

  Las URL están en `data/corpus/sources.csv`.
