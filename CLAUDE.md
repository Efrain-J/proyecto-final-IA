# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Final project for EAFIT SI3003 (Inteligencia Artificial), 2026. The full assignment is in `EAFIT_Proyecto_Final_IA.md` (Spanish); it is the source of truth for requirements and grading. **Hard deadline: Monday 2026-11-16** (moved from Friday 2026-11-13; confirm the exact hour). Code freeze, README, agent evaluation and AI-use declaration are all due then.

Chosen scenario: **3.3 (battery dispatch with an MDP + regulation RAG assistant)**. Three docs to read before planning work:

- `docs/decisions.md`: scope decisions, assumptions and open questions. Add new decisions there.
- `docs/planned_steps.md`: the road map. When you finish a step or hit an obstacle, note it under that step.
- `docs/architecture.md`: package map, dependency rules, interfaces (implemented vs. planned), fallbacks. New modules must follow it; log any change to an interface in `decisions.md`.

Current state: the data layer (XM prices, corpus download) and the full RAG pipeline (cleaning, chunking, Qdrant index, retrieval with BM25 fallback) are implemented and tested. Part 1 is implemented (D15, merged):
- shift-aware Q-learning with a synthetic factory shift schedule (A5)
- 4 / 8 / 12 h batteries
- tuned baselines
- `simulate`
- parallel training
- tuned parameters (`energy.eval.tune_q`)
- the 2024 and 2025 scenarios, with a cross-year check

`energy/config.py` (O8) holds the root-anchored paths and shared settings. The LLM agent (D16) is implemented in `energy/agent/`. The agent evaluation ran live on 2026-11-14 (step 7, 8/11 passed, results in `results/agent_eval/`, see D16). The Streamlit dashboard is implemented (`app.py`, `energy/app/`, step 6). The Part 1 results tables are reproduced by `energy/eval/part1.py` (`results/part1/`). Still to do: the README and report deliverables. Update this file with the real commands as code is added.

## Code documentation conventions

- **Google-style docstrings.** Every module gets a docstring: what it does, how to run it, its inputs/outputs. Public functions and classes get a summary line plus `Args`/`Returns`/`Raises` when they aren't obvious. Types come from the type hints, so `Args` describes meaning and units only.
- Type hints on every public signature. Use dataclasses for anything that crosses a layer boundary.
- Comments explain *why*, not *what*. Point to decision IDs instead of repeating the reasoning (`# one month per request, see D10`).

## Environment

Managed with **uv**, Python 3.12 (`.python-version`). Commit `pyproject.toml` and `uv.lock`.

- Install: `uv sync` (fallback: `python -m venv .venv` + `pip install -r requirements.txt`)
- Run anything: `uv run <cmd>`, e.g. `uv run streamlit run app.py`
- Add a dependency: `uv add <pkg>`, **then always regenerate** `requirements.txt` (a required deliverable): `uv export --no-hashes --no-dev -o requirements.txt`. Never edit `requirements.txt` by hand.
- Tests: `uv run pytest` (offline; real-API tests are marked `network` and run with `uv run pytest -m network`)

## Code layout

- `energy/config.py`: `ROOT`-anchored paths (`PRICES_CSV`, `CHUNKS_PATH`, `QDRANT_DIR`, `POLICIES_DIR`, `RESULTS_DIR`, …), shared settings, and `load_env()`, which only entry points call. Use these instead of relative paths. Run `python -m energy.…` from the repo root, because the package isn't installed.
- `energy/data/io.py`: atomic file writes.
- `energy/data/xm_prices.py`: XM bolsa prices through the XM API called directly (`pydataxm` is incompatible with pandas 3). `uv run python -m energy.data.xm_prices` writes `data/raw/xm/*.json` and `data/processed/prices_hourly.csv`. It's resumable; `--force` re-downloads.
- `energy/data/corpus.py`: RAG documents listed in `data/corpus/sources.csv`. `uv run python -m energy.data.corpus` writes to `data/corpus/raw/`. The manifest columns `selector`, `in_force` and `superseded_by` drive the cleaning and the citations.
- `energy/rag/clean.py`: raw → `data/corpus/clean/<id>.md` (main-content selector, links stripped, `## ARTÍCULO n.` headings for CREG articles in ascending order only). Run with `uv run python -m energy.rag.clean`, after adding or changing a document. Annexes are marked `## ANEXO n.`.
- `energy/rag/chunk.py`: clean → `data/corpus/chunks.jsonl` (committed). Splits by article/annex, each chunk labeled `article` = `"12"` / `"Anexo 1"` / null. Run with `uv run python -m energy.rag.chunk`.
- `energy/rag/index.py`: chunks → Qdrant at `data/qdrant/` with NVIDIA embeddings. The index is committed (D17), except its `.lock` files, so recommit it after a rebuild. Needs `NVIDIA_API_KEY`, takes about 45 s. Run with `uv run python -m energy.rag.index`.
- `energy/rag/retrieve.py`: `open_retriever()` returns Qdrant+NVIDIA, or BM25 over `chunks.jsonl` as fallback; `Hit.citation` gives the citation. Manual check: `uv run python -m energy.rag.retrieve "pregunta" [--bm25]`.
- `energy/data/prices.py`: `load_prices()` and `scenario_split(frame, year)` (train January–September, test October–December, D5).
- `energy/rl/`: Part 1 (D14). `battery.py` (presets, `step()`, the only battery physics), `features.py` (relative price, hour pairs, `Discretizer`), `policies.py` (`NoBattery`, `NightDay`, `PerfectForesight`), `qlearning.py` (`train_q_learning`, `TabularPolicy`), `simulate.py` (`simulate()`, used by the eval, the app and the agent).
- `energy/rl/factory.py` (D15, A5): shift types, `ShiftMix` / `MIXES`, `add_shifts(frame, mix, seed)`, `TEST_SHIFT_SEEDS`. A frame without shift columns means the factory always runs.
- `energy/rl/train.py`: `uv run python -m energy.rl.train --scenarios 2025 [--workers N] [--shift-draws 5] [--mix default]` trains one Q-table per preset in parallel and tunes the price rule. It writes `artifacts/policies/q_<preset>_<year>.npz/.json` and `threshold_<preset>_<year>.json` (committed), and prints the mean test savings over 5 shift patterns, compared with the baselines. It takes about 31 s.
- `energy/eval/part1.py`: `uv run python -m energy.eval.part1 [--presets 8h] [--years 2024 2025] [--seeds 5] [--mixes default mostly_double mostly_single]` evaluates the committed Q-tables and thresholds without retraining, against the rules and perfect foresight, on October–December, over the test shift calendars. Q-learning is compared with the strongest rule, including the price rule with either year's cut-offs (D15). It writes `results/part1/` (`runs.csv`, `summary.csv`, `summary.md`, the tables the report cites) in about 6 s.
- `energy/eval/tune_q.py`: `uv run python -m energy.eval.tune_q [--preset 8h] [--stage all]` tunes the Q-learning parameters on validation months only (January–June training, July–September validation), then tests the default and the winner once on October–December. It writes `results/tuning/` (D15).
- `energy/agent/` (D16): `llm.py` (Groq `ChatFn`, fallback model, `LLMError`), `tools.py` (`search_regulation`, `simulate_battery` over the 5 test shift calendars, `run_tool` never raises), `prompts.py` (Spanish system prompt, `REFUSAL`), `render.py` (the results block: every simulation figure is rendered by code, never by the model), `loop.py` (`run_agent` with `T_MAX`, citation check and repair, `open_agent`). Try it with `uv run python -m energy.agent "pregunta" [--preset 8h] [--start YYYY-MM-DD --end YYYY-MM-DD] [--bm25]`. `settings.py` holds `UserSettings`, the user's battery and period, which `run_agent(settings=...)` uses as defaults. It needs `GROQ_API_KEY`; without it the answer is retrieval-only. Live tests: `uv run pytest -m network`.
- `energy/eval/agent_eval.py` (step 7): `uv run python -m energy.eval.agent_eval [--ids 1,9] [--pause 20] [--bm25] [--fresh]` runs the fixed questions in `energy/eval/agent_questions.yaml` (edit the questions and their expected tools, sources and patterns there) and writes `results/agent_eval/` (`agent_eval.csv`, `transcripts.jsonl`, `summary.md`) after every question. It merges by id with existing outputs, so `--ids` re-runs only those questions, and `--fresh` starts empty. Manual-review notes go in the CSV's `revision_manual` column, and a re-run of a question clears its note. Needs `GROQ_API_KEY`; about 11 questions × ~4k tokens, so run it once on a fresh Groq day.
- `app.py` + `energy/app/` (step 6): the Streamlit dashboard. Run it from the repo root with `uv run streamlit run app.py` (http://localhost:8501). `app.py` is the thin entry point (sidebar, tabs, caching). `energy/app/model.py` holds the pure policy runs, KPIs and labels, averaged over the 5 test shift calendars like the agent's `simulate_battery` (D16). `energy/app/charts.py` holds the Plotly figures. Works without keys: no `GROQ_API_KEY` means retrieved passages only, and no `NVIDIA_API_KEY` means BM25.
- Streamlit Cloud (D17): the hosted demo reads the keys from `st.secrets`. `app.load_secrets()` copies them into the environment only when they aren't set, so a local `.env` always wins.
- Full RAG rebuild order after changing the corpus: `energy.data.corpus` → `energy.rag.clean` → `energy.rag.chunk` → `energy.rag.index`.
- `tests/`: pytest, with HTTP mocked through `httpx.MockTransport`.

## Guiding principle: MVP first

The user's explicit priority, and the assignment's: **a small scope that works end to end beats many features that barely work.** A component that fails during the demo scores zero on functionality and interaction (15% of the grade). When making choices:

- Get a minimal end-to-end path running first (data → Part 1 model → agent with tools → dashboard), then improve.
- Prefer fewer, solid features over breadth. Don't add features beyond what the requirements below need unless asked.
- Every external dependency (API, LLM endpoint) needs a fallback that keeps the app running: local CSV snapshots for data, tool errors returned to the agent as observations rather than crashing.
- The app must run reproducibly from the README with no manual steps by the team.

## Required components (all must be present)

1. **Real Colombian data** (SIATA, IDEAM, AMVA, XM, INS/SIVIGILA, datos.gov.co, Kaggle…), source documented and cited, with a downloaded-historical plan B. `data/` is tracked in git (keep files small); still keep the download script and document it in the README.
2. **Part 1 classical technique implemented by the team** (A*, CSP, MDP/RL, Bayesian network, HMM, or classical ML) — not just a library call that solves it — with a **quantitative metric vs. a baseline**.
3. **LLM agent with RAG** over a relevant corpus and **at least two tools** (tool calling). Manual agentic loop or LangGraph. Must cap iterations (`T_max`) and cite sources. The Part 1 model must be integrated non-trivially (typically exposed as one of the agent's tools).
4. **Dashboard** with ≥2 meaningful visualizations, user-selectable parameters (station, municipality, date, scenario), and a chat with the agent.
5. **Agent evaluation:** a fixed set of 10–15 test questions, recording whether the agent called the correct tool and cited the source.
6. GitHub repo with README, `requirements.txt`, and an AI-use declaration. Also a ≤4-page report.

Systems must not be presented as medical, legal, or safety advice; document limitations.

## Downloading data

When a task requires downloading data (APIs, open-data portals, station/historical series, file dumps), consider using **`asyncio`** (e.g. `aiohttp` / `httpx.AsyncClient`) or **parallel jobs** (e.g. `concurrent.futures.ThreadPoolExecutor`, or `joblib`/multiprocessing for CPU-bound parsing) when there are many independent requests, such as one call per station, municipality, month, or year. Still:

- Bound concurrency (a semaphore or a small worker count) and respect the source's rate limits; add retries with backoff and timeouts.
- Keep download scripts idempotent and resumable: skip files that already exist in `data/` and write each file atomically.
- Fall back to the sequential path or the local CSV snapshot if the parallel version fails, so the app keeps running (MVP first).
- Keep the scripts runnable from the README with a single command.

## Git tree

Don't commit without the users authorization.

## Stack

Chosen to reuse two proven-working reference projects next to this repo (read them for working code, don't copy secrets):
- `../si3003-workshop-rag-Efrain-Jurado/` — working RAG (notebooks): NVIDIA NIM + Qdrant + markitdown + citation prompt + eval CSV.
- `../llm_lab_ia/main_app.py` — working Streamlit app with the Groq SDK.

| Layer | Choice | Notes |
|---|---|---|
| Dashboard | **Streamlit** (`streamlit run app.py`) | Patterns from `llm_lab_ia`: sidebar for parameters, `st.tabs`, `@st.cache_resource` for clients/vector store, `st.write_stream` for streaming, catch SDK errors and show `st.error` instead of crashing. |
| LLM + tool calling | **Groq SDK** (`groq`), OpenAI-style `chat.completions` with `tools` | Manual agent loop with `T_max` cap; tool errors go back to the model as observations. Default to an `openai/gpt-oss-*` model. Key in `GROQ_API_KEY` (env / `.env`). |
| Vector store | **Qdrant** (`qdrant-client`, `langchain-qdrant`) | **Persist to disk** with `QdrantClient(path="data/qdrant")` — never `:memory:` in the app. Build the index once with a separate script; the app only opens it. Probe the embedding dim at runtime, cosine distance. |
| Embeddings | NVIDIA NIM `nvidia/nemotron-3-embed-1b` via `langchain-nvidia-ai-endpoints` (`truncate="END"`) | Proven in the workshop. Key in `NVIDIA_API_KEY`, needed at build time **and** query time. Planned fallback: BM25 over the committed `chunks.jsonl` (see `docs/architecture.md`), not sentence-transformers. |
| Document loading | `markitdown[pdf]` + BeautifulSoup main-content selectors (`energy/rag/clean.py`) | |
| Chunking | CREG: split by `## ARTÍCULO n.` heading, keeping the heading in the chunk text. Others: `RecursiveCharacterTextSplitter`, ~900 chars / 120 overlap, drop chunks < 30 chars | Metadata per chunk: `title`, `source` (doc id), `chunk_index`, `article`, `in_force`, `superseded_by`. Used for citations. |
| Retrieval | `similarity_score_threshold`, k≈5, threshold ≈0.25 | Thresholds tuned for the nemotron embedder in the workshop. |
| Citation prompt | Answer only from context, cite `[title \| source]`, fixed refusal phrase when context is insufficient | From the workshop; makes trap questions in the eval pass. |

API keys live in `.env` (gitignored); never commit them. `.env.example` lists the keys with no values.
