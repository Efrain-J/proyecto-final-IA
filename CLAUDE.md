# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

Final project for EAFIT SI3003 (Inteligencia Artificial), 2026. The full assignment is in `EAFIT_Proyecto_Final_IA.md` (Spanish); it is the source of truth for requirements and grading. **Hard deadline: Friday 2026-11-13, 6:00 p.m.** — code freeze, README, agent evaluation and AI-use declaration all due then.

The repo currently has no code. As the code is written, update this file with the real build/run/test commands and the architecture.

## Guiding principle: MVP first

The user's explicit priority, and the assignment's: **a small scope that works end to end beats many features that barely work.** A component that fails during the demo scores zero on functionality and interaction (15% of the grade). When making choices:

- Get a minimal end-to-end path running first (data → Part 1 model → agent with tools → dashboard), then improve.
- Prefer fewer, solid features over breadth. Don't add features beyond what the requirements below need unless asked.
- Every external dependency (API, LLM endpoint) needs a fallback that keeps the app running: local CSV snapshots for data, tool errors returned to the agent as observations rather than crashing.
- The app must run reproducibly from the README with no manual steps by the team.

## Required components (all must be present)

1. **Real Colombian data** (SIATA, IDEAM, AMVA, XM, INS/SIVIGILA, datos.gov.co, Kaggle…), source documented and cited, with a downloaded-historical plan B. `data/` is gitignored, so the README must include instructions/scripts to obtain the data.
2. **Part 1 classical technique implemented by the team** (A*, CSP, MDP/RL, Bayesian network, HMM, or classical ML) — not just a library call that solves it — with a **quantitative metric vs. a baseline**.
3. **LLM agent with RAG** over a relevant corpus and **at least two tools** (tool calling). Manual agentic loop or LangGraph. Must cap iterations (`T_max`) and cite sources. The Part 1 model must be integrated non-trivially (typically exposed as one of the agent's tools).
4. **Dashboard** with ≥2 meaningful visualizations, user-selectable parameters (station, municipality, date, scenario), and a chat with the agent.
5. **Agent evaluation:** a fixed set of 10–15 test questions, recording whether the agent called the correct tool and cited the source.
6. GitHub repo with README, `requirements.txt`, and an AI-use declaration. Also a ≤4-page report.

Systems must not be presented as medical, legal, or safety advice; document limitations.

## Stack

Chosen to reuse two proven-working reference projects next to this repo (read them for working code, don't copy secrets):
- `../si3003-workshop-rag-Efrain-Jurado/` — working RAG (notebooks): NVIDIA NIM + Qdrant + markitdown + citation prompt + eval CSV.
- `../llm_lab_ia/main_app.py` — working Streamlit app with the Groq SDK.

| Layer | Choice | Notes |
|---|---|---|
| Dashboard | **Streamlit** (`streamlit run app.py`) | Patterns from `llm_lab_ia`: sidebar for parameters, `st.tabs`, `@st.cache_resource` for clients/vector store, `st.write_stream` for streaming, catch SDK errors and show `st.error` instead of crashing. |
| LLM + tool calling | **Groq SDK** (`groq`), OpenAI-style `chat.completions` with `tools` | Manual agent loop with `T_max` cap; tool errors go back to the model as observations. Default to an `openai/gpt-oss-*` model. Key in `GROQ_API_KEY` (env / `.env`). |
| Vector store | **Qdrant** (`qdrant-client`, `langchain-qdrant`) | **Persist to disk** with `QdrantClient(path="data/qdrant")` — never `:memory:` in the app. Build the index once with a separate script; the app only opens it. Probe the embedding dim at runtime, cosine distance. |
| Embeddings | NVIDIA NIM `nvidia/nemotron-3-embed-1b` via `langchain-nvidia-ai-endpoints` (`truncate="END"`) | Proven in the workshop. Key in `NVIDIA_API_KEY`. Fallback if it becomes a problem: local `sentence-transformers` (requires rebuilding the index). |
| Document loading | `markitdown[pdf]` | |
| Chunking | `RecursiveCharacterTextSplitter`, ~900 chars / 120 overlap, drop chunks < 30 chars | Metadata per chunk: `title`, `source` (filename), `chunk_index` — used for citations. |
| Retrieval | `similarity_score_threshold`, k≈5, threshold ≈0.25 | Thresholds tuned for the nemotron embedder in the workshop. |
| Citation prompt | Answer only from context, cite `[title \| source]`, fixed refusal phrase when context is insufficient | From the workshop; makes trap questions in the eval pass. |

API keys live in `.env` (must be gitignored) — never commit them.
