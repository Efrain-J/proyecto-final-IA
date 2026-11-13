# Planned steps

A rough road map for the project, not a checklist. Each step gives the overall idea. As we work, we add the obstacles we hit and a short note on what was actually done. Scope decisions go in `decisions.md`; this file records the path.

The order follows the MVP principle: get a thin version working end to end first, then improve each part.

---

## 1. Data

Get one year of hourly bolsa prices (and demand) from XM through `pydataxm`, and save it as a local CSV snapshot that the rest of the project reads. Add a script that downloads it with a single command.

**Obstacles:**
- `pydataxm` breaks with pandas 3.1 (`freq='M'`). We call XM's API directly instead (D10).
- The API rejects ranges longer than about a month, so we download one month per request.
- 2024 prices average almost three times 2025's, so the MDP needs relative price levels (D5, O7).

**Done (2026-11-12):** `energy/data/xm_prices.py` (`uv run python -m energy.data.xm_prices`) downloads 2024–2025 (17,544 hours, no gaps) to `data/raw/xm/` and builds `data/processed/prices_hourly.csv`. It runs concurrently, retries, and resumes. Tested in `tests/test_xm_prices.py`. Scope changed to two years, prices only (D5).

## 2. Battery environment and MDP (Part 1)

Define the battery problem as an MDP (state, actions, transitions, reward), solve it with value iteration or Q-learning, and compare it against a naive policy and a reference policy, measuring the savings.

**To do first:** O9, the price loader (`energy/data/prices.py`), once the relative price state (O7) is defined. Then follow the Part 1 design in `architecture.md`: one shared `step()` for the battery physics, the `Policy` protocol, and the baselines including perfect foresight.

**Obstacles:**

**Done:**

## 3. Simulation tool

Wrap the trained policy in `simulate(policy, dates)` so it returns the decisions and savings for a date range. The dashboard and the agent both use it.

**Obstacles:**

**Done:**

## 4. RAG corpus and index

Collect the CREG resolutions and XM documents about batteries and the bolsa price, convert them to text, chunk them, and build a Qdrant index on disk with a separate script.

**Obstacles:**
- CREG 098 of 2019 was replaced by CREG 101 113 of 2026. Both are in the corpus (D6).
- The HTML pages were mostly site menus, so we added a cleaning step with per-source selectors (D11).
- Amending resolutions quote other resolutions' articles, which produced duplicate article headings. Only ascending article numbers are marked now (D11).
- CREG 101 113 doesn't explicitly cover a consumer's battery behind the meter. Recorded as a limitation (D6).

**Done (2026-11-12):** the download half. The manifest is `data/corpus/sources.csv` (initially 6 documents), and `energy/data/corpus.py` (`uv run python -m energy.data.corpus`) downloads them to `data/corpus/raw/`; every document converts to more than 28k characters with markitdown. Tested in `tests/test_corpus.py`. Then CREG 131 of 1998 was added (7 documents), and the cleaning step `energy/rag/clean.py` (`uv run python -m energy.rag.clean`) writes `data/corpus/clean/*.md`, with article headings marked and the in-force metadata in the manifest. Tested in `tests/test_clean.py`. Still pending: chunking with article metadata and the Qdrant index.

## 5. LLM agent

Build the agent loop with Groq tool calling, with an iteration cap, at least two tools (document search and `simulate`), and answers that cite their sources.

**To do first:** O8, `energy/config.py` with root-anchored paths and shared settings. Do it before the first module that runs outside the repo root or needs shared settings (agent, eval, app). Load `.env` only in entry points.

**Obstacles:**

**Done:**

## 6. Dashboard

Build a Streamlit app with price vs. decisions, cumulative savings, battery parameters the user can change, and a chat with the agent.

**Obstacles:**

**Done:**

## 7. Agent evaluation

Write 10–15 fixed test questions, including trap questions, and record whether the agent called the right tool and cited a source.

**Obstacles:**

**Done:**

## 8. Deliverables

Write the README with reproducible run instructions, `requirements.txt`, the AI-use declaration, and the report (up to 4 pages), including the limitations.

**Obstacles:**

**Done:**
