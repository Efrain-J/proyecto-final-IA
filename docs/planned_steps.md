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

**Design decided (2026-11-13), see D14:**
- State: hour pair, state of charge, relative price level.
- Reward: relative.
- Episodes: one week each.
- One Q-table per preset (2 h, 3 h and 4 h, D9).
- Split: within each year, train January–September and test October–December; scenario 2025 first, then 2024 (D5).

**To do first:** O9, the price loader (`energy/data/prices.py`), with the trailing 168-hour mean and the within-year split. Then follow the Part 1 design in `architecture.md`: one shared `step()` for the battery physics, the `Policy` protocol, and the baselines including perfect foresight.

**Obstacles:**
- **Miscalibrated test.** The test agent's threshold (Q-learning ≥ 80% of perfect foresight on synthetic data) couldn't be met: even the fixed schedule reaches only 75–84% there. It was replaced with Q-learning ≥ 90% of the fixed schedule.
- **Early stopping never triggered.** With a constant learning rate, near-tied actions kept flipping. Added a learning-rate decay (D14).
- **Q-learning stays slightly below the fixed schedule on real 2025 data** (95–97%). Seven variants didn't change that, so we report it honestly (D14).

**Done (2026-11-13):** written test-first. A Sonnet agent wrote the tests before the code, and a second one wrote the code to pass them.
- `energy/data/prices.py`: load the CSV and split it within a year.
- `energy/rl/battery.py`: presets, `step()` and the lookup tables.
- `energy/rl/features.py`: relative price, hour pairs and price bins.
- `energy/rl/policies.py`: no battery, the fixed schedule and perfect foresight (a dynamic program over the actual prices).
- `energy/rl/qlearning.py`: training, plus `TabularPolicy` with save and load.
- `energy/rl/train.py`: parallel training.

Running `uv run python -m energy.rl.train --scenarios 2025` writes `artifacts/policies/q_{2h,3h,4h}_2025.*`. The results are in D14. 276 tests pass. Still to do: the 2024 scenario (`--scenarios 2024`) and `eval/part1.py` (metrics and plots in `results/`).

**Branch `explore/factory-schedule` (2026-11-13), D15 + A5:**
- **Why:** Q-learning didn't beat a hand rule when the factory always runs.
- **Exploration:** we tried a factory with random shifts (discharge only while it runs) and longer batteries, through three rounds of prototypes in the scratchpad.
- **Implemented test-first:**
  - `energy/rl/factory.py`, the shift generator.
  - Shift-aware masks in `battery`.
  - `ShiftSchedule`, plus a `PriceThreshold` tuned on the training months.
  - A Q-table with the shift dimension, trained on 5 shift patterns.
  - `simulate` adding shifts, and evaluation averaged over 5 test patterns.
- **Result:**
  - For 8 h and 12 h batteries, Q-learning beats the best tuned rule by about 5%.
  - For 4 h the shift schedule still wins, by about 1%.

**Obstacles:**
- **Rigging the comparison.** An untuned price rule made Q-learning look 16% better. Tuning its cut-offs on the training months brought the edge down to about 5%.
- **24 h is too large for a table** (25 charge levels).
- **"Tomorrow's shift" in the state** made the data too sparse.

## 3. Simulation tool

Wrap the trained policy in `simulate(policy, dates)` so it returns the decisions and savings for a date range. The dashboard and the agent both use it.

**Obstacles:**

**Done (2026-11-13), built together with step 2:** `energy/rl/simulate.py`. `simulate(policy, start, end, preset)` takes a policy object or one of the ids `no_battery`, `night_day`, `perfect_foresight`, `q_learning`. It returns a summary and the hourly trace. Impossible actions become idle and are counted, and a missing Q-table falls back to the fixed schedule with a warning.

## 4. RAG corpus and index

Collect the CREG resolutions and XM documents about batteries and the bolsa price, convert them to text, chunk them, and build a Qdrant index on disk with a separate script.

**Obstacles:**
- CREG 098 of 2019 was replaced by CREG 101 113 of 2026. Both are in the corpus (D6).
- The HTML pages were mostly site menus, so we added a cleaning step with per-source selectors (D11).
- Amending resolutions quote other resolutions' articles, which produced duplicate article headings. Only ascending article numbers are marked now (D11).
- CREG 101 113 doesn't explicitly cover a consumer's battery behind the meter. Recorded as a limitation (D6).
- Annex text was cited as the last article (CREG 131 Anexo 1 → "Art. 8"). Annexes are now their own sections (D13).
- Semantic search sometimes ranks the replaced CREG 098 above the current 101 113. The agent prompt must handle `in_force` (D13).

**Done (2026-11-12):** the download half. The manifest is `data/corpus/sources.csv` (initially 6 documents), and `energy/data/corpus.py` (`uv run python -m energy.data.corpus`) downloads them to `data/corpus/raw/`; every document converts to more than 28k characters with markitdown. Tested in `tests/test_corpus.py`. Then CREG 131 of 1998 was added (7 documents), and the cleaning step `energy/rag/clean.py` (`uv run python -m energy.rag.clean`) writes `data/corpus/clean/*.md`, with article headings marked and the in-force metadata in the manifest. Tested in `tests/test_clean.py`. Then chunking, indexing and retrieval (D13): `energy/rag/chunk.py` writes `data/corpus/chunks.jsonl` (984 chunks with article/annex labels), `energy/rag/index.py` builds the Qdrant index with NVIDIA embeddings, and `energy/rag/retrieve.py` provides `open_retriever()` (Qdrant, or BM25 as fallback) and `Hit.citation`. Tested in `tests/test_chunk.py`, `test_index.py`, `test_retrieve.py` (178 tests in total). Step 4 is complete.

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
