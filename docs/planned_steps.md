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

**Parameter tuning (2026-11-13):**
- `energy/eval/tune_q.py` searched the parameters on validation months only. It found γ 0.995, α ending at 0.005, and 10 shift patterns, which were tested once on October–December and adopted.
- **Result:** Q-learning ÷ best rule is now 1.00 for 4 h, 1.07 for 8 h and 1.11 for 12 h (D15).
- The γ that worked best, 0.995, was the highest value tried; a higher one might do better still.

**Scenario 2024 and cross-year transfer (2026-11-13):**
- `--scenarios 2024` trained the 2024 tables, and each year's tables were tested on October–December of both years.
- **Transfer works,** thanks to the relative price state: for 8 h and 12 h, Q-learning ÷ the strongest rule is 1.00–1.15 in every combination. For 4 h the shift schedule always wins.
- **Obstacle:** the price rule tuned on January–September 2024 was weak on October–December 2024, which made Q-learning look 33% better than it is. We now compare against the strongest available rule (D15).
- **Still to do:** move the cross-year script from the scratchpad into `energy/eval/`, with a CSV in `results/`.

**Shift-mix check (2026-11-14), D15:** the tables, trained on the mixed operation only, were tested under all three shift mixes. With mostly double shifts Q-learning holds up or does better. With mostly single shifts its edge over the best price rule disappears (0.95–1.12). Recorded as a limitation; no retraining per mix.

**`energy/eval/part1.py` (started 2026-11-14):** the tests are written (`tests/test_part1.py`), the module isn't. It will evaluate the committed tables without retraining, including the cross-year check, and write `results/part1/` (`runs.csv`, `summary.csv`, `summary.md`) for the report. Until the module exists, `uv run pytest` fails to collect that file.

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

**Done first (2026-11-13):** O8, `energy/config.py`. Paths are anchored at the repo root, the shared settings live in one place, and `load_env()` is called only in entry points. The agent's settings (`T_MAX`, the Groq models) go there when the agent is built.

**Obstacles:**
- **The model's citations were unreliable.** It sometimes wrote a literal `【cita】`, gpt-oss's native brackets, or dropped the brackets altogether. Fixed with normalization, a reminder inside the search results, and one repair round (D16).
- **It applied CREG rules for CND-dispatched SAEB to the user's battery** and called it "permitida y recomendada". The prompt now states the D6 limitation explicitly, and "¿por qué carga…?" questions must call the simulation.
- **It flipped the sign of the comparison** ("5,3 % menos" when it was more), changed a digit when copying the table, and restated a total one peso off. Now code renders a results block from the tool output (`render.py`), and the model never sees those figures; it only explains them (D16).
- **Groq's free tier also has a daily limit** of about 200k tokens for `gpt-oss-120b`. Testing used it up, so the agent fell back to the 20b model, which explains less well. Plan the evaluation and the demo around it.
- **The free Groq tier limits tokens per minute** (8,000 TPM). Back-to-back questions dropped to the 20b model, which follows the format worse, so short waits are now waited out on the same model.

**Done (2026-11-13):** `energy/agent/` (D16), written test-first. Answers are in Spanish, with two tools and the agreed answer formats. Try it with `uv run python -m energy.agent "pregunta"`. Later the same day, the battery and period chosen by the user became the agent's default (`UserSettings`, D16). 556 offline tests and 4 network tests pass; the live check covered regulation, simulation, mixed, trap and no-key questions.

## 6. Dashboard

Build a Streamlit app with price vs. decisions, cumulative savings, battery parameters the user can change, and a chat with the agent.

**Design agreed and implemented (2026-11-14).** The design starts from the customer's questions: the manager of a plant that buys at the bolsa price (D2).
1. Is a battery worth it, and what size?
2. Does a smart controller beat a simple timer?
3. How much do my shifts and the price year matter?
4. What would the battery do in a week?
5. Why did it do that, and is it allowed?

- **UI in Spanish.** No Part 1 results table in the app; those numbers go to the report.
- **Policies shown:** Q-learning, regla de precio, horario por turnos, noche/día. Perfect foresight and no battery are left out of the app.
- **Sidebar:**
  - **Escenario:** "2025 (normal)" or "2024 (El Niño, precios altos)". It defaults to that year's October–December test period, with custom dates in an expander and kept within one year.
  - **Batería:** 4, 8 or 12 h.
  - **Tipo de operación:** Mixto, Principalmente turnos dobles, Principalmente un turno (D15 shift-mix check).
  - **Estado:** whether the Groq key is set and which retriever is in use.
- **Indicators for the whole period:**
  - Q-learning's savings.
  - The difference vs. the best simple rule.
  - Cycles.
- **Tabs:**
  1. **Operación:** one week at a time, chosen with a slider. Price with charge and discharge hours, factory-off hours shaded, and the state of charge, for one policy ("Política mostrada", default Q-learning).
  2. **Comparación:** cumulative savings of the four policies, plus savings by battery size and policy.
  3. **Asistente:** chat with the sidebar's battery and period passed as `UserSettings`. Each answer has an expander with the tools, citations, retriever and stop reason. Groq is used only when a message is sent.
  4. **Acerca de:**
     - data sources and the method
     - the limitations, in plain language: bolsa price not the regulated tariff, synthetic shifts, model trained on the mixed operation, no battery wear, savings measured against no battery
     - "compare with your supplier's quote", since there is no cost data
- **Build:** `energy/app/charts.py` holds pure Plotly functions and the indicator calculations, tested offline. `app.py` is thin, with `st.cache_data` for prices and simulations and `st.cache_resource` for the retriever and agent. There's also an `AppTest` smoke test with no Groq key.
- **Open point:** the agent's `simulate_battery` always uses the mixed operation (D16), so the chat won't follow the "Tipo de operación" selector unless `UserSettings` gains a mix. Decide when building the chat tab.

**Obstacles:**
- **Confusing period selection.** The scenario radio stayed selectable while custom dates were on, and the date picker was three clicks deep. Now one radio has a "Personalizado" option that shows the dates directly.
- **Factory hours were hard to read.** Grey meant "off", so the shading was ambiguous. Factory-running hours are now purple bands, with a legend entry.
- **Averages next to one calendar.** The indicators were means over the 5 shift calendars, but the plot showed calendar 1 only. Averages are now labeled as such, the Operación tab shows the chosen calendar's own metrics, and a selector picks calendars 1–5. On 8 h, Oct–Dec 2025, Q-learning ranges from −0.4 % (calendar 1) to +10.6 % (calendar 4) against the best rule; the mean is +5.3 %.
- **Crowded chart title.** The title collided with the legend. The title is pinned to the top, with a 120 px top margin.
- **Stale cache after a code change.** Streamlit caches simulation results keyed on the function's own code, so the running server had to be restarted after `PolicyRun` changed.
- **Test collection.** A plain `uv run pytest` still fails to collect `tests/test_part1.py` until `energy/eval/part1.py` exists; meanwhile run with `--ignore=tests/test_part1.py`.

**Chat fixes (2026-11-14):** reviewing the Asistente tab offline showed four problems, fixed test-first (interface changes in D16):
- The first answer appeared below the input and jumped above it on the next rerun. The history now lives in a container created before the input.
- Without a Groq key the answer began with a raw error and "Detalles" was empty. It now explains in plain Spanish, lists the retrieved passages with citations, and "Detalles" shows the search engine and the technical error.
- Four example questions show while the chat is empty.
- "Detalles" shows which model answered, and a caption appears when the 20b fallback did.

**Done (2026-11-14):** built test-first with Sonnet agents, as in step 2.
- `app.py`: a thin Streamlit entry point with one function per tab and `main()`, following `../llm_lab_ia/main_app.py`. Run it with `uv run streamlit run app.py` (http://localhost:8501).
- `energy/app/model.py`: pure logic with no Streamlit. It runs the four app policies (Q-learning, price threshold, shift schedule, night/day; no perfect foresight, no battery) and averages them over the 5 test shift calendars, the same way the agent's `simulate_battery` does (D16), so the chat and the charts agree.
- `energy/app/charts.py`: pure Plotly figures for operation, cumulative savings and savings by battery size.
- Tabs: **Operación**, **Comparación**, **Asistente** (chat, with the sidebar battery and period as `UserSettings`; it always simulates the Mixto operation, this resolves the open point above) and **Acerca de**.
- Caching: `st.cache_data` for prices and simulations, `st.cache_resource` for the retriever and agent. Each tab wraps its errors in `st.error`. Groq is called only when a chat message is sent.
- **In-sample warning:** a custom period that overlaps January–September (the training months, D5) shows an `st.info` saying Q-learning's results are optimistic (`model.in_sample`).
- Tests: `tests/test_app_model.py`, `test_app_charts.py`, `test_app_helpers.py`, `test_app.py` (Streamlit `AppTest`, no keys, `.env` never read) and `test_app_consistency.py`.
  - **The consistency test** checks that the dashboard's means equal the agent tool's means (8 h, 2025-10-01..14). Q-learning matches exactly. The baselines differ only by the tool's whole-peso rounding.
  - 679 tests pass in total.

**Still open:**
- A clean-clone check (fresh clone, no `.env`, no Qdrant index, so the BM25 path) together with the README pass.
- A live chat check, which needs the user's authorization because it uses Groq tokens.

## 7. Agent evaluation

Write 10–15 fixed test questions, including trap questions, and record whether the agent called the right tool and cited a source.

**Obstacles:**
- **The planned price-history trap wasn't a trap:** the corpus has XM Paper N°52 on the high-price periods (El Niño 2024), so it became regulation question 4. The replacement trap asks for a battery's cost, which the corpus doesn't cover.
- **A decline in the model's own words** isn't the exact refusal phrase; such answers are left for manual review instead of being scored automatically.

**Written (2026-11-13), live run pending:** 11 questions in `energy/eval/agent_questions.yaml` (4 regulation, 3 simulation, 1 mixed, 3 traps: off-topic, out of corpus, prompt injection), each with its expected and forbidden tools, simulation arguments, sources and answer patterns. `energy/eval/agent_eval.py` runs them in fresh chats with the default settings and writes `results/agent_eval/`. Offline tests in `tests/test_agent_eval.py`.

**Done (2026-11-14):**
- **Live run:** `uv run python -m energy.eval.agent_eval --pause 30`. All answers came from `openai/gpt-oss-120b` with Qdrant + NVIDIA retrieval. The results are in `results/agent_eval/`.
- **Results:**
  - Correct tool: 9/11.
  - Simulation arguments: 4/4.
  - Expected source cited: 6/8.
  - Exact refusal on traps: 2/3.
  - Passed: **8/11**.
- **Manual review** (in the CSV's `revision_manual` column): three real failures, recorded in D16 and the known limitations.
  - Questions 1 and 4 were refused without searching.
  - Question 11's injection succeeded.
- **No second run with a fixed prompt for this delivery** (D16).
- **Before the run:** the runner was changed to write after every question and to merge partial runs by id (`--fresh` to start empty). It was then checked with a one-question smoke run.

## 8. Deliverables

Write the README with reproducible run instructions, `requirements.txt`, the AI-use declaration, and the report (up to 4 pages), including the limitations.

**Obstacles:**

**Done:**
