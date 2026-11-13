# Decisions, assumptions and simplifications

Running log of scope decisions for the final project (scenario 3.3, *Despacho de energía con RL y asistente sobre regulación*, in `EAFIT_Proyecto_Final_IA.md`). Each entry says what we chose, why, and what it costs us. These also feed the limitations section of the README and report.

Status values: **Decided**, **Open**, **Superseded**.

---

## D1. Scenario: 3.3, energy dispatch with RL plus a regulation assistant

**Status:** Decided (2026-11-12)

The problem: decide when to charge or discharge a battery based on the hourly bolsa price (precio de bolsa) of the Colombian wholesale electricity market.

## D2. Target user: a non-regulated commercial or industrial consumer

**Status:** Decided (2026-11-12)

The product is for a **non-regulated user (usuario no regulado)**, such as a factory, mall or hospital, that pays the hourly bolsa price and runs, or is considering, a battery. That user needs two things:

1. **When to charge and discharge** the battery to cut energy cost. This is the Part 1 MDP.
2. **What the rules are** for operating a battery in Colombia. This is the RAG corpus of CREG resolutions and XM documents.

**Why this user and not a household:** regulated households pay a monthly tariff, not the hourly bolsa price, so hourly arbitrage doesn't apply to them.

**Assumption A1:** the user's energy cost is the hourly bolsa price. Real non-regulated contracts often mix bilateral contracts and bolsa exposure. We treat full bolsa exposure as a modeling simplification.

**Assumption A2: the plant's load is always at least the battery's discharge power, and the battery never exports to the grid** (2026-11-12). A battery has a **capacity** (kWh, how much it stores) and a **power** rating (kW, how fast it charges or discharges). For example, a 200 kWh / 100 kW battery delivers at most 100 kWh per hour. If the plant always consumes at least 100 kW, every kWh discharged replaces a kWh bought from the grid at that hour's price. So the reward depends only on the price series and the battery's actions, and we don't need the plant's own load profile, which we don't have (XM demand is national). This is realistic when the battery is small compared with the site's consumption, such as a mall drawing 1–2 MW with a 100 kW battery. Selling to the grid follows other rules and is out of scope.

**Assumption A3: battery wear (degradation) is ignored** (2026-11-12). The reward is only energy cost: price × energy, with round-trip efficiency losses. In reality each cycle wears the battery, so a real operator wouldn't cycle for tiny price differences. Efficiency losses partly make up for this: a charge-discharge cycle only pays off when the price spread exceeds about 10%.

## D3. Battery only, no demand shifting

**Status:** Decided (2026-11-12)

The assignment allows "battery **or** demand shifting." We model only the battery. Demand shifting is the same idea (move energy use from expensive hours to cheap ones), but it needs a model of which loads are flexible and when. The battery MDP has a smaller and cleaner state.

## D4. No solar generation

**Status:** Decided (2026-11-12)

We don't use IDEAM solar radiation and don't model solar panels. That removes one data source and keeps the MDP small. Solar can come back as a future-work item.

## D5. Data window: hourly bolsa prices for 2024 and 2025, prices only

**Status:** Decided (2026-11-12; updated from "one year"; split changed 2026-11-13)

- Hourly national bolsa price (`PrecBolsNaci`, entity `Sistema`, COP/kWh) for **2024-01-01 to 2025-12-31**: 17,544 hours, with no gaps. No demand data, since A2 makes it unnecessary.
- **Split (changed 2026-11-13): two scenarios, each trained and tested within one year.** Train on January–September and test on October–December, split by date, so the agent is never tested on weeks it trained on. **Scenario 2025 comes first.** Scenario 2024 (the El Niño drought) repeats the same setup as a stress test. *Previously:* train on 2024 and test on 2025. We dropped that because it mixes two very different price regimes into a single test, and the within-year version gives each year its own result.
- Snapshot: `data/processed/prices_hourly.csv` (`timestamp`, `price_cop_kwh`), with the raw monthly JSON in `data/raw/xm/`. Both are committed, which is the plan B if the API is down.

**What the data shows:** the two years are very different, which matters for the MDP.

| Year | Min | Mean | Max (COP/kWh) |
|---|---|---|---|
| 2024 (El Niño) | 95.85 | 676.08 | 2675.65 |
| 2025 | 97.89 | 240.27 | 2224.01 |

**Validated (2026-11-12):** our daily averages match XM's official daily weighted price (`PPPrecBolsNaci`) within 1–5% for Sep–Nov 2024 and Apr 2025. The small gap is expected: XM weights by demand, while ours is a plain mean. The extremes are real. The top (≈2,500–2,676 COP/kWh, 2024-09-30 and late Oct to early Nov 2024) is the 2024 dry-season crisis. The bottom (≈96–110) is the price floor in wet periods. Runs of identical hourly prices are normal: generators submit one offer price per day. **We keep all values**; they are the risk the battery protects against.

The average price is lowest around 00:00–04:00 and peaks at **18:00–20:00**; in 2025 the 19:00 average is 468 versus about 190 overnight. Because 2024 averages almost three times 2025, **price levels in the MDP state should be relative** (for example, the price compared with a recent rolling average or percentile), not absolute COP thresholds. Otherwise a policy trained on 2024 would see almost every 2025 hour as "low."

**Hourly and weekly patterns (checked 2026-11-13, 2024–2025):**
- Compared with each day's mean, the price is about 0.87× at 02:00–04:00 and 1.15–1.23× at 18:00–20:00.
- 72% of days peak at 18:00 or 19:00.
- On 78% of days the max/min spread exceeds 12%, which is enough to cover the round-trip losses.
- The hour of day explains only about 29% of the variance within a day, so the state also needs a price level.
- In 2025, compared with the surrounding week's mean, Saturdays are 0.88×, Sundays 0.83× and Wednesdays 1.08×.
- 2025 monthly means range from 520 (January) down to 113 (June).

## D6. RAG corpus: CREG resolutions and XM documents about batteries and the market

**Status:** Decided (2026-11-12), document list Open

The corpus answers the user's regulation questions and explains the bolsa price. The list lives in **`data/corpus/sources.csv`** (id, title, publisher, url, format, notes), and the downloaded files are in `data/corpus/raw/`. Adding a document means adding a row and rerunning the download.

Current documents (downloaded 2026-11-12, all converting to text):

| id | Document | Format |
|---|---|---|
| `creg_101113_2026` | CREG 101 113 of 2026: battery storage (SAEB), **the current rule** | Official HTML |
| `creg_098_2019` | CREG 098 of 2019: SAEB, previous framework | Official HTML |
| `creg_174_2021` | CREG 174 of 2021: small-scale self-generation and distributed generation | Official HTML |
| `xm_glosario` | XM glossary (bolsa price, despacho ideal…) | HTML |
| `xm_faq` | XM frequently asked questions (how the bolsa price is set) | HTML |
| `xm_paper52_precios` | XM paper No. 52: the high prices in the wholesale market | PDF |
| `creg_131_1998` | CREG 131 of 1998: who counts as a non-regulated user (0.1 MW or 55 MWh/month since 2000; amended by CREG 183 of 2009 and 038 of 2010) | Official HTML |

The manifest also has `selector` (the CSS selector for the main content), `in_force` (yes/no) and `superseded_by` (see D11).

- **CREG 098 of 2019 has been replaced** by CREG 101 113 of 2026 (published 2026-06-19). We keep both so the agent can explain the change, but answers about current rules must cite 101 113.
- CREG resolutions come from the official CREG Gestor Normativo as HTML, which is complete and converts well. XM papers are PDFs.
- **Checked (2026-11-12): CREG 101 113 does not explicitly cover our user's battery.** Art. 2 applies it to *agentes* (market participants) implementing a SAEB connected to the SIN. Its definitions cover independent SAEB, SAEB as grid assets, and SAEB inside generation plants. Art. 71, Par. 2 says connecting an independent SAEB through assets owned by non-regulated users will be regulated separately, and until then is not allowed. A consumer's own battery behind the meter, used only to shift its purchases, isn't addressed directly. We treat this as a **known limitation**: the agent should say the corpus doesn't settle it and must not invent a rule. It's also a good evaluation question. (Our reading of the text, not legal advice.)

## D9. Battery options are a fixed set of presets

**Status:** Decided (2026-11-12; preset values and training approach decided 2026-11-13)

The user picks the battery from a fixed list of presets (capacity kWh / power kW / efficiency), not free values. So every policy can be trained ahead of time, and the app never has to train while it runs.

**Decided (2026-11-13):** three presets, all **100 kW with 90% round-trip efficiency** (√0.9 ≈ 94.9% each way):

| Preset | Capacity | Duration | State-of-charge levels |
|---|---|---|---|
| 2 h | 200 kWh | 2 h | 3 (0, ½, full) |
| 3 h | 300 kWh | 3 h | 4 |
| 4 h | 400 kWh | 4 h | 5 |

**One Q-table per preset.** Each preset has its own duration, so this is the same as one per duration.
- **Why not one agent for all batteries:** in tabular Q-learning, adding the preset to the state just splits one table into separate parts that share nothing. The number of charge levels also differs per preset.
- **Real sharing across batteries** would need function approximation (such as a neural network). That's more complex and harder to present as the team's own implementation (MVP).
- **Cost:** training each table takes seconds.

The options considered:

How Q-learning covers the presets:

- **Direct (recommended):** train one Q-table per preset offline, save them to disk, and have `simulate()` load the one matching the selected preset. It's simple and each preset gets its own exact policy.
- **Indirect:** with wear ignored (A3), the reward is proportional to battery size. If we track the state of charge as a percentage, two batteries with the same **duration** (capacity ÷ power, the hours to empty at full power) and the same efficiency have the **same optimal policy**, and the savings just scale with size. For example, 200 kWh / 100 kW and 400 kWh / 200 kW are both 2-hour batteries. So we only need one Q-table per duration, and a preset adds a size multiplier.

The two combine: define the presets as a few durations times a few sizes. Then there's one Q-table per duration, and showing it holds is a nice point for the report. With one size per duration, as decided above, "direct" and "indirect" coincide. Other sizes could be added later by scaling the savings.

## D14. Battery MDP and Q-learning setup

**Status:** Decided (2026-11-13). Closes O1, O4 and O7.

**Decision step:** one hour. The agent acts every hour.

**State:** (hour block, state of charge, price level).
- **Hour block:** the hour of day in **pairs**, 12 values (00–01, 02–03, …, 18–19, 20–21, 22–23). The agent still decides every hour; both hours of a pair share the same state, and the state of charge tells them apart. This halves the hour dimension, so each state is visited twice as often in training.
  - Cost: 20:00, still a peak at 1.15×, is paired with 21:00 at 1.06×. The price level makes up for most of that.
- **State of charge:** a discrete level from 0 to the battery's duration in hours (D9). One hour at full power moves it exactly one level.
- **Price level (closes O7):** the current price divided by the mean of the **previous 168 hours** (7 days), grouped into **4 bins**. The cut points are the quartiles of that ratio in the training set. Being relative, the same table works whether prices average 676 (2024) or 113–520 (2025 months).
  - A cheap weekend shows up as "cheap" relative to the week, so we add **no weekday flag** (see the weekly pattern in D5).
  - The trailing window only uses past prices. The first week of a split takes its warm-up from the hours just before it (December 2024 for 2025). For January 2024, which has no earlier data, the mean is taken over the hours available so far.

**Actions:** charge, idle, discharge, each at full power for one hour.
- Impossible actions are masked rather than penalized: charging when full, or discharging when empty.

**Transitions:** the state of charge moves one level per charge or discharge. Energy accounting, with power P = 100 kW and efficiency η = √0.9 each way:
- Charging stores P × 1 h and **buys** P / η from the grid.
- Discharging takes P × 1 h out of storage and **delivers** P × η to the plant.
- Thanks to A2, every kWh delivered is a kWh the plant does not buy.

**Reward, option A (decided): relative.**
- r = (price ÷ trailing 168-hour mean) × (kWh delivered − kWh bought).
- It matches the relative price state, and the drought's prices don't dominate what is learned.
- **Results are always reported in COP:** `simulate` replays the policy with the real prices.

**Option B (considered, not chosen):** reward in COP, r = price × (kWh delivered − kWh bought).
- Simpler and literal.
- But Q-values learned at one price level are off by about 3× at another (2024 vs. 2025), which makes them noisier, and it doesn't match the relative state. We keep this for the record. It could become an experiment if A behaves badly.

**Episodes:** one **week** (168 hours), starting with the battery empty, discount γ ≈ 0.99.
- The charge level carries over midnight, as in real operation, instead of an artificial reset every day.
- The number of hours learned from is the same as with daily episodes.
- Weekly episodes alone don't teach weekly patterns, because the state has no weekday. The relative price level covers most of that.

**One Q-table per preset** (D9). The table is 12 × (3–5) × 4 states × 3 actions, at most 720 values.

**Baselines (refines O3):**
- **Do nothing:** zero savings.
- **Fixed schedule:** charge 02:00–05:00, discharge 18:00–21:00, as many hours as the battery's duration allows.
- **Perfect foresight:** the upper bound, knowing the week's prices in advance.

Expectation, to state honestly in the report: the daily pattern is regular, so the fixed schedule will be strong. Q-learning should land between it and perfect foresight, gaining on unusual days such as a midday peak, or a flat day where cycling loses money.

**Assumption A4: the current hour's price is known when deciding** (2026-11-13). In reality the bolsa price is settled after the fact, so a consumer doesn't know the exact price of the current hour. Assuming it is known keeps the state simple and strong. The alternative, using only the hour and yesterday's price level, is more realistic but has a weaker signal. A4 makes the savings optimistic and goes in the limitations.

**Implementation settings (2026-11-13):**
- **Learning rate:** α decays linearly from 0.1 to 0.02 over the passes. With a constant α, near-tied greedy actions kept flipping, and early stopping never triggered.
- **Exploration:** ε decays linearly from 1.0 to 0.05.
- **Passes:** at most 300 over the training weeks, with early stopping after 20 unchanged passes.
- **Fixed schedule:** charge from 02:00 and discharge from 18:00, for `duration_h` hours each.
- **Storage:** each Q-table is `float32`, written with `np.savez_compressed` plus a small `.json`, at 1.7–2.7 KB per preset. No pickle.
- **Training runs:** the jobs (preset × year) run in parallel processes. Each job's seed depends only on (base seed, preset, year), so results are identical whatever the number of workers. Training takes 2.7 s per table; the three 2025 tables take 3.6 s in parallel versus 7.0 s one after another.

**Results, scenario 2025 (2026-11-13; test October–December 2025, savings in COP):**

| Preset | Q-learning | Fixed schedule | Perfect foresight | Q ÷ schedule | Q ÷ perfect |
|---|---|---|---|---|---|
| 2 h | 3.59 M | 3.74 M | 4.29 M | 96% | 84% |
| 3 h | 4.51 M | 4.73 M | 5.43 M | 95% | 83% |
| 4 h | 4.84 M | 5.01 M | 6.12 M | 97% | 79% |

**Decided: report this as it is.** Q-learning comes close to the fixed schedule but stays below it. We checked that this isn't a bug:
- It is also slightly below on the training months (0.99).
- No variant beat the schedule (2 h battery, Q ÷ schedule on the test months):

| Variant | Q ÷ schedule |
|---|---|
| D14 as decided | 0.97 |
| Discount 0.999 | 0.88 |
| Discount 1.0 | 0.82 |
| All 24 hours instead of pairs | 0.92 |
| Price compared with the last 24 h | 0.99 |
| One-day episodes | 0.98 |
| 1,000 passes | 0.99 |

Why, for the report:
- **The daily pattern is very regular**, so "charge at night, discharge at 18:00" is close to the best a non-foresighted policy can do. At 02:00 nothing in the state says how expensive this evening will be.
- **Perfect foresight's extra savings come from knowing the future.**
- **The schedule's hours were chosen from the 2024–2025 hourly profile**, which includes the test months, so it has a small information advantage.
- **Q-learning found nearly the same rule on its own** from January–September, without being told the hours.

The test on synthetic data checks Q-learning ≥ 90% of the fixed schedule. A threshold relative to perfect foresight was dropped: there, even the schedule only reaches 75–84%, because perfect foresight profits from the noise.

## D10. Call XM's API directly instead of `pydataxm`

**Status:** Decided (2026-11-12)

`pydataxm` 0.3.17 breaks with pandas 3.1: it calls `pd.date_range(freq='M')`, which now raises `ValueError`. We call the same public API it wraps (`POST https://servapibi.xm.com.co/hourly`) from `energy/data/xm_prices.py`, with our own async client: one request per month (ranges over about 31 days fail), at most 4 at a time, with retries, and resumable. `pydataxm` was removed from the dependencies. Cite the source as XM S.A. E.S.P., the API at servapibi.xm.com.co, metric `PrecBolsNaci`.

## D11. RAG corpus cleaning and metadata

**Status:** Decided (2026-11-12)

- **Cleaning** (`uv run python -m energy.rag.clean`): raw → `data/corpus/clean/<id>.md`, which is committed, so anyone can read exactly what the agent can retrieve. For HTML it keeps only the manifest's `selector` node (`div.panel-documento` for CREG, `#block-xm-content` for XM), so the site menus are dropped. It fails loudly if the selector matches nothing. Then markitdown, links reduced to their text, zero-width characters removed, blank lines collapsed. PDFs skip the selector step.
- **Article-level citations:** a CREG line starting with `ARTÍCULO <n>.` becomes a `## ` heading, so chunks can carry an `article` field and the agent can cite "CREG 101 113 de 2026, Art. 12". Only article numbers higher than the previous one are marked. When a resolution amends another, it quotes the amended article (Art. 80 of 101 113 quotes "ARTÍCULO 6o." of CREG 055 of 1994), and that quoted text has to stay inside the amending article. Result: 108 articles in 101 113, 38 in 098, 31 in 174, 8 in 131.
- **In-force flag:** `in_force=no` and `superseded_by` in the manifest (currently only CREG 098 of 2019 → 101 113 of 2026). It will become chunk metadata, so the agent can say a rule was replaced instead of quoting it as current.

## D13. Chunking, index and retrieval

**Status:** Decided (2026-11-12)

- **Chunking** (`uv run python -m energy.rag.chunk` → `data/corpus/chunks.jsonl`, committed, deterministic). CREG documents are split on `## ARTÍCULO n` **and `## ANEXO n.`** headings, keeping the heading line in the chunk text. Long sections are split further (`RecursiveCharacterTextSplitter`, 900 / 120, drop chunks under 30 characters), and every piece keeps its section label (`"12"`, `"Anexo 1"`). XM documents are split directly. Result: **984 chunks**; 90% of CREG chunks carry an article or annex label, and the rest are preambles.
- **Annexes as their own sections:** the first run labeled all annex text as the last article. For example, CREG 131's eligibility rules (Anexo 1) were cited as "Art. 8". `clean.mark_annexes` now marks bare `ANEXO n.` lines. Bold quoted annexes of other resolutions (`**ANEXO CO.4.**` inside Art. 95 of 101 113) stay inside the amending article, as quoted articles do (D11).
- **Embedding text:** `"{title} — Artículo n"` (or `— Anexo n`), a blank line, then the chunk text. A continuation piece of a long article stays findable by its document and article, while the displayed text stays clean.
- **Index** (`uv run python -m energy.rag.index`): NVIDIA `nemotron-3-embed-1b` (dimension 2048), `qdrant-client` used directly (no `langchain_qdrant`), collection `regulacion`, cosine, at `data/qdrant/` (gitignored, D8). About 42 s for 984 chunks. `.env` is loaded in the entry point only.
- **Retrieval** (`energy/rag/retrieve.py`): `Hit` with a `citation` such as "Resolución CREG 131 de 1998 …, Anexo 1", plus `in_force` / `superseded_by`. `open_retriever()` uses Qdrant + NVIDIA when the key and the index exist, and otherwise falls back to our own **BM25** over `chunks.jsonl`. Manual check: `uv run python -m energy.rag.retrieve "pregunta" [--bm25]`.
- **Threshold 0.25 kept.** In the sample questions, relevant hits score 0.37–0.68, and the out-of-corpus question ("receta de arepas") tops out at 0.12, so it gets 0 hits. Observations for the agent prompt and the evaluation:
  - For "eficiencia mínima… SAEB", the top hit is the **replaced** CREG 098, and the current 101 113 Art. 6 comes second. The agent must use `in_force` to prefer the current rule.
  - For the non-regulated-user limit, the XM FAQ (which quotes CREG 131: 0.1 MW or 55 MWh/month) ranks first. CREG 131 Art. 2 itself is only in BM25's top 5.

## D7. Python environment: uv, Python 3.12, with `requirements.txt` exported

**Status:** Decided (2026-11-12)

- **uv** manages the environment and dependencies. `pyproject.toml` and `uv.lock` are committed, and `.python-version` pins **Python 3.12**, which is safer for library wheels than 3.13.
- The assignment requires a **`requirements.txt`**. We generate it from the lock file with `uv export --no-hashes --no-dev -o requirements.txt` and regenerate it every time a dependency changes. Never edit it by hand.
- The README gives two install paths: `uv sync`, or `python -m venv .venv` + `pip install -r requirements.txt` for anyone without uv.
- We use uv rather than conda because it's faster, installs every package we need from pip, and records exact versions. Conda's advantage, heavy native packages such as CUDA or GDAL, doesn't apply here.

**Deployment:** the requirement is that the app runs from the README on a local machine (the demo and the grader). We don't need a server. If we want a public URL later, Streamlit Community Cloud builds its environment from `requirements.txt`, and the API keys go in its secrets settings.

## D8. Data is tracked in git

**Status:** Decided (2026-11-12)

`data/` is committed to the repo (it's no longer in `.gitignore`), so the CSV snapshot works offline and on any machine, and the README needs no download step to run the app. The download script stays in the repo to show how the data was obtained.

- Keep the committed files small: one year of hourly prices is a small CSV.
- **Decided (2026-11-12): the Qdrant index (`data/qdrant/`) stays gitignored** and is rebuilt with a script. It's binary, rewritten on every rebuild, and has a machine-local lock file. Querying needs `NVIDIA_API_KEY` anyway; without the key, the planned fallback is BM25 over the committed chunks (see `architecture.md`).
- **Resolved:** the corpus PDFs are committed. `.gitignore` no longer excludes `*.pdf`; only the assignment PDF is ignored.
- `docs/` is tracked as well (2026-11-12): the decision log and architecture are evidence for the report.

## D12. Architecture and code documentation conventions

**Status:** Decided (2026-11-12)

An architecture review (Opus, read-only) produced the target design now written up in **`docs/architecture.md`**: package map, dependency rules (`data ← {rl, rag} ← agent ← {eval, app}`), interfaces, fallbacks, and which artifacts are committed. Sections are marked Implemented or Planned. The planned interfaces are the agreed target; refine them when each part is built and log changes here.

Code documentation: Google-style docstrings, a module docstring for every module, type hints on public signatures, and comments that explain *why* by pointing to decision IDs. No per-package READMEs. The README covers setup and the run order, and this file covers rationale.

---

## Open questions

- ~~**O1. MDP specification**~~: resolved in D14.
- **O2. Solver:** leaning toward **Q-learning** (2026-11-12). It learns from the real price series without a price-transition model. Fall back to value iteration if Q-learning doesn't converge or its results are unstable. Both use the same data.
- ~~**O3. Baselines**~~: resolved in D14 (do nothing, the fixed schedule and perfect foresight), with the results for 2025.
- ~~**O4. Battery presets**~~: resolved in D9 (2 h, 3 h and 4 h at 100 kW, 90% round trip, starting empty).
- ~~**O5. Year and split**~~: resolved in D5 (changed 2026-11-13: within-year scenarios, train January–September and test October–December, 2025 first).
- **O6. RAG document list** (see D6): a first version is downloaded. Revisit after reading the documents, especially for behind-the-meter batteries.
- ~~**O7. Price state definition**~~: resolved in D14 (price ÷ the trailing 168-hour mean, 4 quartile bins).
- **O8. To do: `energy/config.py` + root-anchored paths** (deferred 2026-11-12). Today every command must run from the repo root, because the path defaults are relative (`data/...`). Do it when the first module runs from somewhere else (a notebook, the eval, the app) or needs shared settings (model names, `T_max`, thresholds). How:
  - build every path from `Path(__file__).resolve().parents[1]`
  - load `.env` only in entry points, never on import, so tests don't pick up real keys
  - move `USER_AGENT` and the minimum-text-length constant there (currently duplicated)
  - keep the CLI flags

  Trade-off: it's global state that every module depends on, and it changes working code. That's why we're waiting until it's needed.
- ~~**O9**~~: done (2026-11-13), `energy/data/prices.py`. The original note follows.
- **O9. To do: `energy/data/prices.py`** (`load_prices()` + train/test split; deferred 2026-11-12). This is the first step of Part 1, and O7 is now defined (D14). It should provide:
- the within-year split by date (D5)
- the trailing 168-hour mean, computed over the **whole** series before splitting, so each split gets its warm-up from the hours just before it

Until then, nothing reads the CSV.

## Known limitations (for the README and report)

- Full bolsa-price exposure (A1) overstates the savings for a user with bilateral contracts.
- Two years of data (2024–2025). Results may not generalize to other hydrological conditions, such as El Niño versus La Niña years.
- No network charges, taxes or capacity payments in the cost; only the energy price.
- Battery wear is ignored (A3), so the savings are optimistic and the policy may cycle more than a real operator would.
- Only the fixed battery presets are supported (D9).
- The agent knows the current hour's price when deciding (A4), but the real bolsa price is settled after the fact, so the savings are optimistic.
- Each scenario is tested on only three months (October–December) of one year (D5).
- On 2025, Q-learning earns 95–97% of a simple fixed schedule. With such a regular daily price pattern, learning adds little over a well-chosen rule (D14).
- This is an academic prototype, not financial, legal or regulatory advice.
