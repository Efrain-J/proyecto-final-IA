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

**Status:** Decided (2026-11-12; updated from "one year")

- Hourly national bolsa price (`PrecBolsNaci`, entity `Sistema`, COP/kWh) for **2024-01-01 to 2025-12-31**: 17,544 hours, with no gaps. No demand data, since A2 makes it unnecessary.
- **Split:** train on 2024, test on 2025.
- Snapshot: `data/processed/prices_hourly.csv` (`timestamp`, `price_cop_kwh`), with the raw monthly JSON in `data/raw/xm/`. Both are committed, which is the plan B if the API is down.

**What the data shows:** the two years are very different, which matters for the MDP.

| Year | Min | Mean | Max (COP/kWh) |
|---|---|---|---|
| 2024 (El Niño) | 95.85 | 676.08 | 2675.65 |
| 2025 | 97.89 | 240.27 | 2224.01 |

**Validated (2026-11-12):** our daily averages match XM's official daily weighted price (`PPPrecBolsNaci`) within 1–5% for Sep–Nov 2024 and Apr 2025. The small gap is expected: XM weights by demand, while ours is a plain mean. The extremes are real. The top (≈2,500–2,676 COP/kWh, 2024-09-30 and late Oct to early Nov 2024) is the 2024 dry-season crisis. The bottom (≈96–110) is the price floor in wet periods. Runs of identical hourly prices are normal: generators submit one offer price per day. **We keep all values**; they are the risk the battery protects against.

The average price is lowest around 00:00–04:00 and peaks at **18:00–20:00**; in 2025 the 19:00 average is 468 versus about 190 overnight. Because 2024 averages almost three times 2025, **price levels in the MDP state should be relative** (for example, the price compared with a recent rolling average or percentile), not absolute COP thresholds. Otherwise a policy trained on 2024 would see almost every 2025 hour as "low."

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

**Status:** Decided (2026-11-12), preset values and the training approach Open

The user picks the battery from a fixed list of presets (capacity kWh / power kW / efficiency), not free values. So every policy can be trained ahead of time, and the app never has to train while it runs.

How Q-learning covers the presets:

- **Direct (recommended):** train one Q-table per preset offline, save them to disk, and have `simulate()` load the one matching the selected preset. It's simple and each preset gets its own exact policy.
- **Indirect:** with wear ignored (A3), the reward is proportional to battery size. If we track the state of charge as a percentage, two batteries with the same **duration** (capacity ÷ power, the hours to empty at full power) and the same efficiency have the **same optimal policy**, and the savings just scale with size. For example, 200 kWh / 100 kW and 400 kWh / 200 kW are both 2-hour batteries. So we only need one Q-table per duration, and a preset adds a size multiplier.

The two combine: define the presets as a few durations (e.g., 1 h, 2 h, 4 h) times a few sizes. Then there's one Q-table per duration, and showing it holds is a nice point for the report.

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

- **O1. MDP specification:** state discretization (hour, state of charge levels, price bins), action set (charge, idle, discharge), and reward (cost saved minus efficiency losses and degradation).
- **O2. Solver:** leaning toward **Q-learning** (2026-11-12). It learns from the real price series without a price-transition model. Fall back to value iteration if Q-learning doesn't converge or its results are unstable. Both use the same data.
- **O3. Baselines:** naive (no battery, or a fixed night-charge / evening-discharge schedule) and reference (price-threshold rule or perfect-foresight optimum).
- **O4. Battery presets** (see D9): which capacity / power / efficiency combinations to offer, and the initial state of charge. They should be realistic for a commercial or industrial site and consistent with A2 (the site's load stays above the battery's power).
- ~~**O5. Year and split**~~: resolved in D5 (2024 + 2025, train on 2024, test on 2025).
- **O6. RAG document list** (see D6): a first version is downloaded. Revisit after reading the documents, especially for behind-the-meter batteries.
- **O7. Price state definition:** a relative price level instead of absolute COP thresholds, because of the gap between 2024 and 2025 (see D5).
- **O8. To do: `energy/config.py` + root-anchored paths** (deferred 2026-11-12). Today every command must run from the repo root, because the path defaults are relative (`data/...`). Do it when the first module runs from somewhere else (a notebook, the eval, the app) or needs shared settings (model names, `T_max`, thresholds). How:
  - build every path from `Path(__file__).resolve().parents[1]`
  - load `.env` only in entry points, never on import, so tests don't pick up real keys
  - move `USER_AGENT` and the minimum-text-length constant there (currently duplicated)
  - keep the CLI flags

  Trade-off: it's global state that every module depends on, and it changes working code. That's why we're waiting until it's needed.
- **O9. To do: `energy/data/prices.py`** (`load_prices()` + train/test split; deferred 2026-11-12). Do it as the first step of Part 1, after O7. If the relative price level uses a rolling window, the 2025 test set needs the end of 2024 as warm-up, so a plain calendar-year split could be wrong. Until then, nothing reads the CSV.

## Known limitations (for the README and report)

- Full bolsa-price exposure (A1) overstates the savings for a user with bilateral contracts.
- Two years of data (2024–2025). Results may not generalize to other hydrological conditions, such as El Niño versus La Niña years.
- No network charges, taxes or capacity payments in the cost; only the energy price.
- Battery wear is ignored (A3), so the savings are optimistic and the policy may cycle more than a real operator would.
- Only the fixed battery presets are supported (D9).
- This is an academic prototype, not financial, legal or regulatory advice.
