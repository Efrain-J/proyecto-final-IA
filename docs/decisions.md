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
- **Scenario 2024 trained (2026-11-13),** together with a cross-year transfer check: tables trained on one year and tested on the other (see D15).
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

**Replaced on branch `explore/factory-schedule` by D15:** the presets there are 4 h, 8 h and 12 h (400, 800 and 1,200 kWh, still 100 kW and 90% round trip).

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

**On branch `explore/factory-schedule`, D15 extends this setting:** a factory shift schedule, longer batteries, and a state that knows today's shift.

The test on synthetic data checks Q-learning ≥ 90% of the fixed schedule. A threshold relative to perfect foresight was dropped: there, even the schedule only reaches 75–84%, because perfect foresight profits from the noise.

## D15. Factory shift schedule, long batteries and shift-aware Q-learning

**Status:** Decided (2026-11-13), on branch `explore/factory-schedule`. Changes parts of D9 and D14 and keeps the rest.

**Why:** under D14 the factory always runs. The daily price pattern is so regular that a hand-made rule ("charge at 02:00, discharge at 18:00") is near-optimal, and Q-learning only reaches 95–97% of it. When the factory doesn't always run, the decision becomes truly sequential: stored energy is only worth something if a future hour with the factory on comes at a high price. That's where learning helps.

**Prototype evidence (2026-11-13).** Three rounds in the scratchpad, on 2025 prices, training on January–September and testing on October–December. The results are averaged over 4 training seeds × 5 test shift patterns, with the baselines' parameters tuned on the training months:

| Battery | Q-learning ÷ best rule | Best rule | Q-learning ÷ perfect foresight |
|---|---|---|---|
| 4 h | 0.98–1.00 | shift schedule | 73% |
| 6 h | 1.06 | tuned price rule | 70% |
| 8 h | 1.06 | tuned price rule | 69% |
| 12 h | 1.04 | tuned price rule | 65% |
| 24 h | 0.98 | tuned price rule | 55% |

- **Long batteries:** every training seed was at least 1.00× the best rule for 6–12 h.
- **The baseline matters:** before the price rule's cut-offs were tuned, Q-learning seemed to win by 16% at 6 h.
- **Training amount:** doubling the training data or the passes changed nothing, so the long batteries weren't undertrained.
- **24 h is too large for a table:** 25 charge levels.

**The setting:**
- **The factory runs in shifts (A5).** The battery can **discharge only while the factory is on**; this replaces "always" in A2.
  - When the factory is on, its load is at least the battery's power.
  - When it's off, discharging would export energy, so it's not allowed.
  - Charging is allowed at any time.
- **Presets (replaces D9's values):** 4 h, 8 h and 12 h, all at 100 kW and 90% round trip: 400, 800 and 1,200 kWh, with 5, 9 and 13 charge levels.
  - 4 h is the reference case, where a schedule still wins.
  - 8 h and 12 h are where Q-learning wins.
- **State (extends D14):** (hour pair, charge level, price level, **today's shift type**: off, morning, afternoon or full day).
  - Whether the factory is on follows from the hour pair and the shift type, because shift edges fall on pair boundaries.
  - Two alternatives were tested. Only "factory on now" was too little information. Adding "tomorrow is a work day" made the table too sparse for nine months of data.
- **Training:** the January–September prices are **repeated with 5 random shift patterns**, with episodes kept within each pattern. The same prices under different schedules show the agent more situations. Everything else is as in D14: at most 300 passes, early stopping, α decaying from 0.1 to 0.02, weekly episodes, relative reward.
- **Evaluation:** each test result is the **average over 5 test shift patterns**, with fixed seeds never used in training, on October–December.

**Baselines (replaces D14's list):**
- **Do nothing.**
- **Shift schedule:**
  - On work days, charge from 02:00 for as many hours as the battery's duration.
  - Discharge from 18:00 while the factory is on (afternoon or full-day shifts), or during the last hours of a morning shift.
- **Tuned price rule:**
  - Charge when the price level (`rel_price`) is below a cheap cut-off.
  - Discharge when it's above an expensive cut-off and the factory is on.
  - The cut-offs are quantiles of the training price levels: cheap ∈ {10…50%}, expensive ∈ {50…90%}, expensive > cheap. The pair is chosen by the mean training savings over the same 5 shift patterns.
  - In the prototype it picked 50% / 80% for 6–12 h.
- **Perfect foresight,** now limited to hours when the factory is on.
- **The fixed schedule (`NightDay`)** stays as a simple reference.
- **The reported metric:** Q-learning ÷ the best rule.
  - Short batteries favor the schedule and long ones the price rule. Q-learning matches or beats whichever is best without anyone having to choose.
  - In the prototype, the best rule was picked on the test months, which slightly favors the baselines.

**Assumption A5: the factory shift schedule is synthetic** (2026-11-13).
- **Generation:** one random shift type per calendar day, by day of week:
  - Monday–Friday: 45% full day (06:00–22:00), 30% morning (06:00–14:00), 15% afternoon (14:00–22:00), 10% off.
  - Saturday: 40% morning, otherwise off.
  - Sunday: off.
- **Reproducibility:** each pattern comes from a seeded generator. Training, tuning and testing use separate seeds, and the day's shift plan is known at 00:00.
- **The probabilities are invented,** plausible for a factory but not taken from data.
- **What it doesn't model:**
  - Days are independent: no streaks of busy weeks, no maintenance shutdowns.
  - No Colombian public holidays.
  - No seasonal pattern.
  - The load is all or nothing: no base load at night (lighting, refrigeration) that would allow some discharge outside shifts.
  - The shift edges are fixed.
- **Named mixes:** `default` (the one above), `mostly_double` and `mostly_single`. Only `default` is trained. The others show how the trained policy holds up on another factory's pattern, and can be a dashboard setting. They differ only on weekdays; Saturday and Sunday are as above. Values from `MIXES` in `energy/rl/factory.py`, recorded 2026-11-14:

  | Mix (dashboard label) | Off | Morning 06–14 | Afternoon 14–22 | Full day 06–22 | Expected running hours/week |
  |---|---|---|---|---|---|
  | `default` (Mixto) | 10% | 30% | 15% | 45% | ≈ 57 h |
  | `mostly_double` (Principalmente turnos dobles) | 5% | 10% | 10% | 75% | ≈ 71 h |
  | `mostly_single` (Principalmente un turno) | 10% | 45% | 35% | 10% | ≈ 43 h |

  The hours are the expectation: 5 weekdays × (8 h × P(morning or afternoon) + 16 h × P(full)), plus 3.2 h for Saturday (40% × 8 h). For `default` that's 5 × 10.8 + 3.2 ≈ 57 h.

**Implemented (2026-11-13), test-first:** a Sonnet agent wrote the tests, and a second one wrote the code to pass them; 359 tests pass. The new module is `energy/rl/factory.py`; `battery`, `policies`, `qlearning`, `simulate` and `train` were extended.

**Results, scenario 2025** (savings in COP over October–December, mean of the 5 test shift patterns, seed 0):

| Battery | Q-learning | Shift schedule | Tuned price rule | Fixed schedule | Perfect foresight | Q ÷ best rule | Q ÷ perfect |
|---|---|---|---|---|---|---|---|
| 4 h | 2.61 M | **2.70 M** | 2.50 M | 2.64 M | 3.63 M | 0.97 | 72% |
| 8 h | **3.20 M** | 2.47 M | 2.97 M | 2.56 M | 4.56 M | 1.07 | 70% |
| 12 h | **3.13 M** | 2.30 M | 3.07 M | 2.25 M | 4.96 M | 1.02 | 63% |

**Across training seeds 0, 1 and 2,** Q-learning ÷ best rule averages:
- 4 h: 0.99, ranging 0.97–1.02.
- 8 h: 1.05, ranging 1.01–1.07.
- 12 h: 1.06, ranging 1.02–1.08.

That matches the prototype, with a spread of about ±0.03 between seeds.

**Timing and size:**
- Each job takes about 14 s. The three run in 16 s in parallel against 36 s one after another.
- Early stopping never triggers with 5 training patterns, so every job runs all 300 passes.
- The Q-tables are 8 / 15 / 22 KB.
- The threshold files are 60 B. For 8 h the tuned cut-offs are the median and the 80th percentile.

**Implementation choices:**
- **Shift schedule:** its windows don't wrap past midnight. Discharge overrides charge where they overlap, which happens on morning days for 8 h and 12 h, so that rule issues some impossible actions. It's weak for long batteries, and the price rule is the best rule there.
- **Old D14 tables:** a 4-dimensional table can't be loaded as a policy, so `resolve_policy` falls back to the shift schedule instead of crashing.

**One test pattern varies a lot.** On test pattern 1000, 8 h, Q-learning beats the price rule by only 1.6%, so per-pattern results vary. Report the mean over patterns.

**Parameter tuning (2026-11-13), adopted.** Run with `uv run python -m energy.eval.tune_q`; the outputs are in `results/tuning/q_8h_2025_*`.

*Method:* the test months are never used to choose.
- Train on January–June and score on July–September, over 5 validation shift patterns (seeds 2000–2004).
- The score is Q-learning ÷ the tuned price rule, with 4 training seeds per configuration.
- Stage 1 changes one parameter at a time. Stage 2 combines the ones that gained at least 1%.
- Only then are the default and the winner retrained on January–September and tested **once** on October–December.

*Validation, 8 h battery:*

| Change | Effect on the score |
|---|---|
| γ 0.995 | +4.6% |
| 10 shift patterns | +2.6% |
| α ending at 0.005 | +2.0% |
| starting α 0.2 | +1.7% |
| 600 passes | −1.0% (no run stops early) |
| final ε 0.01 or 0.15 | −2 to −3% |
| γ 0.95 | −41%: the policy collapses, because an 8 h battery needs to look further ahead |
| **All three best combined** | **+6.1%** |

*Test, 8 h, 4 seeds:*
- Q-learning ÷ best rule goes from 1.03 (seeds 0.96–1.07) to **1.07** (1.05–1.09).
- Q-learning ÷ perfect foresight goes from 67% to 70%.
- The noise between seeds is about ±4%, so the test gain alone isn't conclusive. The validation gain is consistent, though: every tuned seed beats the default's average.

*Adopted as defaults:* `QConfig.gamma` = 0.995 and `alpha_end` = 0.005, plus `DEFAULT_SHIFT_DRAWS` = 10. A job now takes about 30 s, and the three run in 31 s in parallel.

*Retrained on all presets,* test October–December, Q-learning ÷ best rule for seeds 0 / 1 / 2:

| Battery | Seed 0 | Seed 1 | Seed 2 | Mean | Mean before |
|---|---|---|---|---|---|
| 4 h | 0.99 | 1.00 | 1.00 | **1.00** | 0.99 |
| 8 h | 1.05 | 1.08 | 1.07 | **1.07** | 1.05 |
| 12 h | 1.15 | 1.09 | 1.09 | **1.11** | 1.06 |

Q-learning ÷ perfect foresight is 74% / 70% / 69%. The settings were tuned on 8 h only, and they also helped 12 h.

**Scenario 2024 and cross-year transfer (2026-11-13).** `uv run python -m energy.rl.train --scenarios 2024` trains on January–September 2024, writing `q_*_2024` and `threshold_*_2024`. The tables were then crossed: each year's tables tested on October–December of both years, over the 5 test shift patterns, training seed 0. The script is in the scratchpad and not yet in the repo.

**Compare against the strongest rule.** The price rule's cut-offs don't carry over well across a regime change. Cut-offs tuned on January–September 2024 did worse on October–December 2024 than cut-offs tuned on 2025 (8 h: 8.06 M against 9.67 M). Against the weaker 2024 rule, Q-learning on 8 h looked +33% better, which would mislead. So the cross-year metric is **Q-learning ÷ the strongest rule available**: the shift schedule, the fixed schedule, or either tuned price rule.

**Q-learning ÷ strongest rule:**

| Battery | Trained on | Test Oct–Dec 2024 | Test Oct–Dec 2025 |
|---|---|---|---|
| 4 h | 2024 | 0.91 | 0.95 |
| 4 h | 2025 | 0.83 | 0.99 |
| 8 h | 2024 | 1.10 | 1.06 |
| 8 h | 2025 | 1.02 | 1.05 |
| 12 h | 2024 | 1.00 | 1.07 |
| 12 h | 2025 | 1.15 | 1.15 |

**Savings, October–December 2024 (COP):**

| Battery | Q (trained 2024) | Q (trained 2025) | Price rule (2024 / 2025 cut-offs) | Shift schedule | Perfect foresight |
|---|---|---|---|---|---|
| 8 h | 10.68 M | 9.85 M | 8.06 M / 9.67 M | 6.92 M | 18.76 M |
| 12 h | 11.28 M | 13.00 M | 9.37 M / 11.29 M | 6.14 M | 22.33 M |

**Findings:**
- **The relative-price state carries over between years.** Tables trained on 2025's calm prices do well in the 2024 drought crisis, and the reverse holds too. For 12 h, the 2025 table even beats the 2024 one on 2024.
- **Crisis months are harder for every causal policy.** In October–December 2024 Q-learning captures 50–58% of perfect foresight, against about 70% in 2025: price spikes reward only foresight. Savings in COP are 3–4× higher in 2024.
- **4 h:** the shift schedule wins in every case, and by more in 2024 (Q-learning 0.83–0.91).
- **Caveat:** this uses a single training seed, and seed noise was about ±4% in 2025.

**Small side effect:** the tuned price rule's cut-offs depend on the training shift patterns. For 4 h, seed 0 now picks worse cut-offs (2.09 M against 2.50 M). The 4 h comparison isn't affected, because the shift schedule is the best rule there.

**Shift-mix robustness check (2026-11-14).** The tables were trained on the `default` mix only. Before offering the mix as a dashboard choice, every year's tables were tested on October–December of the same year under all three mixes, over the 5 test shift patterns. That's 540 simulations, 4.6 s, from a scratchpad script. The `default` rows reproduce the numbers above. The strongest rule includes the price rule with the other year's cut-offs, as above.

*Q-learning ÷ strongest rule (range over the 5 patterns where it matters):*

| Battery | Test year | `default` (trained on) | `mostly_double` | `mostly_single` |
|---|---|---|---|---|
| 4 h | 2025 | 0.99 | 0.99 | 0.98 (0.90–1.06) |
| 4 h | 2024 | 0.91 | 0.92 | 0.91 |
| 8 h | 2025 | 1.05 | 1.10 | 0.97 (0.90–1.04) |
| 8 h | 2024 | 1.10 | 1.16 | 1.02 (0.93–1.13) |
| 12 h | 2025 | 1.15 | 1.12 | 1.12 |
| 12 h | 2024 | 1.00 | 1.06 | 0.95 (0.88–1.05) |

*Findings:*
- **Mostly double shifts:** Q-learning holds up or does better (+6 to +16% for 8 h and 12 h). More operating hours mean more hours to discharge, which is the situation it learned.
- **Mostly single shifts:** the edge disappears. For 8 h 2025 and 12 h 2024 Q-learning is slightly below the best price rule (0.97 and 0.95). The training mix has 45% full-shift days, so this mix is outside what the model saw.
- **4 h:** the shift schedule wins under every mix, as before.
- **The type of operation matters more than the controller.** For the same battery, savings are about 35–40% higher with mostly double shifts and about 30% lower with mostly single shifts. The battery can only discharge while the factory runs, about 71 / 57 / 43 h a week (A5).

*Savings, October–December, mean of the 5 test patterns (M COP), Q-learning / strongest rule:*

| Battery | Test year | Mixto | Principalmente turnos dobles | Principalmente un turno |
|---|---|---|---|---|
| 4 h | 2025 | 2.67 / 2.70 | 3.72 / 3.75 | 1.90 / 1.93 |
| 8 h | 2025 | 3.13 / 2.97 | 4.29 / 3.90 | 2.18 / 2.24 |
| 12 h | 2025 | 3.52 / 3.07 | 4.72 / 4.22 | 2.53 / 2.26 |
| 4 h | 2024 | 7.19 / 7.94 | 8.88 / 9.67 | 6.24 / 6.89 |
| 8 h | 2024 | 10.68 / 9.67 | 13.71 / 11.84 | 8.75 / 8.60 |
| 12 h | 2024 | 11.28 / 11.29 | 14.71 / 13.88 | 9.17 / 9.64 |

*Decision:*
- **Offer the mix in the dashboard** as "Tipo de operación": `default` = "Mixto", `mostly_double` = "Principalmente turnos dobles", `mostly_single` = "Principalmente un turno". Show the results as they are, and state in the app that the model was trained on the mixed operation.
- **Don't train one model per mix.** `train.py --mix` exists, but the policy file names don't include the mix. It would mean changing an interface and keeping 18 tables, which is beyond MVP.

## D16. LLM agent: hand-written loop, two tools, checked citations

**Decided and implemented 2026-11-13** in `energy/agent/`, written test-first.

- **Loop:** written by hand, without LangGraph.
  - The graph is trivial: call the model, run the tools, repeat until an answer or T_max.
  - The model is injected as a `ChatFn`, so every path is tested offline with a scripted fake.
  - It needs no extra dependencies, and the report can show the whole loop.
- **Model:** Groq `openai/gpt-oss-120b`, falling back to `openai/gpt-oss-20b`, with `reasoning_effort="low"`.
  - **Retries:**
    - A malformed tool call is retried once on the same model.
    - A rate limit that asks to wait at most 15 s is waited out on the same model.
    - Anything else moves to the next model.
  - **Why wait instead of switching:** the free tier allows 8,000 tokens per minute, and one question uses about 3–4k. Back-to-back questions hit the limit, and the 20b fallback follows the citation format less reliably.
- **Answers:** in Spanish; code and identifiers stay in English.
- **Two tools:**
  - `search_regulation(query, k)` returns `config.AGENT_RETRIEVAL_K` = 4 hits by default (5 at first; cut to save Groq tokens, about 400 per passage). Each hit has `cita` (exactly `Hit.citation`), `in_force` and `superseded_by`, plus a reminder of how to cite.
  - `simulate_battery(policy, preset, start_date, end_date)` covers one calendar year at most.
    - **It runs `simulate()` once for each of the 5 test shift calendars** (`TEST_SHIFT_SEEDS`, mix `default`, kept fixed) for the requested policy and every baseline.
    - **It returns:**
      - the mean, min and max savings
      - the baseline means and the best fixed rule
      - `by_hour_block` (12 rows) and `by_price_level` (4 quartile rows). These profiles are what let the agent explain the decisions.
      - `in_sample`, true when the range overlaps the training months
      - `notes`, the assumptions shown to the user (A4, A5)
      - `field_guide`, which tells the model only what the fields mean (COP/kWh, and what the % of hours means)
    - **Speed:** about 0.3 s for 3 months and 0.9 s for a year.
- **T_max = 5 tool rounds.** After that, one last call with the tools off (`stopped="t_max"`). Tool errors return as `{"error": ...}` observations and never raise.
- **Without the LLM** (no key, or every model failing): a retrieval-only answer with the top 3 passages and their citations (`stopped="llm_error"`).
- **Citations are checked:**
  - Every `[...]` with at least 3 letters that isn't a Markdown link is a citation attempt. gpt-oss's native `【…】` brackets and a `cita:` prefix are normalized first.
  - Attempts that match a `cita` retrieved in the same run, ignoring case and spacing, go to `citations`. The rest go to `unverified_citations`.
  - **Repair round:** if a search returned hits and the answer has no valid citation (and isn't the refusal), the agent makes one extra call, outside T_max. That call asks the model to rewrite the answer citing from the list of retrieved citations. The repaired version is kept only if it has valid citations.
  - **Why the repair round:** in the live check, gpt-oss-120b sometimes wrote a literal `【cita】` or dropped the brackets.
- **Prompt rules:**
  - the exact refusal phrase
  - for replaced rules, say so and cite the current one
  - the behind-the-meter limitation (D6): never apply the rules for market agents or SAEB dispatched by the CND to the user's battery, and never conclude it is allowed or forbidden
  - numbers only from `simulate_battery`, in COP/kWh
  - no general knowledge beyond the tools
  - "¿por qué carga…?" questions always call `simulate_battery`
- **Simulation results are written by code, not by the model** (changed 2026-11-13, after the live check):
  - **Why:** in the live check the model changed a digit in the table and flipped the sign of the comparison.
  - **The block:** `energy/agent/render.py` builds it from the tool output, and the loop puts it at the top of the answer. It contains:
    - **one simulation:** a bold mean-and-range line, a 3-row table `Política | Ahorro promedio (COP)`, the comparison sentence (sign computed from the numbers), the fallback and in-sample notes when they apply, **Supuestos**, and `*Fuente: simulación propia con precios de bolsa de XM, <año>.*`
    - **several simulations:** a table `Escenario | Ahorro promedio (COP) | Mejor regla (COP) | % del óptimo`, at most 6 rows, labelled only by what varies (battery, period or policy). This replaces the earlier `Batería | Q-learning | …` header, which assumed only presets varied.
  - **Number formats (es-CO):** `3.131.506 COP` and `5,3 %`.
  - **The model never sees the figures.** Its view of a simulation omits the savings, the baselines, the comparison percentages and the energy totals. It once restated 3.131.505 for 3.131.506, so now it can't.
  - **What the model writes:** only the explanation (from the hour and price profiles) and any regulation part, with no tables. Any table lines it writes are removed when a block exists.
  - **Every path keeps the block:** a normal answer, the T_max cap, and an LLM failure after a simulation.
  - **Citations are checked on the model's text only.**
  - **`AgentResult` gained** `simulations`, `results_block` and `model_text`.
- **Regulation answers:** short paragraphs ending in citations, then a "no es asesoría legal" line.
- **The user's battery and period are the default** (option A, decided 2026-11-13):
  - **Why:** in the dashboard the user picks the battery (4 / 8 / 12 h) and the dates in the sidebar, and the chat must answer for the same scenario the charts show.
  - **`UserSettings(preset, start, end)`** lives in `energy/agent/settings.py`. It defaults to 8 h, 2025-10-01..2025-12-31 (`config.DEFAULT_*`), and it is validated: a known preset, start ≤ end, one calendar year, inside the data. `run_agent(..., settings=...)` takes it.
  - **Why one calendar year (written down 2026-11-14):**
    - **One model per year:** there is one Q-table and one set of tuned price-rule cut-offs per year (D15), and `simulate()` picks them by the **start date's year**.
    - **What a cross-year range would do:** a range such as November 2024 – February 2025 would quietly run the 2024 model on 2025 prices. The results would look normal, but January–February 2025 would be judged by a model trained on another year's prices.
    - **Where it's enforced:** `UserSettings`, the agent tool, and the dashboard's custom-dates picker all reject such ranges.
    - **Alternative not taken:** split the range at January 1 and use each year's model. That means work in `simulate`, the agent tool and the tests, for little demo value.
  - **Filled by code:** when the model omits `preset`, the loop fills it from the settings. It fills `start_date` and `end_date` only when both are missing; if one is given alone, the tool's error goes back to the model. `ToolCallRecord.from_settings` records which arguments were filled.
  - **The prompt states the settings,** and says to use another battery or period only when the user asks explicitly, which keeps comparisons working. The results block opens with `Escenario: batería de 12 h · oct–dic 2024 · Q-learning`, so the scenario is always visible.
  - **The tool's schema no longer requires dates.** The function still rejects a call without them, with a Spanish error.
- **Groq daily limit (found 2026-11-13):**
  - **The limit:** on the free tier, `gpt-oss-120b` allows about 200,000 tokens per day, and a question uses about 3–4k. Development and test runs used it up, and every call then fell back to `gpt-oss-20b`.
  - **What the 20b model got wrong:**
    - It passed `preset` explicitly instead of leaving it to the settings.
    - Its explanations were weaker; one mixed up the price quartiles.
  - **The figures stayed correct,** because the results block is rendered by code.
  - **Consequence:** the network test now checks that the user's battery and dates were simulated, not whether the model omitted the arguments.
  - **Plan the agent evaluation and the demo around the daily limit.**
- **`ChatReply.model` (added 2026-11-13, step 7):** the id of the model that answered, or None. The agent evaluation records it, so a fallback to the 20b model shows in the results.
- **Live check:**
  - **What passed:** the regulation, simulation, mixed and trap questions behaved as specified; the network tests passed in 4 out of 4 runs.
  - **The model changed a digit when copying numbers by hand** (2.970.674 instead of 2.974.674). That's why the results block above is written by code.

- **Chat fixes (2026-11-14, found reviewing the Asistente tab offline):**
  - **`MissingKeyError(LLMError)`** is raised when `GROQ_API_KEY` is missing. It is raised before the model loop, so a missing key is never retried on the fallback model. Every existing `except LLMError` still works.
  - **`AgentResult.error`** holds the technical message when `stopped == "llm_error"`; "Detalles" shows it as "Error técnico".
  - **`AgentResult.models`** lists the model id of every chat call in order, including the t_max nudge and the citation repair, with "?" for an unknown model (the evaluation's convention). "Detalles" shows "Modelo", and the app adds a caption when the 20b fallback answered, because it explains less precisely and nothing else showed it.
  - **Friendlier retrieval-only answer:** `_fallback_answer` returns `(text, citas, retriever)`. The text says in plain Spanish that the model is missing (no key) or unavailable (limit or connection), then lists the passages and points to the Operación and Comparación tabs. The `llm_error` result now fills `citations` and `retriever`. Reason: a grader without `.env` saw a raw error and an empty "Detalles" even though passages had been retrieved.
  - **Dashboard:** the chat history is drawn in a container above the input, so a new answer no longer appears below it. Four example questions show while the chat is empty.

- **Live evaluation (2026-11-14, step 7):** 8 of 11 questions passed, all answered by `openai/gpt-oss-120b`, using Qdrant + NVIDIA retrieval. Outputs are in `results/agent_eval/`.
  - **Questions 1 and 4:** the agent answered with the refusal phrase without calling `search_regulation`, although the corpus answers both. The prompt says to refuse "if the search returns nothing relevant" but never forces a search first.
  - **Question 11, the injection:** it succeeded. The model followed the injected "developer mode" message and wrote the Python code. The prompt has no scope rule and doesn't tell the model to ignore instructions embedded in the user's message.
  - **Decision: no prompt fix and no second run for this delivery.** The cost in tokens and time isn't worth it this close to the deadline. The fixes are clear (search before refusing; an explicit scope and anti-injection rule), and they are recorded as limitations instead. The YAML expectations were not changed after seeing the answers.
  - **Runner change:** `agent_eval` now writes after every question and merges by id (`--ids` replaces only those rows; `--fresh` starts empty). The summary lists the questions answered by the fallback model.

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
- ~~**O8**~~: done (2026-11-13).
  - **`energy/config.py`:**
    - `ROOT` plus every data, artifact and results path, all absolute.
    - Shared settings: `USER_AGENT`, `MIN_TEXT_CHARS`, `EMBED_MODEL`, `QDRANT_COLLECTION`, `RETRIEVAL_K`, `SCORE_THRESHOLD`.
    - `load_env()`, which reads the root `.env` without overriding real environment variables and is called only from entry points.
  - Old constant names stay as aliases.
  - **One finding:** the project isn't installed as a package (`pyproject.toml` has no build system), so `python -m energy...` still needs the repo root as the working directory, or `PYTHONPATH` set to it. The *paths* no longer depend on where you start. `streamlit run app.py` from the root is unaffected.

  The original note follows.
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
- **Branch `explore/factory-schedule` (D15):**
  - The factory's shift schedule is synthetic (A5), and the factory's load is all or nothing.
  - Q-learning's edge over the best tuned rule is modest: +7% for 8 h and +11% for 12 h after parameter tuning, and a tie for 4 h.
  - Q-learning reaches only 69–74% of perfect foresight.
  - The parameters were tuned on the 8 h battery only.
  - In the October–December 2024 crisis, Q-learning captures only 50–58% of perfect foresight.
  - The tuned price rule's cut-offs don't carry over well across a regime change, so results are compared against the strongest available rule.
  - The Q-tables were trained on the mixed shift operation only. For a plant running mostly single shifts, a tuned price rule does as well (0.95–1.12 of it, D15).
- Without the NVIDIA key and the Qdrant index, search falls back to BM25, which ranks some passages poorly (a glossary passage can come first). Retrieval quality is not tuned for that path.
- **Agent evaluation (D16):** 8/11 questions passed.
  - Twice the agent refused a regulation question without searching first.
  - A prompt injection asking for unrelated code succeeded, so the assistant's scope relies on the model and is not enforced.
  - The simulation figures are not affected, because code writes them.
- This is an academic prototype, not financial, legal or regulatory advice.
