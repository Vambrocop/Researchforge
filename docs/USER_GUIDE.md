# ResearchForge — User Guide

A practical guide for researchers using ResearchForge to analyze data. For the
project overview and install, see [README.md](../README.md); for the engine
internals and how to add methods, see [CLAUDE.md](../CLAUDE.md).

ResearchForge takes a dataset, figures out which analyses are *feasible*,
recommends them with an honest rigor verdict and a methodology scorecard, runs the
one(s) you pick, and saves the code, figures, tables, and a report. It spans ~225
analyses across ~33 method families — the value is breadth plus *honest* guidance
about what actually fits your data.

> The analysis UI surfaces text in **Chinese**; this guide and the developer docs
> are in English.

---

## 1. The workflow at a glance

```
your data.csv
   │  profile        →  type / structure / quality fingerprint
   ▼
recommendations     →  🟢🟡🔴 rigor light + 6-dim methodology score + ⚠ biases
   │  you pick one (optionally override column roles via --config)
   ▼
run                 →  CSV tables + PNG figures + report.md + runnable code
```

Three ways to drive it: the **CLI**, the **web UI**, or the **Python API**.

---

## 2. CLI

```bash
# 0. Front door — health, scale, what to improve. (Run this first in a repo.)
py -3 -m researchforge.cli status

# 1. Profile + see the top recommendations (goal-aware).
py -3 -m researchforge.cli recommend data.csv
py -3 -m researchforge.cli recommend data.csv --goal causal --top 8

# 2. Inspect an analysis's configurable parameters before running.
py -3 -m researchforge.cli params ols_regression

# 3. Run a chosen analysis by id; outputs land in outputs/<id>_<timestamp>/.
py -3 -m researchforge.cli run data.csv ols_regression

# 4. Override the engine's substantive defaults (column roles / params) via JSON.
py -3 -m researchforge.cli run data.csv ols_regression \
    --config '{"outcome":"progression","predictors":["bmi","bp","s5"]}'

# 5. Web UI (upload → recommend → optionally clean → run → download zip).
py -3 -m researchforge.cli web      # then open http://127.0.0.1:8000
```

On Windows use `py -3` (not bare `python`) and set `PYTHONUTF8=1` for clean output.

Other subcommands: `design` (generate a randomized experimental layout *before* you
have data — `rcbd` / `factorial` / `latin_square`), `scorecard`, `benchmark`,
`discover` / `candidates` / `promote` (self-evolution queue). `--help` lists all.

---

## 3. Python API

```python
from researchforge.profiler import profile_dataset
from researchforge.recommender import select_top
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis

fp = profile_dataset("data.csv")                       # 1. profile
for r in select_top(fp, goal="relate", top=6):         # 2. recommend
    print(r.rigor.light, r.entry.id, "—", r.rigor.note, "| score", r.score.overall)

entry = Catalog.load().by_id("ols_regression")          # 3. run
res = run_analysis(fp, entry, config={"outcome": "progression"})
print(res.summary)
print(res.output_dir, res.files)                        # tables / figures / report.md
```

---

## 4. Reading the recommendations

Each recommendation carries three honest signals:

- **🟢🟡🔴 rigor light** — 🟢/🟡 are feasible on *your* data as-is; 🔴 means the
  method needs an informed override (e.g. you must name an instrument) before it
  makes sense. The `note` says why.
- **6-dimension methodology scorecard (0–100)** — popularity, publishability,
  aesthetics (signature figures), difficulty (interpretation/assumption burden — a
  *cost*), fit (suitability to *this* dataset), novelty. These are deterministic,
  offline editorial priors plus the data-specific rigor verdict — **not** live
  trend metrics, and surfaced as such.
- **⚠ bias disclosure** — each method spells out its key assumptions and biases in
  plain language, both in the catalog and again in the run summary.

Use `--goal` to focus the menu: `compare`, `relate`, `causal`, `predict`,
`describe`, `design`, `spatial`, `reduce`, `survival`, … (see
`researchforge/recommender/goals.py` for the full list).

---

## 5. Configuring column roles — the most important knob

The engine auto-detects column roles, but **its defaults are conventions, not
mind-reading**. The single most useful override is the **outcome** (and predictors)
for any regression/ML/effect-size method.

**Why this matters (a real example).** On the diabetes dataset, the disease-
progression target is integer-valued, so the profiler types it as a `count`
column. The regression convention is "first *continuous* column = outcome", so OLS
would default to predicting `age` — a valid model, but not the one you want. Set it
explicitly:

```bash
py -3 -m researchforge.cli run diabetes.csv ols_regression \
    --config '{"outcome":"progression","predictors":["bmi","bp","s1","s5"]}'
```

Run `cli params <id>` to see exactly which keys an analysis accepts; passing an
unknown key or a non-existent column produces a **non-blocking ⚠ warning** in the
run summary (the analysis still runs on its auto defaults) — so a typo never fails
silently. Common keys: `outcome`, `predictors`, `group`, `treatment`, `unit`,
`time`, `x`/`y` (coordinates), `value`, `text`. Per-method keys are documented in
[docs/loop-decisions.md](loop-decisions.md).

---

## 6. What's inside (method families)

A rough map of the ~33 families (see `cli recommend` for what fits *your* data):

- **Regression & GLM** — OLS, logistic, Poisson / negative-binomial / zero-inflated
  counts, quantile, regularized (lasso/ridge/elasticnet), diagnostics, relative
  importance.
- **Causal inference** — DiD (incl. staggered / Callaway–Sant'Anna & robustness),
  RDD (sharp/fuzzy), event study, PSM/IPW, sensitivity (Rosenbaum bounds, E-values),
  and modern causal ML (double/debiased ML, causal forests) via optional libraries.
- **Econometrics / panel** — fixed/random effects, Mundlak/Hausman, dynamic-panel
  GMM, spatial panel.
- **Survival** — Kaplan–Meier, Cox PH (+ time-varying / stratified / diagnostics),
  parametric AFT, competing risks, RMST.
- **Spatial / GIS** — Moran's I (global / local / bivariate), Geary, Getis-Ord,
  kriging/IDW, GWR, Ripley's K, join-count, SKATER regionalization, spatial
  regression.
- **Time series** — ETS/Holt-Winters, ARIMA, state-space, GARCH, changepoints,
  cointegration/VECM, ACF/PACF & Ljung-Box & Hurst diagnostics.
- **Multivariate & ordination** — PCA, factor analysis, MANOVA, LDA, canonical
  correlation, PCoA/NMDS/CA/MCA, RDA.
- **Psychometrics & measurement** — Cronbach's α, McDonald's ω, ICC, Cohen's/
  Fleiss' κ, Bland–Altman, IRT/Rasch family, SEM, latent class/profile,
  moderated-mediation, serial/parallel mediation, moderated moderation.
- **Nonparametric & classical** — t-tests, ANOVA family (one-way/ANCOVA/RM,
  RCBD/Latin-square/split-plot), Kruskal–Wallis/Friedman/Mann–Whitney, chi-square/
  Fisher/McNemar, correlation suite, distribution fitting, effect sizes
  (Cohen's d, Hedges' g, Cliff's δ).
- **Machine learning & interpretability** — random forest, gradient boosting,
  XGBoost, SVM, BART, conformal & quantile prediction intervals; PDP, SHAP, ALE,
  Friedman's H interaction, global surrogate trees; t-SNE, dimensionality reduction,
  network science.
- **Text mining** — LDA topic models, TF-IDF keywords, sentiment (optional backend).
- **Specialized** — MCDA (TOPSIS/VIKOR/PROMETHEE/AHP/entropy), DEA/SFA efficiency,
  configurational/QCA, finance (VaR/EVT), ecology, microbiology, survey methods,
  experimental design / DoE.

---

## 7. Reading the outputs

Every run writes to `outputs/<id>_<timestamp>/`:

- **`report.md`** — the headline summary + the figures/tables, in one place.
- **CSV tables** — estimates, coefficients, diagnostics.
- **PNG figures** — the method's signature plots (English labels).
- **`analysis_code.py`** — runnable code reproducing the core computation.
- The Python `RunResult` also exposes `.summary` (Chinese, with ⚠ disclosures),
  `.estimates` (a float dict), `.files`, and `.output_dir`.

**Read the ⚠ lines.** They are not boilerplate — they state the assumptions the
result rests on (e.g. parallel trends for DiD, the spatial-weights choice for
Moran's I, "explains the model, not causation" for SHAP).

---

## 8. Honest-disclosure philosophy (what to trust)

- **Honest skip / degrade.** If a method doesn't fit your data, or an optional
  backend (R, or a heavy Python library) is missing, the run says so (a "跳过"
  message) and points to an alternative — it does **not** fabricate output.
- **Zero results are reported as zero**; uncertainty is flagged, not hidden.
- **Optional backends.** A handful of gold-standard methods delegate to R packages
  or heavy Python libs; each degrades gracefully when absent. Install the heavy
  Python set with `pip install -e ".[full]"`. R is never fetched at runtime.

---

## 9. Limitations & good practice

- **Set `outcome`/roles explicitly** for any modeling task (see §5) — the auto
  defaults are conventions; an integer target, an ID-like column, or an unusual
  column order can make the default pick something you didn't intend.
- **The scorecard is editorial**, not a live popularity feed — treat it as a
  curated prior, not evidence.
- **Recommendations are feasibility + rigor, not a substitute for judgment.** Read
  the disclosures; a 🟢 light means "runnable and sound on this data shape", not
  "this is the right question for your study".
- **It's research-grade and evolving**, not a finished 1.0 product — outputs should
  be read as you'd read any analysis you ran yourself.
