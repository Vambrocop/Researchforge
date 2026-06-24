# ResearchForge

**A methodology melting-pot engine.** Point it at a dataset and it profiles the
data, figures out which analyses are actually feasible, recommends them with an
honest rigor verdict and a methodology scorecard, then runs the one you pick and
saves the code, figures, tables and a report.

Data in → auto profile (type / structure / quality) → recommend feasible
analyses (🟢🟡🔴 rigor + 6-dim methodology scorecard + ⚠ bias disclosure) → run →
code / figures / tables / report.

> **North star:** more methods + smarter auto-selection. The catalog currently
> spans ~225 analyses across ~33 method families (causal, regression, survival,
> spatial, panel/econometrics, time-series, multivariate, psychometrics,
> nonparametric, machine learning, and more — see [What's inside](#whats-inside)).

ResearchForge is research-grade and actively evolving; it is not a finished
1.0 product. The method recommendation/run UI surfaces text in Chinese; this
README and the developer docs are in English.

---

## Install

Requires **Python 3.12+**. From source:

```bash
git clone <repo-url> researchforge
cd researchforge
pip install -e .
```

That installs the core, pure-Python stack (numpy, pandas, scipy, statsmodels,
scikit-learn, matplotlib, FastAPI/uvicorn for the web UI, etc.). Many analyses
run on this core alone.

**R is optional.** A handful of methods can delegate to gold-standard R packages
through an optional R bridge, but every R-backed method degrades gracefully:
if R (or the package) is missing, the engine falls back to a pure-Python path or
tells you honestly which dependency is needed. R is never fetched at runtime.

Some heavier methods depend on extra Python libraries (survival, panel
econometrics, networks, IRT, latent-class, ordination, GARCH/changepoint, SEM,
modern causal ML). Install them all with the `full` extra:

```bash
pip install -e ".[full]"
```

Each of those methods also degrades gracefully when its optional library is
absent — the rest of the catalog keeps working.

### Windows note

On Windows the bare `python` command is often shadowed by a Microsoft Store stub.
Use **`py -3`** instead (and set `PYTHONUTF8=1` for clean UTF-8 output). All the
command examples below use `py -3`; on macOS/Linux substitute `python`.

---

## Quickstart (CLI)

```bash
# Live project front-door: health score, scale, what to improve. Run this first.
py -3 -m researchforge.cli status

# Profile a dataset and see the top recommended analyses (goal-aware).
py -3 -m researchforge.cli recommend data.csv
py -3 -m researchforge.cli recommend data.csv --goal causal --top 8

# Run a chosen analysis (id from the catalog) and save its outputs.
py -3 -m researchforge.cli run data.csv did

# Override the engine's substantive defaults (column roles / params) via JSON.
py -3 -m researchforge.cli run data.csv ols --config '{"outcome":"yield","predictors":["rain","fert"]}'

# Launch the web UI, then open http://127.0.0.1:8000
py -3 -m researchforge.cli web
```

Other subcommands: `design` (DoE advisory — generate a randomized experimental
layout before you have data: `rcbd` / `factorial` / `latin_square`), `scorecard`
(project self-assessment), `benchmark` (engine quality on known cases),
`discover` / `candidates` / `promote` (self-evolution candidate queue), and
`ingest` (process the skills inbox). Run `py -3 -m researchforge.cli --help` for
the full list.

### Web UI

`py -3 -m researchforge.cli web` starts a FastAPI app (default port 8000):
upload a CSV → see the data fingerprint and recommendations (with rigor lights
and the scorecard) → optionally clean → run an analysis → download the outputs
as a zip.

---

## Quickstart (Python API)

```python
from researchforge.profiler import profile_dataset
from researchforge.recommender import select_top   # or: recommend
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis

# 1. Profile the data.
fp = profile_dataset("data.csv")
print(fp.n_rows, fp.n_cols, "panel:", fp.is_panel, "timeseries:", fp.is_timeseries)

# 2. Recommend the top feasible analyses (optionally focused on a research goal).
for rec in select_top(fp, goal="causal", top=6):
    print(rec.rigor.light, rec.entry.id, rec.entry.method, "—", rec.rigor.note)
    print("   scorecard:", rec.score.overall)

# 3. Run a chosen analysis and save its outputs.
entry = Catalog.load().by_id("ols")
result = run_analysis(fp, entry, config={"outcome": "y"})
print(result.summary)
print(result.output_dir, result.files)
```

`recommend(fp)` returns the full ranked menu; `select_top(fp, goal=..., top=...)`
returns a focused shortlist. `run_analysis(fp, entry, config=...)` accepts an
optional `config` dict of substantive overrides (column roles, anchors, params);
omit it and each analysis still runs on its automatic defaults.

---

## What's inside

The catalog holds ~225 analyses, grouped by methodology family. Highlights by
domain:

- **Causal inference** — difference-in-differences (incl. modern staggered /
  Callaway–Sant'Anna-style and DiD robustness), regression discontinuity, event
  study, sensitivity analysis (Rosenbaum bounds, E-values), and modern causal ML
  (double/debiased ML, causal forests) via optional libraries.
- **Regression & GLM** — OLS, logistic, count models (Poisson / negative
  binomial / zero-inflated), regularized regression, regression diagnostics,
  relative importance.
- **Econometrics / panel** — fixed/random effects, panel estimators, agricultural
  & resource economics methods.
- **Survival** — Kaplan–Meier, Cox PH, parametric and extended survival models.
- **Spatial / GIS** — spatial autocorrelation, kriging/interpolation, geographically
  weighted regression (GWR), soil/compositional methods.
- **Time series** — ARIMA/ETS-style forecasting, state-space, GARCH volatility,
  changepoint detection.
- **Multivariate & ordination** — PCA/factor analysis, MANOVA, ordination
  (PCoA/NMDS/CA/MCA).
- **Psychometrics & measurement** — reliability/agreement (Cronbach's α, ICC,
  Cohen's κ), IRT/Rasch, SEM, latent-class / mixture models, conditional-process
  (moderated-mediation) analysis.
- **Nonparametric & classical statistics** — t-tests, ANOVA family, chi-square /
  contingency tables, distribution fitting, rank-based tests.
- **Machine learning (predictive)** — gradient boosting, SVM, BART, conformal
  prediction, dimensionality reduction, network science / community detection, model interpretability (PDP/SHAP/ALE).
- **Specialized** — MCDA (multi-criteria decision analysis), DEA/SFA efficiency,
  configurational/QCA, finance, ecology, microbiology, experimental design / DoE.

### Methodology scorecard

Every recommendation carries a deterministic, offline 6-dimension scorecard
(0–100): **popularity**, **publishability**, **aesthetics** (strength of its
signature figures), **difficulty** (interpretation/assumption burden — a cost),
**fit** (how well it suits *this* dataset), and **novelty**. These are editorial
priors plus the data-specific rigor verdict, surfaced honestly as such — not
live trend metrics.

### Honest-disclosure philosophy

- **🟢🟡🔴 rigor lights** — green/yellow are feasible on your data; a red light
  means the method needs an informed override before it makes sense.
- **⚠ bias disclosure** — each analysis spells out its key assumptions and
  biases in plain language alongside the results.
- **Honest degrade** — when an optional backend (R or a heavy Python library) is
  missing, the engine falls back or says so, rather than failing silently or
  faking output. Zero results are reported as zero; uncertainty is flagged.

---

## Project status

Current engine version: **v0.8.0** — the Web UI MVP (upload → recommend +
scorecard → run → report) has shipped. The next milestone is **v0.9** (hardening
on real dirty data, packaging/docs, a machine-readable config schema). See
[docs/roadmap.md](docs/roadmap.md) for the full version ladder and the
melting-pot roadmap.

This is research-grade software under active development. There are no published
benchmarks here beyond the in-repo `benchmark` self-check; treat outputs as you
would any analysis you run yourself — read the disclosures.

---

## Development

```bash
pip install -e ".[dev]"          # adds pytest + pytest-xdist
py -3 -m pytest -q               # run the test suite
py -3 -m pytest -m "not slow" -q # fast loop (skips heavy / R-backed tests)
py -3 -m pytest -n 2 -q          # parallel full run (use -n 2, not -n auto)
```

Adding an analysis is a three-part change: a `@register("<id>")` handler in
`researchforge/executor/branches/<family>.py`, a catalog entry in
`researchforge/catalog/entries/*.yaml`, and a test. See `CLAUDE.md` for the
engine architecture and conventions.
