---
name: add-analysis
description: Scaffold and wire a new ResearchForge analysis end-to-end (executor branch + catalog entry + test + double-review + local commit), following the project constitution in CLAUDE.md. Use when adding a new method/analysis id to the engine, e.g. "/add-analysis rdd" or "add a regression-discontinuity analysis".
---

# Add a ResearchForge analysis

Codifies the standard "加一个分析" flow from `CLAUDE.md`. One analysis = ① an executor branch, ② a catalog entry, ③ a test. Follow the steps in order; don't skip the review or full-suite gate.

## 0. Decide the id + family
Pick a snake_case `id` (e.g. `rdd`). Note its family (statistics/causal/sem/meta/ml/spatial/efficiency/configurational/...) and whether it does **real statistical inference** (→ needs review) or is deterministic (reuses already-reviewed helpers → verify by test).

## 1. Executor branch — `researchforge/executor/run.py`
Add `elif entry.id == "<id>":` in the dispatch chain. Copy the conventions from a sibling branch; do NOT reinvent:
- `cfg = config or {}` is already in scope. Read user overrides via `cfg.get("<key>")`; fall back to an auto default that **still runs standalone**. Document new keys in `docs/loop-decisions.md`.
- **Outcome/predictors**: regression-family convention = first continuous = outcome; override via `cfg["outcome"]`/`cfg["predictors"]`.
- **R backend** (if used): go through `researchforge/executor/rbridge.py` — `rbridge.r_available()` + `rbridge.r_package_available("<pkg>")`, else degrade honestly (point to a pure-Python alternative). Guard column names before R formula interpolation: `re.fullmatch(r"[A-Za-z.][A-Za-z0-9._]*", c)`. Pass arbitrary names as `d[["name"]]` vectors when possible (injection-safe). Write the temp CSV into the output dir `d`; delete it in `finally`. Never fetch/run R code from the network at runtime.
- **Artifacts**: CSV + PNG (matplotlib Agg; **English figure labels** — default fonts have no CJK). Fill the `estimates` dict. Write a Chinese `summary` with ⚠ bias/assumption disclosure. Wrap plotting in try/except so a missing lib doesn't abort.
- **profiler gotchas**: integer all-distinct columns profile as `id` (not count/continuous); named-detection should accept `id`. Duration columns may be the `time_col` — don't exclude it for survival-type methods. Binary 0/1 columns are often `treatment_candidates` — don't blanket-exclude them if the method needs them as factors.
- Degrade honestly (clear Chinese message) when preconditions/packages/columns are missing.

## 2. Catalog entry — `catalog/entries/<family>.yaml`
Add an entry: `id, method, domain, family, goal, description, preconditions, produces, executor_ref, biases`. Gate recommendation with the right `preconditions` (add a `Precondition` field + a `recommender/match.py` matcher if no existing field fits — mirror `requires_geo`/`requires_effect_sizes`) so it doesn't pollute every dataset's menu. Put real assumptions in `biases`.

## 3. Test — `tests/test_<id>.py`
- A demo dataset under `data/demo_<id>.csv` (or build it in the test). Use decimal/repeated values so continuous cols aren't profiled as `id`.
- Assert it runs + key estimates are sane. For R/optional-dep methods, `@pytest.mark.skipif(not <available>)`. Add a no-dependency honest-degrade test (wrong/missing data → clear failure message).

## 4. Verify → review → commit
1. `PYTHONUTF8=1 py -3 -m pytest tests/test_<id>.py -q >/tmp/t.log 2>&1; echo EXIT=$?` — check the exit code, don't `| tail` (it masks pytest's code). Smoke via CLI: `py -3 -m researchforge.cli run data/demo_<id>.csv <id>`.
2. **If the method does real inference**: dispatch the `inference-reviewer` subagent (reviewer ≠ builder) on the branch. Apply must-fix items; adopt disclosure polish; you may override a reviewer finding WITH EVIDENCE (an empirical probe) — surface that to the user. Deterministic methods skip review with a one-line justification.
3. Run the **full suite** (`PYTHONUTF8=1 py -3 -m pytest -q`) incl. `test_catalog_consistency` (every catalog id must have a branch). It is R-heavy (~3–5 min) — run it in the background.
4. Local commit only. Multi-line Chinese messages: Write to a temp file then `git commit -F <file>` (the Bash tool is POSIX sh — do NOT use PowerShell `@'...'@`). **Never push** until the user says "今天 ok"; remind them via `git log origin/main..HEAD` if there are unpushed commits.

## Key files
`executor/run.py` (dispatch + branches + helpers) · `executor/rbridge.py` (R bridge) · `catalog/entries/*.yaml` · `catalog/schema.py` (Precondition/AnalysisEntry) · `recommender/match.py` (preconditions) · `recommender/scoring.py` (method score card) · `profiler/` · `web/`. Read `CLAUDE.md` first.
