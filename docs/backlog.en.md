# ResearchForge Backlog (Opus Historical Branch Hardening Scan, 2026-06-12)

> Source: Opus supplement sweep of merged executor branches. Priority P1>P2>P3.
> Each item = issue + one-line reason + affected files.

## P1 — Fix ASAP (Silent error results / correctness risk) ✅ Completed (2026-06-12, Opus dual-review LGTM)

- [ ] **First continuous column treated as dependent variable in `_regression`** (ols/panel_fixed_effects/did): silently picks wrong DV on multi-continuous-column data (same class of bug as the one fixed in random_forest). → Add transparent target column selection + more robust selection method. `executor/run.py`
- [ ] **`group_comparison` may take `unit` id as grouping variable**: doesn't exclude `unit_col`/`time_col`; high-cardinality id as groups → dozens of singleton groups with meaningless ANOVA. `executor/run.py`
- [ ] **`logistic_regression` takes first binary as outcome**: but binaries are usually treatment flags (`treatment_candidates` = all binary columns) → treats treatment as outcome in regression. `executor/run.py`
- [ ] **`iv_regression` in directory but no executor**: falls to else placeholder, runs empty, yet still recommended as viable → misleads, damages trust. → Plug in executor, or gate out from recommendations until ready. `executor/run.py`

## P2 — Should Fix

- [ ] Each branch **lacks transparency**: doesn't state in report "which column chosen as outcome/target" (only RF has notes). → Add one line in summary "dependent variable/target column selection". `executor/run.py` + `_report`
- [ ] **Zero-explanatory-variable regression** silently fits intercept model (`~ 1`) → should warn "no available explanatory variables". `executor/run.py`
- [ ] **did treatment variable** drawn from "all binary columns" → causal estimate may hang wrong variable. → Tighten treatment detection (group×period interaction/name heuristics). `profiler/profile.py` + `run.py`
- [ ] **descriptive_stats lacks high-cardinality guard** (`describe(include="all")` + min_rows 1) → wide tables slow and unreadable; profiler already detects high_cardinality but doesn't use it. `executor/run.py`
- [ ] **estimates key / artifact naming inconsistent** (`feature_importance.png` singular vs `feature_importances.csv` plural; key conventions vary) → downstream (benchmark/reporting) needs predictability. `executor/run.py`

## P3 — Nice-to-have

- [ ] Add **executor-level unit tests** for logistic/group_comparison/arima/random_forest (reach kmeans coverage standard, especially result selection edge cases). `tests/`
- [ ] **group_comparison empty groups cause errors inside scipy** (no per-branch try/except, inconsistent with other branches). `executor/run.py`
- [ ] **infer_kind**: arbitrary two-value text→binary, all-unique text→id, amplifies the above selection bugs. `profiler/types.py`
- [ ] **arima hardcodes order=(1,1,1) + fixed 10 periods**, no stationarity check. → Use auto_arima/ADF for order selection. `executor/run.py`
