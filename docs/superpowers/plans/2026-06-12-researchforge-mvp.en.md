# ResearchForge MVP Implementation Plan (Staged)

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans`. Steps use `- [ ]` checkboxes to track.

**Goal:** Build ResearchForge skeleton and punch through **one agricultural econometrics vertical cut**: scan local data folder → profiling + quality diagnosis + basic cleaning → recommend analyses (with 🟢🟡🔴 rigor review + informed override) → you select → call `empirical-analysis-python` to execute → produce and save code/figures/tables/reports to `outputs/`.

**Architecture:** Python package `researchforge`, three layers Profiler → Recommender (rules + agent) → Executor, plus catalog registry and ingestion. Cross-platform (Win/macOS/Linux, `pathlib`, no hardcoded drive paths). Soft reasoning via Claude Code; bulk grunt work delegated to cheap model subagents (foreman mode). **Each stage end Fable 5 does one code review** (correctness/accuracy/optimization).

**Tech Stack:** Python 3.11+, pandas, openpyxl, pydantic v2, pyyaml, pytest. Executor reuses `empirical-analysis-python` skill. setup uses cross-platform `scripts/setup.py`.

**Staged execution discipline:** One stage at a time → TDD (write failing test → implement → pass → commit) → **Fable 5 review at stage end** → next stage. Stages 2–6 expand to full TDD tasks **when entering that stage** (this plan gives target/files/acceptance first, avoids guessing huge amounts of code at once — deliberate "refine per stage", not placeholder).

---

## File Structure (Determines Decomposition Boundary)

```
researchforge/                 # Engine package
  __init__.py                  # __version__
  profiler/
    __init__.py
    fingerprint.py             # DataFingerprint / ColumnInfo (pydantic models)
    types.py                   # Column type inference
    profile.py                 # profile_dataset(path) -> DataFingerprint
    quality.py                 # Data quality diagnosis
    scan.py                    # Scan data folder
  cleaning/__init__.py  plan.py    # Stage 2
  catalog/__init__.py  schema.py  registry.py  entries/ag_econ.yaml  # Stage 3
  recommender/__init__.py  match.py  recommend.py  rigor.py          # Stage 4
  executor/__init__.py  run.py                                       # Stage 5
  ingestion/__init__.py  ingest.py                                   # Stage 6
  cli.py                       # CLI entry point
tests/                         # pytest, parallel to package
scripts/setup.py               # Cross-platform dependencies/skill install
pyproject.toml
outputs/                       # Run artifacts (gitignore; code/figures/tables/reports)
data/                          # Local data (gitignore raw)
```

---

## Stage 0 — Repository Skeleton + Package + Cross-platform Setup + Smoke Test

**Goal:** Can `pip install -e .`, can `import researchforge`, have runnable tests and CLI `--version`.

### Task 0.1: pyproject + Package Skeleton + Version Smoke Test

**Files:**
- Create: `pyproject.toml`
- Create: `researchforge/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: Write failing test** `tests/test_smoke.py`
```python
import researchforge


def test_version_exposed():
    assert isinstance(researchforge.__version__, str)
    assert researchforge.__version__
```

- [ ] **Step 2: Run confirm fail** `python -m pytest tests/test_smoke.py -q` → Expected: FAIL (ModuleNotFoundError: researchforge)

- [ ] **Step 3: Minimal implementation**
`researchforge/__init__.py`:
```python
__version__ = "0.0.1"
```
`pyproject.toml`:
```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "researchforge"
version = "0.0.1"
description = "Data-driven automated research engine"
requires-python = ">=3.11"
dependencies = ["pandas>=2.0", "openpyxl>=3.1", "pydantic>=2.5", "pyyaml>=6.0"]

[project.optional-dependencies]
dev = ["pytest>=8.0"]

[project.scripts]
rf = "researchforge.cli:main"

[tool.setuptools.packages.find]
include = ["researchforge*"]
```

- [ ] **Step 4: Install and run** `python -m pip install -e ".[dev]"` then `python -m pytest tests/test_smoke.py -q` → Expected: PASS

- [ ] **Step 5: commit** `git add -A && git commit -m "feat(stage0): package skeleton + smoke test"`

### Task 0.2: CLI `--version`

**Files:** Create `researchforge/cli.py`; Test `tests/test_cli.py`

- [ ] **Step 1: Failing test**
```python
import subprocess, sys


def test_cli_version():
    out = subprocess.run([sys.executable, "-m", "researchforge.cli", "--version"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "researchforge" in out.stdout.lower()
```

- [ ] **Step 2: Run confirm fail** → FAIL (no module researchforge.cli)

- [ ] **Step 3: Implement** `researchforge/cli.py`
```python
import argparse
from researchforge import __version__


def main(argv=None):
    p = argparse.ArgumentParser(prog="researchforge")
    p.add_argument("--version", action="store_true")
    args = p.parse_args(argv)
    if args.version:
        print(f"researchforge {__version__}")
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run** `python -m pytest tests/test_cli.py -q` → PASS
- [ ] **Step 5: commit** `git commit -am "feat(stage0): cli --version"`

### Task 0.3: Cross-platform Setup Script + .gitignore outputs/data

**Files:** Create `scripts/setup.py`; Modify `.gitignore`

- [ ] **Step 1:** `scripts/setup.py` (use `sys.executable -m pip install -e .`, check Python>=3.11, pure `pathlib`, runnable on Win/macOS/Linux)
```python
import sys, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    assert sys.version_info >= (3, 11), "Python 3.11+ required"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", f"{ROOT}[dev]"])
    for d in ("outputs", "data/raw"):
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    print("ResearchForge setup complete.")


if __name__ == "__main__":
    main()
```
- [ ] **Step 2:** Confirm `.gitignore` already contains `outputs/`, `data/raw/` (already in repo root .gitignore).
- [ ] **Step 3: Run** `python scripts/setup.py` → Expected: install succeeds, directories created.
- [ ] **Step 4: commit** `git commit -am "feat(stage0): cross-platform setup script"`

**Stage 0 Acceptance:** After `python scripts/setup.py`, `python -m pytest -q` all green, `rf --version` works. **→ Fable 5 review, then Stage 1.**

---

## Stage 1 — Profiler: Data Profiling + Type Identification (include time series) + Quality Diagnosis

**Goal:** `profile_dataset(path)` read CSV/Excel → output `DataFingerprint`; `scan_folder(dir)` scan data folder.

### Task 1.1: DataFingerprint / ColumnInfo Models (`profiler/fingerprint.py`)
- Fields: `ColumnInfo(name, kind, n_missing, n_unique, dtype)`; `DataFingerprint(path, n_rows, n_cols, columns, is_panel, unit_col, time_col, is_timeseries, has_geo, treatment_candidates)`.
- TDD: construction + `.model_dump()` serialization test.

### Task 1.2: Column Type Inference (`profiler/types.py`)
- `infer_kind(series) -> {"continuous","categorical","count","binary","datetime","id","geo"}`.
- TDD: one fixture per kind assertion (e.g., all-unique integers→id; 0/1→binary; parseable dates→datetime; lat/lon range→geo).

### Task 1.3: `profile_dataset(path)` (`profiler/profile.py`)
- pandas read CSV/Excel; infer `infer_kind` per column; panel detection heuristic (exist one id column + one datetime/year column and (id,time) nearly unique → `is_panel=True`, record `unit_col/time_col`); single-series time series detection.
- TDD: fixture `tests/data/panel_province_year.csv` (province×year + 1 numeric outcome + 1 binary treatment) → assert `is_panel`, `unit_col=="province"`, `time_col=="year"`.

### Task 1.4: Quality Diagnosis (`profiler/quality.py`)
- `diagnose(df) -> list[Issue]`: missing rate, duplicate rows, constant columns, type mixing, IQR outliers.
- TDD: construct df with missing/duplicates/constant columns → assert corresponding issues.

### Task 1.5: Folder Scan (`profiler/scan.py`)
- `scan_folder(dir) -> list[Path]` (.csv/.xlsx); `profile_folder(dir) -> dict[Path, DataFingerprint]`.
- TDD: temp directory with 2 csvs → assert both discovered and profiled.

**Stage 1 Acceptance:** Correctly identify panel structure, column types, quality issues on fixture panel CSV. **→ Fable 5 review, then Stage 2.**

---

## Stage 2 — Cleaning (Data Cleaning)
**Goal:** Based Stage 1 diagnosis generate cleaning plan → user confirm → execute + write `outputs/<run>/cleaning_log.json`.
**Files:** `cleaning/plan.py`. **Acceptance:** df with missing/duplicates gets plan, after execution cleaning log drops. (Expand to full TDD when entering.)

## Stage 3 — Catalog (Analysis Catalog)
**Goal:** `catalog/schema.py` (`AnalysisEntry{method, domain, preconditions, produces, executor_ref, biases}`) + `registry.py` (load `entries/*.yaml`, query) + `entries/ag_econ.yaml` (initial entries for panel/cross-section/DID/two-way fixed-effects/IV etc., each with preconditions and biases).
**Acceptance:** Load catalog, filter by conditions return candidates. (Expand to full TDD when entering.)

## Stage 4 — Recommender (Rule Matching + Rigor Review)
**Goal:** `match.py` (fingerprint vs preconditions → candidates) + `rigor.py` (🟢🟡🔴 conclusion + biases + score + informed override) + `recommend.py` (rank menu; agent ranking/explanation optional enhancement).
**Acceptance:** Panel fingerprint → recommendation menu with DID/two-way fixed-effects, each with prerequisite check and bias; 🔴 items overrideable. (Expand to full TDD when entering.)

## Stage 5 — Executor Integration + Artifact Retention
**Goal:** `executor/run.py`: hand selected analysis to `empirical-analysis-python` skill; **write generated code + figures + tables + reports all to `outputs/<timestamp>/`**.
**Acceptance:** Select DID → regression table, figures, report, used code appear in `outputs/`; reproducible. (Expand to full TDD when entering.)

## Stage 6 — Ingestion + Novelty Lens
**Goal:** `ingestion/ingest.py`: handle `skills_inbox/` (skill/papers) → extract register in catalog → archive `_processed/`; novelty lens hang one literature scan on selected analysis, output "blank space" hints.
**Acceptance:** Ingest a skill → catalog adds visible entry and appears in recommendations; selected analysis gets one literature blank space hint. (Expand to full TDD when entering.)

## Stage 7 — Benchmark / Eval Suite (post-MVP, track improvement and optimization)

**Goal:** A set of synthetic cases with "ground truth" + scorer, quantify engine quality and track per version — answer "did our change make it better or worse".

**Why (complement to unit tests)**: Profiling/recommendation/rigor/estimate recovery all are **judgment tasks**, unit tests only keep concrete behavior from regressing; benchmark gives **comparable score**, catch "silent quality drift", objectively verify if optimization effective (catch "Fable 5 regular review + incremental optimize" need).

**Content**:
- Case library: synthetic data (panel/cross-section/time series, with/without treatment, with known ground truth, reuse `synth`) + each case expectation (should identify panel, viable analyses, true effect value).
- Scorer: run engine → compute profiling accuracy / recommendation hit (precision-recall) / **estimate recovery error** / rigor calibration.
- Record: each version score archived in `benchmark/results/<version>.json`, compare trends.

**Acceptance**: `rf benchmark` output score table; after change see score rise/fall. (Expand to full TDD when entering.)

---

## Self-check (against spec)
- Data type identification/time series/quality diagnosis/cleaning → Stage 1–2 ✓
- Recommendation + hybrid C + rigor review + informed override → Stage 3–4 ✓
- Execution + **artifacts (code/figures/tables/reports) land in outputs/** → Stage 5 ✓ (matches spec §6 success criterion 9)
- Skill ingestion + novelty lens → Stage 6 ✓
- Cross-platform/portable → Stage 0 setup ✓
- §8 reserved items (R/ML/DL/GIS/data sources/frontier monitor/GitHub collector/foreman/multi-modal delivery) → **not this plan, later each own plan**.
