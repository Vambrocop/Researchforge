# ResearchForge MVP 实现计划（分阶段）

> **For agentic workers:** REQUIRED SUB-SKILL: `superpowers:subagent-driven-development`（推荐）或 `superpowers:executing-plans`。Steps 用 `- [ ]` 复选框追踪。

**Goal:** 搭好 ResearchForge 骨架，并打通**农业计量一条纵切**：扫描本地数据文件夹 → 画像 + 质量诊断 + 基础清洗 → 推荐分析（带 🟢🟡🔴 严谨度评审 + 知情覆盖）→ 你选 → 调 `empirical-analysis-python` 执行 → 产出**并保存**代码/图/表/报告到 `outputs/`。

**Architecture:** Python 包 `researchforge`，三层 Profiler → Recommender（规则 + agent）→ Executor，外加 catalog 注册表与 ingestion。跨平台（Win/macOS/Linux，`pathlib`、不写死盘符）。软推理经 Claude Code；批量粗活派便宜模型 subagent（包工头模式）。**每阶段末 Fable 5 做一次代码 review**（正确性/准确性/优化）。

**Tech Stack:** Python 3.11+、pandas、openpyxl、pydantic v2、pyyaml、pytest。Executor 复用 `empirical-analysis-python` skill。setup 用跨平台 `scripts/setup.py`。

**分阶段执行纪律：** 一次只做一个 Stage → TDD（先写失败测试 → 实现 → 通过 → commit）→ **阶段末 Fable 5 review** → 再下一阶段。Stage 2–6 在**进入该阶段时**展开为完整 TDD 任务（本计划先给目标/文件/验收，避免一次性写死大量推测代码——这是刻意的"逐阶段细化"，非占位）。

---

## 文件结构（决定分解边界）

```
researchforge/                 # 引擎包
  __init__.py                  # __version__
  profiler/
    __init__.py
    fingerprint.py             # DataFingerprint / ColumnInfo (pydantic 模型)
    types.py                   # 列类型推断
    profile.py                 # profile_dataset(path) -> DataFingerprint
    quality.py                 # 数据质量诊断
    scan.py                    # 扫描数据文件夹
  cleaning/__init__.py  plan.py    # Stage 2
  catalog/__init__.py  schema.py  registry.py  entries/ag_econ.yaml  # Stage 3
  recommender/__init__.py  match.py  recommend.py  rigor.py          # Stage 4
  executor/__init__.py  run.py                                       # Stage 5
  ingestion/__init__.py  ingest.py                                   # Stage 6
  cli.py                       # 命令行入口
tests/                         # pytest，与包同构
scripts/setup.py               # 跨平台依赖/skill 安装
pyproject.toml
outputs/                       # 运行产物（gitignore；代码/图/表/报告）
data/                          # 本地数据（gitignore raw）
```

---

## Stage 0 — 仓库骨架 + 包 + 跨平台 setup + 冒烟测试

**Goal:** 可 `pip install -e .`、可 `import researchforge`、有可运行的测试与 CLI `--version`。

### Task 0.1: pyproject + 包骨架 + 版本冒烟测试

**Files:**
- Create: `pyproject.toml`
- Create: `researchforge/__init__.py`
- Test: `tests/test_smoke.py`

- [ ] **Step 1: 写失败测试** `tests/test_smoke.py`
```python
import researchforge


def test_version_exposed():
    assert isinstance(researchforge.__version__, str)
    assert researchforge.__version__
```

- [ ] **Step 2: 运行确认失败** `python -m pytest tests/test_smoke.py -q` → Expected: FAIL（ModuleNotFoundError: researchforge）

- [ ] **Step 3: 最小实现**
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

- [ ] **Step 4: 安装并运行** `python -m pip install -e ".[dev]"` 然后 `python -m pytest tests/test_smoke.py -q` → Expected: PASS

- [ ] **Step 5: commit** `git add -A && git commit -m "feat(stage0): package skeleton + smoke test"`

### Task 0.2: CLI `--version`

**Files:** Create `researchforge/cli.py`; Test `tests/test_cli.py`

- [ ] **Step 1: 失败测试**
```python
import subprocess, sys


def test_cli_version():
    out = subprocess.run([sys.executable, "-m", "researchforge.cli", "--version"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "researchforge" in out.stdout.lower()
```

- [ ] **Step 2: 运行确认失败** → FAIL（no module researchforge.cli）

- [ ] **Step 3: 实现** `researchforge/cli.py`
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

- [ ] **Step 4: 运行** `python -m pytest tests/test_cli.py -q` → PASS
- [ ] **Step 5: commit** `git commit -am "feat(stage0): cli --version"`

### Task 0.3: 跨平台 setup 脚本 + .gitignore outputs/data

**Files:** Create `scripts/setup.py`; Modify `.gitignore`

- [ ] **Step 1:** `scripts/setup.py`（用 `sys.executable -m pip install -e .`，检查 Python>=3.11，纯 `pathlib`，可在 Win/macOS/Linux 跑）
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
- [ ] **Step 2:** 确认 `.gitignore` 已含 `outputs/`、`data/raw/`（已在仓库根 .gitignore）。
- [ ] **Step 3: 运行** `python scripts/setup.py` → Expected: 安装成功、目录创建。
- [ ] **Step 4: commit** `git commit -am "feat(stage0): cross-platform setup script"`

**Stage 0 验收：** `python scripts/setup.py` 后 `python -m pytest -q` 全绿，`rf --version` 可用。**→ Fable 5 review，再进 Stage 1。**

---

## Stage 1 — Profiler：数据画像 + 类型识别（含时序）+ 质量诊断

**Goal:** `profile_dataset(path)` 读 CSV/Excel → 输出 `DataFingerprint`；`scan_folder(dir)` 扫描数据文件夹。

### Task 1.1: DataFingerprint / ColumnInfo 模型 (`profiler/fingerprint.py`)
- 字段：`ColumnInfo(name, kind, n_missing, n_unique, dtype)`；`DataFingerprint(path, n_rows, n_cols, columns, is_panel, unit_col, time_col, is_timeseries, has_geo, treatment_candidates)`。
- TDD：构造 + `.model_dump()` 序列化测试。

### Task 1.2: 列类型推断 (`profiler/types.py`)
- `infer_kind(series) -> {"continuous","categorical","count","binary","datetime","id","geo"}`。
- TDD：每种 kind 一个 fixture 断言（如全唯一整数序列→id；0/1→binary；可解析日期→datetime；经纬度范围→geo）。

### Task 1.3: `profile_dataset(path)` (`profiler/profile.py`)
- pandas 读 CSV/Excel；逐列 `infer_kind`；面板检测启发式（存在一个 id 列 + 一个 datetime/年份列且 (id,time) 近似唯一 → `is_panel=True`，记 `unit_col/time_col`）；单序列时序检测。
- TDD：fixture `tests/data/panel_province_year.csv`（province×year + 1 数值结果 + 1 处理 0/1）→ 断言 `is_panel`、`unit_col=="province"`、`time_col=="year"`。

### Task 1.4: 质量诊断 (`profiler/quality.py`)
- `diagnose(df) -> list[Issue]`：缺失率、重复行、常数列、类型混杂、IQR 异常值。
- TDD：构造含缺失/重复/常数列的 df → 断言对应 issue。

### Task 1.5: 文件夹扫描 (`profiler/scan.py`)
- `scan_folder(dir) -> list[Path]`（.csv/.xlsx）；`profile_folder(dir) -> dict[Path, DataFingerprint]`。
- TDD：临时目录放 2 个 csv → 断言都被发现并画像。

**Stage 1 验收：** 对 fixture 面板 CSV 正确识别面板结构、列类型、质量问题。**→ Fable 5 review，再进 Stage 2。**

---

## Stage 2 — Cleaning（数据清洗）
**Goal:** 据 Stage 1 诊断生成清洗方案 → 用户确认 → 执行 + 写 `outputs/<run>/cleaning_log.json`。
**Files:** `cleaning/plan.py`。**验收：** 含缺失/重复的 df 得到方案，执行后落清洗日志。（进入时展开 TDD。）

## Stage 3 — Catalog（分析目录）
**Goal:** `catalog/schema.py`（`AnalysisEntry{method, domain, preconditions, produces, executor_ref, biases}`）+ `registry.py`（加载 `entries/*.yaml`、查询）+ `entries/ag_econ.yaml`（面板/截面/DID/双向固定效应/IV 等初始条目，每条带 preconditions 与 biases）。
**验收：** 载入 catalog、按条件过滤返回候选。（进入时展开 TDD。）

## Stage 4 — Recommender（规则匹配 + 严谨度评审）
**Goal:** `match.py`（fingerprint vs preconditions → 候选）+ `rigor.py`（🟢🟡🔴 结论 + 偏差 + 评分 + 知情覆盖）+ `recommend.py`（排序菜单；agent 排序/解释为可选增强）。
**验收：** 面板 fingerprint → 含 DID/双向固定效应的推荐菜单，每条带前提核查与偏差；🔴 项可覆盖。（进入时展开 TDD。）

## Stage 5 — Executor 集成 + 产物保留
**Goal:** `executor/run.py`：把选中分析交给 `empirical-analysis-python` skill；**生成代码 + 图 + 表 + 报告全部写入 `outputs/<timestamp>/`**。
**验收：** 选 DID → `outputs/` 下出现回归表、图、报告、以及所用代码；可复现。（进入时展开 TDD。）

## Stage 6 — Ingestion + 新颖度透镜
**Goal:** `ingestion/ingest.py`：处理 `skills_inbox/`（skill/论文）→ 提炼登记进 catalog → 归档 `_processed/`；新颖度透镜对选中分析挂一次文献扫描，输出"白地"提示。
**验收：** 投一个 skill → catalog 新增可见条目并出现在推荐中；选中分析得到一条文献白地提示。（进入时展开 TDD。）

## Stage 7 — Benchmark / 评测套件（post-MVP，跟踪改进与优化）

**Goal:** 一组带"标准答案"的合成案例 + 评分器，量化引擎质量并按版本跟踪——回答"我们的改动让它变好还是变坏"。

**为什么（与单元测试互补）**：画像/推荐/严谨度/估计回收都是**判断题**，单测只保具体行为不退化；benchmark 给一个**可比分数**，抓"静默质量漂移"、客观验证优化是否有效（接住"Fable 5 定期 review + 一点点优化"的需求）。

**内容**：
- 案例库：合成数据（面板/截面/时序、有无处理，带已知真值，复用 `synth`）+ 每例期望（应识别为面板、应可行的分析、真效应值）。
- 评分器：跑引擎 → 算 画像准确率 / 推荐命中（precision-recall）/ **估计回收误差** / 严谨度校准。
- 记录：每版分数存档 `benchmark/results/<version>.json`，对比看趋势。

**验收**：`rf benchmark` 输出一张分数表；改动后能看出分数升降。（进入时展开 TDD。）

---

## 自检（对照 spec）
- 数据类型识别/时序/质量诊断/清洗 → Stage 1–2 ✓
- 推荐 + 混合C + 严谨度评审 + 知情覆盖 → Stage 3–4 ✓
- 执行 + **产物（代码/图/表/报告）落盘 outputs/** → Stage 5 ✓（对应 spec §6 成功标准 9）
- skill 投递 + 新颖度透镜 → Stage 6 ✓
- 跨平台/便携 → Stage 0 setup ✓
- §8 预留项（R/ML/DL/GIS/数据源/前沿监视/GitHub 采集/包工头/多形态交付）→ **本计划不实现，后续各自立计划**。
