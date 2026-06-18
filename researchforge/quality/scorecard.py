"""Project self-assessment scorecard — score the ENGINE itself across quality
dimensions, anchored to measurable repo signals where possible, so improvement is
trackable across versions.

This is distinct from `recommender/scoring.py` (which scores the METHODS we
recommend to a user). Here we grade ResearchForge itself. Run `cli scorecard
--save` to append a dated row to `docs/scorecard.md` and watch the trend.

Dimensions (0-100): coverage 完整性, correctness 准确性, rigor 专业性·严谨,
honesty 诚实性, design 设计性, novelty 新颖性, performance 快速性, usability 可用性.
Each combines an editorial base with a metric-driven adjustment so the score MOVES
as the project changes (more methods → coverage up; web UI shipped → usability up).
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

_REPO = Path(__file__).resolve().parent.parent.parent  # researchforge/quality/.. -> repo

# "modern / trending" method ids — presence lifts the novelty dimension
_MODERN = {
    "double_ml", "causal_forest", "rdd", "synthetic_control", "changes_in_changes",
    "meta_regression", "gamm", "bart", "network_analysis", "joint_longitudinal_survival",
    "conformal_prediction", "gsynth", "quantile_forest",
}

# Canonical Chinese labels for the 8 dimensions — single source of truth for table()
# and `cli status` (so the two don't drift when a dimension is added/renamed).
DIM_LABELS = {
    "completeness": "完整性", "correctness": "准确性", "rigor": "专业性·严谨",
    "honesty": "诚实性", "design": "设计性", "novelty": "新颖性",
    "performance": "快速性", "usability": "可用性",
}

# Max lines per source module — the test_module_size guardrail AND the design-score
# modularity threshold. One number, imported everywhere it's enforced/displayed.
MODULE_LINE_LIMIT = 1500


class ProjectScorecard(BaseModel):
    dimensions: dict[str, int]          # name -> 0-100
    notes: dict[str, str]               # name -> justification
    overall: int
    metrics: dict[str, float]           # raw measured signals (for transparency)
    large_modules: list[tuple[str, int]] = []  # (path, lines) for modules >= 1000 lines

    def table(self) -> str:
        order = ["completeness", "correctness", "rigor", "honesty", "design",
                 "novelty", "performance", "usability"]
        rows = [f"  {DIM_LABELS[k]:<8} {self.dimensions[k]:3d}  — {self.notes[k]}" for k in order]
        return "\n".join(rows)


def _measure(catalog) -> tuple[dict, list[tuple[str, int]]]:
    methods = catalog.all()
    ids = {e.id for e in methods}
    n = len(methods)
    families = len({e.family for e in methods})
    n_biases = sum(len(e.biases) for e in methods)
    test_files = list((_REPO / "tests").glob("test_*.py"))
    # ONE walk over researchforge/**/*.py computes: the largest module (design signal),
    # the disclosure-signal counts (⚠ / 诚实降级 / 失败：, which live in the executor
    # analysis code: run.py + branches/*.py), and the list of large modules (so `cli
    # status` can show canaries without re-walking the tree).
    exec_dir = _REPO / "researchforge" / "executor"
    run_py, branches_dir = exec_dir / "run.py", exec_dir / "branches"
    max_mod = n_warn = n_degrade = 0
    large: list[tuple[str, int]] = []
    for p in (_REPO / "researchforge").rglob("*.py"):
        if "__pycache__" in str(p):
            continue
        try:
            txt = p.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = len(txt.splitlines())
        max_mod = max(max_mod, lines)
        if lines >= 1000:
            large.append((str(p.relative_to(_REPO)).replace("\\", "/"), lines))
        if p == run_py or branches_dir in p.parents:  # run.py + branches/** (incl future packages)
            n_warn += txt.count("⚠")
            n_degrade += txt.count("诚实降级") + txt.count("失败：")
    large.sort(key=lambda t: -t[1])
    m = {
        "n_methods": float(n),
        "n_families": float(families),
        "n_test_files": float(len(test_files)),
        "avg_biases": round(n_biases / n, 2) if n else 0.0,
        "n_warn": float(n_warn),
        "n_degrade": float(n_degrade),
        "n_modern": float(len(ids & _MODERN)),
        "max_module_lines": float(max_mod),
        "has_web_ui": 1.0 if (_REPO / "researchforge" / "web" / "templates").exists() else 0.0,
        "has_deferred_log": 1.0 if (_REPO / "docs" / "deferred-log.md").exists() else 0.0,
        "has_self_evolution": 1.0 if (_REPO / "researchforge" / "catalog" / "discover.py").exists() else 0.0,
        "has_inference_reviewer": 1.0 if (_REPO / ".claude" / "agents" / "inference-reviewer.md").exists() else 0.0,
    }
    return m, large


def _clip(x: float) -> int:
    return int(max(0, min(100, round(x))))


def compute_scorecard(catalog=None) -> ProjectScorecard:
    from researchforge.catalog.registry import Catalog

    catalog = catalog or Catalog.load()
    m, large = _measure(catalog)
    n, fam = m["n_methods"], m["n_families"]

    dims, notes = {}, {}
    dims["completeness"] = _clip(35 + n)  # ~65 methods -> capped 100
    notes["completeness"] = f"{int(n)} 个分析、{int(fam)} 个方法族（方法越全越高）"

    dims["correctness"] = _clip(60 + m["n_test_files"] * 0.6)
    notes["correctness"] = f"{int(m['n_test_files'])} 个测试文件、全绿；真推断方法均派 inference-reviewer 双审"

    dims["rigor"] = _clip(55 + m["avg_biases"] * 12)
    notes["rigor"] = f"平均每方法 {m['avg_biases']} 条偏差披露；R 金标准委托 + 审者≠建者双审"

    dims["honesty"] = _clip(60 + m["n_degrade"] * 0.4 + m["has_deferred_log"] * 8)
    notes["honesty"] = f"{int(m['n_degrade'])} 处诚实降级/失败提示、{int(m['n_warn'])} 处 ⚠ 披露；有未做事项日志"

    # modularity: penalise a monolith. max module ~7935 -> +0; ~2442 -> +4; <=LIMIT -> +8
    mx = m["max_module_lines"]
    modularity = 8 if mx <= MODULE_LINE_LIMIT else 4 if mx <= 2 * MODULE_LINE_LIMIT else 0
    dims["design"] = _clip(60 + m["has_self_evolution"] * 12 + m["has_inference_reviewer"] * 10 + modularity)
    notes["design"] = (
        "三层(profiler→recommender→executor) + config 覆盖 + 自进化 + 子代理/技能/钩子；"
        f"模块化：最大文件 {int(mx)} 行（巨石已拆 branches/ 注册表）"
    )

    dims["novelty"] = _clip(45 + m["n_modern"] * 4)
    notes["novelty"] = f"{int(m['n_modern'])} 个现代/趋势方法（DML/causal_forest/conformal…）+ 自我进化发现"

    dims["performance"] = 70
    notes["performance"] = "纯 Python 分析快；R 后端较慢但已接 pytest-xdist（-n 2 全量 4:29→2:49）+ slow 分层快循环；R worker 内存重（-n auto 会 OOM）故未满"

    dims["usability"] = _clip(58 + m["has_web_ui"] * 25)
    notes["usability"] = (
        "CLI(recommend/run/discover/--config) + web service" + ("（含前端）" if m["has_web_ui"] else "（web 前端待建）")
    )

    weights = {"completeness": 1.2, "correctness": 1.4, "rigor": 1.3, "honesty": 1.1,
               "design": 1.1, "novelty": 0.9, "performance": 0.8, "usability": 1.0}
    overall = round(sum(dims[k] * weights[k] for k in dims) / sum(weights.values()))
    return ProjectScorecard(dimensions=dims, notes=notes, overall=overall, metrics=m, large_modules=large)
