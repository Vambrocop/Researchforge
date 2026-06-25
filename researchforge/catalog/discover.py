"""Self-evolution — the FRONT of the self-growth pipeline.

Discovers candidate methods the engine doesn't yet have, scores them for
priority (publishability / novelty / popularity), and drops them into the
candidate queue as `pending`. It NEVER promotes: the existing quality gate
(`promote_candidate`, which requires a wired+tested `ready` executor) still
stands between discovery and the live catalog. So the engine can grow its own
backlog, but a human/builder + tests are always the gate to going live.

Sources are pluggable via `fetch_fn` (the same pattern as the novelty lens): the
default is an offline, curated seed of trending methods grounded in a literature
scan, so discovery works with no network and is deterministic/testable.

PHASE 2 (live trend engine, see `catalog/trends.py`): `build_live_fetch_fn()`
returns a `fetch_fn` that enriches each seed with REAL PyPI / CRAN / GitHub signals
(downloads / stars / recency → a 0-100 `momentum`), which `score_candidate` blends
into the discovery priority. `cli discover --live` also writes a momentum snapshot
that the recommendation scorecard reads (no hot-path network). All best-effort with
graceful degrade — offline, it falls straight back to the offline priors. This
scanning is a curation-time tool, never part of the analysis runtime.
"""

from __future__ import annotations

from typing import Callable, Optional

import yaml
from pydantic import BaseModel, Field

from researchforge.catalog.candidates import CatalogCandidate, load_candidates
from researchforge.catalog.registry import Catalog
from researchforge.catalog.schema import AnalysisEntry
from researchforge.recommender.scoring import _DEFAULT, _FAMILY

# Curated seed: trending / high-value methods ResearchForge does NOT yet have,
# grounded in the 2025-26 method scan. Each is a method idea, not a wired analysis.
SEED: list[dict] = [
    # Wave 2 (2026-06-16 reseed — wave 1 RDD/DML/causal_forest/meta_regression/GAMM/
    # changes_in_changes/network/BART/joint_model all implemented). Chosen to avoid a
    # C++/Stan compiler (absent here). The trend engine (phase 2) will refresh this
    # from live CRAN/PyPI/GitHub instead of hand-curation.
    {"id": "conformal_prediction", "method": "Conformal prediction (distribution-free prediction intervals)",
     "family": "ml", "domain": "ml", "goal": "predict",
     "rationale": "分布无关的有限样本覆盖保证的预测区间，2023-25 极热的不确定性量化方法。",
     "sources": ["mapie", "split conformal (hand-roll)"]},
    {"id": "ancom_bc", "method": "ANCOM-BC (analysis of compositions, bias correction)",
     "family": "microbiology", "domain": "microbiology", "goal": "compare",
     "rationale": "微生物组差异丰度的组成性金标准之一，补全 #9（ALDEx2 已接）。",
     "sources": ["ANCOMBC (R, 已装)"]},
    {"id": "gsynth", "method": "Generalized synthetic control (interactive fixed effects)",
     "family": "causal", "domain": "economics", "goal": "explain",
     "rationale": "合成控制 + 交互固定效应，多处理单位/交错采纳，扩展 synthetic_control。",
     "sources": ["gsynth (R)"]},
    {"id": "latent_class_analysis", "method": "Latent class analysis (LCA)",
     "family": "statistics", "domain": "social science", "goal": "explore",
     "rationale": "在分类指标下找潜在亚群（混合模型），社科/市场细分常用。",
     "sources": ["poLCA (R)"]},
    {"id": "gradient_boosting", "method": "Gradient boosting (XGBoost / GBM) with importance",
     "family": "ml", "domain": "ml", "goal": "predict",
     "rationale": "表格数据预测主力，交叉验证 + 特征重要性 + 部分依赖。",
     "sources": ["xgboost", "sklearn GradientBoosting"]},
    {"id": "spatial_panel", "method": "Spatial panel econometrics (SAR/SEM panel)",
     "family": "spatial", "domain": "economics", "goal": "explain",
     "rationale": "面板 + 空间滞后/误差，区域经济/环境扩展（引擎有截面空间回归）。",
     "sources": ["splm (R)"]},
    {"id": "quantile_forest", "method": "Quantile regression forest (prediction intervals)",
     "family": "ml", "domain": "ml", "goal": "predict",
     "rationale": "随机森林给条件分位/预测区间，非参不确定性量化。",
     "sources": ["quantile-forest", "grf (R)"]},
]


class MethodCandidate(BaseModel):
    id: str
    method: str
    family: str
    domain: str = ""
    goal: str = ""
    rationale: str = ""
    sources: list[str] = Field(default_factory=list)
    priority: int = 0  # 0-100 discovery priority
    breakdown: dict[str, int] = Field(default_factory=dict)
    momentum: int = 0          # 0-100 live trend signal (0 = offline / unavailable)
    trend_note: str = ""       # honest note on the trend source


def score_candidate(spec: dict) -> MethodCandidate:
    """Priority for adding a method idea: publishability + novelty weighted most.

    Popularity comes from the offline family rubric (scoring.py). When the LIVE
    trend engine has attached a real ``momentum`` to the spec (via
    ``build_live_fetch_fn``), it is blended in — so a method that is genuinely
    surging on PyPI/GitHub/CRAN outranks a same-family idea that isn't."""
    pop, pub, _aes, _diff, nov = _FAMILY.get(spec.get("family", ""), _DEFAULT)
    momentum = int(spec.get("momentum", 0) or 0)
    has_live = bool(spec.get("trend", {}).get("available")) if spec.get("trend") else momentum > 0
    if has_live:
        priority = round(0.35 * nov + 0.30 * pub + 0.20 * pop + 0.15 * momentum)
    else:
        priority = round(0.40 * nov + 0.35 * pub + 0.25 * pop)
    return MethodCandidate(
        id=spec["id"], method=spec["method"], family=spec.get("family", ""),
        domain=spec.get("domain", ""), goal=spec.get("goal", ""),
        rationale=spec.get("rationale", ""), sources=list(spec.get("sources", [])),
        priority=int(priority),
        breakdown={"novelty": nov, "publishability": pub, "popularity": pop,
                   "momentum": momentum},
        momentum=momentum,
        trend_note=spec.get("trend", {}).get("detail", "") if spec.get("trend") else "",
    )


def build_live_fetch_fn(base_specs: Optional[list[dict]] = None):
    """Return a `fetch_fn` for `discover_candidates` that enriches each seed spec
    with a REAL trend signal (PyPI/CRAN/GitHub via `trends.fetch_trend`). Offline,
    each spec just carries `momentum=0` and scoring falls back to offline priors."""
    from researchforge.catalog.trends import fetch_trend

    specs = base_specs if base_specs is not None else SEED

    def _fetch() -> list[dict]:
        out: list[dict] = []
        for s in specs:
            sig = fetch_trend(s["id"], list(s.get("sources", [])))
            out.append({**s, "momentum": sig.momentum, "trend": sig.as_dict()})
        return out

    return _fetch


def discover_candidates(
    fetch_fn: Optional[Callable[[], list[dict]]] = None,
    catalog: Optional[Catalog] = None,
    persist: bool = False,
) -> list[MethodCandidate]:
    """Discover, dedupe (against live catalog + existing queue), score, and rank
    candidate methods. With persist=True, write the new ones to the candidate
    queue as `pending` (idempotent: overwrites candidate_queue/discovered.yaml).
    Returns the ranked MethodCandidate list. Never promotes."""
    catalog = catalog or Catalog.load()
    specs = list(fetch_fn()) if fetch_fn else SEED
    have = {e.id for e in catalog.all()} | {c.entry.id for c in load_candidates()}
    fresh = [score_candidate(s) for s in specs if s["id"] not in have]
    fresh.sort(key=lambda m: -m.priority)

    if persist and fresh:
        from researchforge.catalog.candidates import _CANDIDATES_DIR

        _CANDIDATES_DIR.mkdir(parents=True, exist_ok=True)
        items = [
            CatalogCandidate(
                entry=AnalysisEntry(
                    id=m.id, method=m.method, domain=m.domain or "statistics",
                    family=m.family, goal=m.goal or "explain",
                    description=m.rationale,
                ),
                source=", ".join(m.sources) or "discovery",
                status="pending",
                notes=f"自动发现(优先级 {m.priority})：执行器未接；需实现+测试才能 promote。",
            ).model_dump()
            for m in fresh
        ]
        (_CANDIDATES_DIR / "discovered.yaml").write_text(
            yaml.safe_dump(items, allow_unicode=True, sort_keys=False), encoding="utf-8"
        )
    return fresh
