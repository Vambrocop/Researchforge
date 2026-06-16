"""Self-evolution — the FRONT of the self-growth pipeline.

Discovers candidate methods the engine doesn't yet have, scores them for
priority (publishability / novelty / popularity), and drops them into the
candidate queue as `pending`. It NEVER promotes: the existing quality gate
(`promote_candidate`, which requires a wired+tested `ready` executor) still
stands between discovery and the live catalog. So the engine can grow its own
backlog, but a human/builder + tests are always the gate to going live.

Sources are pluggable via `fetch_fn` (the same pattern as the novelty lens): the
default is an offline, curated seed of trending methods grounded in a literature
scan, so discovery works with no network and is deterministic/testable. A live
fetcher over CRAN / PyPI / GitHub / Zenodo can be injected as `fetch_fn` later —
that scanning is a curation-time tool, never part of the analysis runtime.
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
    {"id": "rdd", "method": "Regression discontinuity design (RDD)",
     "family": "causal", "domain": "economics", "goal": "explain",
     "rationale": "局部随机的强识别因果设计，高可发表；引擎有 DID/IV/合成控制但缺 RDD。",
     "sources": ["rdrobust (R/Python)"]},
    {"id": "double_ml", "method": "Double / debiased machine learning (DML)",
     "family": "causal", "domain": "economics", "goal": "explain",
     "rationale": "用 ML 学习干扰项再做正交化因果估计，2024-25 工业界/学界热点。",
     "sources": ["DoubleML", "econml"]},
    {"id": "causal_forest", "method": "Causal forest (heterogeneous treatment effects)",
     "family": "causal", "domain": "ml", "goal": "explain",
     "rationale": "估计异质处理效应 CATE，政策/精准农业方向新颖且可发表。",
     "sources": ["grf (R)", "econml"]},
    {"id": "bayesian_sem", "method": "Bayesian structural equation modeling",
     "family": "sem", "domain": "statistics", "goal": "explain",
     "rationale": "放松经典识别约束、给后验不确定性；生态/心理学采用上升。",
     "sources": ["brms", "blavaan (R)"]},
    {"id": "meta_regression", "method": "Meta-regression (moderators)",
     "family": "meta", "domain": "statistics", "goal": "synthesize",
     "rationale": "在 meta 分析上加调节变量解释异质性，是 meta_analysis 的自然延伸。",
     "sources": ["metafor rma(mods=)"]},
    {"id": "gamm", "method": "Generalized additive mixed model (GAMM)",
     "family": "statistics", "domain": "statistics", "goal": "explain",
     "rationale": "GAM 的平滑 + GLMM 的随机效应合体，生态时空数据主力。",
     "sources": ["mgcv gamm/gam(...,random)"]},
    {"id": "joint_longitudinal_survival", "method": "Joint longitudinal–survival model",
     "family": "survival", "domain": "statistics", "goal": "explain",
     "rationale": "同时建纵向标志物与事件时间，临床/可靠性新颖高分。",
     "sources": ["JM", "joineRML (R)"]},
    {"id": "changes_in_changes", "method": "Changes-in-changes / quantile DID",
     "family": "causal", "domain": "economics", "goal": "explain",
     "rationale": "DID 的分布版本，看处理对整个分布的影响，方法学新颖。",
     "sources": ["qte (R)"]},
    {"id": "bart", "method": "Bayesian additive regression trees (BART)",
     "family": "ml", "domain": "ml", "goal": "predict",
     "rationale": "灵活非参贝叶斯，常用于因果(BCF)与预测，近年很流行。",
     "sources": ["dbarts", "bartMachine (R)"]},
    {"id": "network_analysis", "method": "Network / graph analysis",
     "family": "ml", "domain": "ml", "goal": "explore",
     "rationale": "生态互作/社会网络常用，中心性/社团检测，图表美观。",
     "sources": ["igraph", "networkx"]},
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


def score_candidate(spec: dict) -> MethodCandidate:
    """Priority for adding a method idea: publishability + novelty weighted most
    (popularity a minor factor). Uses the offline family rubric from scoring.py."""
    pop, pub, _aes, _diff, nov = _FAMILY.get(spec.get("family", ""), _DEFAULT)
    priority = round(0.40 * nov + 0.35 * pub + 0.25 * pop)
    return MethodCandidate(
        id=spec["id"], method=spec["method"], family=spec.get("family", ""),
        domain=spec.get("domain", ""), goal=spec.get("goal", ""),
        rationale=spec.get("rationale", ""), sources=list(spec.get("sources", [])),
        priority=int(priority),
        breakdown={"novelty": nov, "publishability": pub, "popularity": pop},
    )


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
