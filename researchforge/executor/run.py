"""Executor: run the chosen analysis and persist code / tables / figures / report
to outputs/<timestamp>_<analysis>/. Reuses the empirical-analysis-python stack
(statsmodels + matplotlib)."""

from __future__ import annotations

import datetime
import os
from pathlib import Path

from pydantic import BaseModel, Field

from researchforge.catalog.schema import AnalysisEntry
from researchforge.executor._branch_api import BRANCH_REGISTRY, Ctx
from researchforge.profiler.fingerprint import DataFingerprint
from researchforge.profiler.profile import read_table

_REGRESSION = {"ols_regression", "panel_fixed_effects", "did"}


class RunResult(BaseModel):
    analysis_id: str
    method: str
    output_dir: str
    files: list[str] = Field(default_factory=list)
    report_path: str
    summary: str = ""
    estimates: dict[str, float] = Field(default_factory=dict)


# Helpers now live in executor/_helpers/{core,backends}.py; re-exported here so
# branches/*.py and tests keep importing them from researchforge.executor.run.
from researchforge.executor._helpers.core import (  # noqa: E402
    _THEME_COLORS,
    _coef_plot,
    _conformal_prediction,
    _cost_mask,
    _dea_cross,
    _dea_efficiency,
    _dea_io,
    _entropy_weights,
    _gmm_lags,
    _heatmap,
    _init_mpl_style,
    _io_names,
    _knn_k,
    _mcda_direction_note,
    _mcda_inputs,
    _mcda_rank_plot,
    _minmax01,
    _nca_ceiling,
    _nca_plot,
    _network_via_nx,
    _ordinal_prob_plot,
    _pick_did_treatment,
    _plotly_corr_heatmap,
    _plotly_scatter,
    _qca_anchors,
    _qca_incl_cut,
    _quantile_process_plot,
    _regression,
    _report,
    _resid_plot,
    _run_dir,
    _sem_latents,
    _silhouette_plot,
    _synthetic_control,
    _usda_texture,
    _varimax,
)
from researchforge.executor._helpers.backends import (  # noqa: E402
    _causal_forest_via_econml,
    _dml_via_doubleml,
    _rdd_via_rdrobust,
    _sem_via_semopy,
)
from researchforge.executor._helpers.r_backends import (  # noqa: E402
    _bart_via_r,
    _cic_via_r,
    _cna_via_r,
    _csqca_via_r,
    _diff_abundance_aldex2_via_r,
    _dynamic_gmm_via_r,
    _fsqca_via_r,
    _gam_via_r,
    _gamm_via_r,
    _glmm_via_r,
    _gsynth_via_r,
    _jm_via_r,
    _kriging_via_r,
    _meta_regression_via_r,
    _meta_via_r,
    _panel_qca_via_r,
    _qca_necessity_via_r,
    _sem_via_lavaan,
    _sfa_via_r,
    _spatial_reg_via_r,
)


def run_analysis(
    fp: DataFingerprint,
    entry: AnalysisEntry,
    output_root: str = "outputs",
    override: bool = False,
    config: dict | None = None,
) -> RunResult:
    df = read_table(Path(fp.path))
    # user-supplied overrides for the engine's substantive defaults (column roles,
    # anchors, etc.) — each branch reads cfg.get(<key>) and falls back to its auto
    # default. See docs/loop-decisions.md for the configurable keys per analysis.
    cfg = config or {}
    # Validate the override keys against the entry's declared param spec (if any)
    # and surface problems instead of silently ignoring them. Non-blocking: the
    # analysis still runs on its auto defaults. See catalog/config_schema.py.
    from researchforge.catalog.config_schema import validate_config

    _cfg_warns = validate_config(entry, cfg, fp)
    d = _run_dir(output_root, entry.id)
    _init_mpl_style()
    files: list[str] = []
    summary: list[str] = []
    if _cfg_warns:
        summary.append("⚠ 配置参数提示：" + " ".join(_cfg_warns))
    # Smart-selection nudge (non-binding): if this method takes an `outcome` and the
    # user didn't set one, surface the detected likely outcome so they can config it
    # deliberately (the auto default is "first continuous", which can miss an
    # integer-valued / non-first target). See profiler/roles.py.
    if (fp.likely_outcome and not (cfg.get("outcome") or cfg.get("y"))
            and any(p.name in ("outcome", "y") for p in entry.params)):
        summary.append(
            f"💡 检测到 '{fp.likely_outcome}' 可能是结果变量（{fp.role_hint_reason}）；"
            "若引擎默认选取不符，用 config outcome 指定。"
        )
    estimates: dict[str, float] = {}
    code: list[str] = ["import pandas as pd", f"df = pd.read_csv(r'{fp.path}')", ""]

    # Dispatch via the branch registry: every analysis is a handler in
    # executor/branches/*.py (registered by id); unknown ids get a placeholder report.
    # See executor/_branch_api.py.
    ctx = Ctx(df=df, fp=fp, entry=entry, cfg=cfg, d=d, files=files,
              summary=summary, estimates=estimates, code=code)
    _handler = BRANCH_REGISTRY.get(entry.id)
    if _handler is not None:
        try:
            _handler(ctx)
        except Exception as err:  # noqa: BLE001 — degrade to a report, never crash the run
            summary.append(
                f"⚠ {entry.id} 执行失败：{type(err).__name__}: {str(err)[:200]}"
            )
    else:
        summary.append(f"{entry.method} 暂未接入执行器（需补依赖/封装），仅生成占位报告。")

    (d / "analysis_code.py").write_text("\n".join(code), encoding="utf-8")
    files.append("analysis_code.py")

    (d / "report.md").write_text(
        _report(entry, fp, summary, files, override, estimates), encoding="utf-8")
    files.append("report.md")

    return RunResult(
        analysis_id=entry.id,
        method=entry.method,
        output_dir=str(d),
        files=files,
        report_path=str(d / "report.md"),
        summary="\n".join(summary),
        estimates=estimates,
    )


# Populate BRANCH_REGISTRY: importing the branches package runs each family module's
# @register decorators. Done at the END of run.py so the helpers and run_analysis
# above are already defined when branch modules import them (avoids a circular import).
from researchforge.executor import branches as _branches  # noqa: E402,F401
