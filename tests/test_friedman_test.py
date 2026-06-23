"""Tests for friedman_test: Friedman Q + Kendall's W + Nemenyi post-hoc.

Known-structure checks: a repeated-measures block design with a clear condition
effect (wide form) -> significant Q; a no-effect control -> non-significant;
long-form input via config subject/within/outcome; incomplete-block dropping is
disclosed; an independent Kendall's-W recompute pins the effect-size formula.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="friedman_test",
        method="Friedman test",
        domain="statistics",
        family="nonparametric",
        goal="explain",
        preconditions=Precondition(min_continuous=3, min_rows=4),
    )


def _block_wide(seed: int, n_subj: int, effects):
    """Wide block design: each subject has a random baseline; per-condition
    effects add a fixed shift -> a within-subject condition effect."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, n_subj)
    cols = {}
    for ci, eff in enumerate(effects):
        cols[f"cond{ci + 1}"] = base + eff + rng.normal(0, 0.3, n_subj)
    return pd.DataFrame(cols)


def test_friedman_wide_condition_effect_significant(tmp_path: Path) -> None:
    df = _block_wide(1, 30, effects=[0.0, 2.0, 4.0])  # monotone condition effect
    csv = tmp_path / "fr3.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "friedman_conditions.csv").exists()
    assert (out / "friedman_nemenyi_posthoc.csv").exists()
    assert res.estimates["n_conditions"] == 3.0
    assert res.estimates["n_subjects"] == 30.0
    assert res.estimates["p_value"] < 0.01  # strong condition effect
    assert 0.0 <= res.estimates["kendalls_w"] <= 1.0
    assert res.estimates["kendalls_w"] > 0.3  # strong agreement
    assert res.estimates["n_sig_pairs"] >= 1.0  # Nemenyi flags cond1 vs cond3


def test_friedman_kendalls_w_recompute(tmp_path: Path) -> None:
    """Independently recompute W = chi2/(n*(k-1)) from scipy's Q."""
    df = _block_wide(5, 25, effects=[0.0, 1.5, 3.0])
    csv = tmp_path / "frw.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    mat = df.to_numpy(float)
    Q, _ = sps.friedmanchisquare(*[mat[:, j] for j in range(mat.shape[1])])
    n, k = mat.shape
    w_expected = float(Q) / (n * (k - 1))
    assert abs(res.estimates["chi2_stat"] - float(Q)) < 1e-6
    assert abs(res.estimates["kendalls_w"] - w_expected) < 1e-4


def test_friedman_no_effect_control(tmp_path: Path) -> None:
    df = _block_wide(2, 30, effects=[0.0, 0.0, 0.0])  # no condition shift
    csv = tmp_path / "frflat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["p_value"] > 0.05
    assert res.estimates["n_sig_pairs"] == 0.0


def test_friedman_long_form_config(tmp_path: Path) -> None:
    # long format: subject/condition/value, condition effect present
    rng = np.random.default_rng(3)
    n_subj = 24
    base = rng.normal(0, 1, n_subj)
    rows = []
    for s in range(n_subj):
        for ci, eff in enumerate([0.0, 2.0, 4.0]):
            rows.append({"subj": f"s{s}", "cond": f"c{ci}",
                         "val": base[s] + eff + rng.normal(0, 0.3)})
    df = pd.DataFrame(rows)
    csv = tmp_path / "frlong.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"subject": "subj", "within": "cond", "outcome": "val"})
    assert res.estimates["n_conditions"] == 3.0
    assert res.estimates["n_subjects"] == float(n_subj)
    assert res.estimates["p_value"] < 0.01


def test_friedman_incomplete_block_dropped(tmp_path: Path) -> None:
    df = _block_wide(4, 20, effects=[0.0, 1.5, 3.0])
    # punch a hole: subject 0 missing cond2 -> that whole block must be dropped
    df.loc[0, "cond2"] = np.nan
    csv = tmp_path / "frhole.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["n_subjects"] == 19.0  # one incomplete block dropped
    assert "完整区块" in res.summary or "剔除" in res.summary


def test_friedman_degrade_two_conditions(tmp_path: Path) -> None:
    # only 2 numeric columns -> cannot do Friedman (needs >=3) -> skip
    df = _block_wide(6, 20, effects=[0.0, 2.0])
    csv = tmp_path / "fr2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "chi2_stat" not in res.estimates
