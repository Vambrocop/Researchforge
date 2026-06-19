"""Tests for ahp: AHP criteria weighting via the principal eigenvector + CR check.

Hand-checks: a perfectly consistent pairwise matrix -> CR=0 and weights equal the
generating eigenvector; an inconsistent matrix -> CR>0.1 warning in the summary; the
data-proxy fallback when no pairwise is given; config override; degrade gate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="ahp",
        method="Analytic Hierarchy Process (AHP) weighting",
        domain="evaluation",
        family="mcda",
        goal="describe",
        preconditions=Precondition(min_continuous=2, min_rows=3),
    )


def _df(tmp_path: Path) -> Path:
    # A dominates on every criterion. The .5 on each column keeps it "continuous"
    # (an all-distinct all-whole column would be flagged an id by the profiler).
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C", "D"],
            "c1": [10.5, 7.0, 4.0, 1.0],
            "c2": [9.5, 6.0, 5.0, 2.0],
            "c3": [8.5, 7.0, 3.0, 1.0],
        }
    )
    csv = tmp_path / "ahp.csv"
    df.to_csv(csv, index=False)
    return csv


def test_ahp_consistent_matrix_zero_CR_known_eigenvector(tmp_path: Path) -> None:
    # Perfectly consistent reciprocal matrix a_ij = w_i / w_j for w=[0.5,0.3,0.2].
    w_true = np.array([0.5, 0.3, 0.2])
    A = (w_true[:, None] / w_true[None, :]).tolist()
    csv = _df(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"pairwise": A}
    )
    out = Path(res.output_dir)

    weights = pd.read_csv(out / "weights.csv")
    assert abs(weights["ahp_weight"].sum() - 1.0) < 1e-6
    got = weights.set_index("criterion")["ahp_weight"]
    assert abs(got["c1"] - 0.5) < 2e-3
    assert abs(got["c2"] - 0.3) < 2e-3
    assert abs(got["c3"] - 0.2) < 2e-3
    # Perfectly consistent -> lambda_max ~ n, CI ~ 0, CR ~ 0.
    assert res.estimates["CR"] < 1e-3
    assert abs(res.estimates["lambda_max"] - 3.0) < 1e-3


def test_ahp_inconsistent_matrix_warns(tmp_path: Path) -> None:
    # A moderately inconsistent 3x3 reciprocal matrix (CR=0.117 > 0.1, verified by
    # hand against lambda_max=3.136, CI=(lmax-3)/2, RI(3)=0.58). c1>c2 and c2>c3 only
    # mildly, but c1>>c3 (a13=5) breaks transitivity enough to push CR over 0.1.
    A = [
        [1.0, 3.0, 5.0],
        [1.0 / 3.0, 1.0, 5.0],
        [1.0 / 5.0, 1.0 / 5.0, 1.0],
    ]
    csv = _df(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"pairwise": A}
    )
    assert res.estimates["CR"] > 0.1
    assert ("CR>0.1" in res.summary) or ("不一致" in res.summary)


def test_ahp_data_proxy_fallback_discloses(tmp_path: Path) -> None:
    # No pairwise given -> data proxy (ratio of means), CR ~ 0 (rank-1), disclosed.
    csv = _df(tmp_path)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    weights = pd.read_csv(out / "weights.csv")
    assert abs(weights["ahp_weight"].sum() - 1.0) < 1e-6
    assert res.estimates["CR"] < 1e-3  # rank-1 proxy is perfectly consistent
    assert "数据代理" in res.summary
    scores = pd.read_csv(out / "ahp_scores.csv")
    assert scores.sort_values("rank").iloc[0]["alternative"] == "A"  # A dominates


def test_ahp_weights_drive_ranking(tmp_path: Path) -> None:
    # Heavily weight c2 (where B/C are relatively strong) via a consistent matrix
    # putting all weight on c2 -> ranking should follow c2 ordering.
    df = pd.DataFrame(
        {
            "alt": ["A", "B", "C"],
            "c1": [10.5, 5.0, 1.0],   # favours A
            "c2": [1.0, 5.0, 10.5],   # favours C
        }
    )
    csv = tmp_path / "ahp2.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    w_true = np.array([0.05, 0.95])  # almost all weight on c2
    A = (w_true[:, None] / w_true[None, :]).tolist()
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"), config={"pairwise": A}
    )
    out = Path(res.output_dir)
    scores = pd.read_csv(out / "ahp_scores.csv")
    assert scores.sort_values("rank").iloc[0]["alternative"] == "C"


def test_ahp_precondition_unmet(tmp_path: Path) -> None:
    df = pd.DataFrame({"only_one": [1.1, 2.2, 3.3, 4.4]})  # < 2 numeric criteria
    csv = tmp_path / "one.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
    assert any("连续" in u for u in unmet)
