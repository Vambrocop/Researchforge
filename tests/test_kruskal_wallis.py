"""Tests for kruskal_wallis: Kruskal-Wallis H + epsilon-squared + Dunn post-hoc.

Known-structure checks: three groups with a clear location shift -> significant H,
and Dunn flags the extreme pair; a no-shift negative control -> non-significant.
Plus config override, an independent epsilon-squared recompute (pins correctness),
and honest degrade (only 2 groups -> skip, non-numeric outcome -> skip).
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
        id="kruskal_wallis",
        method="Kruskal-Wallis H test",
        domain="statistics",
        family="nonparametric",
        goal="explain",
        preconditions=Precondition(requires_group=True, min_continuous=1, min_rows=9),
    )


def test_kw_three_groups_shift_significant_and_dunn(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 40
    # A ~ N(0,1), B ~ N(0.5,1) (close), C ~ N(6,1) (far) -> A vs C clearly differ
    a = rng.normal(0, 1, n)
    b = rng.normal(0.5, 1, n)
    c = rng.normal(6, 1, n)
    y = np.concatenate([a, b, c])
    g = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    df = pd.DataFrame({"y": y, "grp": g})  # y first -> outcome by convention
    csv = tmp_path / "kw3.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    out = Path(res.output_dir)

    assert (out / "kruskal_groups.csv").exists()
    assert (out / "kruskal_dunn_posthoc.csv").exists()
    assert res.estimates["n_groups"] == 3.0
    assert res.estimates["n"] == float(3 * n)
    assert res.estimates["p_value"] < 0.01  # strong overall difference
    assert res.estimates["eta_squared_h"] > 0.1  # large effect (C far out)
    assert res.estimates["n_sig_pairs"] >= 1.0

    # Dunn must flag the A-C (and B-C) extreme pair as significant
    dunn = pd.read_csv(out / "kruskal_dunn_posthoc.csv")
    ac = dunn[((dunn.group_a == "A") & (dunn.group_b == "C")) |
              ((dunn.group_a == "C") & (dunn.group_b == "A"))]
    assert bool(ac.iloc[0]["significant_0.05"])


def test_kw_epsilon_squared_recompute(tmp_path: Path) -> None:
    """Independently recompute eps^2 = (H - k + 1)/(n - k) from scipy's H to pin
    the effect-size formula in the branch."""
    rng = np.random.default_rng(7)
    n = 30
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(2, 1, n), rng.normal(4, 1, n)])
    g = np.array(["g1"] * n + ["g2"] * n + ["g3"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "kweff.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    # recompute from raw groups
    H, _ = sps.kruskal(y[:n], y[n:2 * n], y[2 * n:])
    k, N = 3, 3 * n
    eta2h_expected = (float(H) - k + 1) / (N - k)
    # estimates are stored rounded to 4 dp for display, so compare at that precision
    assert abs(res.estimates["h_stat"] - float(H)) < 1e-3
    assert abs(res.estimates["eta_squared_h"] - eta2h_expected) < 1e-3


def test_kw_no_shift_negative_control(tmp_path: Path) -> None:
    rng = np.random.default_rng(2)
    n = 50
    # three groups, identical distribution -> H0 true -> not significant
    y = rng.normal(0, 1, 3 * n)
    g = np.array(["A"] * n + ["B"] * n + ["C"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "kwflat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert res.estimates["p_value"] > 0.05
    assert res.estimates["n_sig_pairs"] == 0.0


def test_kw_config_override(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 30
    a = np.concatenate([rng.normal(0, 1, n), rng.normal(3, 1, n), rng.normal(6, 1, n)])
    noise = rng.normal(0, 1, 3 * n)
    g = np.array(["lo"] * n + ["mid"] * n + ["hi"] * n)
    df = pd.DataFrame({"target": a, "noise": noise, "factor": g})
    csv = tmp_path / "kwovr.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "target", "group": "factor"})
    assert res.estimates["n_groups"] == 3.0
    assert res.estimates["p_value"] < 0.01


def test_kw_degrade_two_groups(tmp_path: Path) -> None:
    # only 2 groups -> should skip (use mann_whitney), no crash
    rng = np.random.default_rng(4)
    n = 30
    y = np.concatenate([rng.normal(0, 1, n), rng.normal(2, 1, n)])
    g = np.array(["A"] * n + ["B"] * n)
    df = pd.DataFrame({"y": y, "grp": g})
    csv = tmp_path / "kw2.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "y", "group": "grp"})
    assert "跳过" in res.summary
    assert "h_stat" not in res.estimates


def test_kw_degrade_non_numeric_outcome(tmp_path: Path) -> None:
    df = pd.DataFrame({
        "label": ["x", "y", "z"] * 6,
        "grp": ["A", "B", "C"] * 6,
    })
    csv = tmp_path / "kwbad.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"outcome": "label", "group": "grp"})
    assert "跳过" in res.summary
    assert "h_stat" not in res.estimates
