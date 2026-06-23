"""Design-aware field-trial ANOVA: RCBD, Latin square, split-plot.

Known-structure data with planted effects. The load-bearing check is split-plot's
TWO error strata: we independently verify the whole-plot F equals the A F-statistic
from an RCBD ANOVA on the whole-plot means (the whole-plot analysis IS an RCBD on
whole-plot totals — same error stratum, identical F), which a naive two-way ANOVA
would get wrong.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _run(csv: Path, aid: str, tmp_path: Path, config=None):
    fp = profile_dataset(csv)
    return run_analysis(fp, _CAT.by_id(aid), output_root=str(tmp_path / "o"), config=config)


# --------------------------------------------------------------------------- #
# RCBD
# --------------------------------------------------------------------------- #
def _rcbd_df(seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trt_eff = {"T1": 0.0, "T2": 2.0, "T3": 4.0, "T4": 6.0}
    blk_eff = {f"B{i}": float(i) for i in range(1, 6)}
    rows = []
    for blk, be in blk_eff.items():
        for trt, te in trt_eff.items():
            rows.append({"treatment": trt, "block": blk,
                         "yield": 10 + te + be + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def test_rcbd_detects_treatment_and_blocking(tmp_path: Path) -> None:
    csv = tmp_path / "rcbd.csv"
    _rcbd_df().to_csv(csv, index=False)
    res = _run(csv, "rcbd_anova", tmp_path)
    e = res.estimates
    assert e["p_treatment"] < 0.05           # strong planted treatment effect
    assert e["n_treatments"] == 4.0 and e["n_blocks"] == 5.0
    assert e["relative_efficiency_blocking"] > 1.0   # graded blocks -> blocking helps
    assert (Path(res.output_dir) / "rcbd_anova_table.csv").exists()
    assert (Path(res.output_dir) / "treatment_means.csv").exists()


def test_rcbd_null_treatment_not_significant(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    rows = []
    for blk in [f"B{i}" for i in range(1, 6)]:
        be = float(blk[-1])
        for trt in ["T1", "T2", "T3", "T4"]:
            rows.append({"treatment": trt, "block": blk, "yield": 10 + be + rng.normal(0, 1)})
    csv = tmp_path / "null.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = _run(csv, "rcbd_anova", tmp_path)
    assert res.estimates["p_treatment"] > 0.05


def test_rcbd_degrades_without_factors(tmp_path: Path) -> None:
    csv = tmp_path / "bad.csv"
    pd.DataFrame({"yield": np.arange(10.0)}).to_csv(csv, index=False)
    res = _run(csv, "rcbd_anova", tmp_path)
    assert "失败" in res.summary
    assert "p_treatment" not in res.estimates


# --------------------------------------------------------------------------- #
# Latin square
# --------------------------------------------------------------------------- #
def _latin_df(t=4, seed=1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    trt_eff = [0.0, 3.0, 6.0, 9.0][:t]
    rows = []
    for i in range(t):
        for j in range(t):
            k = (i + j) % t          # standard cyclic Latin square
            rows.append({"row": f"R{i}", "col": f"C{j}", "treatment": f"T{k}",
                         "yield": 20 + trt_eff[k] + 0.5 * i - 0.3 * j + rng.normal(0, 0.4)})
    return pd.DataFrame(rows)


def test_latin_square_detects_treatment(tmp_path: Path) -> None:
    csv = tmp_path / "latin.csv"
    _latin_df(4).to_csv(csv, index=False)
    res = _run(csv, "latin_square_anova", tmp_path,
               config={"response": "yield", "treatment": "treatment", "row": "row", "col": "col"})
    e = res.estimates
    assert e["square_size"] == 4.0
    assert e["p_treatment"] < 0.05
    assert (Path(res.output_dir) / "latin_square_anova_table.csv").exists()


def test_latin_square_rejects_non_latin_with_matching_marginals(tmp_path: Path) -> None:
    # t×t grid with matching marginals (t treatments, t rows, t cols, n=t²) but
    # treatment aliased to the column -> NOT a Latin square (treatment confounded
    # with the column block). Must be caught, not silently analyzed.
    t = 4
    rng = np.random.default_rng(5)
    rows = []
    for i in range(t):
        for j in range(t):
            rows.append({"row": f"R{i}", "col": f"C{j}", "treatment": f"T{j}",
                         "yield": 20 + 2.0 * j + rng.normal(0, 0.4)})
    csv = tmp_path / "nonlatin.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    res = _run(csv, "latin_square_anova", tmp_path,
               config={"response": "yield", "treatment": "treatment", "row": "row", "col": "col"})
    assert "非真正拉丁方" in res.summary
    assert "p_treatment" not in res.estimates


def test_latin_square_rejects_non_square(tmp_path: Path) -> None:
    # 4 treatments but only 3 rows -> not a t×t square
    df = _latin_df(4)
    df = df[df["row"] != "R3"]
    csv = tmp_path / "notsq.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "latin_square_anova", tmp_path,
               config={"response": "yield", "treatment": "treatment", "row": "row", "col": "col"})
    assert "失败" in res.summary or "跳过" in res.summary


# --------------------------------------------------------------------------- #
# Split-plot — the two error strata
# --------------------------------------------------------------------------- #
def _split_plot_df(r=4, a=3, b=3, seed=2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    whole_eff = [2.0 * i for i in range(a)]      # 0,2,4,...
    sub_eff = [5.0 * i for i in range(b)]        # 0,5,10,... strong sub-plot effect
    rows = []
    for rep in range(r):
        block_eff = rng.normal(0, 1.0)
        for ai in range(a):
            wp_err = rng.normal(0, 1.0)          # whole-plot error (block×A)
            for bi in range(b):
                y = (30 + block_eff + whole_eff[ai] + wp_err
                     + sub_eff[bi] + rng.normal(0, 0.5))
                rows.append({"block": f"R{rep}", "whole_plot": f"A{ai}",
                             "sub_plot": f"B{bi}", "yield": y})
    return pd.DataFrame(rows)


def test_split_plot_strata_and_effects(tmp_path: Path) -> None:
    df = _split_plot_df(r=4, a=3, b=3)
    csv = tmp_path / "sp.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "split_plot_anova", tmp_path,
               config={"response": "yield", "block": "block",
                       "whole_plot": "whole_plot", "sub_plot": "sub_plot"})
    e = res.estimates
    # correct error-stratum degrees of freedom
    assert e["df_whole_plot_error"] == (4 - 1) * (3 - 1)        # (r-1)(a-1) = 6
    assert e["df_sub_plot_error"] == 3 * (4 - 1) * (3 - 1)      # a(r-1)(b-1) = 12
    # strong planted sub-plot effect is significant
    assert e["p_sub_plot"] < 0.05
    tab = pd.read_csv(Path(res.output_dir) / "split_plot_anova_table.csv")
    assert len(tab) == 6
    assert set(tab["tested_against"].dropna()) >= {"vs WP error", "vs sub error"}


def test_split_plot_whole_plot_F_matches_rcbd_on_whole_means(tmp_path: Path) -> None:
    """Independent check: the whole-plot analysis is an RCBD on the whole-plot means,
    so its A F-statistic must equal split-plot's f_whole_plot."""
    df = _split_plot_df(r=5, a=3, b=4, seed=7)
    csv = tmp_path / "sp2.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "split_plot_anova", tmp_path,
               config={"response": "yield", "block": "block",
                       "whole_plot": "whole_plot", "sub_plot": "sub_plot"})
    f_wp = res.estimates["f_whole_plot"]

    import statsmodels.api as sm
    from statsmodels.formula.api import ols

    means = (df.groupby(["block", "whole_plot"])["yield"].mean().reset_index()
             .rename(columns={"yield": "y"}))
    m = ols("y ~ C(whole_plot) + C(block)", data=means).fit()
    aov = sm.stats.anova_lm(m, typ=2)
    f_a_ref = float(aov.loc["C(whole_plot)", "F"])
    assert abs(f_wp - f_a_ref) < 1e-6


def test_split_plot_unbalanced_skips_honestly(tmp_path: Path) -> None:
    df = _split_plot_df(r=4, a=3, b=3)
    df = df.iloc[:-1]                    # drop one cell -> unbalanced
    csv = tmp_path / "unbal.csv"
    df.to_csv(csv, index=False)
    res = _run(csv, "split_plot_anova", tmp_path,
               config={"response": "yield", "block": "block",
                       "whole_plot": "whole_plot", "sub_plot": "sub_plot"})
    assert "跳过" in res.summary
    assert "f_whole_plot" not in res.estimates
