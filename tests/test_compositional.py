"""compositional 家族——闭合数据上普通统计量会说谎，而引擎当时把它排在第 2 位。

Dogfood measurement on a sand / silt / clay frame (rows summing to 100), catalog as it stood:

    catalog methods for compositional data    0 of 306
    `correlation`                             rank 2 of 306, with no warning at all
    raw corr(sand, silt) = -0.292
    raw corr(sand, clay) = -0.657

Both negatives are artefacts of closure, not findings. With a constant row sum, every
component satisfies Sum_j cov(x_i, x_j) = 0 — the covariances of one part with the rest are
FORCED to sum to zero, so most pairs come out negative whatever the underlying quantities do.
Pearson wrote this down in 1897; the engine was serving it as its top substantive method.

The tests below pin both halves, because either alone is useless: the methods have to be
CORRECT (checked against closed-form properties of the Aitchison geometry), and they have to
be SURFACED while the correlation methods are pushed down and the reason disclosed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend
from researchforge.recommender.affinity import has_closed_composition

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _soil(n=400, seed=1, zeros=0.0):
    """sand + silt + clay = 100, plus an unrelated `yield_t` so the closed set is a SUBSET."""
    rng = np.random.default_rng(seed)
    raw = rng.dirichlet([3, 2, 5], n) * 100
    if zeros:
        mask = rng.random(n) < zeros
        raw[mask, 1] = 0.0
        raw[mask] = raw[mask] / raw[mask].sum(axis=1, keepdims=True) * 100
    return pd.DataFrame({"sand": raw[:, 0].round(2), "silt": raw[:, 1].round(2),
                         "clay": raw[:, 2].round(2),
                         "yield_t": (2 + 0.03 * raw[:, 0] + rng.normal(0, 0.4, n)).round(3)})


def _open_positive(n=400, seed=2):
    """Positive columns that do NOT close — the shape that must not be mistaken for one."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.gamma(2, 1, n).round(3), "y": rng.gamma(3, 1, n).round(3),
                         "z": rng.gamma(4, 1, n).round(3)})


# ── the shape fact ───────────────────────────────────────────────────────────
def test_closure_is_detected_as_a_subset_not_the_whole_frame(tmp_path):
    """The closed set is usually a SUBSET: testing every numeric column at once finds
    nothing here, because `yield_t` sits beside the three parts."""
    fp = _fp(_soil(), tmp_path, name="soil.csv")
    assert fp.closed_components == ["sand", "silt", "clay"]
    assert has_closed_composition(fp) is True


@pytest.mark.parametrize("frame,name", [(_open_positive, "open"),
                                        (lambda: pd.DataFrame(
                                            {f"q{i}": np.random.default_rng(3).integers(1, 6, 300)
                                             for i in range(5)}), "likert")])
def test_ordinary_positive_columns_are_not_a_composition(frame, name, tmp_path):
    fp = _fp(frame(), tmp_path, name=f"{name}.csv")
    assert fp.closed_components == []
    assert has_closed_composition(fp) is False


def test_a_two_part_frame_is_not_enough(tmp_path):
    """p + q = 1 is closed but degenerate — one free dimension, nothing to analyse."""
    rng = np.random.default_rng(4)
    p = rng.beta(2, 3, 300)
    fp = _fp(pd.DataFrame({"p": p.round(4), "q": (1 - p).round(4)}), tmp_path, name="two.csv")
    assert fp.closed_components == []


# ── the statistics ───────────────────────────────────────────────────────────
def test_the_profile_reports_closure_free_association(tmp_path):
    fp = _fp(_soil(), tmp_path, name="prof.csv")
    res = run_analysis(fp, _CAT.by_id("compositional_profile"), output_root=str(tmp_path / "o"))
    e = res.estimates
    assert e["n_components"] == 3.0
    # the Aitchison mean is itself a composition — it closes
    means = [e[f"aitchison_mean_{c}"] for c in ("sand", "silt", "clay")]
    assert sum(means) == pytest.approx(1.0, abs=1e-3)
    # every raw pair comes out negative here; that IS the artefact being disclosed
    assert e["raw_corr_negative_pairs"] == e["raw_corr_pairs"] == 3.0
    assert "variation_matrix.csv" in res.files
    assert "伪相关" in res.summary and "Pearson 1897" in res.summary


def test_the_variation_matrix_matches_its_definition(tmp_path):
    """tau_ij = Var(ln(x_i/x_j)) — computed from the CSV the run wrote, against numpy."""
    df = _soil()
    fp = _fp(df, tmp_path, name="var.csv")
    res = run_analysis(fp, _CAT.by_id("compositional_profile"), output_root=str(tmp_path / "p"))
    import pathlib

    T = pd.read_csv(pathlib.Path(res.output_dir) / "variation_matrix.csv", index_col=0)
    A = df[["sand", "silt", "clay"]].to_numpy(float)
    A = A / A.sum(axis=1, keepdims=True)
    L = np.log(A)
    assert T.loc["sand", "clay"] == pytest.approx(float(np.var(L[:, 0] - L[:, 2], ddof=1)),
                                                  rel=1e-6)
    assert T.loc["sand", "sand"] == pytest.approx(0.0, abs=1e-12)   # tau_ii = 0
    assert T.loc["silt", "sand"] == pytest.approx(T.loc["sand", "silt"], rel=1e-12)  # symmetric


def test_aitchison_pca_uses_d_minus_one_dimensions(tmp_path):
    """CLR coordinates carry a zero-sum constraint, so D parts give D-1 free directions.
    With D=3 the first two components must explain everything."""
    fp = _fp(_soil(), tmp_path, name="pca.csv")
    res = run_analysis(fp, _CAT.by_id("aitchison_pca"), output_root=str(tmp_path / "q"))
    e = res.estimates
    assert e["pc1_pc2_explained"] == pytest.approx(1.0, abs=1e-6)
    assert e["clr_last_eigenvalue"] == pytest.approx(0.0, abs=1e-8)
    # ...and the raw-parts decomposition is degenerate too, which is the point being made
    assert e["raw_last_eigenvalue"] < 1e-3
    assert "退化" in res.summary


def test_too_many_zeros_is_refused_rather_than_patched(tmp_path):
    """Log-ratios need strictly positive parts. Below the threshold zeros are repaired by
    multiplicative replacement and disclosed; above it, repairing becomes inventing."""
    fp = _fp(_soil(zeros=0.35), tmp_path, name="zeros.csv")
    res = run_analysis(fp, _CAT.by_id("compositional_profile"), output_root=str(tmp_path / "r"))
    assert "跳过：" in res.summary and "编造" in res.summary
    assert not res.estimates


def test_rare_zeros_are_replaced_and_disclosed(tmp_path):
    fp = _fp(_soil(zeros=0.03), tmp_path, name="fewzeros.csv")
    res = run_analysis(fp, _CAT.by_id("compositional_profile"), output_root=str(tmp_path / "s"))
    assert "乘法替换" in res.summary, res.summary[:300]
    assert res.estimates["n_components"] == 3.0


def test_config_components_wins(tmp_path):
    rng = np.random.default_rng(6)
    n = 300
    raw = rng.dirichlet([2, 2, 2, 2], n)
    df = pd.DataFrame({f"p{i}": raw[:, i].round(5) for i in range(4)})
    fp = _fp(df, tmp_path, name="cfg.csv")
    res = run_analysis(fp, _CAT.by_id("compositional_profile"), output_root=str(tmp_path / "t"),
                       config={"components": ["p0", "p1", "p2"]})
    assert res.estimates["n_components"] == 3.0


# ── selection: surface the family, demote what closure invalidates ──────────
def _ranks(fp, ids):
    got = [getattr(getattr(r, "entry", r), "id", "?") for r in recommend(fp, _CAT)]
    return {i: (got.index(i) + 1 if i in got else None) for i in ids}, got


def test_the_compositional_family_is_surfaced_and_correlation_is_pushed_down(tmp_path):
    """Measured before this tilt: `correlation` rank 2 of 306 with no warning; the two
    compositional methods did not exist. Both halves matter — surfacing the right method is
    only half the job when the wrong one is sitting at the top."""
    fp = _fp(_soil(), tmp_path, name="sel.csv")
    r, got = _ranks(fp, ["compositional_profile", "aitchison_pca", "correlation",
                         "correlation_matrix"])
    assert r["compositional_profile"] is not None and r["compositional_profile"] <= 5, got[:6]
    assert r["aitchison_pca"] is not None and r["aitchison_pca"] <= 5, got[:6]
    assert r["correlation"] > 50, r
    assert r["correlation_matrix"] > 50, r


def test_the_closure_warning_reaches_the_recommendation(tmp_path):
    """A demotion the reader cannot see is not a disclosure."""
    fp = _fp(_soil(), tmp_path, name="note.csv")
    for rec in recommend(fp, _CAT):
        if getattr(getattr(rec, "entry", rec), "id", "") == "correlation":
            note = " ".join(str(getattr(rec.score, "note", "")).split())
            assert "闭合成分数据" in note and "Pearson 1897" in note, note[:200]
            assert "compositional_profile" in note
            return
    pytest.fail("correlation was not in the recommendations at all")


def test_open_data_is_left_alone(tmp_path):
    """The symmetric half: correlation must keep its normal standing on ordinary data."""
    fp = _fp(_open_positive(), tmp_path, name="openrank.csv")
    r, _ = _ranks(fp, ["correlation", "compositional_profile"])
    assert r["correlation"] is not None and r["correlation"] <= 20, r
    assert r["compositional_profile"] is None or r["compositional_profile"] > 20, r
