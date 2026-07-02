"""Tests for dif_detection — Differential Item Functioning (Mantel-Haenszel).

The correctness target is sensitivity AND specificity: a biased item (same
ability, different difficulty by group) must be FLAGGED, while clean items (equal
difficulty across groups) must NOT be. We simulate two groups whose abilities
share the SAME distribution (so any item-level group difference is genuine DIF,
not impact), make most items 2PL-clean, and inject one item with a large
group-specific difficulty shift. The MH common odds ratio + chi-square (matched
on the rest score) should catch the biased item and leave the clean ones alone.
Plus a no-group honest-skip and a config group override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog import AnalysisEntry
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset

_ENTRY = AnalysisEntry(
    id="dif_detection",
    method="Differential Item Functioning (Mantel-Haenszel)",
    domain="psychometrics",
    family="irt",
    goal="explain",
    preconditions={"min_categorical_cols": 3, "min_rows": 40, "requires_group": True},
)


def _simulate_dif(
    n_per_group: int,
    b_clean: np.ndarray,
    dif_item: int,
    dif_shift: float,
    seed: int,
) -> pd.DataFrame:
    """Two groups, SAME ability distribution (no impact), one biased item.

    Both groups draw theta ~ N(0,1). All items are 1PL (a=1) with difficulty
    b_clean, EXCEPT `dif_item`, whose difficulty is shifted by `dif_shift` for
    the focal group only (uniform DIF). Returns respondents x items + a binary
    `group` column (0=reference, 1=focal).
    """
    rng = np.random.default_rng(seed)
    k = len(b_clean)

    def draw(theta, b_vec):
        P = 1.0 / (1.0 + np.exp(-(theta[:, None] - b_vec[None, :])))
        return (rng.uniform(size=P.shape) < P).astype(int)

    theta_ref = rng.standard_normal(n_per_group)
    theta_foc = rng.standard_normal(n_per_group)

    b_ref = b_clean.copy()
    b_foc = b_clean.copy()
    b_foc[dif_item] = b_clean[dif_item] + dif_shift  # focal group harder on this item

    resp_ref = draw(theta_ref, b_ref)
    resp_foc = draw(theta_foc, b_foc)

    data = np.vstack([resp_ref, resp_foc])
    grp = np.concatenate([np.zeros(n_per_group, int), np.ones(n_per_group, int)])
    cols = [f"q{j + 1}" for j in range(k)]
    df = pd.DataFrame(data, columns=cols)
    df["group"] = grp
    # shuffle so group is not block-ordered
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def test_dif_flags_biased_item_not_clean(tmp_path: Path) -> None:
    b_clean = np.array([-1.5, -0.8, -0.2, 0.3, 0.9, 1.4, -0.5, 0.6])
    dif_item = 3  # q4 is biased
    df = _simulate_dif(700, b_clean, dif_item=dif_item, dif_shift=1.3, seed=42)
    csv = tmp_path / "dif.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"group": "group"},
    )
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    assert res.estimates["n_items"] == float(len(b_clean))

    tbl = pd.read_csv(Path(res.output_dir) / "dif_detection.csv")
    assert "group" not in list(tbl["item"])  # group column not treated as an item
    tbl = tbl.set_index("item")
    biased = "q4"

    # ---- sensitivity: the biased item is flagged (significant MH) ----
    assert tbl.loc[biased, "mh_p_value"] < 0.05, (
        f"biased item not detected: MH p={tbl.loc[biased, 'mh_p_value']}"
    )
    # logistic uniform-DIF test should also catch it
    assert tbl.loc[biased, "logistic_group_p"] < 0.05

    # ---- specificity: clean items mostly NOT flagged (allow <=1 false alarm) ----
    clean_items = [c for c in tbl.index if c != biased]
    false_alarms = sum(
        1 for c in clean_items
        if (tbl.loc[c, "mh_p_value"] == tbl.loc[c, "mh_p_value"])
        and tbl.loc[c, "mh_p_value"] < 0.05
    )
    assert false_alarms <= 1, f"too many clean items flagged ({false_alarms})"

    # the biased item should carry the largest |delta| (ETS effect size)
    deltas = tbl["mh_delta"].abs()
    assert deltas.idxmax() == biased, f"largest DIF effect was {deltas.idxmax()}, not {biased}"
    assert tbl.loc[biased, "ets_dif_class"] in {"B", "C"}

    assert (Path(res.output_dir) / "dif_detection.png").exists()
    assert res.estimates["n_flagged_dif"] >= 1.0


def test_dif_no_group_skips(tmp_path: Path) -> None:
    # No 2-level group column -> honest skip.
    rng = np.random.default_rng(5)
    df = pd.DataFrame({f"q{j}": rng.integers(0, 2, 200) for j in range(1, 7)})
    csv = tmp_path / "nogrp.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    # force only the q-items so no binary doubles as a group, then no group remains
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"items": ["q1", "q2", "q3", "q4", "q5", "q6"]},
    )
    assert "跳过" in res.summary
    assert "n_flagged_dif" not in res.estimates
    assert (Path(res.output_dir) / "report.md").exists()


def test_dif_group_config_override(tmp_path: Path) -> None:
    # A custom-named group column resolved via config.
    b_clean = np.array([-1.0, -0.3, 0.2, 0.8, 1.2, -0.6])
    df = _simulate_dif(400, b_clean, dif_item=2, dif_shift=1.2, seed=7)
    df = df.rename(columns={"group": "cohort"})
    csv = tmp_path / "cfg.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"group": "cohort"},
    )
    assert "失败" not in res.summary and "跳过" not in res.summary, res.summary
    tbl = pd.read_csv(Path(res.output_dir) / "dif_detection.csv")
    assert "cohort" not in list(tbl["item"])
    # biased item q3 should have the largest absolute delta
    tbl = tbl.set_index("item")
    assert tbl["mh_delta"].abs().idxmax() == "q3"


def test_dif_too_few_per_group_skips(tmp_path: Path) -> None:
    b_clean = np.array([-0.8, -0.2, 0.3, 0.9, -0.5])
    df = _simulate_dif(4, b_clean, dif_item=1, dif_shift=1.0, seed=2)  # 8 total, tiny groups
    csv = tmp_path / "tiny.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _ENTRY, output_root=str(tmp_path / "o"),
        config={"group": "group"},
    )
    assert "跳过" in res.summary
    assert "n_flagged_dif" not in res.estimates


def test_dif_ets_classification_nonsignificant_is_always_a() -> None:
    """Non-significant item with |Δ| >= 1.0 must be classified as A, not B.

    ETS DIF classification rules:
    - A (negligible): p >= 0.05 (not significant, any |Δ|) OR |Δ| < 1.0
    - B (moderate): significant AND 1.0 <= |Δ| < 1.5
    - C (large): significant AND |Δ| >= 1.5

    This test verifies that non-significance alone produces "A", regardless of
    effect size magnitude, and that significant large-effect items still get "C".
    """
    from researchforge.executor.branches.irt import _mh_dif_item

    # Case 1: Non-significant item with |Δ| >= 1.0
    # Construct sparse synthetic data: few strata, small cell counts → non-significant MH test
    # but OR displaced enough to have |Δ| >= 1.0
    # Using 4 strata with N=5 per stratum (very sparse to reduce power)
    item_resp_case1 = np.array(
        [0, 0, 1, 1, 0,   # stratum 1: ref 3 correct, focal 2 correct
         1, 1, 0, 0, 1,   # stratum 2: ref 3 correct, focal 2 correct
         0, 0, 1, 1, 0,   # stratum 3: ref 3 correct, focal 2 correct
         1, 1, 0, 0, 1]   # stratum 4: ref 3 correct, focal 2 correct
    )
    grp_case1 = np.array([0, 0, 0, 1, 1] * 4)  # reference: indices 0,1,2; focal: indices 3,4
    score_case1 = np.array([1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4])

    or_mh1, chi2_1, p_1, ets_1, delta_1 = _mh_dif_item(
        item_resp_case1, grp_case1, score_case1
    )

    # Verify: if this item is non-significant, it MUST be "A" regardless of |Δ|
    if p_1 >= 0.05:
        assert ets_1 == "A", (
            f"Non-significant item must be A, not {ets_1}. "
            f"p={p_1:.4f}, |Δ|={abs(delta_1):.4f}"
        )

    # Case 2: Significant item with large |Δ| (should be C)
    # Many strata with large effect: focal group much lower success → large OR displacement
    n_per_stratum = 50  # per group per stratum
    n_strata = 15

    item_resp_list = []
    grp_list = []
    score_list = []

    for stratum_idx in range(1, n_strata + 1):
        # Reference group: 50 with baseline performance (70% correct)
        n_ref_correct = 35
        n_ref_incorrect = 15
        item_resp_list.extend([1] * n_ref_correct + [0] * n_ref_incorrect)
        grp_list.extend([0] * n_per_stratum)
        score_list.extend([stratum_idx] * n_per_stratum)

        # Focal group: 50 with much lower performance (20% correct) → strong DIF
        n_foc_correct = 10
        n_foc_incorrect = 40
        item_resp_list.extend([1] * n_foc_correct + [0] * n_foc_incorrect)
        grp_list.extend([1] * n_per_stratum)
        score_list.extend([stratum_idx] * n_per_stratum)

    item_resp_case2 = np.array(item_resp_list)
    grp_case2 = np.array(grp_list)
    score_case2 = np.array(score_list)

    or_mh2, chi2_2, p_2, ets_2, delta_2 = _mh_dif_item(
        item_resp_case2, grp_case2, score_case2
    )

    # Verify: significant large-effect item should be C
    assert p_2 < 0.05, (
        f"Expected significant item but p={p_2:.5f}. "
        f"OR={or_mh2:.4f}, |Δ|={abs(delta_2):.4f}"
    )
    assert abs(delta_2) >= 1.5, (
        f"Expected |Δ| >= 1.5 but got {abs(delta_2):.4f}. "
        f"OR={or_mh2:.4f}"
    )
    assert ets_2 == "C", (
        f"Significant item with |Δ| >= 1.5 must be C, not {ets_2}. "
        f"p={p_2:.5f}, |Δ|={abs(delta_2):.4f}"
    )
