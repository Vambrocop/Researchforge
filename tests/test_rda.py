"""Tests for rda (Redundancy analysis — constrained ordination via R vegan::rda,
optional + graceful degrade)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions

_VEGAN = rbridge.r_available() and rbridge.r_package_available("vegan")


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="rda",
        method="Redundancy analysis (constrained ordination)",
        domain="ecology",
        family="ecology",
        goal="explain",
        preconditions=Precondition(min_count_cols=2, min_continuous=1, min_rows=6),
    )


def _make_df(seed: int = 0) -> pd.DataFrame:
    """Community composition driven by two environmental predictors -> RDA should
    capture a substantial constrained fraction when vegan is present."""
    rng = np.random.default_rng(seed)
    n = 40
    env1 = rng.normal(0, 1, n)
    env2 = rng.normal(0, 1, n)
    sp0 = np.clip((20 + 6 * env1 + rng.normal(0, 1.5, n)).round(), 0, None).astype(int)
    sp1 = np.clip((20 - 6 * env1 + rng.normal(0, 1.5, n)).round(), 0, None).astype(int)
    sp2 = np.clip((20 + 6 * env2 + rng.normal(0, 1.5, n)).round(), 0, None).astype(int)
    sp3 = np.clip((20 - 6 * env2 + rng.normal(0, 1.5, n)).round(), 0, None).astype(int)
    return pd.DataFrame(
        {"sp0": sp0, "sp1": sp1, "sp2": sp2, "sp3": sp3, "env1": env1, "env2": env2}
    )


def test_rda_graceful_degrade_or_run(tmp_path: Path) -> None:
    """ALWAYS green: with vegan -> real result; without -> honest skip pointing to
    nmds/permanova. Never crashes."""
    csv = tmp_path / "rda.csv"
    _make_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    joined = " ".join(res.summary)

    if _VEGAN:
        assert "global_F" in res.estimates, f"expected RDA estimates; summary={res.summary}"
    else:
        # honest degrade: mentions vegan and points to a pure-Python alternative
        assert "vegan" in joined
        assert "nmds" in joined or "permanova" in joined


@pytest.mark.skipif(not _VEGAN, reason="R vegan not installed")
def test_rda_live(tmp_path: Path) -> None:
    """Live test (only when vegan present): environment-driven community -> the
    constrained fraction is non-trivial and the global test is significant."""
    csv = tmp_path / "rda.csv"
    _make_df().to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))

    assert (Path(res.output_dir) / "rda_variance.csv").exists()
    assert (Path(res.output_dir) / "rda_significance.csv").exists()
    assert res.estimates["constrained_variance_pct"] > 20, (
        f"constrained% too low: {res.estimates['constrained_variance_pct']}"
    )
    assert res.estimates["global_p"] < 0.05, (
        f"global_p should be significant: {res.estimates['global_p']}"
    )


def test_rda_precondition_unmet(tmp_path: Path) -> None:
    """Continuous-only dataset -> count-cols precondition unmet."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(0, 1, 20), "y": rng.normal(0, 1, 20)})
    csv = tmp_path / "cont.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, _unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok
