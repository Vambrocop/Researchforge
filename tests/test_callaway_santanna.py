"""Tests for callaway_santanna: Callaway & Sant'Anna (2021) group-time ATT via
the R `did` package. The R path is OPTIONAL + graceful-degrade per CLAUDE.md, so
the precondition / honest-degrade tests always run (even without R); the
empirical recovery test only runs when R + `did` are installed.

Empirical check: a synthetic STAGGERED-adoption panel with a KNOWN dynamic
effect — units adopt at different cohorts, the treatment effect is 0 before
treatment and jumps to +tau (constant) once treated, on top of unit fixed
effects and a common time trend (so untreated potential outcomes are parallel).
We assert (a) the analysis completes, (b) the overall ATT recovers ~tau,
(c) pre-treatment event-study estimates hover near 0 while post are near tau.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import rbridge, run_analysis
from researchforge.profiler import profile_dataset
from researchforge.recommender.match import check_preconditions

TAU = 3.0  # true post-treatment ATT (constant across exposure)


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="callaway_santanna",
        method="Callaway & Sant'Anna (2021) — group-time ATT (staggered DiD)",
        domain="economics",
        family="causal",
        goal="explain",
        preconditions=Precondition(is_panel=True, requires_treatment=True, min_rows=60),
    )


def _make_staggered_panel(
    tau: float = TAU, n_per_cohort: int = 12, n_time: int = 10, seed: int = 0
) -> pd.DataFrame:
    """Staggered adoption with a KNOWN constant dynamic effect.

    Cohorts first-treated at t=4 and t=7, plus a never-treated group. Outcome:
        y_it = unit_FE_i + 0.5*t (common trend) + tau*1[t >= g_i] + noise
    Parallel trends hold in the untreated potential outcome (unit FE + common
    trend), and the post effect is a clean +tau jump (0 pre). We emit BOTH an
    explicit gname column ('first_treat', 0 = never) and a 0/1 'treated'
    indicator so the derivation path can be exercised too.
    """
    rng = np.random.default_rng(seed)
    times = list(range(1, n_time + 1))
    cohorts = [4, 7, 0]  # 0 = never-treated
    rows = []
    uid = 0
    for g in cohorts:
        for _ in range(n_per_cohort):
            unit_fe = rng.normal(0.0, 1.0)
            for t in times:
                treated_now = 1 if (g != 0 and t >= g) else 0
                y = (
                    unit_fe
                    + 0.5 * t
                    + tau * treated_now
                    + rng.normal(0.0, 0.4)
                )
                rows.append(
                    {
                        "firm": uid,
                        "year": t,
                        "y": round(float(y), 5),
                        "first_treat": g,       # explicit gname (0 = never)
                        "treated": treated_now,  # 0/1 indicator for derivation
                    }
                )
            uid += 1
    return pd.DataFrame(rows)


def _r_did() -> bool:
    return rbridge.r_available() and rbridge.r_package_available("did")


def test_callaway_santanna_recovers_att_explicit_gname(tmp_path: Path) -> None:
    """With an explicit first-treatment-period column, CS must recover overall
    ATT ~ tau and a flat-near-0 pre-trend / near-tau post event study."""
    if not _r_did():
        pytest.skip("R `did` package not available")

    df = _make_staggered_panel(tau=TAU)
    csv = tmp_path / "stag.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat"},
    )
    out = Path(res.output_dir)

    assert (out / "callaway_santanna_event_study.csv").exists(), res.summary
    att = res.estimates.get("att_overall")
    assert att is not None, res.summary
    # (b) overall ATT recovers ~tau (sign + roughly)
    assert att > 0, f"ATT wrong sign: {att}"
    assert abs(att - TAU) < 0.9, f"overall ATT did not recover tau={TAU}: got {att}"

    # (c) pre-treatment event-study ~0, post ~tau
    es = pd.read_csv(out / "callaway_santanna_event_study.csv")
    pre = es[es["event_time"] < 0]
    post = es[es["event_time"] >= 0]
    assert len(pre) >= 1 and len(post) >= 1, f"missing leads/lags: {es}"
    # pre-treatment leads should hover near 0 (parallel-trends evidence). The mean
    # is the robust check; extreme-e leads have few obs so allow a looser per-point band.
    assert abs(pre["att"].mean()) < 0.7, f"pre-trend not near 0: {pre[['event_time','att']]}"
    assert pre["att"].abs().max() < 1.2, f"a pre-trend lead far from 0: {pre[['event_time','att']]}"
    assert abs(post["att"].mean() - TAU) < 0.9, f"post ATT not near tau: {post[['event_time','att']]}"
    # pre-trend flag is reported (a single extreme-e lead can be a chance false
    # positive, so we assert it exists rather than forcing it clean)
    assert "pretrend_violation" in res.estimates


def test_callaway_santanna_recovers_att_derived_gname(tmp_path: Path) -> None:
    """Same recovery but gname DERIVED from a 0/1 treatment indicator
    (first period each unit turns on; 0 = never)."""
    if not _r_did():
        pytest.skip("R `did` package not available")

    df = _make_staggered_panel(tau=TAU, seed=1)
    csv = tmp_path / "stag2.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "treatment": "treated"},
    )
    att = res.estimates.get("att_overall")
    assert att is not None, res.summary
    assert att > 0 and abs(att - TAU) < 0.9, f"derived-gname ATT off: {att}"
    # not-yet-treated control auto-selected only if no never; here never-treated exist
    assert res.estimates.get("n_never_treated", 0.0) > 0


def test_callaway_santanna_not_yet_treated_control(tmp_path: Path) -> None:
    """control_group='notyettreated' must also recover ~tau."""
    if not _r_did():
        pytest.skip("R `did` package not available")

    df = _make_staggered_panel(tau=TAU, seed=3)
    csv = tmp_path / "stag3.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={
            "unit": "firm", "time": "year", "outcome": "y",
            "gname": "first_treat", "control_group": "notyettreated",
        },
    )
    att = res.estimates.get("att_overall")
    assert att is not None, res.summary
    assert att > 0 and abs(att - TAU) < 0.9, f"not-yet-treated ATT off: {att}"
    assert "尚未处理" in res.summary  # control group disclosed


def test_callaway_santanna_degrades_without_did(tmp_path: Path) -> None:
    """Without R/`did` the handler must degrade honestly (no crash) and point at
    the pure-Python staggered_did / event_study alternatives. If `did` IS
    installed this asserts the success path produced an ATT instead."""
    df = _make_staggered_panel(tau=TAU)
    csv = tmp_path / "stag.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    res = run_analysis(
        fp, _entry(), output_root=str(tmp_path / "o"),
        config={"unit": "firm", "time": "year", "outcome": "y", "gname": "first_treat"},
    )
    if not _r_did():
        assert "did" in res.summary
        assert ("staggered_did" in res.summary) or ("event_study" in res.summary)
    else:
        assert "att_overall" in res.estimates


def test_callaway_santanna_precondition_unmet_not_panel(tmp_path: Path) -> None:
    """A flat cross-section (no panel structure) must fail the precondition AND,
    if executed anyway, the handler must skip honestly without crashing — runs
    even without R."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "y": rng.normal(0, 1, 50),
            "x": rng.normal(0, 1, 50),
            "treated": rng.integers(0, 2, 50),
        }
    )
    csv = tmp_path / "flat.csv"
    df.to_csv(csv, index=False)

    fp = profile_dataset(csv)
    ok, unmet = check_preconditions(fp, _entry().preconditions)
    assert not ok and unmet, "flat cross-section should not satisfy panel precondition"

    # executing anyway must degrade honestly (no panel -> honest skip message)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "Callaway-Sant'Anna" in res.summary
    assert "att_overall" not in res.estimates
