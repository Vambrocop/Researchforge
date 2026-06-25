import numpy as np
import pandas as pd

from researchforge.profiler import profile_dataset
from researchforge.recommender import (
    apply_diagnostic_ranking,
    build_plan,
    recommend,
    select_top,
)
from researchforge.synth import make_panel

_ORDER = {"green": 0, "yellow": 1, "red": 2}


def test_recommend_panel(tmp_path):
    csv = tmp_path / "panel.csv"
    make_panel(n_units=6, n_periods=6, treated=True, seed=5).to_csv(csv, index=False)
    recs = recommend(profile_dataset(csv))
    by_id = {r.entry.id: r for r in recs}

    assert by_id["did"].feasible
    assert by_id["did"].rigor.light in {"green", "yellow"}
    assert by_id["descriptive_stats"].feasible
    assert recs[0].feasible  # best recommendation first
    lights = [r.rigor.light for r in recs]
    assert lights == sorted(lights, key=_ORDER.get)  # green -> yellow -> red


def test_did_red_on_cross_section(tmp_path):
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 13)],
            "y": [2.0, 4, 5, 4, 5, 7, 8, 9, 8, 9, 11, 12],
        }
    )
    csv = tmp_path / "cs.csv"
    df.to_csv(csv, index=False)
    by_id = {r.entry.id: r for r in recommend(profile_dataset(csv))}

    assert by_id["did"].feasible is False
    assert by_id["did"].rigor.light == "red"
    assert by_id["did"].rigor.unmet  # explains why


# --------------------------------------------------------------------------- #
# diagnostic-aware ranking (smarter auto-selection, deeper)
# --------------------------------------------------------------------------- #
def _overdispersed_csv(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.normal(size=300),
                       "events": rng.negative_binomial(2, 0.2, 300)})  # overdispersed count
    csv = tmp_path / "od.csv"
    df.to_csv(csv, index=False)
    return csv


def test_diagnostic_ranking_boosts_preferred_demotes_argued_against(tmp_path):
    fp = profile_dataset(_overdispersed_csv(tmp_path))
    plan = build_plan(fp)
    recs = apply_diagnostic_ranking(recommend(fp), plan)
    by_id = {r.entry.id: r for r in recs}
    # overdispersion → negbin preferred (fit up), poisson argued against (fit down)
    assert by_id["negative_binomial_regression"].diagnostic_fit > 0
    assert "诊断契合" in by_id["negative_binomial_regression"].diagnostic_note
    assert by_id["poisson_regression"].diagnostic_fit < 0
    # within the same rigor tier, the preferred method now outranks the argued-against one
    order = [r.entry.id for r in recs]
    if by_id["negative_binomial_regression"].rigor.light == by_id["poisson_regression"].rigor.light:
        assert order.index("negative_binomial_regression") < order.index("poisson_regression")


def test_diagnostic_ranking_preserves_rigor_tiers(tmp_path):
    # the nudge must NOT break green→yellow→red primacy
    fp = profile_dataset(_overdispersed_csv(tmp_path))
    recs = apply_diagnostic_ranking(recommend(fp), build_plan(fp))
    lights = [r.rigor.light for r in recs]
    assert lights == sorted(lights, key=_ORDER.get)


def test_recommend_stays_pure_without_plan(tmp_path):
    # recommend() itself is unchanged: no diagnostic_fit applied
    fp = profile_dataset(_overdispersed_csv(tmp_path))
    recs = recommend(fp)
    assert all(r.diagnostic_fit == 0 for r in recs)


def test_select_top_diagnostic_aware_toggle(tmp_path):
    fp = profile_dataset(_overdispersed_csv(tmp_path))
    aware = select_top(fp, top=30, diagnostic_aware=True)
    plain = select_top(fp, top=30, diagnostic_aware=False)
    # diagnostic-aware pool carries the nudge; the plain pool does not
    assert any(r.diagnostic_fit != 0 for r in aware)
    assert all(r.diagnostic_fit == 0 for r in plain)
