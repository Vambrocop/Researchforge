"""Golden-selection regression suite — a falsifiable safety net for the recommender.

Stage 1 of the smarter-auto-selection plan. Each case is a small dataset whose
*correct* method family is unambiguous from the data structure (panel → panel
estimators, survival data → survival models, an edge list → network methods, …).
The test asserts that at least one appropriate method appears in the top-K of the
feasible recommendations the engine returns (`select_top`, the real CLI/web path).

This pins selection QUALITY so later `fit`/ranking changes (Stages 3–4) can't quietly
regress it — and measures the gap they must close. Cases the current selector already
gets right are strict assertions (the net); cases it currently gets WRONG are marked
``xfail(strict=True)`` with the reason, so when a later stage fixes one it will XPASS
and force us to promote it to a hard assertion (a ratchet that proves the improvement).

Datasets are generated deterministically (fixed seed) so the suite is reproducible;
the "golden" part is the committed accept-sets + currently-correct flags below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from researchforge.profiler import profile_dataset
from researchforge.recommender import select_top

_TOP_K = 6


# ── dataset builders (each returns a DataFrame with an unambiguous "right" method) ──
def _panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n = 120
    firm = np.repeat(np.arange(20), 6)
    year = np.tile(np.arange(6), 20)
    fe = np.repeat(rng.normal(0, 1, 20), 6)
    cap = rng.normal(0, 1, n)
    return pd.DataFrame({"firm": firm, "year": year, "cap": cap.round(3),
                         "sales": rng.normal(0, 1, n).round(3),
                         "invest": (0.5 * cap + fe + rng.normal(0, 0.5, n)).round(3)})


def _timeseries() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    t = np.arange(140)
    return pd.DataFrame({"month": t, "sales": (np.cumsum(rng.normal(0.1, 1, 140)) + 50).round(3)})


def _binary_outcome() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 220
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n); x3 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.4 + 0.9 * x1 - 0.6 * x2)))
    return pd.DataFrame({"approved": rng.binomial(1, p), "income": x1.round(3),
                         "age": (40 + 8 * x2).round(3), "score": x3.round(3)})


def _overdispersed_count() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 220
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n)
    mu = np.exp(0.6 + 0.4 * x1)
    y = rng.poisson(mu) + rng.poisson(mu * 2)  # var >> mean
    return pd.DataFrame({"visits": y, "x1": x1.round(3), "x2": x2.round(3)})


def _survival() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 220
    dur = rng.exponential(10, n).round(2)
    return pd.DataFrame({"duration": dur, "event": rng.binomial(1, 0.6, n),
                         "age": (60 + 10 * rng.normal(0, 1, n)).round(2)})


def _two_group() -> pd.DataFrame:
    rng = np.random.default_rng(6)
    n = 160
    g = rng.binomial(1, 0.5, n)
    return pd.DataFrame({"group": g, "outcome": (rng.normal(0, 1, n) + 0.6 * g).round(3)})


def _edgelist() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 320
    return pd.DataFrame({"source": [f"u{i}" for i in rng.integers(0, 40, n)],
                         "target": [f"u{i}" for i in rng.integers(0, 40, n)]})


def _geo() -> pd.DataFrame:
    rng = np.random.default_rng(8)
    n = 160
    return pd.DataFrame({"lat": rng.uniform(30, 40, n).round(4),
                         "lon": rng.uniform(-120, -110, n).round(4),
                         "temp": rng.normal(15, 5, n).round(3)})


def _linear_regression() -> pd.DataFrame:
    # plain cross-section regression: a guard that the smarter ranking doesn't over-rotate
    # toward specialised methods and drop ordinary regressors off the top.
    rng = np.random.default_rng(20)
    n = 150
    x1 = rng.normal(0, 1, n); x2 = rng.normal(0, 1, n); x3 = rng.normal(0, 1, n)
    y = 2 + 1.5 * x1 - 0.8 * x2 + rng.normal(0, 1, n)
    return pd.DataFrame({"y": y.round(3), "x1": x1.round(3), "x2": x2.round(3), "x3": x3.round(3)})


def _two_categorical() -> pd.DataFrame:
    # ordinary cross-section with two LOW-cardinality categoricals + a continuous outcome.
    # Guard: this must NOT be read as an edge list (few distinct values ≠ node ids), so
    # network methods must not float to the top via the structure floor.
    rng = np.random.default_rng(21)
    n = 160
    region = rng.choice(["N", "S", "E", "W"], n)
    sector = rng.choice(["a", "b", "c"], n)
    y = rng.normal(0, 1, n) + (region == "N") * 0.5
    return pd.DataFrame({"region": region, "sector": sector, "y": y.round(3)})


def _case(name, build, accept, currently_ok, why, reject=None):
    payload = {"name": name, "build": build, "accept": set(accept), "reject": set(reject or ())}
    marks = () if currently_ok else (pytest.mark.xfail(reason=why, strict=True),)
    return pytest.param(payload, id=name, marks=marks)


# accept-sets = methods that are an appropriate primary choice for that data structure
GOLDEN = [
    _case("panel", _panel,
          {"panel_fixed_effects", "random_effects", "first_difference", "mundlak",
           "dynamic_panel_gmm", "hausman_taylor", "system_gmm"},
          currently_ok=True, why=""),
    _case("timeseries", _timeseries,
          {"arima", "exponential_smoothing", "theta_method", "garch", "bayesian_state_space"},
          currently_ok=True, why=""),
    _case("overdispersed_count", _overdispersed_count,
          {"negative_binomial_regression", "zero_inflated_negbin", "zero_inflated_poisson", "tweedie_glm"},
          currently_ok=True, why=""),
    _case("binary_outcome", _binary_outcome,
          {"logistic_regression"},
          currently_ok=True, why=""),  # Stage 4: fit-driven ranking surfaces logistic
    _case("survival", _survival,
          {"survival_analysis", "parametric_survival", "cox_ph_diagnostics", "stratified_cox",
           "time_varying_cox", "competing_risks", "rmst", "bayesian_survival"},
          currently_ok=True, why=""),  # Stage 4: survival family affinity surfaces these
    _case("two_group", _two_group,
          {"group_comparison", "anova_oneway", "mann_whitney", "kruskal_wallis"},
          currently_ok=True, why=""),  # Stage 4: requires_group precondition bonus
    _case("edgelist", _edgelist,
          {"community_detection", "centrality_suite", "network_analysis", "link_prediction",
           "stochastic_block_model", "ergm", "epidemic_model"},
          currently_ok=True, why=""),  # Stage 4: structure-floor for requires_edgelist methods
    _case("geo", _geo,
          {"moran_i", "local_moran", "kriging", "idw_interpolation", "gwr", "spatial_regression",
           "getis_ord", "getis_ord_gi"},
          currently_ok=True, why=""),  # Stage 4: spatial family + requires_geo bonus
    _case("linear_regression", _linear_regression,
          {"ols_regression", "robust_regression", "regularized_regression", "random_forest",
           "gradient_boosting", "gam", "gamm", "quantile_regression", "bayesian_regression"},
          currently_ok=True, why=""),  # guard: a valid regressor stays on top of plain data
    _case("two_categorical", _two_categorical,
          {"group_comparison", "anova_oneway", "kruskal_wallis", "ols_regression",
           "chi_square_test", "loglinear", "manova"},
          currently_ok=True, why="",  # a real group/association method belongs on top
          reject={"community_detection", "centrality_suite", "network_analysis",
                  "link_prediction", "stochastic_block_model", "ergm", "epidemic_model"}),
]


def _top_feasible_ids(df: pd.DataFrame, tmp_path: Path, k: int = _TOP_K) -> list[str]:
    csv = tmp_path / "g.csv"
    df.to_csv(csv, index=False)
    fp = profile_dataset(csv)
    return [r.entry.id for r in select_top(fp, top=k)]


@pytest.mark.parametrize("case", GOLDEN)
def test_golden_selection(case, tmp_path: Path) -> None:
    ids = _top_feasible_ids(case["build"](), tmp_path)
    hit = case["accept"] & set(ids)
    assert hit, (
        f"{case['name']}: expected an appropriate method "
        f"{sorted(case['accept'])} in top-{_TOP_K} feasible, got {ids}"
    )
    bad = case["reject"] & set(ids)
    assert not bad, (
        f"{case['name']}: inappropriate method(s) {sorted(bad)} in top-{_TOP_K} feasible, got {ids}"
    )


def test_selection_scoreboard(tmp_path: Path) -> None:
    """Informational + a floor: how many golden cases the selector currently nails.
    Asserts the count never drops below today's baseline (the currently_ok cases)."""
    baseline = sum(1 for c in GOLDEN if not c.marks)  # currently_ok cases (no xfail mark)
    correct, lines = 0, []
    for c in GOLDEN:
        payload = c.values[0]
        ids = _top_feasible_ids(payload["build"](), tmp_path)
        hit = bool(payload["accept"] & set(ids))
        correct += hit
        lines.append(f"  {payload['name']:20s} {'OK ' if hit else 'MISS'}  top={ids[:4]}")
    print(f"\nGolden selection scoreboard: {correct}/{len(GOLDEN)} correct\n" + "\n".join(lines))
    assert correct >= baseline, f"selection regressed below baseline {baseline}: only {correct} correct"
