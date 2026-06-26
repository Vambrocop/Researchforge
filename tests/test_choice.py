"""Discrete-choice family: mnl_choice (multinomial logit) + conditional_logit.

Recovery cases: an MNL where a predictor shifts class probabilities fits with a
positive pseudo-R²; a conditional logit simulated from a known utility recovers the
attribute sign(s) and WTP = -beta_attr/beta_cost. Plus honest-degrade paths.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str) -> AnalysisEntry:
    return AnalysisEntry(id=eid, method=method, domain="economics", family="choice",
                         goal="explain", preconditions=Precondition(min_rows=1))


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# mnl_choice — multinomial logit
# --------------------------------------------------------------------------- #
def test_mnl_choice_fits_with_signal(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    # 3-class softmax over utilities (0, 1.5x, -1.5x) -> x shifts class membership
    u = np.column_stack([np.zeros(n), 1.5 * x, -1.5 * x])
    p = np.exp(u) / np.exp(u).sum(axis=1, keepdims=True)
    y = np.array([rng.choice(["A", "B", "C"], p=p[i]) for i in range(n)])
    df = pd.DataFrame({"x": x, "choice": y})
    res = run_analysis(profile_dataset(_csv(tmp_path, "mnl.csv", df)),
                       _entry("mnl_choice", "Multinomial logit"),
                       output_root=str(tmp_path / "o"),
                       config={"outcome": "choice", "predictors": ["x"]})
    e = res.estimates
    assert e["pseudo_r2"] > 0.05
    assert e["lr_p"] < 0.05
    assert e["n_alternatives"] == 3.0


def test_mnl_choice_degrades_on_binary(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"x": rng.normal(size=40), "y": rng.integers(0, 2, 40)})
    res = run_analysis(profile_dataset(_csv(tmp_path, "bin.csv", df)),
                       _entry("mnl_choice", "Multinomial logit"),
                       output_root=str(tmp_path / "o"), config={"outcome": "y"})
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# conditional_logit — McFadden, with WTP
# --------------------------------------------------------------------------- #
def test_conditional_logit_recovers_wtp(tmp_path: Path) -> None:
    # simulate choice sets from U = 1.0*quality - 0.5*price + Gumbel noise
    # -> WTP_quality = -beta_quality/beta_price = -(1.0)/(-0.5) = 2.0
    rng = np.random.default_rng(2)
    n_sets, n_alt = 400, 3
    rows = []
    b_q, b_p = 1.0, -0.5
    for s in range(n_sets):
        quality = rng.normal(0, 1, n_alt)
        price = rng.normal(0, 1, n_alt)
        gumbel = rng.gumbel(0, 1, n_alt)
        u = b_q * quality + b_p * price + gumbel
        chosen_idx = int(np.argmax(u))
        for j in range(n_alt):
            rows.append({"set": s, "quality": quality[j], "price": price[j],
                         "chosen": 1 if j == chosen_idx else 0})
    df = pd.DataFrame(rows)
    res = run_analysis(profile_dataset(_csv(tmp_path, "clogit.csv", df)),
                       _entry("conditional_logit", "Conditional logit"),
                       output_root=str(tmp_path / "o"),
                       config={"choice_id": "set", "chosen": "chosen",
                               "attributes": ["quality", "price"], "cost": "price"})
    e = res.estimates
    assert e["converged"] == 1.0
    assert e["beta__quality"] > 0 and e["beta__price"] < 0  # correct signs
    assert math.isclose(e["wtp__quality"], 2.0, rel_tol=0.35)  # -bq/bp ~ 2.0


def test_conditional_logit_degrades_without_choice_id(tmp_path: Path) -> None:
    df = pd.DataFrame({"quality": [1.0, 2.0, 3.0, 4.0], "chosen": [1, 0, 1, 0]})
    res = run_analysis(profile_dataset(_csv(tmp_path, "noid.csv", df)),
                       _entry("conditional_logit", "Conditional logit"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
