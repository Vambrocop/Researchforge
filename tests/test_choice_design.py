"""离散选择设计：引擎对着 conjoint 表推荐 IRT 和序数回归。

Measured on a 120 x 8 x 3 conjoint (respondent x task x alternative, one `chosen` per set),
catalog as it stood:

    conditional_logit   rank 76 of 306
    mnl_choice          rank 44 of 306
    headline            descriptive_stats, dif_detection (IRT), proportional_odds_logit,
                        ordered_probit, multinomial_logit

A conjoint table looks like "a few categorical attributes and a binary column" to every
generic signal, so the family built for exactly this stayed buried on its own shape.

The signature that identifies the design is also the thing the estimator conditions on:
**exactly one chosen alternative per choice set**, with at least two alternatives in each.
That holds for a choice-based survey and, identically, for a matched case-control study
(one case per matched set) — both are conditional logit, so one signal serving both is
principled rather than a coincidence.

It also surfaced an under-declaration worth its own note: `conditional_logit` declared only
`min_rows: 10`, which made it feasible on ANY numeric frame and cost it the
precondition-specificity bonus. On its own conjoint frame it ranked 13 while `mnl_choice`
— less apt here, but declaring one more precondition — ranked 2. Declaring what a method
actually requires is both an honesty fix and a ranking fix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.profiler import profile_dataset
from researchforge.recommender import check_preconditions, recommend
from researchforge.recommender.affinity import has_choice_design

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _conjoint(n_resp=120, n_task=8, n_alt=3, seed=1):
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(n_resp):
        for t in range(n_task):
            pick = rng.integers(0, n_alt)
            for a in range(n_alt):
                rows.append({"respondent": f"r{r:03d}", "task": t, "alt": a,
                             "price": float(rng.choice([10, 20, 30])),
                             "brand": rng.choice(list("ABC")),
                             "warranty": int(rng.choice([1, 2, 3])),
                             "chosen": int(a == pick)})
    return pd.DataFrame(rows)


def _matched_case_control(n_strata=80, per=4, seed=2):
    """One case per matched set — the other classic conditional-logit design."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"stratum": f"s{s:02d}", "exposure": float(rng.normal()),
                          "age": float(rng.normal(50, 8)), "case": int(i == 0)}
                         for s in range(n_strata) for i in range(per)])


def _plain_binary_outcome(n=600, seed=3):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(0, 1, n).round(3), "g": rng.integers(0, 5, n),
                         "y": (rng.random(n) < 0.4).astype(int)})


def _staggered_panel(n_unit=40, n_t=10, seed=4):
    """`policy_on` switches on and STAYS on, so it sums to more than 1 per unit."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame([{"unit": f"u{u:02d}", "year": 2000 + t,
                          "policy_on": int(t >= 5) if u % 2 == 0 else 0,
                          "y": float(rng.normal())}
                         for u in range(n_unit) for t in range(n_t)])


# ── the shape fact ───────────────────────────────────────────────────────────
def test_a_conjoint_table_is_recognised_by_its_choice_sets(tmp_path):
    fp = _fp(_conjoint(), tmp_path, name="cj.csv")
    assert fp.choice_flag == "chosen"
    assert fp.choice_set_cols == ["respondent", "task"]
    assert has_choice_design(fp) is True


def test_a_matched_case_control_is_the_same_design(tmp_path):
    """Not a coincidence that one signal covers both: conditional logit is defined on the
    stratification, and 'one case per matched set' is that stratification."""
    fp = _fp(_matched_case_control(), tmp_path, name="mcc.csv")
    assert fp.choice_flag == "case" and fp.choice_set_cols == ["stratum"]


@pytest.mark.parametrize("frame,name", [(_plain_binary_outcome, "plain"),
                                        (_staggered_panel, "panel")])
def test_ordinary_binary_columns_are_not_a_choice_design(frame, name, tmp_path):
    """The two shapes most likely to be confused for one: a binary outcome with a grouping
    column, and a staggered-adoption panel whose treatment switches on and stays on."""
    fp = _fp(frame(), tmp_path, name=f"{name}.csv")
    assert fp.choice_flag is None and fp.choice_set_cols == []
    assert has_choice_design(fp) is False


def test_detection_stays_cheap_on_a_wide_frame(tmp_path):
    """The search is bounded (capped candidates, pairs only after singles, a hard try
    limit) because it runs on every profile."""
    import time

    rng = np.random.default_rng(5)
    n = 5000
    df = pd.DataFrame({**{f"g{i}": rng.integers(0, 7, n) for i in range(6)},
                       **{f"b{i}": (rng.random(n) < 0.3).astype(int) for i in range(6)},
                       **{f"x{i}": rng.normal(0, 1, n).round(3) for i in range(6)}})
    csv = tmp_path / "wide.csv"
    df.to_csv(csv, index=False)
    t0 = time.perf_counter()
    fp = profile_dataset(csv)
    assert time.perf_counter() - t0 < 5.0
    assert fp.choice_flag is None


# ── the precondition: honest feasibility, and the specificity it earns ──────
def test_conditional_logit_is_infeasible_without_a_choice_design(tmp_path):
    """It declared only `min_rows: 10`, so it was judged feasible on any numeric frame.
    Conditional logit has nothing to condition on there — that is a hard requirement."""
    pre = _CAT.by_id("conditional_logit").preconditions
    ok, unmet = check_preconditions(_fp(_plain_binary_outcome(), tmp_path, name="p.csv"), pre)
    assert ok is False and any("离散选择设计" in u for u in unmet), unmet
    ok2, unmet2 = check_preconditions(_fp(_conjoint(), tmp_path, name="c.csv"), pre)
    assert ok2 is True and unmet2 == []


def _rank(fp, target):
    ids = [getattr(getattr(r, "entry", r), "id", "?") for r in recommend(fp, _CAT)]
    return (ids.index(target) + 1) if target in ids else None, ids


def test_the_choice_family_leads_on_a_conjoint_table(tmp_path):
    fp = _fp(_conjoint(), tmp_path, name="rank.csv")
    r_cl, ids = _rank(fp, "conditional_logit")
    r_mnl, _ = _rank(fp, "mnl_choice")
    assert r_cl is not None and r_cl <= 3, ids[:6]
    assert r_mnl is not None and r_mnl <= 5, ids[:6]
    # the method that used to take the headline slot is an item-response model; it has no
    # business leading here, so it must at least fall behind the choice family
    r_dif, _ = _rank(fp, "dif_detection")
    assert r_dif is None or r_dif > r_cl, (r_dif, r_cl)


def test_the_matched_design_also_leads_to_conditional_logit(tmp_path):
    fp = _fp(_matched_case_control(), tmp_path, name="mcc_rank.csv")
    r, ids = _rank(fp, "conditional_logit")
    assert r is not None and r <= 3, ids[:6]


def test_plain_data_does_not_summon_the_choice_family(tmp_path):
    """The symmetric half — and now it is an infeasibility, not merely a low rank."""
    fp = _fp(_plain_binary_outcome(), tmp_path, name="plain_rank.csv")
    r, _ = _rank(fp, "conditional_logit")
    assert r is None or r > 50, r
