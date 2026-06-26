"""Tests for the MARKETING-ANALYTICS family: rfm_segmentation,
customer_lifetime_value, market_basket.

Known-value cases are hand-computed in the docstrings; honest-degrade paths
assert the Chinese "跳过" message and no crash. Fixed seeds / fixed toy data.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry(eid: str, method: str, goal: str = "describe") -> AnalysisEntry:
    return AnalysisEntry(
        id=eid,
        method=method,
        domain="business",
        family="marketing",
        goal=goal,
        preconditions=Precondition(min_rows=1),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# 1) rfm_segmentation
# --------------------------------------------------------------------------- #
def _rfm_transactions() -> pd.DataFrame:
    """Five customers with strictly ordered R/F/M so quintiles map cleanly to 1..5.

    Reference (max) date in the data = 2023-12-30 (C1's most recent order).
      C1: last 2023-12-30 (R=0 days), 5 orders, spend 100*5=500  -> R5 F5 M5
      C2: last 2023-12-20 (R=10),     4 orders, spend  80*4=320  -> R4 F4 M4
      C3: last 2023-12-01 (R=29),     3 orders, spend  60*3=180  -> R3 F3 M3
      C4: last 2023-11-01 (R=59),     2 orders, spend  40*2= 80  -> R2 F2 M2
      C5: last 2023-09-01 (R=120),    1 order,  spend  20*1= 20  -> R1 F1 M1
    With 5 distinct values, ceil(rank_pct*5) gives 1..5; Recency reversed so the
    most-recent customer (smallest days) scores R=5. C1 -> (R5,F5,M5) -> Champions.
    """
    rows = []
    plan = {
        "C1": ("2023-12-30", 5, 100.0),
        "C2": ("2023-12-20", 4, 80.0),
        "C3": ("2023-12-01", 3, 60.0),
        "C4": ("2023-11-01", 2, 40.0),
        "C5": ("2023-09-01", 1, 20.0),
    }
    for cust, (last, n_orders, amt) in plan.items():
        last_ts = pd.Timestamp(last)
        for k in range(n_orders):
            # spread earlier orders backwards; the LAST (k=0) order is `last`
            rows.append({
                "cust_id": cust,
                "order_date": (last_ts - pd.Timedelta(days=200 * k)).strftime("%Y-%m-%d"),
                "amount": amt,
            })
    return pd.DataFrame(rows)


def test_rfm_values_and_champion(tmp_path: Path) -> None:
    """C1 has the smallest recency, highest frequency & monetary -> R5/F5/M5 ->
    Champions. Recency(C1) relative to the max date in the data (2023-12-30) = 0.
    Frequency(C1)=5, Monetary(C1)=500."""
    df = _rfm_transactions()
    csv = _csv(tmp_path, "tx.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("rfm_segmentation", "RFM"),
                       output_root=str(tmp_path / "o"),
                       config={"customer": "cust_id", "date": "order_date",
                               "amount": "amount"})
    e = res.estimates
    assert e["n_customers"] == 5.0
    assert e["n_champions"] >= 1.0
    out = Path(res.output_dir)
    cust_csv = out / "rfm_customers.csv"
    assert cust_csv.exists()
    t = pd.read_csv(cust_csv)
    t["customer"] = t["customer"].astype(str)
    c1 = t[t["customer"] == "C1"].iloc[0]
    # max date in data = 2023-12-30 (C1's last order) -> C1 recency = 0 days
    assert math.isclose(c1["recency_days"], 0.0, abs_tol=1e-6)
    assert c1["frequency"] == 5
    assert math.isclose(c1["monetary"], 500.0, abs_tol=1e-6)
    assert c1["R"] == 5 and c1["F"] == 5 and c1["M"] == 5
    assert c1["segment"] == "Champions"
    # C5 is the worst on all three -> R1/F1/M1
    c5 = t[t["customer"] == "C5"].iloc[0]
    assert c5["R"] == 1 and c5["F"] == 1 and c5["M"] == 1
    assert (out / "rfm_segment_counts.csv").exists()


def test_rfm_degrade_no_date(tmp_path: Path) -> None:
    """Customer + amount but no parseable date -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "nodate.csv", pd.DataFrame({
        "cust_id": ["A", "B", "A"],
        "amount": [10.0, 20.0, 30.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("rfm_segmentation", "RFM"),
                       output_root=str(tmp_path / "o"),
                       config={"customer": "cust_id", "amount": "amount"})
    assert "跳过" in res.summary
    assert "n_customers" not in res.estimates


def test_rfm_degrade_no_amount(tmp_path: Path) -> None:
    """Only id + date, no numeric amount -> honest 跳过."""
    csv = _csv(tmp_path, "noamt.csv", pd.DataFrame({
        "cust_id": ["A", "B", "C"],
        "label": ["x", "y", "z"],
        "order_date": ["2023-01-01", "2023-02-01", "2023-03-01"],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("rfm_segmentation", "RFM"),
                       output_root=str(tmp_path / "o"),
                       config={"customer": "cust_id", "date": "order_date"})
    assert "跳过" in res.summary


# --------------------------------------------------------------------------- #
# 2) customer_lifetime_value
# --------------------------------------------------------------------------- #
def test_clv_historical_mean(tmp_path: Path) -> None:
    """Per-customer monetary: A=100 (2 orders 60+40), B=200 (1 order), C=300 (1).
       historical CLV per customer = 100, 200, 300 -> mean=200, median=200.
       Top-decile share: ceil(0.1*3)=1 customer -> 300 / 600 = 0.5."""
    df = pd.DataFrame({
        "cust_id": ["A", "A", "B", "C"],
        "amount": [60.0, 40.0, 200.0, 300.0],
    })
    csv = _csv(tmp_path, "clv.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("customer_lifetime_value", "CLV", goal="predict"),
                       output_root=str(tmp_path / "o"),
                       config={"customer": "cust_id", "amount": "amount"})
    e = res.estimates
    assert e["n_customers"] == 3.0
    assert math.isclose(e["mean_clv"], 200.0, abs_tol=1e-6)
    assert math.isclose(e["median_clv"], 200.0, abs_tol=1e-6)
    assert math.isclose(e["top_decile_share"], 0.5, abs_tol=1e-6)
    out = Path(res.output_dir)
    assert (out / "clv_customers.csv").exists()
    # no retention given -> projection is NaN
    assert math.isnan(e["projected_clv"])


def test_clv_retention_formula(tmp_path: Path) -> None:
    """Discounted-margin CLV = m*r/(1+d-r). With m=100, r=0.8, d=0.1:
       100*0.8/(1.1-0.8) = 80/0.3 = 266.6667."""
    df = pd.DataFrame({
        "cust_id": ["A", "B", "C"],
        "amount": [100.0, 100.0, 100.0],
    })
    csv = _csv(tmp_path, "clv.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("customer_lifetime_value", "CLV", goal="predict"),
                       output_root=str(tmp_path / "o"),
                       config={"customer": "cust_id", "amount": "amount",
                               "margin": 100.0, "retention": 0.8, "discount": 0.1})
    e = res.estimates
    assert math.isclose(e["projected_clv"], 266.6667, abs_tol=1e-3)
    assert math.isclose(e["retention_used"], 0.8, abs_tol=1e-9)
    assert math.isclose(e["discount_used"], 0.1, abs_tol=1e-9)


def test_clv_degrade_no_amount(tmp_path: Path) -> None:
    """No numeric amount column -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "noamt.csv", pd.DataFrame({
        "cust_id": ["A", "B"],
        "label": ["x", "y"],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("customer_lifetime_value", "CLV", goal="predict"),
                       output_root=str(tmp_path / "o"),
                       config={"customer": "cust_id"})
    assert "跳过" in res.summary
    assert "mean_clv" not in res.estimates


# --------------------------------------------------------------------------- #
# 3) market_basket
# --------------------------------------------------------------------------- #
def _basket_long() -> pd.DataFrame:
    """Five transactions where {bread, butter} co-occur strongly (long form).
       T1: bread, butter
       T2: bread, butter
       T3: bread, butter, milk
       T4: bread, milk
       T5: milk
    N=5. support(bread)=4/5=0.8, support(butter)=3/5=0.6,
    support({bread,butter})=3/5=0.6.
    confidence(bread->butter)=0.6/0.8=0.75; lift=0.75/0.6=1.25 (>1)."""
    rows = []
    baskets = {
        "T1": ["bread", "butter"],
        "T2": ["bread", "butter"],
        "T3": ["bread", "butter", "milk"],
        "T4": ["bread", "milk"],
        "T5": ["milk"],
    }
    for tid, items in baskets.items():
        for it in items:
            rows.append({"order_id": tid, "item": it})
    return pd.DataFrame(rows)


def test_market_basket_bread_butter_lift(tmp_path: Path) -> None:
    """Hand-checked: rule bread->butter has support=0.6, confidence=0.75,
    lift=1.25 (>1)."""
    df = _basket_long()
    csv = _csv(tmp_path, "basket.csv", df)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("market_basket", "Apriori", goal="relate"),
                       output_root=str(tmp_path / "o"),
                       config={"transaction": "order_id", "item": "item",
                               "min_support": 0.01, "min_confidence": 0.3,
                               "max_len": 3})
    e = res.estimates
    assert e["n_transactions"] == 5.0
    assert e["n_rules"] >= 1.0
    assert e["top_lift"] > 1.0
    out = Path(res.output_dir)
    rules = pd.read_csv(out / "market_basket_rules.csv")
    assert {"antecedent", "consequent", "support", "confidence", "lift"}.issubset(
        rules.columns)
    # find the bread -> butter rule and check its hand-computed metrics.
    bb = rules[(rules["antecedent"] == "{bread}")
               & (rules["consequent"] == "{butter}")]
    assert len(bb) == 1
    row = bb.iloc[0]
    assert math.isclose(row["support"], 0.6, abs_tol=1e-6)
    assert math.isclose(row["confidence"], 0.75, abs_tol=1e-6)
    assert math.isclose(row["lift"], 1.25, abs_tol=1e-6)


def test_market_basket_onehot(tmp_path: Path) -> None:
    """Same baskets as the long-form case but as a one-hot 0/1 matrix; the
    bread->butter rule must reproduce the hand-checked lift=1.25."""
    onehot = pd.DataFrame({
        "bread":  [1, 1, 1, 1, 0],
        "butter": [1, 1, 1, 0, 0],
        "milk":   [0, 0, 1, 1, 1],
    })
    csv = _csv(tmp_path, "onehot.csv", onehot)
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("market_basket", "Apriori", goal="relate"),
                       output_root=str(tmp_path / "o"),
                       config={"min_support": 0.01, "min_confidence": 0.3,
                               "max_len": 3})
    e = res.estimates
    assert e["n_transactions"] == 5.0
    assert e["n_rules"] >= 1.0
    out = Path(res.output_dir)
    rules = pd.read_csv(out / "market_basket_rules.csv")
    bb = rules[(rules["antecedent"] == "{bread}")
               & (rules["consequent"] == "{butter}")]
    assert len(bb) == 1
    assert math.isclose(bb.iloc[0]["lift"], 1.25, abs_tol=1e-6)


def test_market_basket_degrade(tmp_path: Path) -> None:
    """Neither a transaction+item long form nor a one-hot 0/1 matrix -> 跳过."""
    csv = _csv(tmp_path, "bad.csv", pd.DataFrame({
        "note": ["a", "b", "c"],
        "value": [3.5, 7.1, 9.9],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry("market_basket", "Apriori", goal="relate"),
                       output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_rules" not in res.estimates
