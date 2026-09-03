"""Wave H4 — the smart-selection nudge states what the branch REALLY modeled.

The nudge used to be emitted BEFORE dispatch, claiming "已自动选取 'y'（高置信）" purely from
the fingerprint's role hint — with no check that the running branch honoured it. A branch still
defaulting to ``cont[0]`` then produced a self-contradicting report ("已自动选取 y" in the nudge,
"结果 x1" in the summary). ``resolve_outcome`` now records what it actually bound, and
``run_analysis`` builds the nudge afterwards from that, also exposing it on ``RunResult.outcome``.

Pinned here: the bound case states the truth, a MISMATCH is surfaced instead of hidden, a branch
that doesn't use the shared resolver makes no binding claim, and the nudge can never name an
outcome different from the one actually bound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.executor.run import capture_bound_outcome, resolve_outcome
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _nudge(res):
    return next((ln for ln in res.summary.split("\n") if ln.startswith("💡")), "")


def test_recorder_captures_resolve_outcome(tmp_path):
    """resolve_outcome records its binding inside a capture block (the mechanism)."""
    fp = _fp(pd.DataFrame({"x1": [1.0, 2, 3], "y": [2.0, 4, 6]}), tmp_path)
    with capture_bound_outcome() as rec:
        chosen = resolve_outcome(fp, {}, ["x1", "y"])
    assert rec == [chosen]


def test_recorder_is_noop_outside_capture(tmp_path):
    """Calling resolve_outcome outside a run must not raise or leak state."""
    fp = _fp(pd.DataFrame({"x1": [1.0, 2, 3], "y": [2.0, 4, 6]}), tmp_path)
    assert resolve_outcome(fp, {}, ["x1", "y"]) in {"x1", "y"}  # returns a candidate, no raise


def _high_conf_frame(seed=0, n=150):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # decoys FIRST, the unambiguous DV name last
    return pd.DataFrame({"x1": x1.round(3), "x2": x2.round(3),
                         "y": (2 * x1 - x2 + rng.normal(0, 1, n)).round(3)})


def test_bound_outcome_is_reported_and_nudge_agrees(tmp_path):
    fp = _fp(_high_conf_frame(), tmp_path)
    assert fp.likely_outcome == "y" and fp.likely_outcome_confidence == "high"
    res = run_analysis(fp, _CAT.by_id("robust_regression"), output_root=str(tmp_path / "o"))
    assert res.outcome == "y"                      # RunResult carries the real binding
    assert "已自动选取 'y'" in _nudge(res)          # and the nudge states it
    assert "结果 y" in res.summary or "'y'" in res.summary


def test_mismatch_is_surfaced_not_hidden(tmp_path):
    """A LOW-confidence positional hint that the branch does NOT bind must be reported as a
    mismatch (this is the case that used to produce a self-contradicting report)."""
    rng = np.random.default_rng(3)
    n = 150
    df = pd.DataFrame({
        "x1": rng.normal(0, 1, n).round(3),
        "x2": rng.normal(0, 1, n).round(3),
        "age": rng.integers(20, 70, n),   # last numeric → LOW-confidence positional hint
    })
    fp = _fp(df, tmp_path)
    if fp.likely_outcome is None:        # role detector found nothing → nothing to assert
        return
    res = run_analysis(fp, _CAT.by_id("robust_regression"), output_root=str(tmp_path / "o"))
    if res.outcome and res.outcome != fp.likely_outcome:
        nud = _nudge(res)
        assert f"'{res.outcome}'" in nud and f"'{fp.likely_outcome}'" in nud
        assert "已自动选取" not in nud   # no false "auto-selected" claim on a mismatch


def test_no_binding_claim_when_resolver_unused(tmp_path):
    """A branch that never calls resolve_outcome leaves outcome=None and the nudge must not
    claim anything was auto-selected."""
    fp = _fp(_high_conf_frame(seed=1), tmp_path)
    for cid in ("correlation", "pca", "descriptive_stats"):
        entry = _CAT.by_id(cid)
        if entry is None or not any(p.name in ("outcome", "y") for p in entry.params):
            continue
        res = run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"))
        if res.outcome is None:
            assert "已自动选取" not in _nudge(res)


def test_nudge_never_names_an_outcome_other_than_the_bound_one(tmp_path):
    """Cross-method invariant: whenever a nudge claims auto-selection, the named column IS the
    one the branch bound — the self-contradiction H4 was opened for is impossible."""
    fp = _fp(_high_conf_frame(seed=2), tmp_path)
    for cid in ("robust_regression", "ols_regression", "influence_diagnostics",
                "quantile_regression", "gam"):
        entry = _CAT.by_id(cid)
        if entry is None:
            continue
        res = run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"))
        nud = _nudge(res)
        if "已自动选取" in nud:
            assert res.outcome and f"'{res.outcome}'" in nud, f"{cid}: nudge/binding disagree"
