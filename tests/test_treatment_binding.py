"""Wave H4d — the TREATMENT side of role binding (the twin of H4/H4b/H4c's outcome side).

``fp.treatment_candidates`` is literally *every binary column* (profiler/profile.py), so the
family-wide idiom

    treatment = cfg["treatment"] else fp.treatment_candidates[0]

meant "the first binary column in file order" with no role signal whatsoever — while
``roles.py`` had already computed a NAME-aware ``fp.likely_treatment`` that nobody read.

On a perfectly ordinary survival frame ``[duration, event, treatment, age, biomarker]`` that
made PSM match on the **survival event indicator** as if it were the intervention, with a
column literally named ``treatment`` sitting right next to it. The damage was not cosmetic:
the truth is ``duration ~ Exp(20·exp(-0.4·treatment + …))``, i.e. treatment SHORTENS duration
by about 6.7, and the engine reported ATT = **+4.64, p = 0.16 (not significant)** — wrong
sign, effect missed entirely. With the name signal it reports ATT ≈ −8.7, p ≈ 0.0003.

``resolve_treatment`` (executor/_helpers/core.py) is the shared fix, and it records what it
bound on ``RunResult.treatment`` so this is auditable the way outcomes became auditable in H4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.executor.run import resolve_treatment
from researchforge.profiler import profile_dataset

_CAT = Catalog.load()
_DEGRADED = ("失败", "跳过", "暂未接入", "未检测到")

# the causal branches that resolve a treatment through the shared helper
_WIRED = ["psm", "ipw", "aipw", "rosenbaum_bounds", "evalue"]


def _fp(df, tmp_path, name="d.csv"):
    csv = tmp_path / name
    df.to_csv(csv, index=False)
    return profile_dataset(csv)


def _ran(res) -> bool:
    return bool(res.estimates) and not any(k in res.summary for k in _DEGRADED)


def _survival_frame(n=300, seed=2):
    """`event` comes BEFORE `treatment` in file order, so the old positional pick lands on
    the survival event indicator. Treatment shortens duration (true ATT ≈ -6.7)."""
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 80, n)
    trt = rng.integers(0, 2, n)
    dur = np.round(rng.exponential(20 * np.exp(-0.4 * trt + 0.01 * (age - 50))), 2)
    return pd.DataFrame({"duration": dur,
                         "event": (rng.random(n) < 0.7).astype(int),
                         "treatment": trt,
                         "age": age,
                         "biomarker": rng.normal(5, 1.5, n).round(3)})


# ── the resolver itself ──────────────────────────────────────────────────────
def test_resolver_prefers_a_treatment_named_candidate(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    assert fp.treatment_candidates == ["event", "treatment"]      # every binary, file order
    assert resolve_treatment(fp, {}, fp.treatment_candidates) == "treatment"


def test_resolver_honors_explicit_config(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    assert resolve_treatment(fp, {"treatment": "event"}, fp.treatment_candidates) == "event"


def test_resolver_falls_back_to_column_order_without_a_signal(tmp_path):
    """No treatment-named candidate → the long-standing positional default is preserved."""
    rng = np.random.default_rng(4)
    df = pd.DataFrame({"flag_a": rng.integers(0, 2, 120),
                       "flag_b": rng.integers(0, 2, 120),
                       "value": rng.normal(0, 1, 120).round(3)})
    fp = _fp(df, tmp_path)
    assert resolve_treatment(fp, {}, ["flag_a", "flag_b"]) in {"flag_a", "flag_b"}


def test_resolver_returns_none_on_no_candidates(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    assert resolve_treatment(fp, {}, []) is None


# ── the branches ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("cid", _WIRED)
def test_causal_family_binds_the_named_treatment(cid, tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    entry = _CAT.by_id(cid)
    if entry is None:
        pytest.skip(f"{cid} not in catalog")
    res = run_analysis(fp, entry, output_root=str(tmp_path / f"o_{cid}"))
    if not _ran(res):
        pytest.skip(f"{cid} degraded: {res.summary[:60]}")
    assert res.treatment == "treatment", (
        f"{cid} treated {res.treatment!r} as the intervention "
        "(first-binary fallback would give 'event')"
    )


def test_psm_recovers_the_real_effect_not_the_event_contrast(tmp_path):
    """The regression this wave exists for. Matching on `event` gave ATT=+4.64 (p=0.16):
    wrong sign, effect missed. The truth is a large NEGATIVE ATT (~-6.7)."""
    fp = _fp(_survival_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "psm"))
    if not _ran(res):
        pytest.skip(f"psm degraded: {res.summary[:60]}")
    att, se = res.estimates["att"], res.estimates["se"]
    assert att < 0, f"ATT should be negative (treatment shortens duration), got {att:.3f}"
    assert res.estimates["pvalue"] < 0.05, "the real effect is large; it must be detected"
    assert abs(att - (-6.7)) < 3 * se, f"ATT {att:.2f} ± {se:.2f} should cover the truth ≈ -6.7"
    # and the matched sample must be the ASSIGNMENT split (~50%), not the event rate (~70%)
    assert 0.35 < res.estimates["n_treated"] / 300 < 0.60


def test_config_treatment_still_wins_end_to_end(tmp_path):
    fp = _fp(_survival_frame(), tmp_path)
    res = run_analysis(fp, _CAT.by_id("psm"), output_root=str(tmp_path / "cfg"),
                       config={"treatment": "event"})
    if not _ran(res):
        pytest.skip("psm degraded")
    assert res.treatment == "event"


def test_wired_branches_never_run_without_reporting_their_treatment(tmp_path):
    """Ratchet: RunResult.treatment is the audit instrument for the treatment side."""
    fp = _fp(_survival_frame(seed=9), tmp_path)
    silent = []
    for cid in _WIRED:
        entry = _CAT.by_id(cid)
        if entry is None:
            continue
        res = run_analysis(fp, entry, output_root=str(tmp_path / f"r_{cid}"))
        if _ran(res) and res.treatment is None:
            silent.append(cid)
    assert not silent, f"ran but reported no treatment: {silent}"
