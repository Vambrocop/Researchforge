"""Self-evolution phase 2 — the live trend engine (catalog.trends).

HTTP is mocked at the single choke point (`trends._get_json`) so these are
deterministic and need no network. The cache/snapshot paths are redirected to a
tmp dir by the autouse `_isolate_trend_artifacts` fixture in conftest. One real
PyPI smoke is included but skipped when the network is unreachable.
"""

from __future__ import annotations

import researchforge.catalog.trends as trends
from researchforge.catalog.discover import build_live_fetch_fn, score_candidate
from researchforge.catalog.trends import (
    TrendSignal,
    compute_momentum,
    fetch_trend,
    load_snapshot,
    parse_sources,
    write_snapshot,
)


# ── a controllable fake HTTP layer ───────────────────────────────────────────
def _fake_http(spec: dict):
    """Return a _get_json stand-in that answers by URL substring from `spec`:
    keys pypi/pypi_dl/cran/github map to canned payloads (None = unreachable)."""
    def _get(url, headers=None):
        if "pypi.org/pypi/" in url:
            return {"info": {"version": "1.0"}} if spec.get("pypi") else None
        if "pypistats.org" in url:
            dl = spec.get("pypi_dl")
            return {"data": {"last_month": dl}} if dl is not None else None
        if "crandb.r-pkg.org" in url:
            return {"Package": "x"} if spec.get("cran") else None
        if "api.github.com/search" in url:
            g = spec.get("github")
            if not g:
                return None
            import urllib.parse as _up

            q = _up.parse_qs(_up.urlparse(url).query).get("q", ["x"])[0]
            return {"items": [{"full_name": f"org/{q}", "stargazers_count": g[0],
                              "pushed_at": g[1]}]}
        return None
    return _get


# ── parse_sources ────────────────────────────────────────────────────────────
def test_parse_sources_splits_r_and_pypi():
    py, r = parse_sources(["mapie", "grf (R)", "splm (R)", "split conformal (hand-roll)", "xgboost"])
    assert "mapie" in py and "xgboost" in py
    assert "grf" in r and "splm" in r
    assert "split" in py  # first token of "split conformal", annotation dropped


# ── compute_momentum (pure, real-number → 0-100) ─────────────────────────────
def test_momentum_high_for_popular_package():
    sig = TrendSignal(name="x", pypi_downloads=1_000_000, github_stars=40_000,
                      github_pushed_days=0, available=True)
    assert compute_momentum(sig) >= 90


def test_momentum_low_for_obscure_and_presence_only():
    assert compute_momentum(TrendSignal(name="x", pypi_downloads=50, github_stars=5,
                                        github_pushed_days=400)) < 20
    # only an existence flag -> faint presence score
    assert compute_momentum(TrendSignal(name="x", cran_exists=True)) == 25
    assert compute_momentum(TrendSignal(name="x")) == 0


def test_momentum_renormalizes_when_signals_missing():
    # only GitHub stars available -> still a fair (single-component) read
    sig = TrendSignal(name="x", github_stars=100_000, github_pushed_days=0)
    assert compute_momentum(sig) >= 90


# ── fetch_trend with mocked HTTP ─────────────────────────────────────────────
def test_fetch_trend_blends_live_signals(monkeypatch):
    monkeypatch.setattr(trends, "_get_json",
                        _fake_http({"pypi": True, "pypi_dl": 500_000,
                                    "github": (30_000, "2099-01-01T00:00:00Z")}))
    sig = fetch_trend("gradient_boosting", ["xgboost"], use_cache=False)
    assert sig.available is True
    assert sig.pypi_exists is True and sig.pypi_downloads == 500_000
    assert sig.github_stars == 30_000
    assert sig.momentum >= 80
    assert "实时趋势" in sig.detail


def test_github_rejects_unrelated_repo(monkeypatch):
    # a bare-name search that returns a same-named-but-unrelated repo is discarded,
    # so its stars aren't attributed to this method (honesty guard).
    def _get(url, headers=None):
        if "pypi.org/pypi/" in url:
            return {"info": {"version": "1.0"}}
        if "api.github.com/search" in url:
            return {"items": [{"full_name": "someone/totally-different",
                               "stargazers_count": 99999, "pushed_at": "2099-01-01T00:00:00Z"}]}
        return None

    monkeypatch.setattr(trends, "_get_json", _get)
    sig = fetch_trend("mymethod", ["mypkg"], use_cache=False)
    assert sig.pypi_exists is True       # pypi still counted (exact match)
    assert sig.github_stars is None      # unrelated repo's stars rejected


def test_fetch_trend_degrades_offline(monkeypatch):
    monkeypatch.setattr(trends, "_get_json", lambda url, headers=None: None)
    sig = fetch_trend("anything", ["somepkg", "rpkg (R)"], use_cache=False)
    assert sig.available is False
    assert sig.momentum == 0
    assert "不可用" in sig.detail


def test_fetch_trend_uses_cache(monkeypatch):
    calls = {"n": 0}
    base = _fake_http({"pypi": True, "pypi_dl": 1000, "cran": True})

    def counting(url, headers=None):
        calls["n"] += 1
        return base(url, headers)

    monkeypatch.setattr(trends, "_get_json", counting)
    fetch_trend("cached_method", ["pkg"], use_cache=True)
    after_first = calls["n"]
    assert after_first > 0
    fetch_trend("cached_method", ["pkg"], use_cache=True)  # served from disk cache
    assert calls["n"] == after_first  # no further HTTP


# ── score_candidate blends momentum ──────────────────────────────────────────
def test_score_candidate_blends_live_momentum():
    base = {"id": "x", "method": "X", "family": "ml"}
    offline = score_candidate(base)
    live = score_candidate({**base, "momentum": 95, "trend": {"available": True}})
    assert live.momentum == 95
    assert live.priority != offline.priority  # live signal moves the priority


def test_build_live_fetch_fn_enriches_specs(monkeypatch):
    monkeypatch.setattr(trends, "_get_json",
                        _fake_http({"pypi": True, "pypi_dl": 200_000,
                                    "github": (10_000, "2099-01-01T00:00:00Z")}))
    fn = build_live_fetch_fn([{"id": "z", "method": "Z", "family": "ml", "sources": ["pkg"]}])
    out = fn()
    assert out[0]["momentum"] > 0 and out[0]["trend"]["available"] is True


# ── snapshot write → scorecard read (the closed loop, no hot-path network) ────
def test_snapshot_feeds_scorecard(monkeypatch):
    from researchforge.catalog.schema import AnalysisEntry
    from researchforge.profiler.fingerprint import DataFingerprint
    from researchforge.recommender.rigor import RigorVerdict
    from researchforge.recommender.scoring import _trend_snapshot, score_method

    path = write_snapshot([
        {"id": "quantile_forest", "family": "ml", "momentum": 60, "available": True},
        {"id": "other", "family": "ml", "momentum": 80, "available": True},
    ])
    assert path is not None
    snap = load_snapshot()
    assert snap["by_id"]["quantile_forest"] == 60 and "ml" in snap["by_family"]

    _trend_snapshot.cache_clear()  # pick up the just-written snapshot
    fp = DataFingerprint(path="x", n_rows=100, n_cols=5, columns=[])
    rv = RigorVerdict(light="green", score=90)
    # by_id hit: gradient_boosting isn't in by_id, falls to by_family mean (70)
    s_fam = score_method(fp, AnalysisEntry(id="gradient_boosting", method="m", domain="ml",
                                           family="ml", goal="predict"), rv)
    assert "实时趋势" in s_fam.note
    # by_id hit overrides family
    s_id = score_method(fp, AnalysisEntry(id="quantile_forest", method="m", domain="ml",
                                          family="ml", goal="predict"), rv)
    assert "实时趋势" in s_id.note
    _trend_snapshot.cache_clear()


def test_no_snapshot_uses_editorial_prior(monkeypatch):
    from researchforge.catalog.schema import AnalysisEntry
    from researchforge.profiler.fingerprint import DataFingerprint
    from researchforge.recommender.rigor import RigorVerdict
    from researchforge.recommender.scoring import _trend_snapshot, score_method

    _trend_snapshot.cache_clear()  # isolated tmp dir has no snapshot
    fp = DataFingerprint(path="x", n_rows=100, n_cols=5, columns=[])
    s = score_method(fp, AnalysisEntry(id="z", method="m", domain="ml", family="ml",
                                       goal="predict"), RigorVerdict(light="green", score=90))
    assert "离线编辑先验" in s.note


# ── real-network smoke (skipped if unreachable) ──────────────────────────────
def test_real_pypi_smoke():
    import pytest

    real = trends._get_json("https://pypi.org/pypi/numpy/json")
    if not real:
        pytest.skip("network unavailable")
    assert real.get("info", {}).get("name", "").lower() == "numpy"
