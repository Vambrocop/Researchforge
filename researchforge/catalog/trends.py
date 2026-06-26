"""Self-evolution phase 2 — the LIVE trend engine.

Phase 1 (discover.py) curated candidate methods from an offline seed and scored
them on static editorial priors. Phase 2 adds REAL signals: it asks PyPI / CRAN /
GitHub how widely used and how recently active each method's package is, and turns
those raw numbers into a 0-100 ``momentum`` score that feeds both the discovery
priority and the recommendation scorecard's popularity dimension.

Design (mirrors the R bridge — `executor/rbridge.py`):
- **Optional + graceful degrade.** Every fetch is best-effort with a short timeout;
  if `requests` is missing, the network is down, or a host rate-limits us, the
  signal is simply ``None`` and the caller falls back to the offline prior. Nothing
  here ever raises into the engine.
- **Honest.** Momentum is computed from real fetched counts (or marked unavailable);
  it is never fabricated. The blend weights are editorial but operate on real data,
  and that is disclosed wherever a momentum is surfaced.
- **Curation-time only.** This runs from ``cli discover --live`` / the snapshot
  refresh — NEVER from the analysis runtime or the per-recommend hot path. Results
  are cached on disk (``~/.researchforge/trend_cache.json``, 7-day TTL) so repeated
  runs don't hammer the public APIs.
"""

from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

_CACHE_DIR = Path.home() / ".researchforge"
_CACHE_FILE = _CACHE_DIR / "trend_cache.json"
_SNAPSHOT_FILE = _CACHE_DIR / "trend_snapshot.json"
_CACHE_TTL_S = 7 * 24 * 3600  # 7 days — trends move slowly; don't hammer the APIs
_SNAPSHOT_TTL_S = 30 * 24 * 3600  # the scorecard tolerates a month-old snapshot
_TIMEOUT_S = 8


class TrendSignal(BaseModel):
    """Live popularity/recency signals for one method (probed via its packages)."""

    name: str
    pypi_exists: Optional[bool] = None
    pypi_downloads: Optional[int] = None   # last-month downloads (pypistats)
    github_stars: Optional[int] = None
    github_pushed_days: Optional[int] = None  # days since the repo's last push
    cran_exists: Optional[bool] = None
    momentum: int = 0           # 0-100 blended trend score (real-data, editorial weights)
    available: bool = False     # True iff at least one live signal was fetched
    detail: str = ""            # honest note on what was / wasn't reachable

    def as_dict(self) -> dict:
        return self.model_dump()


# ── HTTP layer (single choke point, easy to mock in tests) ───────────────────
def _get_json(url: str, headers: Optional[dict] = None) -> Optional[dict]:
    """Best-effort GET → parsed JSON, or None on any failure. Never raises."""
    try:
        import requests
    except Exception:
        return None
    try:
        resp = requests.get(url, headers=headers or {"User-Agent": "ResearchForge-trends"},
                            timeout=_TIMEOUT_S)
        if resp.status_code != 200:
            return None
        return resp.json()
    except Exception:
        return None


# ── individual source probes ─────────────────────────────────────────────────
def _pypi(pkg: str) -> tuple[Optional[bool], Optional[int]]:
    """(exists, last-month downloads). Existence from the PyPI JSON API; downloads
    from pypistats (best-effort — may rate-limit, then downloads is None)."""
    meta = _get_json(f"https://pypi.org/pypi/{pkg}/json")
    exists = None if meta is None else bool(meta.get("info"))
    downloads = None
    if exists:
        stats = _get_json(f"https://pypistats.org/api/packages/{pkg.lower()}/recent")
        if stats and isinstance(stats.get("data"), dict):
            downloads = stats["data"].get("last_month")
    return exists, downloads


def _cran(pkg: str) -> Optional[bool]:
    """Whether the package is on CRAN (crandb metadata API)."""
    meta = _get_json(f"https://crandb.r-pkg.org/{pkg}")
    if meta is None:
        return None
    return bool(meta.get("Package"))


def _github(query: str) -> tuple[Optional[int], Optional[int]]:
    """(stars, days-since-last-push) for the best-matching repo via GitHub search.
    Unauthenticated search is rate-limited; degrades to (None, None)."""
    data = _get_json(
        f"https://api.github.com/search/repositories?q={query}&sort=stars&order=desc&per_page=1",
        headers={"User-Agent": "ResearchForge-trends", "Accept": "application/vnd.github+json"},
    )
    if not data or not data.get("items"):
        return None, None
    top = data["items"][0]
    # honesty guard: only trust the match if the repo name actually contains the
    # query token — a bare-name search can otherwise surface a same-named but
    # unrelated repo and attribute its stars to this method.
    full = (top.get("full_name") or top.get("name") or "").lower()
    if query.lower() not in full:
        return None, None
    stars = top.get("stargazers_count")
    pushed_days = None
    pushed = top.get("pushed_at")
    if pushed:
        try:
            from datetime import datetime, timezone

            dt = datetime.strptime(pushed, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            pushed_days = max(0, int((datetime.now(timezone.utc) - dt).days))
        except Exception:
            pushed_days = None
    return stars, pushed_days


# ── source-hint parsing ──────────────────────────────────────────────────────
def parse_sources(sources: list[str]) -> tuple[list[str], list[str]]:
    """Split a candidate's free-text `sources` into (pypi/github names, cran names).

    `sources` look like "mapie", "xgboost", "grf (R)", "splm (R)", "split conformal
    (hand-roll)". A trailing "(R)" marks a CRAN package; "(hand-roll)"/"(...)" notes
    are dropped; the remaining bare tokens are treated as PyPI/GitHub package names."""
    py_names: list[str] = []
    r_names: list[str] = []
    for s in sources:
        s = s.strip()
        if not s:
            continue
        is_r = bool(re.search(r"\(\s*R[\s,)]", s + ")")) or s.lower().endswith("(r)")
        name = re.sub(r"\(.*?\)", "", s).strip()       # drop any "(...)" annotation
        name = re.split(r"[\s/]", name)[0].strip()      # first token ("split conformal" → "split")
        if not name or " " in name or len(name) < 2:
            continue
        if is_r:
            r_names.append(name)
        else:
            py_names.append(name)
    return py_names, r_names


# ── momentum normalization (real numbers → 0-100) ────────────────────────────
def _log_score(value: Optional[int], lo_exp: float, hi_exp: float) -> Optional[float]:
    """Map a count onto 0-100 on a log10 scale between 10**lo_exp and 10**hi_exp."""
    if value is None:
        return None
    x = math.log10(max(0, value) + 1)
    return max(0.0, min(100.0, (x - lo_exp) / (hi_exp - lo_exp) * 100.0))


def compute_momentum(sig: TrendSignal) -> int:
    """Blend the available signals into 0-100. Components: PyPI downloads (log),
    GitHub stars (log), recency of last push; CRAN presence is a small additive
    bonus. Missing components are dropped and the rest renormalized — so a method
    with only one live signal still gets a fair (if lower-confidence) read."""
    parts: list[tuple[float, float]] = []  # (weight, score)
    dl = _log_score(sig.pypi_downloads, 2, 6)   # 100/mo → 0, 1M/mo → 100
    if dl is not None:
        parts.append((0.45, dl))
    stars = _log_score(sig.github_stars, 1, 5)  # 10★ → 0, 100k★ → 100
    if stars is not None:
        parts.append((0.35, stars))
    if sig.github_pushed_days is not None:
        recency = max(0.0, min(100.0, 100.0 - sig.github_pushed_days / 3.65))  # today→100, 1y→0
        parts.append((0.20, recency))
    if not parts:
        # only existence flags available → faint presence signal
        base = 25.0 if (sig.pypi_exists or sig.cran_exists) else 0.0
        return int(round(base))
    wsum = sum(w for w, _ in parts)
    score = sum(w * s for w, s in parts) / wsum
    if sig.cran_exists:
        score = min(100.0, score + 5.0)  # gold-standard packaging bonus
    return int(round(score))


# ── disk cache ───────────────────────────────────────────────────────────────
def _load_cache() -> dict:
    try:
        if _CACHE_FILE.exists():
            return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def _save_cache(cache: dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception:
        pass


# ── top-level: fetch one method's trend signal (cached) ──────────────────────
def fetch_trend(name: str, sources: list[str], use_cache: bool = True) -> TrendSignal:
    """Probe the live sources for one method and return its TrendSignal. Cached on
    disk for 7 days keyed by name. Degrades to available=False (momentum 0) offline."""
    cache = _load_cache() if use_cache else {}
    hit = cache.get(name)
    if hit and (time.time() - hit.get("_fetched_at", 0)) < _CACHE_TTL_S:
        return TrendSignal(**{k: v for k, v in hit.items() if k != "_fetched_at"})

    py_names, r_names = parse_sources(sources)
    sig = TrendSignal(name=name)
    reached = []

    for pkg in py_names:
        exists, downloads = _pypi(pkg)
        if exists is not None:
            sig.pypi_exists = bool(sig.pypi_exists) or exists
            reached.append("pypi")
        if downloads is not None:
            sig.pypi_downloads = max(sig.pypi_downloads or 0, downloads)
        stars, pushed = _github(pkg)
        if stars is not None:
            sig.github_stars = max(sig.github_stars or 0, stars)
            if pushed is not None and (sig.github_pushed_days is None or pushed < sig.github_pushed_days):
                sig.github_pushed_days = pushed
            reached.append("github")
    for pkg in r_names:
        on_cran = _cran(pkg)
        if on_cran is not None:
            sig.cran_exists = bool(sig.cran_exists) or on_cran
            reached.append("cran")

    sig.available = bool(reached)
    sig.momentum = compute_momentum(sig) if sig.available else 0
    if sig.available:
        sig.detail = "实时趋势（来源：" + "、".join(sorted(set(reached))) + "；动量基于真实下载/星标/活跃度）"
    else:
        sig.detail = "实时趋势不可用（离线/无 requests/限流）——回退离线编辑先验"

    if use_cache and sig.available:
        cache[name] = {**sig.as_dict(), "_fetched_at": time.time()}
        _save_cache(cache)
    return sig


# ── trend snapshot (curation-time write → hot-path read by the scorecard) ─────
def write_snapshot(rows: list[dict]) -> Optional[Path]:
    """Persist a momentum snapshot the scorecard can read WITHOUT any network.

    `rows` are dicts with at least `id`, `family`, `momentum`, `available`. We store
    a per-id map and a per-family mean (over available rows), timestamped. Returns
    the path written, or None if nothing live was available / the write failed."""
    live = [r for r in rows if r.get("available") and r.get("momentum")]
    if not live:
        return None
    by_id = {r["id"]: int(r["momentum"]) for r in live}
    fam: dict[str, list[int]] = {}
    for r in live:
        if r.get("family"):
            fam.setdefault(r["family"], []).append(int(r["momentum"]))
    by_family = {k: int(round(sum(v) / len(v))) for k, v in fam.items()}
    payload = {"by_id": by_id, "by_family": by_family, "fetched_at": time.time()}
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_FILE.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        return _SNAPSHOT_FILE
    except Exception:
        return None


def load_snapshot() -> Optional[dict]:
    """Load the momentum snapshot if present and not older than the TTL, else None.
    Pure file read — safe on the per-recommend hot path (never touches the network)."""
    try:
        if not _SNAPSHOT_FILE.exists():
            return None
        snap = json.loads(_SNAPSHOT_FILE.read_text(encoding="utf-8"))
        if (time.time() - snap.get("fetched_at", 0)) > _SNAPSHOT_TTL_S:
            return None
        return snap
    except Exception:
        return None


