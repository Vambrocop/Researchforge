"""CI guard: a catalog entry's declared params must COVER the config keys its
handler actually reads (declared ⊇ read).

Why this exists
---------------
``AnalysisEntry.params`` (machine-readable ParamSpec list) is the single source of
truth consumed by ``catalog.config_schema.validate_config`` at run time. validate
only **warns**, but if an entry UNDER-declares a key its handler genuinely reads,
a user who legitimately passes that key gets a spurious "未知参数 '<key>'" warning
("已忽略、回退默认") even though the analysis really does honour it. ~30 families were
backfilled by hand; this test stops that from regressing.

What it checks
--------------
For every catalog entry that HAS declared params, the set of config keys its
handler reads (own body PLUS the bodies of helpers it transitively calls) must be a
SUBSET of the declared param names.

Entries that read config keys but declare NO params are a *separate* concern (with
no spec, validate stays silent — no spurious warning is possible), so they are
reported as a warning list but do NOT fail the test.

Precise helper attribution (avoids false positives)
---------------------------------------------------
A naive "attribute every module-level cfg key to every handler in the file" over-
reports. We build a call graph and propagate a helper's cfg keys to a caller only
through edges that actually carry the user config, defeating two false-positive shapes:

  1. Helper never called (croston). ``forecasting.py`` has a ``_detect_period`` helper
     that reads ``seasonal_periods``; ``exponential_smoothing`` / ``theta_method`` call
     it, but ``croston`` does NOT — so croston must not get ``seasonal_periods``.
     Exercised by ``test_croston_not_flagged``.
  2. Helper called, but NOT with the user config object (cna). ``configurational.py``'s
     ``_branch_cna`` calls ``_qca_incl_cut({"incl_cut": cfg.get("con")}, 0.8)`` — the
     helper reads ``incl_cut`` from a *synthesised* dict, not the user's config, so cna
     must not be flagged for ``incl_cut`` (it really exposes ``con``/``cov``).

So a call edge propagates a callee's cfg keys ONLY IF the call site passes a
config-bearing argument (a bare ``cfg`` / ``config`` / ``ctx`` Name). Edges that pass
only literals/other locals (the cna case) do not propagate. Helpers that take ``ctx``
(e.g. ``_detect_period(ctx, y)``) still propagate, since ``ctx`` carries ``ctx.cfg``.

Self-contained: handlers/entries are discovered dynamically (no hardcoded id list).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from researchforge.catalog.registry import Catalog

REPO = Path(__file__).resolve().parents[1]
BRANCHES_DIR = REPO / "researchforge" / "executor" / "branches"
HELPERS_DIR = REPO / "researchforge" / "executor" / "_helpers"

# Regex for cfg-key reads, BOTH quote styles, on cfg AND config receivers:
#   cfg.get("K"  /  cfg['K']  /  config.get("K"  /  config["K"]
_KEY_RE = re.compile(
    r"""(?:cfg|config)            # receiver
        (?:                       # one of:
            \.get\(\s*["']([^"']+)["']   #   .get("K"  / .get('K'
          | \[\s*["']([^"']+)["']\s*\]   #   ["K"]     / ['K']
        )
    """,
    re.VERBOSE,
)


def _source_segment(src: str, node: ast.AST) -> str:
    """Exact source text of an AST node (function body), for regex scanning."""
    seg = ast.get_source_segment(src, node)
    return seg if seg is not None else ""


def _cfg_keys_in(text: str) -> set[str]:
    """All config keys read in a chunk of source text (both quote styles, both
    cfg and config receivers)."""
    keys: set[str] = set()
    for m in _KEY_RE.finditer(text):
        keys.add(m.group(1) or m.group(2))
    return keys


# Argument names that carry the user config into a helper.
_CONFIG_ARGS = {"cfg", "config", "ctx"}


def _config_carrying_callees(node: ast.FunctionDef) -> set[str]:
    """Names of functions ``node`` calls AND passes a config-bearing argument to.

    A call counts only if one of its (positional or keyword) arguments is a bare Name
    in ``_CONFIG_ARGS`` (``cfg`` / ``config`` / ``ctx``) — i.e. the user config actually
    flows into that callee. Calls that pass only literals / synthesised dicts / other
    locals do NOT count (this is what keeps the cna ``incl_cut`` false positive out).

    Matches both bare-name calls (``foo(cfg)``) and attribute calls' final attr
    (``x.foo(cfg)``). We only ever resolve these names against known helper defs, so
    over-collecting harmless names is fine — they won't map to a helper."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if not isinstance(sub, ast.Call):
            continue
        passes_config = any(
            isinstance(a, ast.Name) and a.id in _CONFIG_ARGS for a in sub.args
        ) or any(
            isinstance(kw.value, ast.Name) and kw.value.id in _CONFIG_ARGS
            for kw in sub.keywords
        )
        if not passes_config:
            continue
        f = sub.func
        if isinstance(f, ast.Name):
            names.add(f.id)
        elif isinstance(f, ast.Attribute):
            names.add(f.attr)
    return names


class _Module:
    """Parsed module: per-function own cfg keys + config-carrying callees.

    (Handler id -> function mapping comes from the LIVE ``BRANCH_REGISTRY`` instead of
    the source decorators, so dynamic forms like ``@register(*_REGRESSION)`` resolve.)"""

    def __init__(self, path: Path):
        self.path = path
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        # function name -> own cfg keys read directly in its body
        self.func_keys: dict[str, set[str]] = {}
        # function name -> callees it passes the user config to (config-carrying edges)
        self.func_calls: dict[str, set[str]] = {}
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body_text = _source_segment(src, node)
                self.func_keys[node.name] = _cfg_keys_in(body_text)
                self.func_calls[node.name] = _config_carrying_callees(node)


def _parse_modules() -> tuple[list[_Module], list[_Module]]:
    """Parse every branch module (recursing into subpackages) plus the shared
    _helpers/*.py. Returns (branch_modules, helper_modules)."""
    branch_mods: list[_Module] = []
    for p in sorted(BRANCHES_DIR.rglob("*.py")):
        if "__pycache__" in str(p) or p.name == "__init__.py":
            continue
        branch_mods.append(_Module(p))
    helper_mods: list[_Module] = []
    for p in sorted(HELPERS_DIR.glob("*.py")):
        if p.name == "__init__.py":
            continue
        helper_mods.append(_Module(p))
    return branch_mods, helper_mods


def _build_global_helper_index(
    branch_mods: list[_Module], helper_mods: list[_Module]
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Across ALL modules, build name -> (own cfg keys, callee names).

    Helper names are global by convention (the shared resolvers live in _helpers
    and are imported by branch modules; branch-local helpers are unique enough).
    If a name is defined in more than one module we UNION their keys/calls — a safe
    over-approximation that can only make the check stricter, never miss a real key.
    """
    keys: dict[str, set[str]] = {}
    calls: dict[str, set[str]] = {}
    for mod in branch_mods + helper_mods:
        for name, ks in mod.func_keys.items():
            keys.setdefault(name, set()).update(ks)
        for name, cs in mod.func_calls.items():
            calls.setdefault(name, set()).update(cs)
    return keys, calls


def _transitive_keys(
    start_func: str,
    own_keys: set[str],
    own_calls: set[str],
    global_keys: dict[str, set[str]],
    global_calls: dict[str, set[str]],
    max_depth: int = 4,
) -> set[str]:
    """cfg keys read by ``start_func`` directly PLUS keys read by any helper it
    transitively reaches THROUGH config-carrying call edges (bounded depth, cycle-safe).
    A callee only contributes if it is a known function (resolves in ``global_keys``) —
    unknown calls (stdlib, pandas) are ignored, giving precise attribution."""
    read: set[str] = set(own_keys)
    seen: set[str] = {start_func}
    frontier = [(c, 0) for c in own_calls]
    while frontier:
        callee, depth = frontier.pop()
        if callee in seen or depth > max_depth:
            continue
        seen.add(callee)
        if callee in global_keys:
            read |= global_keys[callee]
        for nxt in global_calls.get(callee, set()):
            if nxt not in seen:
                frontier.append((nxt, depth + 1))
    return read


def _live_handler_funcs() -> dict[str, str]:
    """id -> handler function name, taken from the LIVE ``BRANCH_REGISTRY``.

    Using the populated registry (rather than only the source decorators) makes the
    test robust to dynamic registration such as ``@register(*_REGRESSION)`` in
    regression.py, which the AST scan cannot expand. Importing
    ``researchforge.executor.run`` imports the branches package, running every
    ``@register`` decorator."""
    import researchforge.executor.run  # noqa: F401  (side effect: populate registry)
    from researchforge.executor._branch_api import BRANCH_REGISTRY

    return {hid: fn.__name__ for hid, fn in BRANCH_REGISTRY.items()}


def _read_keys_for_handlers() -> dict[str, set[str]]:
    """Map every registered handler id -> the set of config keys it transitively reads.

    Helper resolution is MODULE-LOCAL plus the shared ``_helpers/`` package — never
    across sibling branch modules. This is essential: several branch modules define a
    same-named branch-local helper (e.g. ``_resolve_xy`` exists in both spatial_extra.py
    [reads x/y] and ml_supervised.py [reads outcome/predictors]); a global union by name
    would cross-contaminate (svm←x/y, acf_pacf←is_returns) and produce false positives.
    """
    branch_mods, helper_mods = _parse_modules()
    # shared helpers are global by design (imported across families)
    h_keys: dict[str, set[str]] = {}
    h_calls: dict[str, set[str]] = {}
    for mod in helper_mods:
        for n, ks in mod.func_keys.items():
            h_keys.setdefault(n, set()).update(ks)
        for n, cs in mod.func_calls.items():
            h_calls.setdefault(n, set()).update(cs)
    # handler/branch-local function name -> its defining branch module
    fname_to_mod: dict[str, _Module] = {}
    for mod in branch_mods:
        for n in mod.func_keys:
            fname_to_mod.setdefault(n, mod)

    out: dict[str, set[str]] = {}
    for hid, fname in _live_handler_funcs().items():
        mod = fname_to_mod.get(fname)
        if mod is None:
            out[hid] = set()
            continue
        # combined lookup = shared helpers OVERLAID by this handler's own module
        combo_keys = {**h_keys, **mod.func_keys}
        combo_calls = {**h_calls, **mod.func_calls}
        out[hid] = _transitive_keys(
            fname, combo_keys.get(fname, set()), combo_calls.get(fname, set()),
            combo_keys, combo_calls,
        )
    return out


def _declared_params() -> dict[str, set[str]]:
    """entry id -> declared param names. (executor_ref is free-form — 'numpy',
    'semopy', 'py::id', … — so do NOT filter on it; read-keys come from the live
    registry, which covers every registered handler regardless of executor_ref text.)"""
    cat = Catalog.load()
    return {e.id: {p.name for p in e.params} for e in cat.all()}


def _entries_with_params() -> set[str]:
    cat = Catalog.load()
    return {e.id for e in cat.all() if e.params}


def test_declared_params_cover_read_keys() -> None:
    """For every entry that DECLARES params, read_keys ⊆ declared names."""
    read = _read_keys_for_handlers()
    declared = _declared_params()
    with_params = _entries_with_params()

    gaps: list[str] = []
    for entry_id in sorted(with_params):
        names = declared.get(entry_id, set())
        keys = read.get(entry_id, set())
        missing = keys - names
        if missing:
            gaps.append(
                f"{entry_id}: handler reads {sorted(missing)} "
                f"but declares {sorted(names)}"
            )

    assert not gaps, (
        "Catalog entries UNDER-declare config params their handlers read "
        "(users passing these keys get a spurious '未知参数' warning). "
        "Add the missing param(s) to the entry yaml (mirror correlation_suite.yaml):\n"
        + "\n".join(gaps)
    )


def test_croston_not_flagged() -> None:
    """Regression: precise call-graph attribution must NOT attribute
    ``seasonal_periods`` (read by the _detect_period helper) to croston, which never
    calls that helper. Guards the very false positive this test design exists for."""
    read = _read_keys_for_handlers()
    assert "croston" in read, "croston handler not discovered"
    assert "seasonal_periods" not in read["croston"], (
        "croston was wrongly attributed 'seasonal_periods' — call-graph attribution "
        f"is over-reporting. croston reads: {sorted(read['croston'])}"
    )
    # Sanity: the handlers that DO call _detect_period should get the key, proving
    # the attribution is discriminating (not just empty for everyone).
    assert "seasonal_periods" in read.get("exponential_smoothing", set()), (
        "exponential_smoothing should read 'seasonal_periods' via _detect_period — "
        "attribution may be too narrow."
    )


def test_report_entries_with_read_keys_but_no_params() -> None:
    """Informational (never fails): entries whose handler reads config keys but that
    declare NO params. validate stays silent for these (no spec) so no spurious
    warning fires — but they're candidates for future backfill."""
    read = _read_keys_for_handlers()
    declared = _declared_params()
    undocumented = {
        eid: sorted(keys)
        for eid, keys in read.items()
        if keys and eid in declared and not declared[eid]
    }
    if undocumented:
        lines = [f"  {eid}: reads {keys}" for eid, keys in sorted(undocumented.items())]
        print(
            "\n[info] entries that READ config keys but declare NO params "
            "(no spurious warning — validate is silent without a spec — but "
            "candidates for backfill):\n" + "\n".join(lines)
        )
    # Always passes; this is a backlog report, not a gate.
    assert True
