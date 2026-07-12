"""Shared role-hint constants and helpers for the experimental-design family.

Used by 2+ handlers; kept here (family-local) rather than in executor/_helpers/
because they are specific to design-aware analyses (block/treatment/plot/G×E roles).
"""

from __future__ import annotations

from researchforge.profiler.semantics import ROLE_HINTS

# column-name hints for auto-detecting roles when config doesn't specify them
# (single source: researchforge/profiler/semantics.py — Wave L ColumnSemantics C0)
_BLOCK_HINTS = ROLE_HINTS["block"]
_TRT_HINTS = ROLE_HINTS["treatment"]
_ROW_HINTS = ("row", "lane")
_COL_HINTS = ("col", "column", "position")
_GENO_HINTS = ("genotype", "geno", "variety", "cultivar", "hybrid", "line", "entry", "accession")
_ENV_HINTS = ("environment", "env", "site", "location", "loc", "year", "season", "trial")


def _degenerate_fit(model, y_vals) -> bool:
    """True if an OLS fit has too little residual error (near-saturated design) for a
    trustworthy F-test — residual MS ~ 0 or residual df < 1 makes F explode spuriously."""
    import numpy as np

    rdf = float(model.df_resid)
    mse = float(model.mse_resid)
    yv = float(np.var(np.asarray(y_vals, dtype=float)))
    return rdf < 1 or not np.isfinite(mse) or mse < 1e-9 * max(yv, 1e-12)


def _pick_geno_env(fp, df, cfg, y):
    """Resolve genotype + environment role columns for G×E methods (AMMI / GGE).

    Config overrides accept ANY column (a genotype/env factor may profile as count/id,
    not categorical). Auto-default: name-hint match first, else leftover categorical/id
    columns in declaration order. Returns (genotype, environment, guessed)."""
    _excl = {fp.unit_col, fp.time_col}
    role_cols = [c.name for c in fp.columns
                 if c.kind in {"categorical", "binary", "count", "id"} and c.name not in _excl]
    g_cfg = cfg.get("genotype") if cfg.get("genotype") in df.columns else None
    e_cfg = cfg.get("environment") if cfg.get("environment") in df.columns else None
    genotype, environment = g_cfg, e_cfg

    def _pick(hints, taken):
        for c in role_cols:
            if c == y or c in taken:
                continue
            if any(h in c.lower() for h in hints):
                return c
        return None

    if genotype is None:
        genotype = _pick(_GENO_HINTS, {environment})
    if environment is None:
        environment = _pick(_ENV_HINTS, {genotype})
    leftover = [c for c in role_cols if c not in {genotype, environment} and c != y]
    if genotype is None and leftover:
        genotype = leftover.pop(0)
    if environment is None and leftover:
        environment = leftover.pop(0)
    guessed = not (g_cfg is not None and e_cfg is not None)
    return genotype, environment, guessed


def _ge_means_matrix(sub, genotype, environment, y):
    """Genotype×environment table of cell means (genotypes=rows, environments=cols),
    dropping any row/col with missing cells so SVD has a complete matrix."""
    import pandas as pd  # noqa: F401

    mat = sub.groupby([genotype, environment], observed=True)[y].mean().unstack()
    mat = mat.dropna(axis=0, how="any").dropna(axis=1, how="any")
    return mat
