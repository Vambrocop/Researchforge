"""Tests for LEONTIEF INPUT-OUTPUT ANALYSIS (resource family / economics):
``input_output_analysis``.

Known-value cases are hand-computed in the docstrings; honest-degrade paths assert
the Chinese "跳过" message and no crash.

Hand calc for A = [[0.2, 0.1], [0.3, 0.2]]:
    I - A = [[0.8, -0.1], [-0.3, 0.8]],  det = 0.64 - 0.03 = 0.61
    L = (1/0.61) * [[0.8, 0.1], [0.3, 0.8]]
      = [[1.3114754, 0.1639344], [0.4918033, 1.3114754]]
    output multipliers (column sums of L):
        col0 = 1.3114754 + 0.4918033 = 1.8032787
        col1 = 0.1639344 + 1.3114754 = 1.4754098
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from researchforge.catalog.schema import AnalysisEntry, Precondition
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset


def _entry() -> AnalysisEntry:
    return AnalysisEntry(
        id="input_output_analysis",
        method="Leontief Input-Output Analysis",
        domain="economics",
        family="resource",
        goal="explain",
        preconditions=Precondition(min_rows=2),
    )


def _csv(tmp_path: Path, name: str, df: pd.DataFrame) -> Path:
    p = tmp_path / name
    df.to_csv(p, index=False)
    return p


# --------------------------------------------------------------------------- #
# (a) known coefficient matrix -> hand-computed L, multipliers, linkages
# --------------------------------------------------------------------------- #
def test_io_coefficient_matrix_known_value(tmp_path: Path) -> None:
    """A = [[0.2,0.1],[0.3,0.2]] is already a coefficient matrix (all in [0,1),
    column sums 0.5 and 0.3 < 1). Output multipliers = [1.8032787, 1.4754098];
    backward linkages = [1.1, 0.9] (mean multiplier = 1.6393443)."""
    csv = _csv(tmp_path, "io.csv", pd.DataFrame({
        "agri": [0.2, 0.3],
        "energy": [0.1, 0.2],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    e = res.estimates
    assert e["n_sectors"] == 2.0
    # spectral radius of A: eigenvalues 0.2 +/- sqrt(0.1*0.3)=0.2+/-0.17320508 ->
    # rho = 0.37320508 (< 1, productive)
    assert math.isclose(e["spectral_radius_A"], 0.3732051, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(e["multiplier__agri"], 1.8032787, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(e["multiplier__energy"], 1.4754098, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(e["mean_output_multiplier"], 1.6393443, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(e["max_output_multiplier"], 1.8032787, rel_tol=0, abs_tol=1e-5)

    out = Path(res.output_dir)
    assert (out / "io_multipliers.csv").exists()
    tbl = pd.read_csv(out / "io_multipliers.csv")
    assert {"sector", "output_multiplier", "backward_linkage", "forward_linkage"}.issubset(tbl.columns)
    bl = dict(zip(tbl["sector"].astype(str), tbl["backward_linkage"]))
    assert math.isclose(bl["agri"], 1.1, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(bl["energy"], 0.9, rel_tol=0, abs_tol=1e-5)
    # forward linkages = normalized row sums; row sums = [1.4754098, 1.8032787],
    # mean same 1.6393443 -> [0.9, 1.1]
    fl = dict(zip(tbl["sector"].astype(str), tbl["forward_linkage"]))
    assert math.isclose(fl["agri"], 0.9, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(fl["energy"], 1.1, rel_tol=0, abs_tol=1e-5)


def test_io_flow_matrix_with_total_output(tmp_path: Path) -> None:
    """Raw flow matrix Z with config total_output. Pick Z so that Z / x̂ reproduces
    A = [[0.2,0.1],[0.3,0.2]]: with x = [10, 20],
        column 0 (agri using): z = a*x0 = [0.2*10, 0.3*10] = [2, 3]
        column 1 (energy using): z = a*x1 = [0.1*20, 0.2*20] = [2, 4]
    so Z = [[2, 2], [3, 4]] and total_output = [10, 20] -> same A -> same multipliers."""
    csv = _csv(tmp_path, "z.csv", pd.DataFrame({
        "sector": ["agri", "energy"],
        "agri": [2.0, 3.0],
        "energy": [2.0, 4.0],
        "x": [10.0, 20.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"total_output": "x", "sectors": "sector"})
    e = res.estimates
    assert e["n_sectors"] == 2.0
    assert math.isclose(e["multiplier__agri"], 1.8032787, rel_tol=0, abs_tol=1e-5)
    assert math.isclose(e["multiplier__energy"], 1.4754098, rel_tol=0, abs_tol=1e-5)
    assert "Z + total_output" in res.summary


def test_io_raw_flow_without_total_output_degrades(tmp_path: Path) -> None:
    """A raw flow matrix with NO total_output and values outside [0,1) cannot yield a
    PRODUCTIVE Leontief model: column-normalizing by column sums gives a column-
    stochastic A (ρ=1, (I−A) singular). The handler must honestly degrade and ask for
    total_output rather than fabricate a degenerate model."""
    csv = _csv(tmp_path, "z.csv", pd.DataFrame({
        "agri": [2.0, 3.0],
        "energy": [2.0, 4.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "total_output" in res.summary  # tells the user how to proceed
    assert "n_sectors" not in res.estimates  # nothing fabricated


# --------------------------------------------------------------------------- #
# (b) non-productive matrix (spectral radius >= 1) -> honest 跳过
# --------------------------------------------------------------------------- #
def test_io_nonproductive_honest_degrade(tmp_path: Path) -> None:
    """Z = [[6,6],[6,6]] with total_output = [10,10] -> A = [[0.6,0.6],[0.6,0.6]],
    eigenvalues {1.2, 0} -> spectral radius 1.2 >= 1 -> not productive ->
    (I - A) inverse not economically valid -> honest 跳过, no crash, no multipliers."""
    csv = _csv(tmp_path, "z.csv", pd.DataFrame({
        "a": [6.0, 6.0],
        "b": [6.0, 6.0],
        "x": [10.0, 10.0],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"total_output": "x"})
    assert "跳过" in res.summary
    assert "n_sectors" not in res.estimates
    assert "spectral_radius_A" not in res.estimates


# --------------------------------------------------------------------------- #
# (c) final-demand vector -> x = L * f matches hand calc
# --------------------------------------------------------------------------- #
def test_io_final_demand_required_output(tmp_path: Path) -> None:
    """A = [[0.2,0.1],[0.3,0.2]], f = [100, 50]:
        L = [[1.3114754, 0.1639344], [0.4918033, 1.3114754]]
        x = L @ f:
          x0 = 1.3114754*100 + 0.1639344*50 = 131.14754 + 8.196721 = 139.344262
          x1 = 0.4918033*100 + 1.3114754*50 = 49.180328 + 65.573770 = 114.754098"""
    csv = _csv(tmp_path, "io.csv", pd.DataFrame({
        "sector": ["agri", "energy"],
        "agri": [0.2, 0.3],
        "energy": [0.1, 0.2],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"sectors": "sector", "final_demand": [100.0, 50.0]})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "io_multipliers.csv")
    assert "required_output" in tbl.columns
    req = dict(zip(tbl["sector"].astype(str), tbl["required_output"]))
    assert math.isclose(req["agri"], 139.344262, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(req["energy"], 114.754098, rel_tol=0, abs_tol=1e-4)


def test_io_final_demand_from_column(tmp_path: Path) -> None:
    """final_demand given as a COLUMN name resolves the same x = L*f."""
    csv = _csv(tmp_path, "io.csv", pd.DataFrame({
        "sector": ["agri", "energy"],
        "agri": [0.2, 0.3],
        "energy": [0.1, 0.2],
        "fd": [100.0, 50.0],
    }))
    fp = profile_dataset(csv)
    # NOTE: 'fd' is a numeric column, so it would be a 3rd numeric column. With 2
    # rows the matrix is the first 2 numeric cols (agri, energy); fd is left over
    # and used only as the final-demand vector via config.
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"),
                       config={"sectors": "sector", "final_demand": "fd"})
    out = Path(res.output_dir)
    tbl = pd.read_csv(out / "io_multipliers.csv")
    assert "required_output" in tbl.columns
    req = dict(zip(tbl["sector"].astype(str), tbl["required_output"]))
    assert math.isclose(req["agri"], 139.344262, rel_tol=0, abs_tol=1e-4)
    assert math.isclose(req["energy"], 114.754098, rel_tol=0, abs_tol=1e-4)


# --------------------------------------------------------------------------- #
# honest degrade: not square / no numeric
# --------------------------------------------------------------------------- #
def test_io_not_square_degrade(tmp_path: Path) -> None:
    """3 numeric columns but only 2 rows -> not square -> honest 跳过."""
    csv = _csv(tmp_path, "ns.csv", pd.DataFrame({
        "a": [0.1, 0.2],
        "b": [0.1, 0.2],
        "c": [0.1, 0.2],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_sectors" not in res.estimates


def test_io_no_numeric_degrade(tmp_path: Path) -> None:
    """No usable numeric matrix -> honest 跳过, no crash."""
    csv = _csv(tmp_path, "txt.csv", pd.DataFrame({
        "label": ["a", "b", "c"],
        "note": ["x", "y", "z"],
    }))
    fp = profile_dataset(csv)
    res = run_analysis(fp, _entry(), output_root=str(tmp_path / "o"))
    assert "跳过" in res.summary
    assert "n_sectors" not in res.estimates
