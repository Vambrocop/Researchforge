from pathlib import Path

from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.synth import make_panel


def test_run_did_produces_outputs(tmp_path):
    csv = tmp_path / "panel.csv"
    make_panel(n_units=6, n_periods=6, treated=True, seed=11).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("did")

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    out = Path(res.output_dir)

    assert out.exists()
    assert (out / "report.md").exists()
    assert (out / "coefficients.csv").exists()
    assert (out / "analysis_code.py").exists()
    assert "Difference" in res.method
    assert res.summary


def test_run_descriptive_produces_table(tmp_path):
    csv = tmp_path / "panel.csv"
    make_panel(seed=12).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("descriptive_stats")

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"))
    assert (Path(res.output_dir) / "table_describe.csv").exists()
