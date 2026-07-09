"""run_meta.json — every run_analysis output dir carries a reproducibility manifest
(engine version, data hash, config, timestamp). See CLAUDE.md task note: complements
(does not replace) study.py's own study_meta.json."""

from __future__ import annotations

import json
from pathlib import Path

from researchforge import __version__
from researchforge.catalog import Catalog
from researchforge.executor import run_analysis
from researchforge.profiler import profile_dataset
from researchforge.synth import make_panel


def test_run_writes_run_meta_json(tmp_path):
    csv = tmp_path / "panel.csv"
    make_panel(seed=13).to_csv(csv, index=False)
    fp = profile_dataset(csv)
    entry = Catalog.load().by_id("descriptive_stats")

    res = run_analysis(fp, entry, output_root=str(tmp_path / "outputs"),
                        config={"foo": "bar"})
    out = Path(res.output_dir)
    meta_path = out / "run_meta.json"

    assert meta_path.exists()
    assert "run_meta.json" in res.files
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    assert meta["engine_version"] == __version__
    assert meta["analysis_id"] == "descriptive_stats"
    assert meta["method"] == entry.method
    assert meta["data_path"] == str(fp.path)
    assert isinstance(meta["data_sha256"], str) and len(meta["data_sha256"]) == 64
    assert all(c in "0123456789abcdef" for c in meta["data_sha256"])
    assert meta["n_rows"] == fp.n_rows
    assert meta["n_cols"] == fp.n_cols
    assert meta["config"] == {"foo": "bar"}
    assert meta["timestamp"]
    assert "report.md" in meta["files"]
