from researchforge.profiler.scan import profile_folder, scan_folder
from researchforge.synth import make_panel


def test_scan_and_profile_folder(tmp_path):
    make_panel(seed=1).to_csv(tmp_path / "a.csv", index=False)
    make_panel(seed=2).to_csv(tmp_path / "b.csv", index=False)
    (tmp_path / "notes.txt").write_text("ignore me")

    found = scan_folder(tmp_path)
    assert len(found) == 2

    profiles = profile_folder(tmp_path)
    assert len(profiles) == 2
    assert all(fp.is_panel for fp in profiles.values())
