import subprocess
import sys


def test_cli_version():
    result = subprocess.run(
        [sys.executable, "-m", "researchforge.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "researchforge" in result.stdout.lower()


def test_cli_recommend_runs(tmp_path, capsys):
    from researchforge.cli import main
    from researchforge.synth import make_panel

    csv = tmp_path / "p.csv"
    make_panel(seed=2).to_csv(csv, index=False)

    assert main(["recommend", str(csv)]) == 0
    assert "可做的分析" in capsys.readouterr().out
