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
