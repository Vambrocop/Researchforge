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
    out = capsys.readouterr().out
    assert "推荐 top" in out  # focused, goal-aware menu (was the full "可做的分析" list)


def test_cli_pick_binary_picks_logistic(tmp_path, capsys):
    import numpy as np
    import pandas as pd

    from researchforge.cli import main

    rng = np.random.default_rng(3)
    n = 220
    x1 = rng.normal(0, 1, n)
    p = 1 / (1 + np.exp(-(0.4 + 0.9 * x1)))
    csv = tmp_path / "b.csv"
    pd.DataFrame({"approved": rng.binomial(1, p), "income": x1.round(3),
                  "age": rng.normal(40, 8, n).round(2), "score": rng.normal(0, 1, n).round(3)}).to_csv(csv, index=False)

    assert main(["pick", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "🎯 快速推荐" in out
    assert "logistic_regression" in out          # the right call for a binary outcome
    assert "descriptive_stats" not in out.split("其他候选")[0]  # trivial helper skipped as THE pick
    assert "run " in out                          # prints the exact run command


def test_cli_pick_survival(tmp_path, capsys):
    import numpy as np
    import pandas as pd

    from researchforge.cli import main

    rng = np.random.default_rng(5)
    n = 200
    csv = tmp_path / "s.csv"
    pd.DataFrame({"duration": rng.exponential(10, n).round(2), "event": rng.binomial(1, 0.6, n),
                  "age": rng.normal(60, 10, n).round(1)}).to_csv(csv, index=False)

    assert main(["pick", str(csv)]) == 0
    out = capsys.readouterr().out
    # a survival model is THE pick, with the data-diagnostic reason shown
    assert any(m in out for m in ("survival_analysis", "stratified_cox", "parametric_survival"))
    assert "生存" in out
