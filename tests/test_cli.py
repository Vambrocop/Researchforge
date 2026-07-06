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


def test_cli_clean_preview_and_apply(tmp_path, capsys):
    import numpy as np
    import pandas as pd

    from researchforge.cli import main

    rng = np.random.default_rng(4)
    n = 200
    common = rng.choice(["A", "B", "C"], 150)
    rare = [f"r{i}" for i in range(25)]
    grp = list(common) + list(rng.choice(rare, 50))
    rng.shuffle(grp)
    df = pd.DataFrame({"grp": grp, "x": rng.normal(0, 1, n).round(3), "const": 1})
    df.loc[:9, "x"] = np.nan  # missing to impute
    csv = tmp_path / "dirty.csv"
    df.to_csv(csv, index=False)

    # preview: shows the plan, writes nothing
    assert main(["clean", str(csv)]) == 0
    out = capsys.readouterr().out
    assert "清理计划" in out and "collapse_rare" in out and "预览" in out
    assert not (tmp_path / "dirty_cleaned.csv").exists()

    # apply: writes a cleaned CSV + a log; const column gone, missing imputed
    outp = tmp_path / "cleaned.csv"
    assert main(["clean", str(csv), "--apply", "--out", str(outp)]) == 0
    out = capsys.readouterr().out
    assert "已应用" in out
    assert outp.exists()
    cleaned = pd.read_csv(outp)
    assert "const" not in cleaned.columns
    assert cleaned["x"].isna().sum() == 0
    assert outp.with_suffix(".cleaning.json").exists()


def test_cli_clean_on_clean_data(tmp_path, capsys):
    import numpy as np
    import pandas as pd

    from researchforge.cli import main

    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(0, 1, 40).round(3), "b": rng.normal(0, 1, 40).round(3)})
    csv = tmp_path / "ok.csv"
    df.to_csv(csv, index=False)
    assert main(["clean", str(csv)]) == 0
    # no actionable steps -> honest "nothing to clean", not a fabricated plan
    out = capsys.readouterr().out
    assert "未发现需要清理" in out or "清理计划" in out  # either honest-empty or advisory-only


def _dirty_df(seed: int = 7, n: int = 200):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    common = rng.choice(["A", "B", "C"], n - 30)
    rare = [f"r{i}" for i in range(20)]
    grp = list(common) + list(rng.choice(rare, 30))
    rng.shuffle(grp)
    df = pd.DataFrame({
        "grp": grp,
        "x": rng.normal(0, 1, n).round(3),
        "const": 1,
    })
    df.loc[:9, "x"] = np.nan
    return df


def test_cli_run_clean_flag_auto_cleans_then_runs(tmp_path, capsys):
    from researchforge.cli import main

    csv = tmp_path / "dirty.csv"
    _dirty_df().to_csv(csv, index=False)

    assert main(["run", str(csv), "descriptive_stats", "--clean"]) == 0
    out = capsys.readouterr().out
    assert "已应用" in out  # disclosure of applied cleaning steps
    assert "已执行" in out  # analysis actually ran afterward
    assert (tmp_path / "dirty_cleaned.csv").exists()


def test_cli_run_without_clean_flag_skips_cleaning(tmp_path, capsys):
    from researchforge.cli import main

    csv = tmp_path / "dirty2.csv"
    _dirty_df(seed=8).to_csv(csv, index=False)

    assert main(["run", str(csv), "descriptive_stats"]) == 0
    out = capsys.readouterr().out
    assert "已执行" in out
    assert "已应用" not in out  # no auto-clean disclosure without --clean
    assert not (tmp_path / "dirty2_cleaned.csv").exists()


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
