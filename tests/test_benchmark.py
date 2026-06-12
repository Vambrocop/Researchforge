from researchforge.benchmark import run_benchmark


def test_benchmark_quality(tmp_path):
    rep = run_benchmark(work_dir=str(tmp_path))

    assert rep.n_cases == 3
    assert rep.profile_accuracy == 1.0  # panel vs not-panel classified correctly
    assert rep.recommendation_score >= 0.8  # mostly-correct feasibility calls
    assert rep.recovery_pass_rate >= 0.5  # known effects recovered within tolerance
