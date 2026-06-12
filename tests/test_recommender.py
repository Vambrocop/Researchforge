import pandas as pd

from researchforge.profiler import profile_dataset
from researchforge.recommender import recommend
from researchforge.synth import make_panel

_ORDER = {"green": 0, "yellow": 1, "red": 2}


def test_recommend_panel(tmp_path):
    csv = tmp_path / "panel.csv"
    make_panel(n_units=6, n_periods=6, treated=True, seed=5).to_csv(csv, index=False)
    recs = recommend(profile_dataset(csv))
    by_id = {r.entry.id: r for r in recs}

    assert by_id["did"].feasible
    assert by_id["did"].rigor.light in {"green", "yellow"}
    assert by_id["descriptive_stats"].feasible
    assert recs[0].feasible  # best recommendation first
    lights = [r.rigor.light for r in recs]
    assert lights == sorted(lights, key=_ORDER.get)  # green -> yellow -> red


def test_did_red_on_cross_section(tmp_path):
    df = pd.DataFrame(
        {
            "x": [float(i) for i in range(1, 13)],
            "y": [2.0, 4, 5, 4, 5, 7, 8, 9, 8, 9, 11, 12],
        }
    )
    csv = tmp_path / "cs.csv"
    df.to_csv(csv, index=False)
    by_id = {r.entry.id: r for r in recommend(profile_dataset(csv))}

    assert by_id["did"].feasible is False
    assert by_id["did"].rigor.light == "red"
    assert by_id["did"].rigor.unmet  # explains why
