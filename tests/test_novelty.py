from researchforge.recommender import novelty_hint


def test_novelty_stub_offline():
    hint = novelty_hint("DID on agricultural panel")
    assert hint.status == "not_run"
    assert hint.query


def test_novelty_with_injected_search():
    hint = novelty_hint("topic", search_fn=lambda q: ["paper A", "paper B"])
    assert hint.status == "scanned"
    assert len(hint.sources) == 2
