import researchforge


def test_version_exposed():
    assert isinstance(researchforge.__version__, str)
    assert researchforge.__version__
