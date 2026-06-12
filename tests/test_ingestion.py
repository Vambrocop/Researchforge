from researchforge.ingestion import ingest_inbox


def test_ingest_skill_and_markdown(tmp_path):
    inbox = tmp_path / "inbox"
    skill = inbox / "my-skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: my-skill\ndescription: does X\n---\nbody", encoding="utf-8"
    )
    (inbox / "note.md").write_text(
        "---\nname: note\ndescription: a note\n---\n", encoding="utf-8"
    )
    (inbox / "README.md").write_text("readme, should be skipped", encoding="utf-8")
    manifest = tmp_path / "ingested.json"

    items = ingest_inbox(inbox=inbox, manifest=manifest)
    names = {i.name for i in items}

    assert "my-skill" in names
    assert "note" in names
    assert "README" not in {i.name for i in items}
    assert (inbox / "_processed" / "my-skill").exists()  # archived
    assert manifest.exists()


def test_ingest_empty_inbox(tmp_path):
    assert ingest_inbox(inbox=tmp_path / "nope", manifest=tmp_path / "m.json") == []
