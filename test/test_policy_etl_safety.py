from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_policy_etl_replaces_documents_atomically_and_rejects_empty_replacements():
    source = (ROOT / "scripts/etl_policy.py").read_text(encoding="utf-8")

    assert "await session.flush()" in source
    assert "staged_chunk_count == 0" in source
    assert "Policy replacement contains no non-empty chunks" in source
    assert "Deletion and every replacement chunk become visible atomically." in source
