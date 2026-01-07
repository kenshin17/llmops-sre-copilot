from sre_copilot.ingestion.pipeline import chunk_text


def test_chunk_text_splits_long_string():
    text = " ".join(["word"] * 120)
    chunks = chunk_text(text, max_len=50)
    assert len(chunks) > 1
    assert all(len(c) > 0 for c in chunks)
