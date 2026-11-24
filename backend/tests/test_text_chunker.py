import pytest

from app.services.text_chunker import chunk_text


# --------------------------------------------------
# TEST 1: Empty Input
# --------------------------------------------------

def test_empty_text_returns_empty_list():

    assert chunk_text("") == []
    assert chunk_text("   ") == []


# --------------------------------------------------
# TEST 2: Invalid Chunk Size
# --------------------------------------------------

def test_invalid_chunk_size():

    with pytest.raises(ValueError):
        chunk_text("Some sample text", chunk_size=0)


# --------------------------------------------------
# TEST 3: Invalid Chunk Overlap
# --------------------------------------------------

def test_invalid_chunk_overlap():

    with pytest.raises(ValueError):
        chunk_text(
            "Some sample text",
            chunk_size=100,
            chunk_overlap=100,
        )


# --------------------------------------------------
# TEST 4: Short Text Produces One Chunk
# --------------------------------------------------

def test_short_text_produces_one_chunk():

    text = "Employees receive 30 days of annual leave."

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert text in chunks[0]["text"]


# --------------------------------------------------
# TEST 5: Chunk Metadata
# --------------------------------------------------

def test_chunk_metadata():

    text = (
        "This is a sample paragraph containing "
        "enough information to test chunk metadata. "
    ) * 10

    chunks = chunk_text(
        text,
        chunk_size=200,
        chunk_overlap=40,
    )

    for index, chunk in enumerate(chunks):

        assert chunk["chunk_index"] == index
        assert "text" in chunk
        assert "start_char" in chunk
        assert "end_char" in chunk
        assert "character_count" in chunk

        assert chunk["character_count"] == len(chunk["text"])


# --------------------------------------------------
# TEST 6: Paragraph Boundary Preservation
# --------------------------------------------------

def test_paragraph_boundary_preservation():

    text = (
        "1. Annual Leave\n"
        "Employees receive annual paid leave according "
        "to company policy.\n"
        "2. Sick Leave\n"
        "Employees must notify their manager about "
        "sickness as soon as possible.\n"
    )

    chunks = chunk_text(
        text,
        chunk_size=120,
        chunk_overlap=0,
    )

    assert len(chunks) >= 2

    combined_text = "\n".join(
        chunk["text"] for chunk in chunks
    )

    assert "1. Annual Leave" in combined_text
    assert "2. Sick Leave" in combined_text


# --------------------------------------------------
# TEST 7: Overlap Between Consecutive Chunks
# --------------------------------------------------

def test_chunk_overlap():

    text = " ".join(
        f"word{i}" for i in range(200)
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        chunk_overlap=50,
    )

    assert len(chunks) > 1

    assert any(
        chunks[index]["text"][-20:]
        in chunks[index + 1]["text"]
        for index in range(len(chunks) - 1)
    )


# --------------------------------------------------
# TEST 8: Long Text Produces Multiple Chunks
# --------------------------------------------------

def test_long_text_produces_multiple_chunks():

    text = (
        "Employees must follow company policies "
        "and procedures when requesting leave. "
    ) * 30

    chunks = chunk_text(
        text,
        chunk_size=300,
        chunk_overlap=50,
    )

    assert len(chunks) > 1


# --------------------------------------------------
# TEST 9: Chunk Size Is Respected
# --------------------------------------------------

def test_chunk_size_is_respected():

    text = " ".join(
        f"word{i}" for i in range(300)
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        chunk_overlap=40,
    )

    for chunk in chunks:
        assert len(chunk["text"]) <= 200
