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

# --------------------------------------------------
# TEST 10: Content Preservation
# --------------------------------------------------

def test_chunking_does_not_drop_document_words():

    text = " ".join(
        f"unique_word_{i}" for i in range(300)
    )

    chunks = chunk_text(
        text,
        chunk_size=200,
        chunk_overlap=40,
    )

    source_words = set(text.split())

    chunk_words = set(
        word
        for chunk in chunks
        for word in chunk["text"].split()
    )

    assert source_words.issubset(chunk_words)


# --------------------------------------------------
# TEST 11: Character Offset Accuracy
# --------------------------------------------------

def test_chunk_character_offsets_match_normalized_text():

    text = (
        "First paragraph contains useful information.\n"
        "\n"
        "Second paragraph contains additional information.\n"
        "\n"
        "Third paragraph contains more useful information."
    )

    normalized_text = "\n".join(
        " ".join(paragraph.split())
        for paragraph in text.replace("\r\n", "\n").split("\n")
        if paragraph.strip()
    )

    chunks = chunk_text(
        text,
        chunk_size=100,
        chunk_overlap=20,
    )

    for chunk in chunks:

        expected_text = normalized_text[
            chunk["start_char"]:chunk["end_char"]
        ].strip()

        assert chunk["text"] == expected_text

def test_numbered_headings_stay_with_their_content():
    text = """
    1. ANNUAL LEAVE

    Employees receive 30 days of paid annual leave.
    Requests must be approved by the manager.

    2. SICK LEAVE

    Employees must notify their manager when they are sick.
    Medical certificates may be required.

    3. IT SUPPORT

    Employees must submit a support request.
    IT Support will assess the issue and arrange repairs.
    """

    chunks = chunk_text(text, chunk_size=120, chunk_overlap=20)

    combined_chunks = [chunk["text"] for chunk in chunks]

    assert any(
        "2. SICK LEAVE" in chunk
        and "Employees must notify their manager" in chunk
        for chunk in combined_chunks
    )

    assert any(
        "3. IT SUPPORT" in chunk
        and "Employees must submit a support request" in chunk
        for chunk in combined_chunks
    )
