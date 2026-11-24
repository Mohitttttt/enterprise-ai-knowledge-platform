from app.services.text_chunker import chunk_text


sample_text = """
Enterprise AI Knowledge Intelligence Platform.

This platform allows employees to upload documents,
search company knowledge and retrieve relevant information.

The system uses document extraction, text chunking,
vector embeddings and semantic search.

Retrieval Augmented Generation allows AI systems
to answer questions using relevant document context.
""" * 20


chunks = chunk_text(sample_text)

print("=" * 60)
print("TEXT CHUNKING TEST")
print("=" * 60)

print(f"Total chunks generated: {len(chunks)}")

for chunk in chunks:

    print("\n" + "-" * 60)

    print(f"Chunk Index: {chunk['chunk_index']}")
    print(f"Characters: {chunk['character_count']}")
    print(f"Start: {chunk['start_char']}")
    print(f"End: {chunk['end_char']}")

    print("\nText Preview:")
    print(chunk["text"][:150])