from app.services.embedding_service import (
    generate_embedding,
    generate_embeddings,
    EMBEDDING_DIMENSION,
)


def main():

    print("=" * 60)
    print("EMBEDDING GENERATION TEST")
    print("=" * 60)

    texts = [
        "How can I reset my company password?",
        "I forgot my login credentials and need account recovery.",
        "What is the company's annual leave policy?",
    ]

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(texts)

    print(f"\nTotal embeddings generated: {len(embeddings)}")

    for index, embedding in enumerate(embeddings):

        print(f"\nText {index + 1}: {texts[index]}")
        print(f"Vector dimensions: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")

    assert all(
        len(embedding) == EMBEDDING_DIMENSION
        for embedding in embeddings
    )

    print("\nDimension validation: PASSED")

    # Test single-text embedding
    single_embedding = generate_embedding(
        "Explain enterprise knowledge management."
    )

    print(
        f"Single embedding dimensions: {len(single_embedding)}"
    )

    print("\nAll embedding tests passed!")


if __name__ == "__main__":
    main()