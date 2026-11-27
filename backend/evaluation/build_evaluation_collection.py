import numpy as np

from app.services.vector_store import VectorStore, vector_store


EVALUATION_COLLECTION = "enterprise_evaluation"


def main():
    source = vector_store.collection

    # Read original production records.
    records = source.get(
        include=["documents", "metadatas", "embeddings"]
    )

    ids = records["ids"]
    documents = records["documents"]
    metadatas = records["metadatas"]
    embeddings = records["embeddings"]

    if not ids:
        raise RuntimeError("Production collection is empty.")

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(embeddings)
    ):
        raise RuntimeError("Production record counts do not match.")

    if len(set(ids)) != len(ids):
        raise RuntimeError("Duplicate IDs found in production.")

    if any(embedding is None for embedding in embeddings):
        raise RuntimeError("Some production embeddings are missing.")

    print(f"Source records: {len(ids)}")

    # Create or access the separate evaluation collection.
    evaluation_store = VectorStore(
        collection_name=EVALUATION_COLLECTION
    )

    existing_count = evaluation_store.collection.count()

    # Never overwrite existing evaluation records.
    if existing_count != 0:
        raise RuntimeError(
            f"Evaluation collection already contains "
            f"{existing_count} records. Nothing was changed."
        )

    # Convert NumPy arrays into ordinary Python lists.
    embedding_lists = [
        embedding.tolist()
        for embedding in embeddings
    ]

    # Copy original records without re-chunking or re-embedding.
    evaluation_store.add_documents(
        ids=ids,
        embeddings=embedding_lists,
        documents=documents,
        metadatas=metadatas,
    )

    # Verify copied records.
    copied = evaluation_store.collection.get(
        ids=ids,
        include=["documents", "metadatas", "embeddings"],
    )

    copied_by_id = {
        record_id: index
        for index, record_id in enumerate(copied["ids"])
    }

    if set(copied["ids"]) != set(ids):
        raise RuntimeError("Copied record IDs do not match.")

    for index, record_id in enumerate(ids):
        copied_index = copied_by_id[record_id]

        if copied["documents"][copied_index] != documents[index]:
            raise RuntimeError(
                f"Document mismatch: {record_id}"
            )

        if copied["metadatas"][copied_index] != metadatas[index]:
            raise RuntimeError(
                f"Metadata mismatch: {record_id}"
            )

        if not np.allclose(
            copied["embeddings"][copied_index],
            embeddings[index],
        ):
            raise RuntimeError(
                f"Embedding mismatch: {record_id}"
            )

    print("\nEvaluation collection created successfully.")
    print(f"Collection: {EVALUATION_COLLECTION}")
    print(f"Verified records: {len(copied['ids'])}")
    print("Documents: MATCH")
    print("Metadata: MATCH")
    print("Embeddings: MATCH")
    print("\nProduction collection was not modified.")


if __name__ == "__main__":
    main()
