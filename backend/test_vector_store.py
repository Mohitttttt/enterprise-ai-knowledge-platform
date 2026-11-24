from app.services.embedding_service import generate_embeddings
from app.services.vector_store import vector_store


def main():

    texts = [
        "Employees can reset their passwords through the company portal.",
        "Annual leave entitlement is 30 days per year.",
        "The IT department provides technical support.",
    ]

    embeddings = generate_embeddings(texts)

    ids = [
        "test_doc_chunk_1",
        "test_doc_chunk_2",
        "test_doc_chunk_3",
    ]

    metadatas = [
        {"source": "test_document.txt", "chunk_index": 0},
        {"source": "test_document.txt", "chunk_index": 1},
        {"source": "test_document.txt", "chunk_index": 2},
    ]

    vector_store.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print("Stored vectors:", vector_store.count())

    query = "How do I recover my company login?"

    query_embedding = generate_embeddings([query])[0]

    results = vector_store.search(
        query_embedding=query_embedding,
        n_results=2,
    )

    print("\nSearch query:", query)
    print("\nSearch results:")

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        print("\nDocument:", document)
        print("Metadata:", metadata)
        print("Distance:", distance)


if __name__ == "__main__":
    main()