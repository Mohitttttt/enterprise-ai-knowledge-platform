from app.services.vector_store import vector_store


def main():

    print("\n--- DOCUMENT MANAGEMENT TEST ---")

    # 1. List documents
    documents = vector_store.get_documents()

    print("\nUploaded documents:")

    for document in documents:
        print(document)

    # 2. Retrieve document details
    if documents:

        document_id = documents[0]["document_id"]

        print("\nDocument details:")

        details = vector_store.get_document(document_id)

        print(details)

        # 3. Test deletion separately later
        print("\nDocument deletion method is available.")

    else:
        print("\nNo uploaded documents with document IDs found.")

    print("\nTotal vectors:", vector_store.count())


if __name__ == "__main__":
    main()