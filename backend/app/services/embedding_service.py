from functools import lru_cache

from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load the embedding model once and reuse it.
    """
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully!")

    return model


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Convert multiple text passages into normalized vectors.
    """

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Input texts cannot contain empty strings.")

    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return embeddings.tolist()


def generate_embedding(text: str) -> list[float]:
    """
    Generate an embedding for a single text passage.
    """

    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    return generate_embeddings([text])[0]
