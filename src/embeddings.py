
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """
    Load the Sentence Transformer model.
    """
    embedding_model = SentenceTransformer(MODEL_NAME)

    return embedding_model


def generate_embeddings(
    texts,
    embedding_model,
    batch_size=32
):
    """
    Generate normalized text embeddings.
    """

    if not texts:
        raise ValueError("No text provided for embedding generation.")

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")
