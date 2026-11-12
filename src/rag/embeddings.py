from sentence_transformers import SentenceTransformer
from config import (
    EMBEDDING_MODEL,
    EMBEDDING_DIMENSION
)

class EmbeddingService:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def encode(self, texts: list[str]):

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True
        )

        if embeddings.shape[1] != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION} dimensions, "
                f"got {embeddings.shape[1]}"
            )

        return embeddings

if __name__ == "__main__":

    embedding_service = EmbeddingService()

    texts = [
        "Shipping to Cairo takes 2 to 4 days.",
        "Delivery to Alexandria takes 3 to 5 days.",
        "The wireless mouse costs 350 EGP."
    ]

    vectors = embedding_service.encode(texts)

    print("\nNumber of vectors:", len(vectors))
    print("Vector dimensions:", vectors.shape[1])
    print("First vector:", vectors[0][:10])
