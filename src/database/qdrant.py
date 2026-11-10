
from qdrant_client import QdrantClient, models
from config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    EMBEDDING_DIMENSION
)

def get_qdrant_client():

    if not QDRANT_URL:
        raise ValueError("QDRANT_URL is missing!")

    if not QDRANT_API_KEY:
        raise ValueError("QDRANT_API_KEY is missing!")

    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        timeout=30
    )

def create_collection(client):

    if client.collection_exists(QDRANT_COLLECTION):
        print(f"Collection already exists: {QDRANT_COLLECTION}")
        return

    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIMENSION,
            distance=models.Distance.COSINE
        )
    )

    print(f"Collection created: {QDRANT_COLLECTION}")

def test_connection():

    client = get_qdrant_client()

    print("Successfully connected to Qdrant!")

    create_collection(client)

if __name__ == "__main__":
    test_connection()
