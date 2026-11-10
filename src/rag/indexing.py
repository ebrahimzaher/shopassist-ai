
from uuid import uuid5, NAMESPACE_URL
from qdrant_client import models
from config import STORE_ID, QDRANT_COLLECTION
from database import get_qdrant_client
from rag import load_all_pdfs
from rag import create_chunks
from rag import EmbeddingService

def index_documents():
    print("Loading PDF documents...")
    documents = load_all_pdfs()

    if not documents:
        print("No documents found!")
        return

    print("\nCreating chunks...")
    chunks = create_chunks(documents)
    print(f"Total chunks: {len(chunks)}")

    print("\nGenerating embeddings...")
    embedding_service = EmbeddingService()

    texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_service.encode(texts)

    if len(chunks) != len(vectors):
        raise ValueError("Chunks and embeddings count mismatch!")

    client = get_qdrant_client()

    print("\nCreating payload index...")

    client.create_payload_index(
        collection_name=QDRANT_COLLECTION,
        field_name="store_id",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )

    print("Payload index is ready!")

    store_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="store_id",
                match=models.MatchValue(value=STORE_ID)
            )
        ]
    )

    print("\nRemoving old store data...")

    client.delete(
        collection_name=QDRANT_COLLECTION,
        points_selector=models.FilterSelector(
            filter=store_filter
        ),
        wait=True
    )

    print("Old store data removed.")

    points = []

    for chunk, vector in zip(chunks, vectors):
        point_id = str(
            uuid5(
                NAMESPACE_URL,
                f"{STORE_ID}:{chunk['source']}:{chunk['page']}:{chunk['chunk_index']}"
            )
        )

        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector.tolist(),
                payload={
                    "text": chunk["text"],
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_index": chunk["chunk_index"],
                    "store_id": STORE_ID
                }
            )
        )

    print("\nUploading vectors to Qdrant...")

    batch_size = 32

    for start in range(0, len(points), batch_size):
        batch = points[start:start + batch_size]

        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=batch,
            wait=True
        )

        uploaded = min(start + batch_size, len(points))
        print(f"Uploaded {uploaded}/{len(points)}")

    result = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=store_filter,
        exact=True
    )

    print("\nIndexing completed successfully!")
    print(f"Total stored points: {result.count}")

if __name__ == "__main__":
    index_documents()
