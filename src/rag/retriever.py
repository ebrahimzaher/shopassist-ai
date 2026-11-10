
from qdrant_client import models
from config import (
    STORE_ID,
    QDRANT_COLLECTION,
    TOP_K,
    SIMILARITY_THRESHOLD
)
from database import get_qdrant_client
from rag import EmbeddingService

class Retriever:
    def __init__(self):
        self.client = get_qdrant_client()
        self.embedding_service = EmbeddingService()

    def search(self, query: str, top_k: int = TOP_K):
        if not query.strip():
            return []

        query_vector = self.embedding_service.encode([query])[0]

        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector.tolist(),
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="store_id",
                        match=models.MatchValue(value=STORE_ID)
                    )
                ]
            ),
            limit=top_k,
            with_payload=True
        ).points

        relevant_results = []

        for result in results:
            if result.score < SIMILARITY_THRESHOLD:
                continue

            relevant_results.append({
                "text": result.payload["text"],
                "source": result.payload["source"],
                "page": result.payload["page"],
                "score": result.score
            })

        return relevant_results

if __name__ == "__main__":
    retriever = Retriever()

    question = "How much is the wireless mouse?"

    results = retriever.search(question)

    print("\nQuestion:", question)
    print("\nRetrieved documents:")

    if not results:
        print("No relevant information found.")

    for index, result in enumerate(results, start=1):
        print(f"\nResult {index}")
        print("Score:", round(result["score"], 3))
        print("Source:", result["source"])
        print("Page:", result["page"])
        print("Text:", result["text"])
