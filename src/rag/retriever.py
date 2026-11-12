import logging

from qdrant_client import models

from config import (
    STORE_ID,
    QDRANT_COLLECTION,
    TOP_K,
    SIMILARITY_THRESHOLD
)

from database import get_qdrant_client
from rag.embeddings import EmbeddingService

logger = logging.getLogger(__name__)

STOP_WORDS = {
    # Arabic
    "كام", "كم", "بكام", "سعر", "اسعار", "ما", "هو",
    "هي", "هل", "من", "في", "على", "عن", "ايه",
    "ماهو", "ممكن", "عايز", "اعرف",

    # English
    "how", "much", "what", "is", "the", "of",
    "price", "cost", "for", "do", "you", "have"
}

SYNONYMS = {
    "الماوس": "mouse",  "ماوس": "mouse",    "mouse": "mouse",
    "اللاسلكي": "wireless", "لاسلكي": "wireless", "wireless": "wireless",
    "الشاحن": "charger", "شاحن": "charger",  "charger": "charger",
    "الكابل": "cable",  "كابل": "cable",    "cable": "cable",
    "باور": "power",    "بنك": "bank",
}


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, and collapse whitespace."""
    import re
    import unicodedata

    text = text.lower().strip()
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = text.replace("\u0640", "")
    text = re.sub(r"[أإآ]", "ا", text)
    text = text.replace("ى", "ي")
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _keywords(text: str) -> set:
    words = _normalize(text).split()
    return {SYNONYMS.get(w, w) for w in words if w not in STOP_WORDS}


def _keyword_score(query: str, document: str) -> float:
    qk = _keywords(query)
    dk = _keywords(document)
    if not qk:
        return 0.0
    return len(qk & dk) / len(qk)


def _product_code_match(query: str, document: str) -> bool:
    import re
    pattern = r"\bNS-\d+\b"
    q_codes = set(re.findall(pattern, query.upper()))
    d_codes = set(re.findall(pattern, document.upper()))
    return bool(q_codes & d_codes)


class Retriever:

    def __init__(self):
        self.client = get_qdrant_client()
        self.embedding_service = EmbeddingService()

    def search(self, query: str, top_k: int = TOP_K) -> list[dict]:
        if not query or not query.strip():
            return []

        query_vector = self.embedding_service.encode([query])[0]

        candidate_limit = min(max(top_k * 5, 20), 50)

        raw = self.client.query_points(
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
            limit=candidate_limit,
            with_payload=True
        ).points

        logger.debug("Candidates retrieved: %d", len(raw))

        results = []

        for hit in raw:
            payload = hit.payload or {}
            text = payload.get("text", "")
            semantic = float(hit.score)

            if semantic < SIMILARITY_THRESHOLD:
                continue

            kw_score = _keyword_score(query, text)
            code_bonus = 0.35 if _product_code_match(query, text) else 0.0
            final = semantic + 0.20 * kw_score + code_bonus

            results.append({
                "text": text,
                "source": payload.get("source", "Unknown"),
                "page": payload.get("page", 0),
                "score": semantic,
                "keyword_score": kw_score,
                "final_score": final,
            })

        results.sort(key=lambda r: r["final_score"], reverse=True)

        logger.debug(
            "Top-%d results: %s",
            top_k,
            [
                f"{r['source']} sem={r['score']:.2f} kw={r['keyword_score']:.2f}"
                for r in results[:top_k]
            ]
        )

        return results[:top_k]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    retriever = Retriever()
    q = "كام سعر الماوس اللاسلكي؟"
    docs = retriever.search(q)
    print(f"\nQuestion: {q}")
    for i, d in enumerate(docs, 1):
        print(f"\nResult {i}: {d['source']} | score={d['score']:.3f}")
        print(d["text"][:200])
