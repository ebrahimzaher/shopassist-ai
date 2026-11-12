
import re

from rag.ingestion import load_all_pdfs
from config import CHUNK_SIZE, CHUNK_OVERLAP


PRODUCT_PATTERN = re.compile(
    r"(?=Product\s+NS-\d+|المنتج\s+NS-\d+)",
    re.IGNORECASE
)


def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
):
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            boundary = text.rfind(" ", start, end)

            if boundary > start + chunk_size * 0.7:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = end - overlap

        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def split_product_catalog(text: str):
    sections = PRODUCT_PATTERN.split(text)

    chunks = []

    for section in sections:
        section = section.strip()

        if not section:
            continue

        # Keep each product together when possible.
        if PRODUCT_PATTERN.match(section):
            product_chunks = split_text(
                section,
                chunk_size=800,
                overlap=50
            )
            chunks.extend(product_chunks)
        else:
            # Catalog introduction or general instructions.
            chunks.extend(split_text(section))

    return chunks


def create_chunks(documents):
    all_chunks = []

    for document in documents:
        source = document["source"]

        if "products_catalog" in source.lower():
            text_chunks = split_product_catalog(document["text"])
        else:
            text_chunks = split_text(document["text"])

        for index, chunk in enumerate(text_chunks):
            all_chunks.append({
                "text": chunk,
                "source": source,
                "page": document["page"],
                "chunk_index": index
            })

    return all_chunks


if __name__ == "__main__":
    documents = load_all_pdfs()
    chunks = create_chunks(documents)

    print(f"Total pages: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")

    if chunks:
        print("\nFirst chunk:")
        print(chunks[0]["text"])

        print("\nMetadata:")
        print({
            "source": chunks[0]["source"],
            "page": chunks[0]["page"],
            "chunk_index": chunks[0]["chunk_index"]
        })

        print("\nProduct chunks:")

        for chunk in chunks:
            if "products_catalog" in chunk["source"].lower():
                print("\nSource:", chunk["source"])
                print("Chunk index:", chunk["chunk_index"])
                print("Text:", chunk["text"][:300])
