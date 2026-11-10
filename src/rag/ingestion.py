from pathlib import Path
import pymupdf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

def extract_text_from_pdf(pdf_path: Path):
    extracted_pages = []

    with pymupdf.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()

            if text:
                extracted_pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number
                })

    return extracted_pages

def load_all_pdfs():
    documents = []

    for pdf_path in DOCUMENTS_DIR.glob("*.pdf"):
        pages = extract_text_from_pdf(pdf_path)
        documents.extend(pages)

        print(f"Loaded: {pdf_path.name} ({len(pages)} pages)")

    return documents

if __name__ == "__main__":
    documents = load_all_pdfs()

    print(f"\nTotal pages extracted: {len(documents)}")

    if documents:
        print("\nFirst extracted text:")
        print(documents[0]["text"][:500])
    else:
        print("\nNo PDF text found.")
