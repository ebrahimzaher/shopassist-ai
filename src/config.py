import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_NAME = "ShopAssist AI"
STORE_ID = os.getenv("STORE_ID", "nile_store")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

QDRANT_COLLECTION = "shopassist_documents"

EMBEDDING_MODEL = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

EMBEDDING_DIMENSION = 384

CHUNK_SIZE = 500
CHUNK_OVERLAP = 80

TOP_K = 3
SIMILARITY_THRESHOLD = 0.40
