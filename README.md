<div align="center">

# 🛒 ShopAssist AI

### RAG-Powered Customer Support Bot for Egyptian E-Commerce

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Telegram](https://img.shields.io/badge/Telegram%20Bot-API%2022.8-2CA5E0?style=flat-square&logo=telegram&logoColor=white)](https://core.telegram.org/bots/api)
[![Groq](https://img.shields.io/badge/LLM-Groq-F55036?style=flat-square&logo=groq&logoColor=white)](https://groq.com/)
[![Qdrant](https://img.shields.io/badge/Vector%20DB-Qdrant-DC2626?style=flat-square)](https://qdrant.tech/)
[![HuggingFace](https://img.shields.io/badge/Embeddings-HuggingFace-FFD21E?style=flat-square&logo=huggingface&logoColor=black)](https://huggingface.co/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

**ShopAssist AI** is a conversational Telegram bot built on **Retrieval-Augmented Generation (RAG)**. It serves as an intelligent customer support assistant for Egyptian e-commerce stores, answering customer questions in Arabic and English with strict factual accuracy — using only information from your own store documents.

---

## ✨ Key Features

| Feature | Details |
|---|---|
| 🧠 **Full RAG Pipeline** | Retrieves relevant store documents before every answer, with hybrid semantic + keyword reranking |
| 🌐 **Bilingual** | Understands Egyptian Arabic (colloquial & formal) and English naturally |
| ⚡ **Fast Inference** | Powered by Groq Cloud LPUs for low-latency text generation |
| 🛡️ **Anti-Hallucination** | Strictly answers from retrieved documents only — never invents prices or policies |
| 🏢 **Multi-Tenant Ready** | Isolates each store's data in Qdrant via `store_id` filtering |
| 🔄 **Fully Async** | Uses `asyncio.to_thread` to handle concurrent messages without blocking |
| 🐳 **Docker-Ready** | Containerized and deployable on any cloud platform |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   Customer (Telegram)                    │
└──────────────────────┬───────────────────────────────────┘
                       │  User Message
                       ▼
┌──────────────────────────────────────────────────────────┐
│                 Telegram Bot Handler                     │
│            (python-telegram-bot v22.8)                   │
│                                                          │
│  1. Check for greeting / small-talk shortcuts            │
│  2. Dispatch to asyncio.to_thread  (non-blocking)        │
└──────────────────────┬───────────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │      RAG Pipeline       │
          │                         │
          │  ① Encode Query         │
          │    (MiniLM-L12-v2)      │
          │          │              │
          │  ② Vector Search        │
          │    (Qdrant, top-20)     │
          │          │              │
          │  ③ Hybrid Re-rank       │
          │    Semantic + Keyword   │
          │    + Product Code       │
          │          │              │
          │  ④ Build Context        │
          │    (Top-3 chunks)       │
          │          │              │
          │  ⑤ Call Groq LLM        │
          │    (temp=0.2)           │
          └────────────┬────────────┘
                       │  Arabic / English Answer
                       ▼
                Customer ✅
```

---

## 📂 Project Structure

```
shopassist-ai/
│
├── documents/                        # Store knowledge base (PDF files)
│   ├── 01_store_information.pdf      # General store information
│   ├── 02_shipping_policy.pdf        # Shipping & delivery policy
│   ├── 03_returns_refunds_policy.pdf # Returns & refund policy
│   ├── 04_payment_methods.pdf        # Available payment methods
│   └── 05_products_catalog.pdf       # Product catalog with prices
│
├── src/                              # Application source code
│   ├── config.py                     # Centralized configuration & env loader
│   ├── main.py                       # Bot entry point
│   │
│   ├── bot/
│   │   ├── __init__.py
│   │   └── handlers.py               # Telegram command & message handlers
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   └── qdrant.py                 # Qdrant client setup & collection management
│   │
│   ├── rag/
│   │   ├── __init__.py               # Exports main RAG classes
│   │   ├── ingestion.py              # PDF text extraction (PyMuPDF)
│   │   ├── chunking.py               # Word-aware overlapping text splitter
│   │   ├── embeddings.py             # SentenceTransformer wrapper (384-dim)
│   │   ├── indexing.py               # Full indexing pipeline (PDF → Qdrant)
│   │   ├── retriever.py              # Hybrid retriever: Semantic + Keyword + Code
│   │   └── generator.py              # Groq LLM answer generator
│   │
│   └── api/
│       └── __init__.py               # FastAPI endpoints (optional)
│
├── Dockerfile                        # Docker image definition
├── .dockerignore
├── pyproject.toml                    # Project metadata & dependencies (PEP 621)
├── requirements.txt                  # Dependencies for Docker builds
├── .env.example                      # Environment variables template
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Bot Framework** | `python-telegram-bot 22.8` | Receive & send messages via Telegram API |
| **LLM** | `Groq Cloud` | Generate context-grounded answers at high speed |
| **Vector DB** | `Qdrant Cloud` | Store and search document embeddings |
| **Embedding Model** | `paraphrase-multilingual-MiniLM-L12-v2` | Encode Arabic & English text into 384-dim vectors |
| **PDF Parser** | `PyMuPDF (MuPDF)` | Extract structured text from PDF documents |
| **Config** | `python-dotenv` | Load secrets and settings from `.env` |
| **Containerization** | `Docker` | Package and deploy on any cloud environment |

---

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- A free [Groq Cloud](https://console.groq.com/) account
- A free [Qdrant Cloud](https://cloud.qdrant.io/) cluster
- A Telegram Bot Token from [@BotFather](https://t.me/BotFather)

---

### 1. Clone the Repository

```bash
git clone https://github.com/ebrahimzaher/shopassist-ai.git
cd shopassist-ai
```

### 2. Create a Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -e .
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
```

Open `.env` and fill in your credentials:

```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token

GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-20b

QDRANT_URL=https://your-cluster-id.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key

STORE_ID=nile_store
```

> **Important — Groq Model Lifecycle:** Groq periodically deprecates or updates model names. If a model is retired, you can switch to any active model in your Groq account (such as `llama-3.3-70b-versatile` or `llama-3.1-8b-instant`) simply by updating `GROQ_MODEL` in `.env` without modifying any code. Check the [Groq Supported Models documentation](https://console.groq.com/docs/models) for the latest list of active models.

---

### 5. Initialize the Qdrant Collection

Create the vector collection before indexing:

```bash
cd src
python -c "from database.qdrant import test_connection; test_connection()"
```

### 6. Index Your Store Documents

Place your store PDF files in the `documents/` folder, then run:

```bash
cd src
python -m rag.indexing
```

What happens during indexing:

```
Loading PDF documents...        ← Reads all PDF files from documents/
Creating chunks...              ← Splits text into overlapping chunks
Generating embeddings...        ← Encodes each chunk into a 384-dim vector
Creating payload index...       ← Enables fast filtering by store_id
Removing old store data...      ← Cleans up stale vectors for this store
Uploading vectors to Qdrant...  ← Uploads in batches of 32
Indexing completed! ✅
```

### 7. Run the Bot

```bash
cd src
python main.py
```

On successful startup you will see:

```
Initializing RAG services...
RAG services initialized successfully!
ShopAssist AI Telegram Bot is running...
```

---

## 🐳 Docker Deployment

```bash
# Build the image
docker build -t shopassist-ai .

# Run with your .env file
docker run -d \
  --name shopassist-bot \
  --env-file .env \
  --restart unless-stopped \
  shopassist-ai
```

---

## 🧠 How the RAG Pipeline Works

### Phase 1 — Indexing (runs once)

```
PDF Files
   ↓
PyMuPDF  →  Extract text, page by page
   ↓
Chunking →  Split into overlapping chunks (500 chars, 80 overlap)
            Product catalog pages use per-product chunking (800 chars)
   ↓
SentenceTransformer  →  Encode each chunk  →  384-dim vector
   ↓
Qdrant  →  Store vector + payload { text, source, page, store_id }
```

### Phase 2 — Query & Answer (runs for every message)

```
Customer Question
   ↓
SentenceTransformer  →  Encode question  →  384-dim query vector
   ↓
Qdrant  →  Retrieve top-20 candidates (filtered by store_id)
   ↓
Hybrid Re-ranking:
   final_score = semantic_score
               + 0.20 × keyword_overlap_score
               + 0.35 × product_code_exact_match_bonus
   ↓
Top-3 chunks above similarity threshold (0.40)
   ↓
Groq LLM  →  System prompt + Context + Question  →  Answer
   ↓
Telegram  →  Deliver answer to customer ✅
```

---

## 🛡️ Anti-Hallucination Guardrails

The system enforces 7 core guardrail rules in the LLM system prompt:

1. **No fabrication** — Only answer using factual information present in the retrieved context; never invent prices, policies, or products.
2. **Graceful handling of unmentioned services** — If a service or feature (e.g., installments) is absent from the context, treat it as currently unavailable rather than hallucinating terms.
3. **First-person support persona** — Never refer to the system as "the robot" or "the chatbot"; communicate naturally as an authentic customer support representative.
4. **No internal metadata leaks** — Never expose internal terms such as "documents", "files", "PDFs", "source", or "context" to customers.
5. **No false transaction claims** — Never claim that an order was placed, a refund was approved, or a payment was executed.
6. **Contact integrity** — Only provide contact details explicitly documented in the retrieved store knowledge base.
7. **Strict language mirroring** — Detect the customer's language and respond entirely in the matching language (Arabic or English) without mixing.

---

## ⚙️ Configuration Reference

All settings can be overridden via environment variables or edited directly in `src/config.py`:

| Setting | Default | Description |
|---|---|---|
| `GROQ_MODEL` | `openai/gpt-oss-20b` | Groq LLM model name |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | HuggingFace embedding model |
| `EMBEDDING_DIMENSION` | `384` | Vector dimensionality |
| `CHUNK_SIZE` | `500` | Maximum characters per chunk |
| `CHUNK_OVERLAP` | `80` | Character overlap between adjacent chunks |
| `TOP_K` | `3` | Number of chunks sent to the LLM as context |
| `SIMILARITY_THRESHOLD` | `0.40` | Minimum cosine similarity score to include a chunk |
| `STORE_ID` | `nile_store` | Store identifier for Qdrant multi-tenant isolation |

---

## 📊 Benchmark & Evaluation (35 Queries)

The system was evaluated using an automated benchmark suite (`scripts/evaluate_benchmark.py`) across **35 diverse queries** spanning 7 categories: Products & Pricing, Shipping & Logistics, Returns & Refunds, Payment Methods, Store Info, Guardrail Stress Tests (adversarial / prompt injection / out-of-domain), and Conversational small-talk.

### Key Measured Metrics:

| Metric | Result | Definition |
|---|---|---|
| **Guardrail Compliance** | **100.0%** (35/35) | Programmatic check verifying 0 internal metadata leaks (`document`, `PDF`, `مستندات`), 0 false order/refund confirmations, and 0 third-person persona breaks. |
| **Language Consistency** | **100.0%** (35/35) | Automated script detection verifying that the response script (Arabic Unicode `\u0600-\u06FF` vs. Latin) strictly matched the customer query language. |
| **Retrieval Hit Rate** | **82.9%** (29/35) | Percentage of queries where at least one knowledge chunk scored above the cosine cutoff (`SIMILARITY_THRESHOLD = 0.40`). |
| **Vector Retrieval Latency** | **234.4 ms** (Median) | Time to compute MiniLM query embedding + Qdrant vector retrieval + lexical keyword reranking (P95: 302.6 ms). |
| **End-to-End Latency** | **5.96 s** (Median) | Complete user turnaround time: retrieval pipeline + Groq LLM API generation over network. |

### Failure Analysis & Iteration Insights:

- **Language Consistency Resolution (100.0% - 35/35)**: Initial benchmark runs revealed 3 language mismatches on English out-of-domain questions (`M04` crypto, `G02` appliances, `G05` prompt injection) where 0 relevant chunks met the similarity threshold, triggering a hardcoded Arabic fallback template. Introducing query-language detection on the 0-chunk fallback path in `generator.py` resolved this, bringing consistency to a perfect 35/35.
- **Retrieval Drop Root Cause (6 / 35)**: 4 out of the 6 drops were **intended safety drops** where out-of-domain or adversarial queries correctly scored below the 0.40 threshold. The remaining 2 (`P03` 20W charger cable, `P07` speaker) stemmed from a cross-lingual vocabulary gap between colloquial Egyptian terms (e.g., using "سماعة" instead of "speaker") and English PDF catalog entries, causing cosine scores to narrowly miss the threshold.

To reproduce the benchmark on your own setup:
```bash
python scripts/evaluate_benchmark.py
```

Detailed per-query logs and breakdowns are exported to [`benchmark_results.json`](benchmark_results.json).

---

## 💬 Example Interactions

| Customer Message | Bot Response |
|---|---|
| `"How much is the wireless mouse?"` | States the exact price from the product catalog |
| `"What are the shipping fees to Alexandria?"` | Explains shipping fees and delivery timelines |
| `"Can I return an item after two weeks?"` | Details the 14-day return and refund policy conditions |
| `"What payment methods do you accept?"` | Lists all available payment methods (Cash on Delivery, Cards, Digital Wallets, Installments) |
| `"Hi, who are you and how can you help?"` | Responds with a polite greeting and lists supported topics |

---

## 👨‍💻 Author

**Ebrahim Zaher**

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
