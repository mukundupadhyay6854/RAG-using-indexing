# NCERT RAG System using Qdrant, DeepSeek & FastAPI

A Retrieval-Augmented Generation (RAG) system built on NCERT textbooks that enables users to ask questions and receive answers grounded in textbook content.

Unlike traditional page-based retrieval, this version performs **section-level indexing**, where each chapter section becomes an independent vector. This significantly improves retrieval precision and answer quality.

---

# Features

* NCERT PDF ingestion using PyMuPDF
* Section-level indexing
* Automatic chapter and section extraction
* SentenceTransformer embeddings
* Qdrant vector database
* DeepSeek LLM via OpenRouter
* FastAPI backend
* Semantic search over textbook content
* Source references with chapter and section metadata

---

# Project Structure

```text
RAG-using-indexing/
│
├── books/
│   ├── Biology.pdf
│   └── History.pdf
│
├── vector_db/
│   ├── metadata.json
│   ├── embeddings.npy
│   └── qdrant_db/
│
├── ingest.py
├── generate_embeddings.py
├── load_to_qdrant.py
├── rag.py
├── app.py
│
├── requirements.txt
├── .env
└── README.md
```

---

# Architecture

```text
NCERT PDFs
        ↓
Section Extraction
        ↓
Metadata Generation
        ↓
Embedding Generation
        ↓
Qdrant Vector Store
        ↓
Semantic Retrieval
        ↓
DeepSeek (OpenRouter)
        ↓
Final Answer
```

---

# Metadata Format

Each indexed section is stored as:

```json
{
  "subject": "Biology",
  "chapter": "MICROBES IN HUMAN WELFARE",
  "section": "8.2.1 Fermented Beverages",
  "text": "Microbes especially yeasts..."
}
```

Each section becomes a separate vector.

---

# Embedding Model

```text
BAAI/bge-base-en-v1.5
```

Vector Size:

```text
768
```

Similarity Metric:

```text
Cosine Similarity
```

---

# LLM

This project uses:

```text
DeepSeek Chat
```

through:

```text
OpenRouter
```

---

# Installation

## 1. Clone Repository

```bash
git clone <your-repository-url>
cd RAG-using-indexing
```

---

## 2. Create Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux/Mac:

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If OpenAI SDK is missing:

```bash
pip install openai
```

---

# API Key Setup

Create a file named:

```text
.env
```

Add:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
```

Get a free API key from:

https://openrouter.ai

---

# Add Textbooks

Place NCERT PDFs inside:

```text
books/
```

Example:

```text
books/
├── Biology.pdf
└── History.pdf
```

Currently tested on:

* Biology
* History

---

# Build the Knowledge Base

## Step 1: Extract Sections

```bash
python ingest.py
```

Output:

```text
vector_db/metadata.json
```

---

## Step 2: Generate Embeddings

```bash
python generate_embeddings.py
```

Output:

```text
vector_db/embeddings.npy
```

---

## Step 3: Load Vectors into Qdrant

```bash
python load_to_qdrant.py
```

Output:

```text
vector_db/qdrant_db/
```

---

# Run Local CLI Version

```bash
python rag.py
```

Example:

```text
Enter your question:

What is Microsporogenesis?
```

Output:

```text
Answer
References
```

---

# Run API Server

Start FastAPI:

```bash
uvicorn app:app --reload
```

Server:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

# Example API Request

POST

```text
/ask
```

Request:

```json
{
  "question": "What is Microsporogenesis?"
}
```

Response:

```json
{
  "question": "What is Microsporogenesis?",
  "answer": "...",
  "references": [
    {
      "subject": "Biology",
      "chapter": "SEXUAL REPRODUCTION IN FLOWERING PLANTS",
      "section": "1.2.1 Stamen, Microsporangium and Pollen Grain",
      "score": 0.72
    }
  ]
}
```

---

# Development Workflow

Whenever textbooks change:

```bash
python ingest.py
python generate_embeddings.py
python load_to_qdrant.py
```

Then restart:

```bash
uvicorn app:app --reload
```

---

# Current Limitations

* Geography PDFs are not yet supported.
* Heading extraction is based primarily on numbered sections.
* Some exercise questions may still be indexed.
* Chapter detection in History can occasionally mislabel headings.

---

# Future Improvements

* Hierarchical heading extraction
* Subsection support
* Better History chapter detection
* Reranking for retrieval
* Multi-book support
* Citation-aware answers
* Local LLM support via Ollama

---

# Author

Mukund Upadhyay

B.Tech Artificial Intelligence

NCERT RAG System using:

* PyMuPDF
* SentenceTransformers
* Qdrant
* DeepSeek
* FastAPI
