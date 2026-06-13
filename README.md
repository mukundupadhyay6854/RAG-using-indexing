# NCERT RAG System using Qdrant, Gemini & FastAPI

An AI-powered Retrieval-Augmented Generation (RAG) system that allows users to ask questions from NCERT textbooks and receive accurate answers along with source references (Subject, Book, Page Number).

---

## Features

- PDF-based Knowledge Base
- Page-Level Indexing
- Semantic Search using Embeddings
- Vector Database using Qdrant
- Answer Generation using Gemini
- FastAPI REST API
- Swagger Documentation
- Postman Integration
- Multi-Book Support
- Source Attribution with Page Numbers

---

## Architecture

```text
PDF Books
    ↓
Text Extraction
    ↓
Page Indexing
    ↓
Embedding Generation
    ↓
Qdrant Vector Database
    ↓
User Question
    ↓
Embedding Generation
    ↓
Semantic Search
    ↓
Relevant Pages Retrieved
    ↓
Gemini LLM
    ↓
Answer + References
```

---

## Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Vector Database | Qdrant |
| Embedding Model | BAAI/bge-base-en-v1.5 |
| LLM | Gemini 2.5 Flash |
| API Framework | FastAPI |
| API Testing | Swagger UI, Postman |
| PDF Processing | PyMuPDF (fitz) |
| Environment Management | python-dotenv |

---

## Project Structure

```text
RAG/
│
├── books/
│   ├── Biology.pdf
│   ├── Geography.pdf
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
├── requirements.txt
├── .env
└── README.md
```

---

## Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/mukundupadhyay6854/RAG-using-indexing.git

cd RAG-using-indexing
```

---

### 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate:

#### Windows

```bash
venv\Scripts\activate
```

#### Linux/Mac

```bash
source venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Gemini API Key

Create a `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

---

## Data Pipeline

### Step 1: Extract Text from PDFs

```bash
python ingest.py
```

Output:

```text
Metadata Created Successfully
Total Pages Indexed: XXX
```

---

### Step 2: Generate Embeddings

```bash
python generate_embeddings.py
```

Output:

```text
Embeddings Saved Successfully
Shape: (XXX, 768)
```

---

### Step 3: Load Data into Qdrant

```bash
python load_to_qdrant.py
```

Output:

```text
Upload Successful
Vectors Stored: XXX
```

---

## Running the RAG System

### Terminal-Based Version

```bash
python rag.py
```

Example:

```text
Enter your question:
What is Harappan Civilization?
```

Output:

```text
Answer:
...

References:
History.pdf | Page 1
History.pdf | Page 5
```

---

## FastAPI Integration

Start the API server:

```bash
uvicorn app:app
```

Output:

```text
Uvicorn running on:
http://127.0.0.1:8000
```

---

## Swagger Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

Interactive API documentation will be available.

---

## API Endpoints

### Health Check

#### Request

```http
GET /
```

#### Response

```json
{
    "message": "NCERT RAG API Running"
}
```

---

### Ask Question

#### Request

```http
POST /ask
```

#### Request Body

```json
{
    "question": "What is Harappan Civilization?"
}
```

#### Response

```json
{
    "question": "What is Harappan Civilization?",
    "answer": "...",
    "references": [
        {
            "subject": "History",
            "book": "History.pdf",
            "page": 1,
            "score": 0.7714
        }
    ]
}
```

---

## Adding New Books

Place the new PDF inside:

```text
books/
```

Example:

```text
books/
├── Biology.pdf
├── Geography.pdf
├── History.pdf
├── Physics.pdf
└── Chemistry.pdf
```

Then run:

```bash
python ingest.py
python generate_embeddings.py
python load_to_qdrant.py
```

No code changes required.

---

## Current Capabilities

- Multiple NCERT books
- Cross-book retrieval
- Semantic search
- Source references
- REST API support
- FastAPI integration
- Postman testing
- Page-level indexing

---

## Future Improvements

- Hybrid Chunking + Page Indexing
- Metadata Filtering
- Chapter-Level Retrieval
- User Authentication
- Docker Deployment
- Cloud Qdrant Integration
- Streamlit/Web Interface
- Conversation Memory

---

## Author

**Mukund Upadhyay**

B.Tech Artificial Intelligence  
SRM Institute of Science and Technology

GitHub:
https://github.com/mukundupadhyay6854

LinkedIn:
https://www.linkedin.com/in/mukund-upadhyay

---

## License

This project is intended for educational and research purposes.