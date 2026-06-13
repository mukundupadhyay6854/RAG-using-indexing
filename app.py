import os

from dotenv import load_dotenv
import google.generativeai as genai

from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

# =====================================
# FastAPI App
# =====================================

app = FastAPI(
    title="NCERT RAG API"
)

# =====================================
# Load Models Once
# =====================================

print("Loading Gemini...")

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

print("Loading Embedding Model...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)

print("Connecting Qdrant...")

client = QdrantClient(
    path="./vector_db/qdrant_db"
)

COLLECTION_NAME = "ncert_collection"

# =====================================
# Request Schema
# =====================================

class QuestionRequest(BaseModel):
    question: str

# =====================================
# Root Endpoint
# =====================================

@app.get("/")
def root():

    return {
        "message": "NCERT RAG API Running"
    }

# =====================================
# Ask Endpoint
# =====================================

@app.post("/ask")
def ask_question(
    request: QuestionRequest
):

    question = request.question

    query_embedding = model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=10
    ).points

    context = ""

    references = []

    for result in results:

        payload = result.payload

        context += payload["text"]
        context += "\n\n"

        references.append(
            {
                "subject": payload["subject"],
                "book": payload["book"],
                "page": payload["page"],
                "score": round(result.score, 4)
            }
        )

    prompt = f"""
You are an NCERT textbook assistant.

Answer ONLY using the provided NCERT context.

If the answer is not available in the context, reply:

I could not find the answer in the provided NCERT content.

Context:
{context}

Question:
{question}

Provide a concise and accurate answer.
"""

    response = llm.generate_content(
        prompt
    )

    return {
        "question": question,
        "answer": response.text,
        "references": references
    }