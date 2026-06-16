import os

from dotenv import load_dotenv
from openai import OpenAI

from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

# =====================================
# Load Environment Variables
# =====================================

load_dotenv()

# =====================================
# OpenRouter Client
# =====================================

print("Loading OpenRouter...")

llm = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
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
        limit=5
    ).points

    context = ""

    references = []

    for result in results:

        payload = result.payload

        context += f"""
Subject: {payload.get('subject', 'Unknown Subject')}
Chapter: {payload.get('chapter', 'Unknown Chapter')}
Section: {payload.get('section', 'Unknown Section')}

{payload.get('text', '')}

----------------------------------------
"""

        references.append(
            {
                "subject": payload.get(
                    "subject",
                    "Unknown Subject"
                ),

                "chapter": payload.get(
                    "chapter",
                    "Unknown Chapter"
                ),

                "section": payload.get(
                    "section",
                    "Unknown Section"
                ),

                "score": round(
                    result.score,
                    4
                )
            }
        )

    prompt = f"""
You are an NCERT textbook assistant.

Answer ONLY using the provided NCERT context.

If the answer is not available in the context, reply exactly:

I could not find the answer in the provided NCERT content.

Context:
{context}

Question:
{question}

Provide a concise and accurate answer.
"""

    try:

        response = llm.chat.completions.create(
            model="deepseek/deepseek-chat",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        answer = response.choices[0].message.content

    except Exception as e:

        answer = f"LLM Error: {str(e)}"

    return {
        "question": question,
        "answer": answer,
        "references": references
    }