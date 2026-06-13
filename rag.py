import os

from dotenv import load_dotenv
import google.generativeai as genai

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)

client = QdrantClient(
    path="./vector_db/qdrant_db"
)

COLLECTION_NAME = "ncert_collection"

question = input(
    "\nEnter your question: "
)

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

sources = []

for result in results:

    payload = result.payload

    context += payload["text"]
    context += "\n\n"

    sources.append(
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

If the answer is not present in the context, reply:

I could not find the answer in the provided NCERT content.

Context:
{context}

Question:
{question}

Give a concise answer.
"""

response = llm.generate_content(
    prompt
)

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)

print(response.text)

print("\n" + "=" * 80)
print("REFERENCES")
print("=" * 80)

for idx, source in enumerate(
    sources,
    start=1
):

    print(f"\n[{idx}]")

    print(
        f"Subject : {source['subject']}"
    )

    print(
        f"Book    : {source['book']}"
    )

    print(
        f"Page    : {source['page']}"
    )

    print(
        f"Score   : {source['score']}"
    )

client.close()