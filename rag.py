import os

from dotenv import load_dotenv
import google.generativeai as genai

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
# Gemini Model
# =====================================

llm = genai.GenerativeModel(
    "gemini-2.5-flash"
)

# =====================================
# Embedding Model
# =====================================

print("Loading embedding model...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)

# =====================================
# Qdrant Connection
# =====================================

client = QdrantClient(
    path="./vector_db/qdrant_db"
)

COLLECTION_NAME = "ncert_collection"

# =====================================
# User Question
# =====================================

question = input(
    "\nEnter your question: "
)

# =====================================
# Generate Query Embedding
# =====================================

query_embedding = model.encode(
    question,
    normalize_embeddings=True
).tolist()

# =====================================
# Retrieve Relevant Pages
# =====================================

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

# =====================================
# Prompt
# =====================================

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

# =====================================
# Gemini Response
# =====================================

response = llm.generate_content(
    prompt
)

# =====================================
# Print Answer
# =====================================

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)

print(response.text)

# =====================================
# References
# =====================================

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