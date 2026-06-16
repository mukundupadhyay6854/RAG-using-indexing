import os

from dotenv import load_dotenv
from openai import OpenAI

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
# Retrieve Relevant Sections
# =====================================

results = client.query_points(
    collection_name=COLLECTION_NAME,
    query=query_embedding,
    limit=5
).points

context = ""

sources = []

for result in results:

    payload = result.payload

    context += f"""
Subject: {payload.get('subject', 'Unknown Subject')}
Chapter: {payload.get('chapter', 'Unknown Chapter')}
Section: {payload.get('section', 'Unknown Section')}

{payload.get('text', '')}

----------------------------------------
"""

    sources.append(
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
# LLM Response
# =====================================

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

# =====================================
# Print Answer
# =====================================

print("\n" + "=" * 80)
print("ANSWER")
print("=" * 80)

print(answer)

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
        f"Chapter : {source['chapter']}"
    )

    print(
        f"Section : {source['section']}"
    )

    print(
        f"Score   : {source['score']}"
    )

client.close()