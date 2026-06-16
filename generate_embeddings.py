import json
import numpy as np

from sentence_transformers import SentenceTransformer

print("Loading model on GPU...")

model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cuda"
)

with open(
    "vector_db/metadata.json",
    "r",
    encoding="utf-8"
) as f:

    sections = json.load(f)

texts = []

for section in sections:

    text = f"""
Subject: {section['subject']}
Chapter: {section['chapter']}
Section: {section['section']}

{section['text']}
"""

    texts.append(text)

print(
    f"Generating embeddings for {len(texts)} sections..."
)

embeddings = model.encode(
    texts,
    batch_size=16,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

np.save(
    "vector_db/embeddings.npy",
    embeddings
)

print("\nEmbeddings Saved Successfully")
print(
    "Shape:",
    embeddings.shape
)