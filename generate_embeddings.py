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

    pages = json.load(f)

texts = [
    f"""
    Subject: {page['subject']}
    Book: {page['book']}
    Page: {page['page']}

    {page['text']}
    """
    for page in pages
]

print(
    f"Generating embeddings for {len(texts)} pages..."
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