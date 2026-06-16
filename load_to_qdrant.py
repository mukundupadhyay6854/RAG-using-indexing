import json
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

print("Loading metadata...")

with open(
    "vector_db/metadata.json",
    "r",
    encoding="utf-8"
) as f:

    metadata = json.load(f)

print("Loading embeddings...")

embeddings = np.load(
    "vector_db/embeddings.npy"
)

print(
    f"Metadata Records: {len(metadata)}"
)

print(
    f"Embeddings Shape: {embeddings.shape}"
)

# =====================================
# Validation
# =====================================

if len(metadata) != len(embeddings):
    raise ValueError(
        "Metadata and embeddings count do not match!"
    )

# =====================================
# Connect Qdrant
# =====================================

client = QdrantClient(
    path="./vector_db/qdrant_db"
)

COLLECTION_NAME = "ncert_collection"

# =====================================
# Delete Existing Collection
# =====================================

collections = client.get_collections()

existing_collections = [
    collection.name
    for collection in collections.collections
]

if COLLECTION_NAME in existing_collections:

    print(
        "Deleting existing collection..."
    )

    client.delete_collection(
        collection_name=COLLECTION_NAME
    )

# =====================================
# Create Collection
# =====================================

print(
    "Creating collection..."
)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=768,
        distance=Distance.COSINE
    )
)

# =====================================
# Build Points
# =====================================

points = []

for idx, (section, embedding) in enumerate(
    zip(metadata, embeddings)
):

    point = PointStruct(
        id=idx,

        vector=embedding.tolist(),

        payload={
            "subject": section["subject"],
            "chapter": section["chapter"],
            "section": section["section"],
            "text": section["text"]
        }
    )

    points.append(point)

# =====================================
# Upload
# =====================================

print(
    f"Uploading {len(points)} sections..."
)

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

# =====================================
# Verify
# =====================================

info = client.get_collection(
    COLLECTION_NAME
)

print("\nUpload Successful")

print(
    f"Vectors Stored: {info.points_count}"
)

client.close()