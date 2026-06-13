import fitz
import os
import json

BOOKS_FOLDER = "books"

all_pages = []

for pdf_file in os.listdir(BOOKS_FOLDER):

    if not pdf_file.endswith(".pdf"):
        continue

    pdf_path = os.path.join(
        BOOKS_FOLDER,
        pdf_file
    )

    subject = os.path.splitext(
        pdf_file
    )[0]

    print(
        f"Processing {pdf_file}"
    )

    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):

        text = doc[page_num].get_text().strip()

        if not text:
            continue

        all_pages.append(
            {
                "subject": subject,
                "book": pdf_file,
                "page": page_num + 1,
                "text": text
            }
        )

    doc.close()

os.makedirs(
    "vector_db",
    exist_ok=True
)

with open(
    "vector_db/metadata.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_pages,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nMetadata Created Successfully")
print(
    f"Total Pages Indexed: {len(all_pages)}"
)