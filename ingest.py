import fitz
import os
import json
import re

BOOKS_FOLDER = "books"

SUPPORTED_BOOKS = {
    "Biology.pdf",
    "History.pdf"
}

OUTPUT_FILE = "vector_db/metadata.json"

MIN_SECTION_CHARS = 200


def is_noise(line):
    line = line.strip()

    if not line:
        return True

    if line.isdigit():
        return True

    if line.startswith("Reprint"):
        return True

    noise_words = {
        "BIOLOGY",
        "HISTORY",
        "THEMES IN INDIAN HISTORY"
    }

    return line.upper() in noise_words


def is_heading_number(line):
    """
    Matches:
    8.1
    8.2
    8.2.1
    9.3.2
    1.
    2.
    """

    line = line.strip()

    patterns = [
        r"^\d+\.$",
        r"^\d+\.\d+$",
        r"^\d+\.\d+\.\d+$",
        r"^\d+\.\d+\.\d+\.\d+$"
    ]

    return any(
        re.match(pattern, line)
        for pattern in patterns
    )


def is_inline_heading(line):
    """
    Matches:
    8.1 Microbes in Household Products
    8.2.1 Fermented Beverages
    1. Beginnings
    2.1 Agricultural Technologies
    """

    line = line.strip()

    patterns = [
        r"^\d+\.\s+.+",
        r"^\d+\.\d+\s+.+",
        r"^\d+\.\d+\.\d+\s+.+",
        r"^\d+\.\d+\.\d+\.\d+\s+.+"
    ]

    return any(
        re.match(pattern, line)
        for pattern in patterns
    )


def get_clean_lines(doc):

    lines = []

    for page in doc:

        text = page.get_text()

        for line in text.splitlines():

            line = line.strip()

            if is_noise(line):
                continue

            lines.append(line)

    return lines


def save_section(
    all_sections,
    subject,
    chapter,
    section,
    buffer
):

    if not section:
        return

    text = "\n".join(buffer).strip()

    if len(text) < MIN_SECTION_CHARS:
        return

    all_sections.append(
        {
            "subject": subject,
            "chapter": chapter,
            "section": section,
            "text": text
        }
    )


all_sections = []

for pdf_file in os.listdir(BOOKS_FOLDER):

    if pdf_file not in SUPPORTED_BOOKS:
        print(f"Skipping {pdf_file}")
        continue

    print(f"\nProcessing {pdf_file}")

    pdf_path = os.path.join(
        BOOKS_FOLDER,
        pdf_file
    )

    subject = os.path.splitext(pdf_file)[0]

    doc = fitz.open(pdf_path)

    lines = get_clean_lines(doc)

    doc.close()

    current_chapter = None
    current_section = None
    buffer = []

    i = 0

    while i < len(lines):

        line = lines[i].strip()

        # ------------------------------------------
        # Chapter Detection
        # ------------------------------------------

        if line.upper().startswith("CHAPTER"):

            save_section(
                all_sections,
                subject,
                current_chapter,
                current_section,
                buffer
            )

            current_section = None
            buffer = []

            chapter_lines = []

            j = i + 1

            while j < len(lines):

                candidate = lines[j].strip()

                if (
                    is_inline_heading(candidate)
                    or is_heading_number(candidate)
                ):
                    break

                if candidate:
                    chapter_lines.append(candidate)

                if len(chapter_lines) >= 2:
                    break

                j += 1

            current_chapter = " ".join(
                chapter_lines
            ).upper()

            print(
                f"Chapter Found: {current_chapter}"
            )

            i += 1
            continue

        # ------------------------------------------
        # Inline Headings
        # ------------------------------------------

        if is_inline_heading(line):

            heading = line

            if i + 1 < len(lines):

                nxt = lines[i + 1].strip()

                if (
                    len(nxt.split()) <= 4
                    and not is_inline_heading(nxt)
                    and not is_heading_number(nxt)
                ):
                    heading += " " + nxt
                    i += 1

            save_section(
                all_sections,
                subject,
                current_chapter,
                current_section,
                buffer
            )

            current_section = heading
            buffer = []

            i += 1
            continue

        # ------------------------------------------
        # Split Headings
        # Example:
        #
        # 8.2
        # MICROBES IN INDUSTRIAL PRODUCTS
        # ------------------------------------------

        if is_heading_number(line):

            if i + 1 < len(lines):

                next_line = lines[i + 1].strip()

                heading = (
                    f"{line} {next_line}"
                )

                save_section(
                    all_sections,
                    subject,
                    current_chapter,
                    current_section,
                    buffer
                )

                current_section = heading
                buffer = []

                i += 2
                continue

        # ------------------------------------------
        # Content
        # ------------------------------------------

        if current_section:
            buffer.append(line)

        i += 1

    save_section(
        all_sections,
        subject,
        current_chapter,
        current_section,
        buffer
    )

os.makedirs(
    "vector_db",
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        all_sections,
        f,
        ensure_ascii=False,
        indent=4
    )

print("\nMetadata Created Successfully")
print(
    f"Total Sections Indexed: {len(all_sections)}"
)