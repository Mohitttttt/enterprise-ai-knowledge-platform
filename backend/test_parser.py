from pathlib import Path

from app.services.document_parser import (
    extract_text,
    DocumentParsingError,
)


UPLOAD_DIR = Path("uploads")


def main():
    files = [
        file
        for file in UPLOAD_DIR.iterdir()
        if file.suffix.lower() in {".pdf", ".docx", ".txt"}
    ]

    if not files:
        print("No documents found in uploads folder.")
        return

    for file_path in files:

        print("\n" + "=" * 60)
        print(f"Testing: {file_path.name}")
        print("=" * 60)

        try:
            extracted_text = extract_text(file_path)

            print("\nExtraction successful!")
            print(f"Characters extracted: {len(extracted_text)}")

            print("\nFirst 1000 characters:")
            print(extracted_text[:1000])

        except DocumentParsingError as error:
            print(f"\nExtraction failed: {error}")


if __name__ == "__main__":
    main()