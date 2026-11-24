from pathlib import Path

from docx import Document
from pypdf import PdfReader


class DocumentParsingError(Exception):
    """Raised when a document cannot be parsed."""


def extract_text(file_path: str | Path) -> str:
    """
    Extract text from PDF, DOCX or TXT documents.

    Args:
        file_path: Path to the uploaded document.

    Returns:
        Extracted and cleaned text.
    """

    path = Path(file_path)
    extension = path.suffix.lower()

    if not path.exists():
        raise DocumentParsingError("Document file does not exist.")

    try:
        if extension == ".pdf":
            text = _extract_pdf(path)

        elif extension == ".docx":
            text = _extract_docx(path)

        elif extension == ".txt":
            text = _extract_txt(path)

        else:
            raise DocumentParsingError(
                f"Unsupported document format: {extension}"
            )

        cleaned_text = _clean_text(text)

        if not cleaned_text.strip():
            raise DocumentParsingError(
                "No readable text found. The document may be empty "
                "or contain scanned images requiring OCR."
            )

        return cleaned_text

    except DocumentParsingError:
        raise

    except Exception as exc:
        raise DocumentParsingError(
            f"Failed to extract document text: {str(exc)}"
        ) from exc


def _extract_pdf(path: Path) -> str:
    """Extract text from PDF pages."""

    reader = PdfReader(str(path))

    extracted_pages = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        extracted_pages.append(page_text)

    return "\n".join(extracted_pages)


def _extract_docx(path: Path) -> str:
    """Extract text from Word paragraphs and tables."""

    document = Document(str(path))

    extracted_content = []

    # Extract paragraphs
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            extracted_content.append(paragraph.text)

    # Extract tables
    for table in document.tables:
        for row in table.rows:
            row_text = [
                cell.text.strip()
                for cell in row.cells
                if cell.text.strip()
            ]

            if row_text:
                extracted_content.append(" | ".join(row_text))

    return "\n".join(extracted_content)


def _extract_txt(path: Path) -> str:
    """Extract text from plain text files."""

    return path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    )


def _clean_text(text: str) -> str:
    """Normalize extracted text."""

    lines = [
        " ".join(line.split())
        for line in text.splitlines()
    ]

    return "\n".join(
        line for line in lines if line
    )