from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

from werkzeug.datastructures import FileStorage

from PyPDF2 import PdfReader


ALLOWED_EXTENSIONS = {".txt", ".pdf"}


def allowed_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def extract_text(file_storage: FileStorage) -> str:
    """Extracts text content from supported resume file types."""
    suffix = Path(file_storage.filename or "").suffix.lower()
    if suffix == ".txt":
        file_storage.stream.seek(0)
        data = file_storage.stream.read()
        return data.decode("utf-8", errors="ignore")
    if suffix == ".pdf":
        file_storage.stream.seek(0)
        return _extract_pdf_text(file_storage.stream)
    raise ValueError("Unsupported file type. Please upload a PDF or TXT file.")


def _extract_pdf_text(file_stream: BinaryIO) -> str:
    reader = PdfReader(file_stream)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)
