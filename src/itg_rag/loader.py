import hashlib
import logging
import re
import unicodedata
from pathlib import Path

from itg_rag.errors import RAGError
from itg_rag.schemas import Document

logger = logging.getLogger(__name__)


def clean_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text).replace("\ufeff", "")
    return "\n".join(
        line
        for raw in text.splitlines()
        if (line := re.sub(r"\s+", " ", raw).strip())
    )


def load_documents(root: Path) -> list[Document]:
    if not root.is_dir():
        raise RAGError(f"Knowledge directory not found: {root}")
    documents = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in {".txt", ".md"} or not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(
            root.resolve()
        ):
            raise RAGError(f"Symlink outside knowledge directory: {path.name}")
        try:
            text = clean_text(path.read_text(encoding="utf-8-sig"))
        except (OSError, UnicodeError):
            raise RAGError(
                f"Cannot read UTF-8 document: {path.name}"
            ) from None
        if not text:
            logger.warning("Skipping empty document: %s", path.name)
            continue
        name = path.relative_to(root).as_posix()
        documents.append(
            Document(
                document_id=hashlib.sha256(name.encode()).hexdigest()[:24],
                source_name=name,
                text=text,
            )
        )
    if not documents:
        raise RAGError("No non-empty .txt or .md documents found")
    logger.info("Loaded %d documents", len(documents))
    return documents
