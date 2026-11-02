import uuid

from itg_rag.schemas import Chunk, Document


def chunk_documents(
    documents: list[Document], size: int = 900, overlap: int = 150
) -> list[Chunk]:
    if size < 1 or not 0 <= overlap < size:
        raise ValueError("Require size > 0 and 0 <= overlap < size")
    chunks = []
    for document in documents:
        start = 0
        index = 0
        while start < len(document.text):
            end = min(start + size, len(document.text))
            if end < len(document.text):
                boundary = max(
                    document.text.rfind("\n", start, end),
                    document.text.rfind(" ", start, end),
                )
                if boundary > start + max(overlap, size // 2):
                    end = boundary
            text = document.text[start:end].strip()
            actual_start = (
                start
                + len(document.text[start:end])
                - len(document.text[start:end].lstrip())
            )
            if text:
                chunks.append(
                    Chunk(
                        document_id=document.document_id,
                        source_name=document.source_name,
                        chunk_id=str(
                            uuid.uuid5(
                                uuid.NAMESPACE_URL,
                                f"itg-rag:{document.document_id}:{index}",
                            )
                        ),
                        text=text,
                        start=actual_start,
                        end=actual_start + len(text),
                    )
                )
                index += 1
            if end == len(document.text):
                break
            start = end - overlap
    return chunks
