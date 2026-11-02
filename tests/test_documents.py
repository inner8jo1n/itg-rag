from pathlib import Path

import pytest

from itg_rag.chunking import chunk_documents
from itg_rag.errors import RAGError
from itg_rag.loader import clean_text, load_documents
from itg_rag.schemas import Document


def test_normalization() -> None:
    assert clean_text("\ufeff A\t  B \r\n\n\n C \u00a0 D ") == "A B\nC D"


def test_loader_nested_ids_and_empties(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "a.txt").write_text("first", encoding="utf-8")
    (tmp_path / "nested/a.txt").write_text("second", encoding="utf-8")
    (tmp_path / "empty.md").write_text(" \n ", encoding="utf-8")
    (tmp_path / "skip.csv").write_text("skip", encoding="utf-8")
    docs = load_documents(tmp_path)
    assert [d.source_name for d in docs] == ["a.txt", "nested/a.txt"]
    assert len({d.document_id for d in docs}) == 2
    (tmp_path / "a.txt").write_text("edited", encoding="utf-8")
    assert load_documents(tmp_path)[0].document_id == docs[0].document_id


def test_loader_invalid_utf8(tmp_path: Path) -> None:
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe")
    with pytest.raises(RAGError, match="UTF-8"):
        load_documents(tmp_path)


def test_loader_outside_symlink(tmp_path: Path) -> None:
    (tmp_path / "external.txt").symlink_to("/etc/hosts")
    with pytest.raises(RAGError, match="Symlink"):
        load_documents(tmp_path)


def test_empty_corpus(tmp_path: Path) -> None:
    with pytest.raises(RAGError, match="No non-empty"):
        load_documents(tmp_path)


@pytest.mark.parametrize("text", ["x" * 2000, "слово " * 500, "a\nb\n" * 500])
@pytest.mark.parametrize("overlap", [0, 25, 99])
def test_chunking_preserves_text_and_offsets(text: str, overlap: int) -> None:
    doc = Document(document_id="id", source_name="a.txt", text=text.strip())
    chunks = chunk_documents([doc], size=100, overlap=overlap)
    assert len(chunks) > 1
    covered = set()
    for chunk in chunks:
        assert chunk.text == doc.text[chunk.start : chunk.end]
        assert 0 < len(chunk.text) <= 100
        covered.update(range(chunk.start, chunk.end))
    assert all(
        i in covered for i, char in enumerate(doc.text) if not char.isspace()
    )
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)
    assert chunks == chunk_documents([doc], size=100, overlap=overlap)


@pytest.mark.parametrize("size, overlap", [(0, 0), (100, 100), (10, -1)])
def test_invalid_chunking(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        chunk_documents([], size, overlap)


def test_demo_corpus() -> None:
    docs = load_documents(Path("knowledge_base"))
    assert len(docs) >= 5
    assert len(chunk_documents(docs)) > len(docs)
