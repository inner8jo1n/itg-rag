from pathlib import Path

import pytest
from pydantic import ValidationError

from itg_rag.config import Settings
from itg_rag.errors import RAGError
from itg_rag.main import emit, main


@pytest.mark.parametrize(
    "values",
    [
        {"chunk_size": 100, "chunk_overlap": 100},
        {"top_k": 0},
        {"max_distance": 3},
        {"collection": "lowercase"},
    ],
)
def test_invalid_settings(values: dict) -> None:
    with pytest.raises(ValidationError):
        Settings.model_validate(values)


def test_missing_keys_and_redacted_repr() -> None:
    with pytest.raises(RAGError, match="API keys"):
        Settings().require_key()
    settings = Settings(api_key="private-key")
    assert "private-key" not in repr(settings)
    assert settings.require_key(embedding=True) == "private-key"


def test_index_signature_catches_changed_embedding() -> None:
    settings = Settings()
    changed = Settings(embedding_model="different")
    assert settings.index_signature() != changed.index_signature()
    assert (
        settings.index_signature() == Settings(api_key="new").index_signature()
    )


def test_cli_load_needs_no_keys(tmp_path: Path, monkeypatch, capsys) -> None:
    (tmp_path / "sample.txt").write_text("Hello", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KNOWLEDGE_DIR", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["itg-rag", "load"])
    assert main() == 0
    assert "sample.txt" in capsys.readouterr().out


def test_cli_bad_config_no_traceback(monkeypatch, capsys) -> None:
    monkeypatch.setenv("TOP_K", "0")
    monkeypatch.setattr("sys.argv", ["itg-rag", "load"])
    assert main() == 1
    assert "top_k" in capsys.readouterr().err


def test_output_nested_directory(tmp_path: Path) -> None:
    path = tmp_path / "results" / "answer.json"
    emit({"message": "Привет"}, path)
    assert "Привет" in path.read_text(encoding="utf-8")
