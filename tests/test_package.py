from importlib import import_module
from importlib.metadata import distribution


def test_package_is_installed() -> None:
    package = import_module("itg_rag")
    installed = distribution("itg-rag")

    assert package.__file__ is not None
    assert installed.metadata["Name"] == "itg-rag"
