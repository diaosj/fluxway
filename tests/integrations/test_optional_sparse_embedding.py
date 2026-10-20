from __future__ import annotations

import importlib
import sys
from contextlib import contextmanager
from collections.abc import Iterator


@contextmanager
def _without_sparse_embedding() -> Iterator[None]:
    prefixes = ("sparse_embedding", "fluxway.integrations.sparse_embedding")
    saved = {
        name: module
        for name, module in sys.modules.items()
        if any(name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes)
    }
    for name in saved:
        sys.modules.pop(name, None)
    try:
        yield
    finally:
        for name in tuple(sys.modules):
            if any(
                name == prefix or name.startswith(f"{prefix}.") for prefix in prefixes
            ):
                sys.modules.pop(name, None)
        sys.modules.update(saved)


class _SparseEmbeddingBlocker:
    def find_spec(self, fullname: str, path: object = None, target: object = None):
        if fullname == "sparse_embedding" or fullname.startswith("sparse_embedding."):
            raise ModuleNotFoundError(
                "sparse_embedding is required by this integration",
                name="sparse_embedding",
            )
        return None


def test_core_and_integration_namespace_do_not_import_optional_dependency():
    with _without_sparse_embedding():
        sys.meta_path.insert(0, _SparseEmbeddingBlocker())
        try:
            importlib.import_module("fluxway")
            importlib.import_module("fluxway.integrations")
        finally:
            sys.meta_path.pop(0)

        assert "sparse_embedding" not in sys.modules


def test_explicit_sparse_embedding_integration_import_requires_dependency():
    with _without_sparse_embedding():
        sys.meta_path.insert(0, _SparseEmbeddingBlocker())
        try:
            try:
                importlib.import_module("fluxway.integrations.sparse_embedding")
            except ModuleNotFoundError as error:
                assert error.name == "sparse_embedding"
            else:
                raise AssertionError("missing optional dependency was not reported")
        finally:
            sys.meta_path.pop(0)
