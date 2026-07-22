from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = REPO_ROOT / "fluxway/example/recommendation"


def _class(path: str, name: str) -> ast.ClassDef:
    module = ast.parse((EXAMPLE_ROOT / path).read_text())
    return next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    )


def _parameters(class_node: ast.ClassDef, method_name: str) -> tuple[str, ...]:
    method = next(
        node
        for node in class_node.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == method_name
    )
    return tuple(argument.arg for argument in method.args.args)


def test_recommendation_reader_and_dense_use_context_checkpoint_contract() -> None:
    reader = _class("reader.py", "RecommendationReader")
    dense = _class("dense_spmd.py", "DenseSPMD")

    assert _parameters(reader, "save_state") == ("self", "context")
    assert _parameters(reader, "restore_state") == ("self", "state", "context")
    assert _parameters(dense, "save_state") == ("self", "context")
    assert _parameters(dense, "checkpoint_restore_layout") == ("self", "context")
    assert _parameters(dense, "restore_state") == ("self", "state", "context")


def test_recommendation_sparse_uses_distributed_prepare_commit_contract() -> None:
    sparse = _class("sparse_spmd.py", "SparseEmbeddingSPMD")

    assert _parameters(sparse, "prepare_checkpoint_artifacts") == (
        "self",
        "context",
    )
    assert _parameters(sparse, "save_state") == ("self", "context")
    assert _parameters(sparse, "checkpoint_metadata") == (
        "self",
        "saved_state",
        "context",
    )
    assert _parameters(sparse, "commit_checkpoint_artifacts") == (
        "self",
        "gathered_metadata",
        "context",
    )
    assert _parameters(sparse, "abort_checkpoint_artifacts") == (
        "self",
        "context",
    )
    assert _parameters(sparse, "restore_state") == ("self", "state", "context")
