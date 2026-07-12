import ast
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_DIR = REPO_ROOT / "fluxway" / "example" / "recommendation"


def _load_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _decorator_has_resource_kw(decorator: ast.AST, decorator_names: set[str]) -> bool:
    if not isinstance(decorator, ast.Call):
        return False
    if not isinstance(decorator.func, ast.Name):
        return False
    if decorator.func.id not in decorator_names:
        return False
    return any(keyword.arg == "resource" for keyword in decorator.keywords)


def test_recommendation_spmd_classes_declare_resources():
    files = [
        EXAMPLE_DIR / "input_spmd.py",
        EXAMPLE_DIR / "sparse_spmd.py",
        EXAMPLE_DIR / "dense_spmd.py",
        EXAMPLE_DIR / "reader.py",
    ]

    for path in files:
        module = _load_module(path)
        classes = [node for node in module.body if isinstance(node, ast.ClassDef)]
        assert classes, f"expected at least one class in {path}"

        for cls in classes:
            assert any(
                _decorator_has_resource_kw(decorator, {"spmd", "spmd_source"})
                for decorator in cls.decorator_list
            ), f"{path.name}:{cls.name} should declare explicit resource requirements"


def test_recommendation_train_uses_fluxway_launch():
    module = _load_module(EXAMPLE_DIR / "train.py")
    launch_calls = []

    for node in ast.walk(module):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        # Accept both `launch(...)` and `fluxway.launch(...)` call styles
        is_bare_launch = isinstance(func, ast.Name) and func.id == "launch"
        is_attr_launch = (
            isinstance(func, ast.Attribute)
            and func.attr == "launch"
            and isinstance(func.value, ast.Name)
            and func.value.id == "fluxway"
        )
        if is_bare_launch or is_attr_launch:
            launch_calls.append(node)

    assert launch_calls, (
        "expected train.py to call fluxway.launch() for zero-config Ray initialization"
    )


def _is_controller_resource_summary_remote_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != "remote":
        return False
    owner = func.value
    if not isinstance(owner, ast.Attribute):
        return False
    if owner.attr != "get_resource_allocation_summary":
        return False
    return isinstance(owner.value, ast.Name) and owner.value.id == "controller"


def test_recommendation_train_prints_controller_resource_summary():
    module = _load_module(EXAMPLE_DIR / "train.py")

    assert any(
        _is_controller_resource_summary_remote_call(node) for node in ast.walk(module)
    ), "train.py should query controller.get_resource_allocation_summary.remote()"


def test_sparse_spmd_reserves_embedding_store_capacity():
    module = _load_module(EXAMPLE_DIR / "sparse_spmd.py")
    init_fn = next(
        node
        for node in ast.walk(module)
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )

    reserve_calls = [
        node
        for node in ast.walk(init_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reserve_per_shard"
    ]
    shard_count_calls = [
        node
        for node in ast.walk(init_fn)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "internal_shard_count"
    ]

    assert reserve_calls, "SparseEmbeddingSPMD should reserve sparse store capacity"
    assert shard_count_calls, (
        "SparseEmbeddingSPMD should size reserve counts from internal shard count"
    )


def test_sparse_spmd_balanced_counts_rejects_invalid_capacity():
    module = _load_module(EXAMPLE_DIR / "sparse_spmd.py")
    fn = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_balanced_counts"
    )

    raises = [node for node in ast.walk(fn) if isinstance(node, ast.Raise)]
    assert len(raises) == 2, "_balanced_counts should reject invalid inputs directly"


def test_dense_spmd_marks_jax_distributed_worker_process():
    module = _load_module(EXAMPLE_DIR / "dense_spmd.py")
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "DenseSPMD"
    )

    decorator_names = {
        decorator.id
        for decorator in cls.decorator_list
        if isinstance(decorator, ast.Name)
    }
    assert "requires_worker_process" in decorator_names


def test_sparse_spmd_marks_embedding_worker_process():
    module = _load_module(EXAMPLE_DIR / "sparse_spmd.py")
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "SparseEmbeddingSPMD"
    )

    decorator_names = {
        decorator.id
        for decorator in cls.decorator_list
        if isinstance(decorator, ast.Name)
    }

    assert "requires_worker_process" in decorator_names


def test_input_spmd_marks_preprocessing_worker_process():
    module = _load_module(EXAMPLE_DIR / "input_spmd.py")
    cls = next(
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == "InputSPMD"
    )

    decorator_names = {
        decorator.id
        for decorator in cls.decorator_list
        if isinstance(decorator, ast.Name)
    }

    assert "requires_worker_process" in decorator_names


def test_recommendation_visualization_export_is_explicit(monkeypatch, tmp_path):
    from fluxway.example.recommendation import train

    calls = []
    monkeypatch.setattr(
        "fluxway.visualize_graph",
        lambda graph, directory, *, overwrite: calls.append(
            (graph, directory, overwrite)
        )
        or directory,
    )
    graph = object()
    train._export_graph_visualization(
        graph,
        SimpleNamespace(
            graph_visualization_dir=None,
            overwrite_graph_visualization=False,
        ),
    )
    assert calls == []

    destination = tmp_path / "bundle"
    train._export_graph_visualization(
        graph,
        SimpleNamespace(
            graph_visualization_dir=destination,
            overwrite_graph_visualization=True,
        ),
    )
    assert calls == [(graph, destination, True)]
