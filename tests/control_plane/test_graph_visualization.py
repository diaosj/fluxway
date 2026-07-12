import json
import builtins
from pathlib import Path
from typing import Any

import pytest

import fluxway as fw
from fluxway.common.decorators import operator, spmd


_REP = {"type": "static", "axis": []}


@spmd(mesh_axis_names=("data",), mesh_axis_shapes=(1,))
class _VisualizationSPMD:
    @operator(name="produce", output_sharding={"left": _REP, "right": _REP})
    def produce(self) -> dict[str, Any]:
        return {"left": 1, "right": 2}

    @operator(name="consume", input_sharding={"first": _REP, "second": _REP})
    def consume(self, first: Any, second: Any) -> None:
        del first, second


def _build_graph():
    builder = fw.Builder()
    builder.add_node("source", _VisualizationSPMD, description="producer")
    builder.add_node("target", _VisualizationSPMD, description="consumer")
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"left": "first", "right": "second"},
    )
    return builder.build("visualization-test")


def test_visualize_graph_writes_manifest_and_model_explorer_projection(tmp_path):
    output = fw.visualize_graph(_build_graph(), tmp_path / "bundle")

    assert output == tmp_path / "bundle"
    assert sorted(path.name for path in output.iterdir()) == [
        "manifest.json",
        "model_explorer.json",
    ]

    manifest = json.loads((output / "manifest.json").read_text())
    projection = json.loads((output / "model_explorer.json").read_text())
    graph = projection["graphs"][0]

    assert graph["id"] == manifest["manifest_id"]
    assert graph["groupNodeAttributes"][""]["manifest_id"] == manifest["manifest_id"]
    assert len(graph["nodes"]) == 2

    target = next(node for node in graph["nodes"] if node["label"] == "consume")
    attrs = {item["key"]: item["value"] for item in target["attrs"]}
    assert attrs["spmd_id"] == "target"
    assert attrs["op_id"] == "consume"
    assert attrs["function_id"].endswith("._VisualizationSPMD.consume")
    assert (
        attrs["input_signature_id"]
        == manifest["spmds"][1]["operators"][0]["input_signature_id"]
    )
    assert len(target["incomingEdges"]) == 2
    input_metadata = {
        item["id"]: {attr["key"]: attr["value"] for attr in item["attrs"]}
        for item in target["inputsMetadata"]
    }
    assert {metadata["tuuid"] for metadata in input_metadata.values()} == {"1", "2"}
    assert {metadata["__tensor_tag"] for metadata in input_metadata.values()} == {
        "first",
        "second",
    }


def test_visualize_graph_preserves_special_characters_and_empty_description(tmp_path):
    builder = fw.Builder()
    builder.add_node('source/<&"', _VisualizationSPMD, description="")
    builder.add_node("target", _VisualizationSPMD)
    builder.add_func_edge(
        'source/<&"',
        "produce",
        "target",
        "consume",
        tensor_map={"left": "first", "right": "second"},
    )
    graph = builder.build('graph/<&"')

    output = fw.visualize_graph(graph, tmp_path / "bundle")
    manifest = json.loads((output / "manifest.json").read_text())
    projection = json.loads((output / "model_explorer.json").read_text())

    assert manifest["graph_name"] == 'graph/<&"'
    assert manifest["spmds"][0]["spmd_id"] == 'source/<&"'
    assert manifest["spmds"][0]["description"] == ""
    assert projection["label"] == 'graph/<&"'
    source = next(
        node
        for node in projection["graphs"][0]["nodes"]
        if {item["key"]: item["value"] for item in node["attrs"]}["spmd_id"]
        == 'source/<&"'
    )
    assert source["namespace"] == "source%2F%3C%26%22"


def test_visualize_graph_protects_existing_directories(tmp_path):
    graph = _build_graph()
    destination = tmp_path / "bundle"
    destination.mkdir()

    with pytest.raises(FileExistsError):
        fw.visualize_graph(graph, destination)
    with pytest.raises(ValueError, match="must contain exactly"):
        fw.visualize_graph(graph, destination, overwrite=True)

    destination.rmdir()
    fw.visualize_graph(graph, destination)
    fw.visualize_graph(graph, destination, overwrite=True)

    projection_path = destination / "model_explorer.json"
    projection = json.loads(projection_path.read_text())
    projection["graphs"][0]["id"] = "tampered"
    projection_path.write_text(json.dumps(projection))
    with pytest.raises(ValueError, match="does not match manifest"):
        fw.visualize_graph(graph, destination, overwrite=True)


def test_visualize_graph_restores_bundle_when_directory_swap_fails(
    monkeypatch, tmp_path
):
    graph = _build_graph()
    destination = fw.visualize_graph(graph, tmp_path / "bundle")
    original = {path.name: path.read_bytes() for path in destination.iterdir()}
    real_replace = Path.replace

    def fail_new_bundle_replace(path, target):
        if (
            path.parent == destination.parent
            and path.name.startswith(f".{destination.name}.")
            and not path.name.startswith(f".{destination.name}.backup.")
            and Path(target) == destination
        ):
            raise OSError("proof directory swap failure")
        return real_replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_new_bundle_replace)
    with pytest.raises(OSError, match="proof directory swap failure"):
        fw.visualize_graph(graph, destination, overwrite=True)

    assert {path.name: path.read_bytes() for path in destination.iterdir()} == original


def test_view_graph_uses_model_explorer_config(monkeypatch, tmp_path):
    bundle = fw.visualize_graph(_build_graph(), tmp_path / "bundle")
    calls = {}

    import model_explorer  # pyrefly: ignore [missing-import]

    monkeypatch.setattr(
        model_explorer,
        "visualize_from_config",
        lambda **kwargs: calls.update(kwargs),
    )

    fw.view_graph(bundle, host="127.0.0.2", port=8090, open_browser=False)

    assert calls["host"] == "127.0.0.2"
    assert calls["port"] == 8090
    assert calls["no_open_in_browser"] is True
    assert calls["config"].model_sources == [
        {
            "url": str((bundle / "model_explorer.json").resolve()),
            "adapterId": "builtin_json",
        }
    ]


def test_visualization_dependency_and_service_failures_are_explicit(
    monkeypatch, tmp_path
):
    graph = _build_graph()
    bundle = fw.visualize_graph(graph, tmp_path / "bundle")
    real_import = builtins.__import__

    def reject_model_explorer(name, *args, **kwargs):
        if name == "model_explorer":
            raise ImportError("proof missing dependency")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_model_explorer)
    with pytest.raises(FileExistsError):
        fw.visualize_graph(graph, bundle)
    with pytest.raises(ImportError, match=r"fluxway\[visualization\]"):
        fw.visualize_graph(graph, tmp_path / "missing-dependency")
    with pytest.raises(ImportError, match=r"fluxway\[visualization\]"):
        fw.visualize_graph(graph, bundle, overwrite=True)

    monkeypatch.setattr(builtins, "__import__", real_import)

    import model_explorer  # pyrefly: ignore [missing-import]

    def fail_service(**kwargs):
        raise RuntimeError("service failed")

    monkeypatch.setattr(model_explorer, "visualize_from_config", fail_service)
    with pytest.raises(RuntimeError, match="service failed"):
        fw.view_graph(bundle, open_browser=False)


def test_graph_manifest_rejects_direct_graph_construction():
    with pytest.raises(ValueError, match="no computation manifest"):
        fw.graph_manifest(fw.Graph(name="legacy"))
