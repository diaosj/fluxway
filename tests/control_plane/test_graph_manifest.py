from typing import Any

import pytest
from pydantic import ValidationError
from ray import cloudpickle

import fluxway as fw
from fluxway.common.decorators import operator, spmd


_REP = {"type": "static", "axis": []}


@spmd(mesh_axis_names=("data",), mesh_axis_shapes=(2,))
class _ManifestSPMD:
    @operator(name="produce", output_sharding={"value": _REP})
    def produce(self) -> dict[str, Any]:
        return {"value": 1}

    @operator(
        name="consume",
        input_sharding={"item": _REP},
        output_sharding={"result": _REP},
        static_argument=["scale"],
    )
    def consume(self, item: Any, scale: int = 1) -> dict[str, Any]:
        return {"result": item * scale}

    @operator(
        name="loop",
        input_sharding={"token": _REP},
        output_sharding={"next": _REP},
    )
    def loop(self, token: Any) -> dict[str, Any]:
        return {"next": token}


def _build_graph():
    builder = fw.Builder()
    builder.add_node("source", _ManifestSPMD, description="source node")
    builder.add_node("target", _ManifestSPMD, description="target node")
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
    )
    return builder.build("manifest-test")


def test_graph_manifest_is_complete_deterministic_and_frozen():
    first = fw.graph_manifest(_build_graph())
    second = fw.graph_manifest(_build_graph())

    assert first.manifest_id == second.manifest_id
    assert first.graph_name == "manifest-test"
    assert [spmd.spmd_id for spmd in first.spmds] == ["source", "target"]

    source = first.spmds[0]
    assert source.class_id == f"{_ManifestSPMD.__module__}.{_ManifestSPMD.__qualname__}"
    assert source.description == "source node"
    assert source.mesh == (("data", 2),)

    produce = next(op for op in source.operators if op.op_id == "produce")
    assert produce.spmd_id == "source"
    assert produce.function_id.endswith("._ManifestSPMD.produce")
    assert produce.outputs.data == ("value",)
    assert len(produce.input_signature_id) == 64
    assert len(produce.output_signature_id) == 64

    transfer = first.transfers[0]
    assert transfer.source.spmd_id == "source"
    assert transfer.source.op_id == "produce"
    assert transfer.source.name == "value"
    assert transfer.target.spmd_id == "target"
    assert transfer.target.op_id == "consume"
    assert transfer.target.name == "item"
    assert transfer.tuuid == 1

    with pytest.raises(Exception):
        first.graph_name = "changed"  # type: ignore


def test_graph_manifest_survives_json_round_trip():
    manifest = fw.graph_manifest(_build_graph())
    restored = type(manifest).model_validate_json(manifest.model_dump_json())

    assert restored == manifest
    assert restored.manifest_id == manifest.manifest_id


def test_operator_facts_are_shared_with_runtime_metadata():
    graph = _build_graph()
    assert graph.manifest is not None
    manifest_operator = next(
        operator
        for spmd in graph.manifest.spmds
        if spmd.spmd_id == "source"
        for operator in spmd.operators
        if operator.op_id == "produce"
    )

    runtime_facts = (
        graph.nodes["source"].spmd_config.operator_metas["produce"].graph_facts
    )
    assert runtime_facts is not None
    assert runtime_facts is graph.manifest.runtime_operator_facts("source", "produce")
    assert runtime_facts.operator is manifest_operator


def test_graph_manifest_survives_ray_serialization():
    graph = _build_graph()
    assert graph.manifest is not None
    restored = cloudpickle.loads(cloudpickle.dumps(graph))  # pyright: ignore[reportPrivateImportUsage]
    assert restored.manifest is not None

    assert restored.manifest == graph.manifest
    assert restored.manifest.manifest_id == graph.manifest.manifest_id
    restored_operator = restored.manifest.spmds[0].operators[0]
    restored_runtime_facts = (
        restored.nodes["source"].spmd_config.operator_metas["produce"].graph_facts
    )
    assert restored_runtime_facts is not None
    assert restored_runtime_facts.operator == restored_operator


def test_rebuilding_builder_does_not_mutate_existing_graph():
    builder = fw.Builder()
    builder.add_node("source", _ManifestSPMD)
    builder.add_node("target", _ManifestSPMD)
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
    )
    first = builder.build("first")
    first_manifest = fw.graph_manifest(first)
    second = builder.build("second")
    second_manifest = fw.graph_manifest(second)

    assert fw.graph_manifest(first) is first_manifest
    assert (
        first.nodes["source"].spmd_config.graph_manifest_id
        == first_manifest.manifest_id
    )
    assert (
        second.nodes["source"].spmd_config.graph_manifest_id
        == second_manifest.manifest_id
    )
    assert first_manifest.manifest_id != second_manifest.manifest_id


def test_graph_runtime_manifest_reference_is_frozen():
    graph = _build_graph()
    with pytest.raises(ValidationError, match="Field is frozen"):
        graph.nodes["source"].spmd_config.graph_manifest_id = "changed"


def test_graph_manifest_rejects_replaced_runtime_operator_facts():
    graph = _build_graph()
    other = fw.Builder()
    other.add_node("source", _ManifestSPMD)
    other.add_node("target", _ManifestSPMD)
    other.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
    )
    other_graph = other.build("other")
    assert other_graph.manifest is not None
    config = graph.nodes["source"].spmd_config
    meta = config.operator_metas["produce"]
    config.operator_metas["produce"] = meta.model_copy(
        update={
            "graph_facts": other_graph.manifest.runtime_operator_facts(
                "source", "produce"
            )
        }
    )

    with pytest.raises(ValueError, match="does not match its manifest facts"):
        fw.graph_manifest(graph)


def test_graph_manifest_rejects_tampered_identity_and_signatures():
    manifest = fw.graph_manifest(_build_graph())
    tampered_manifest = manifest.model_dump(mode="json")
    tampered_manifest["graph_name"] = "different"
    with pytest.raises(ValidationError, match="manifest_id"):
        type(manifest).model_validate(tampered_manifest)

    tampered_operator = manifest.model_dump(mode="json")
    tampered_operator["spmds"][0]["operators"][0]["input_signature_id"] = "bad"
    with pytest.raises(ValidationError, match="input_signature_id"):
        type(manifest).model_validate(tampered_operator)


def test_graph_manifest_preserves_transfer_attributes():
    def condition() -> bool:
        return True

    builder = fw.Builder()
    builder.add_node("source", _ManifestSPMD)
    builder.add_node("target", _ManifestSPMD)
    symmetric_to = ("target", "consume", "source", "produce")
    builder.add_func_edge(
        "source",
        "produce",
        "target",
        "consume",
        tensor_map={"value": "item"},
        condition_fn=condition,
        symmetric_to=symmetric_to,
    )
    builder.add_func_edge(
        "source",
        "loop",
        "source",
        "loop",
        tensor_map={"next": "token"},
        local_only=True,
    )

    manifest = fw.graph_manifest(builder.build("transfer-attributes"))
    remote, local = manifest.transfers
    assert remote.condition_id == f"{condition.__module__}.{condition.__qualname__}"
    assert remote.symmetric_to == symmetric_to
    assert remote.local_only is False
    assert local.condition_id is None
    assert local.symmetric_to is None
    assert local.local_only is True
