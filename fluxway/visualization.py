from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
import shutil
import tempfile
from urllib.parse import quote

from fluxway.common.types import ComputationGraphManifest, Graph
from fluxway.common.types.manifest import (
    OperatorFacts,
    OperatorPort,
    SPMDFacts,
    TransferFacts,
)


def graph_manifest(graph: Graph) -> ComputationGraphManifest:
    if graph.manifest is None:
        raise ValueError(
            "Graph has no computation manifest. Build it with fluxway.Builder."
        )
    manifest = graph.manifest
    manifest_spmd_ids = {spmd.spmd_id for spmd in manifest.spmds}
    if set(graph.nodes) != manifest_spmd_ids:
        raise ValueError("Graph nodes do not match computation manifest SPMDs.")
    for spmd in manifest.spmds:
        config = graph.nodes[spmd.spmd_id].spmd_config
        if config.graph_manifest_id != manifest.manifest_id:
            raise ValueError(
                f"Graph node {spmd.spmd_id!r} does not reference its manifest."
            )
        if config.graph_spmd_facts != spmd:
            raise ValueError(
                f"Graph node {spmd.spmd_id!r} does not reference its SPMD facts."
            )
        for operator in spmd.operators:
            meta = config.operator_metas.get(operator.op_id)
            expected = manifest.runtime_operator_facts(spmd.spmd_id, operator.op_id)
            if meta is None or meta.graph_facts != expected:
                raise ValueError(
                    f"Graph operator {spmd.spmd_id}.{operator.op_id} does not "
                    "match its manifest facts."
                )
    return manifest


def visualize_graph(
    graph: Graph,
    directory: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    manifest = graph_manifest(graph)
    destination = Path(directory)
    destination_exists = destination.exists()
    if destination_exists and not overwrite:
        raise FileExistsError(f"Visualization directory already exists: {destination}")
    model_explorer = _model_explorer_module()
    projection = _model_explorer_projection(manifest, model_explorer)
    if destination_exists:
        _validate_bundle(destination, model_explorer)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_json(temporary / "manifest.json", manifest.model_dump(mode="json"))
        _write_json(temporary / "model_explorer.json", projection)
        if destination_exists:
            backup = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination.name}.backup.", dir=destination.parent
                )
            )
            backup.rmdir()
            destination.replace(backup)
            try:
                temporary.replace(destination)
            except OSError:
                backup.replace(destination)
                raise
            shutil.rmtree(backup)
        else:
            temporary.replace(destination)
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return destination


def view_graph(
    bundle: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    open_browser: bool = True,
) -> None:
    directory = Path(bundle)
    model_explorer = _model_explorer_module()
    _validate_bundle(directory, model_explorer)
    model_path = (directory / "model_explorer.json").resolve()
    config = model_explorer.config()
    config.add_model_from_path(str(model_path), adapterId="builtin_json")
    model_explorer.visualize_from_config(
        config=config,
        host=host,
        port=port,
        no_open_in_browser=not open_browser,
    )


def _model_explorer_module():
    try:
        import model_explorer  # pyrefly: ignore [missing-import]
    except ImportError as exc:
        raise ImportError(
            "Graph visualization requires `ai-edge-model-explorer`. "
            "Install the `fluxway[visualization]` extra."
        ) from exc
    return model_explorer


def _read_bundle_manifest(directory: Path) -> ComputationGraphManifest:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Bundle is missing manifest.json: {directory}")
    return ComputationGraphManifest.model_validate_json(
        manifest_path.read_text(encoding="utf-8")
    )


def _validate_bundle(directory: Path, model_explorer) -> None:
    if not directory.is_dir():
        raise ValueError(f"Visualization bundle is not a directory: {directory}")
    expected_files = {"manifest.json", "model_explorer.json"}
    actual_files = {path.name for path in directory.iterdir()}
    if actual_files != expected_files:
        raise ValueError(
            f"Visualization bundle must contain exactly {sorted(expected_files)}, "
            f"got {sorted(actual_files)}."
        )
    manifest = _read_bundle_manifest(directory)
    projection_path = directory / "model_explorer.json"
    projection = json.loads(projection_path.read_text(encoding="utf-8"))
    if projection != _model_explorer_projection(manifest, model_explorer):
        raise ValueError("model_explorer.json does not match manifest.json.")


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _node_id(operator: OperatorFacts) -> str:
    encoded = f"{operator.spmd_id}\0{operator.op_id}".encode()
    return f"op_{sha256(encoded).hexdigest()}"


def _namespace(spmd_id: str) -> str:
    return quote(spmd_id, safe="")


def _key_values(graph_builder, values: dict[str, str]):
    return [
        graph_builder.KeyValue(key=key, value=value) for key, value in values.items()
    ]


def _operator_attributes(
    manifest_id: str, spmd: SPMDFacts, operator: OperatorFacts
) -> dict[str, str]:
    values = {
        "manifest_id": manifest_id,
        "spmd_id": operator.spmd_id,
        "spmd_class_id": spmd.class_id,
        "spmd_description": spmd.description,
        "spmd_mesh": json.dumps(spmd.mesh, ensure_ascii=False),
        "op_id": operator.op_id,
        "function_id": operator.function_id,
        "input_signature_id": operator.input_signature_id,
        "output_signature_id": operator.output_signature_id,
    }
    if spmd.resource is not None:
        values.update(
            {
                "spmd_num_cpus": str(spmd.resource.num_cpus),
                "spmd_num_gpus": str(spmd.resource.num_gpus),
                "spmd_memory_gb": str(spmd.resource.memory_gb),
                "spmd_resource_count": str(spmd.resource.count),
            }
        )
    return values


def _transfer_attributes(
    transfer: TransferFacts, *, endpoint: OperatorPort, peer: OperatorPort
) -> dict[str, str]:
    values = {
        "__tensor_tag": endpoint.name,
        "kind": endpoint.kind,
        "tuuid": str(transfer.tuuid),
        "peer_spmd_id": peer.spmd_id,
        "peer_op_id": peer.op_id,
        "peer_port": peer.name,
        "local_only": str(transfer.local_only).lower(),
    }
    if transfer.condition_id is not None:
        values["condition_id"] = transfer.condition_id
    if transfer.symmetric_to is not None:
        values["symmetric_to"] = ".".join(transfer.symmetric_to)
    return values


def _model_explorer_projection(
    manifest: ComputationGraphManifest, model_explorer
) -> dict:
    graph_builder = model_explorer.graph_builder
    operators = {
        (operator.spmd_id, operator.op_id): operator
        for spmd in manifest.spmds
        for operator in spmd.operators
    }
    spmds = {spmd.spmd_id: spmd for spmd in manifest.spmds}
    nodes = {
        key: graph_builder.GraphNode(
            id=_node_id(operator),
            label=operator.op_id,
            namespace=_namespace(operator.spmd_id),
            attrs=_key_values(
                graph_builder,
                _operator_attributes(
                    manifest.manifest_id,
                    spmds[operator.spmd_id],
                    operator,
                ),
            ),
        )
        for key, operator in operators.items()
    }
    connected_inputs: set[tuple[str, str, str]] = set()
    connected_outputs: set[tuple[str, str, str]] = set()
    for index, transfer in enumerate(manifest.transfers):
        source_key = (transfer.source.spmd_id, transfer.source.op_id)
        target_key = (transfer.target.spmd_id, transfer.target.op_id)
        source = nodes[source_key]
        target = nodes[target_key]
        output_id = f"transfer_{index}_output"
        input_id = f"transfer_{index}_input"
        target.incomingEdges.append(
            graph_builder.IncomingEdge(
                sourceNodeId=source.id,
                sourceNodeOutputId=output_id,
                targetNodeInputId=input_id,
            )
        )
        source.outputsMetadata.append(
            graph_builder.MetadataItem(
                id=output_id,
                attrs=_key_values(
                    graph_builder,
                    _transfer_attributes(
                        transfer,
                        endpoint=transfer.source,
                        peer=transfer.target,
                    ),
                ),
            )
        )
        target.inputsMetadata.append(
            graph_builder.MetadataItem(
                id=input_id,
                attrs=_key_values(
                    graph_builder,
                    _transfer_attributes(
                        transfer,
                        endpoint=transfer.target,
                        peer=transfer.source,
                    ),
                ),
            )
        )
        connected_outputs.add((*source_key, transfer.source.name))
        connected_inputs.add((*target_key, transfer.target.name))

    for key, operator in operators.items():
        node = nodes[key]
        for kind, names in (
            ("data", operator.inputs.data),
            ("control", operator.inputs.control),
            ("static", operator.inputs.static),
        ):
            for index, name in enumerate(names):
                if (*key, name) in connected_inputs:
                    continue
                node.inputsMetadata.append(
                    graph_builder.MetadataItem(
                        id=f"{kind}_input_{index}",
                        attrs=_key_values(
                            graph_builder,
                            {"__tensor_tag": name, "kind": kind},
                        ),
                    )
                )
        for kind, names in (
            ("data", operator.outputs.data),
            ("control", operator.outputs.control),
        ):
            for index, name in enumerate(names):
                if (*key, name) in connected_outputs:
                    continue
                node.outputsMetadata.append(
                    graph_builder.MetadataItem(
                        id=f"{kind}_output_{index}",
                        attrs=_key_values(
                            graph_builder,
                            {"__tensor_tag": name, "kind": kind},
                        ),
                    )
                )

    group_attributes = {"": {"manifest_id": manifest.manifest_id}}
    for spmd in manifest.spmds:
        values = {
            "spmd_id": spmd.spmd_id,
            "class_id": spmd.class_id,
            "description": spmd.description,
            "mesh": json.dumps(spmd.mesh, ensure_ascii=False),
        }
        if spmd.resource is not None:
            values.update(
                {
                    "num_cpus": str(spmd.resource.num_cpus),
                    "num_gpus": str(spmd.resource.num_gpus),
                    "memory_gb": str(spmd.resource.memory_gb),
                    "resource_count": str(spmd.resource.count),
                }
            )
        if spmd.priority is not None:
            values["priority"] = spmd.priority
        if spmd.placement_strategy is not None:
            values["placement_strategy"] = spmd.placement_strategy
        group_attributes[_namespace(spmd.spmd_id)] = values

    graph = graph_builder.Graph(
        id=manifest.manifest_id,
        nodes=list(nodes.values()),
        groupNodeAttributes=group_attributes,
    )
    collection = graph_builder.GraphCollection(
        label=manifest.graph_name,
        graphs=[graph],
    )
    return asdict(collection)
