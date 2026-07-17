import pickle

import numpy as np

from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.tensor_backend import use_backend
from fluxway.common.types.sharding import AlgebraicRouting, P, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.spmd.worker.data_plane import (
    SuccessorRouteContext,
    build_step_frame_groups,
    compile_send_plan,
)
from fluxway.transport.manifest import (
    ENTRY_EXPECTED_COUNT,
    ENTRY_ROUTE_CODE,
    FRAME_ENTRY_COUNT,
)
from tests.compiled_graph_fixtures import compiled_spmd_spec


def test_mirror_dataplane_build_uses_serialized_step_transfer_cache_receipt():
    with use_backend("hybrid"):
        mesh = DeviceMesh(axis_names=("x",), device_shape=(4,), device_ids=np.arange(4))
        dynamic = ShardingSpec.hash(P("x"))
        mirror = ShardingSpec.follow("lookup.unique_ids")
        upstream = OperatorSpec(
            name="upstream",
            output_arguments={"data": ("unique_ids",)},
            output_sharding={"unique_ids": dynamic},
        )
        lookup = OperatorSpec(
            name="lookup",
            input_arguments={"data": ("unique_ids",)},
            output_arguments={"data": ("sparse_features_dim_8",)},
            input_sharding={"unique_ids": dynamic},
            output_sharding={"sparse_features_dim_8": mirror},
        )
        dense = OperatorSpec(
            name="dense",
            input_arguments={"data": ("sparse_features_dim_8",)},
            input_sharding={"sparse_features_dim_8": dynamic},
        )
        compiled_graph = compile_graph(
            name="serialized-mirror-cache",
            spmds=(
                compiled_spmd_spec(
                    "Lookup",
                    mesh,
                    {"lookup": lookup, "upstream": upstream},
                ),
                compiled_spmd_spec("Dense", mesh, {"dense": dense}),
            ),
            dependencies=(
                DependencySpec(
                    source=OperatorKey("Lookup", "upstream"),
                    target=OperatorKey("Lookup", "lookup"),
                    transfers=(
                        TensorTransfer(
                            OperatorPort("unique_ids"),
                            OperatorPort("unique_ids"),
                        ),
                    ),
                    local_only=True,
                ),
                DependencySpec(
                    source=OperatorKey("Lookup", "lookup"),
                    target=OperatorKey("Dense", "dense"),
                    transfers=(
                        TensorTransfer(
                            OperatorPort("sparse_features_dim_8"),
                            OperatorPort("sparse_features_dim_8"),
                        ),
                    ),
                ),
            ),
        )
        assert isinstance(dynamic.routing, AlgebraicRouting)
        lookup_plan = compiled_graph.spmd("Lookup")
        incoming = next(
            dependency
            for dependency in lookup_plan.incoming
            if dependency.spec.target.op_id == "lookup"
        )
        outgoing = next(
            dependency
            for dependency in lookup_plan.outgoing
            if dependency.spec.source.op_id == "lookup"
        )
        input_transfer_code = incoming.transfers[0].transfer_code
        output_transfer_code = outgoing.transfers[0].transfer_code

        source_ref = TransportTensor(
            np.array([3, 7, 11], dtype=np.int64),
            origin_indices=ExplicitIndex(np.array([20, 10, 30], dtype=np.int64)),
            global_shape=(64,),
            receipt=SymmetryReceipt(4, np.array([2, 0, 3], dtype=np.int32)),
        )
        serialized_cache = pickle.loads(pickle.dumps({input_transfer_code: source_ref}))
        lookup_output = TransportTensor(
            np.ones((3, 8), dtype=np.float32),
            global_shape=(3, 8),
        )
        route_contexts = (
            SuccessorRouteContext(
                context_index=0,
                dependencies=(outgoing,),
                effective_rank=1,
            ),
        )
        compiled_plan = compile_send_plan(
            graph_view=lookup_plan,
            source_op_id="lookup",
            rank=1,
            successor_contexts=route_contexts,
            sender_spmd_code=9,
            interleave_unit_size=None,
        )

        frame_groups = build_step_frame_groups(
            compiled_plan=compiled_plan,
            flat_results=[lookup_output],
            step_transfer_cache=serialized_cache,
            exec_step=0,
        )

        assert frame_groups
        assert {frame_arg[0] for _, frame_arg in frame_groups} == set(range(mesh.size))
        for _, frame_arg in frame_groups:
            target_rank, header, payloads = frame_arg
            assert 0 <= target_rank < mesh.size
            assert int(header.frame_i32[FRAME_ENTRY_COUNT]) == 1
            assert int(header.entry_i32[0, ENTRY_ROUTE_CODE]) == output_transfer_code
            assert int(header.entry_i32[0, ENTRY_EXPECTED_COUNT]) == mesh.size
            assert payloads
