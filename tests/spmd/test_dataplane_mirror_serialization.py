import pickle
from types import SimpleNamespace

import numpy as np

from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import use_backend
from fluxway.common.types.sharding import AlgebraicRouting, P, ShardingSpec
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.spmd.worker.data_plane import (
    SuccessorRouteContext,
    build_step_frame_groups,
    compile_send_plan,
)
from fluxway.transport.manifest import EDGE_EXPECTED_COUNT, EDGE_TUUID, FRAME_EDGE_COUNT


def test_mirror_dataplane_build_uses_serialized_step_tuuid_cache_receipt():
    with use_backend("hybrid"):
        mesh = DeviceMesh(axis_names=("x",), device_shape=(4,), device_ids=np.arange(4))
        dynamic = ShardingSpec.hash(P("x"))
        mirror = ShardingSpec.follow("lookup.unique_ids")
        mirror.resolve(lambda path: 100 if path == "lookup.unique_ids" else None)
        assert isinstance(dynamic.routing, AlgebraicRouting)
        mirror.resolve_sharding(dynamic.routing)

        source_ref = TransportTensor(
            np.array([3, 7, 11], dtype=np.int64),
            origin_indices=ExplicitIndex(np.array([20, 10, 30], dtype=np.int64)),
            global_shape=(64,),
            receipt=SymmetryReceipt(4, np.array([2, 0, 3], dtype=np.int32)),
        )
        serialized_cache = pickle.loads(pickle.dumps({100: source_ref}))
        lookup_output = TransportTensor(
            np.ones((3, 8), dtype=np.float32),
            global_shape=(3, 8),
        )
        successor = SimpleNamespace(
            mesh=mesh,
            tuuid_sharding={200: dynamic},
            is_interleaved=False,
            interleave_unit_size=1,
        )

        route_contexts = (
            SuccessorRouteContext(
                context_index=0,
                spmd_id="Dense",
                info=successor,
                effective_rank=1,
            ),
        )
        compiled_plan = compile_send_plan(
            op_name="lookup",
            rank=1,
            mesh=mesh,
            output_uuids=(200,),
            output_sharding={200: mirror},
            successor_contexts=route_contexts,
            tuuid_to_name={
                100: "lookup.unique_ids",
                200: "lookup.sparse_features_dim_8",
            },
            sender_spmd_code=9,
            sender_world_size=4,
            interleave_unit_size=None,
        )

        frame_groups = build_step_frame_groups(
            compiled_plan=compiled_plan,
            flat_results=[lookup_output],
            step_tuuid_cache=serialized_cache,
            exec_step=0,
        )

        assert frame_groups
        assert {frame_arg[0] for _, frame_arg in frame_groups} == set(range(mesh.size))
        for _, frame_arg in frame_groups:
            target_rank, header, payloads = frame_arg
            assert 0 <= target_rank < mesh.size
            assert int(header.frame_i32[FRAME_EDGE_COUNT]) == 1
            assert int(header.edge_i32[0, EDGE_TUUID]) == 200
            assert int(header.edge_i32[0, EDGE_EXPECTED_COUNT]) == mesh.size
            assert payloads
