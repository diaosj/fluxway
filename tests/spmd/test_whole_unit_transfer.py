import pytest
import numpy as np
from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
)
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types.sharding import ShardingSpec, P
from fluxway.common.mesh import DeviceMesh
from fluxway.common.transfer import WholeUnitRouting
from fluxway.common.tensor_backend import set_hybrid_device
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.spmd.worker.operator_manager.operator_manager import OperatorManager
from fluxway.spmd.worker.data_plane import (
    compile_send_plan,
    SuccessorRouteContext,
    build_step_frame_groups,
)
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.transport.manifest import ENTRY_EXPECTED_COUNT
from tests.compiled_graph_fixtures import compile_test_graph, spmd_config_from_graph


class Producer:
    def produce(self):
        pass


class Consumer:
    def consume(self, value: TransportTensor):
        pass


@pytest.mark.asyncio
async def test_ordinary_operator_whole_unit_frames():
    set_hybrid_device("cpu")
    for sources, assignment in (
        (1, (0,)),
        (1, (0, 0, 0)),
        (2, (1, 0)),
        (2, (1, 0, 1, 0)),
    ):
        targets = len(assignment)
        src = OperatorSpec(
            name="produce",
            owner=Producer,
            is_source=False,
            output_arguments={"data": ("value",)},
            output_sharding={"value": ShardingSpec.chunk(P("s"))},
        )
        dst = OperatorSpec(
            name="consume",
            owner=Consumer,
            input_arguments={"data": ("value",)},
            input_sharding={"value": ShardingSpec.chunk(P("d"))},
        )
        graph = compile_test_graph(
            name="ordinary-whole-unit",
            spmds={
                "src": (DeviceMesh(("s",), (sources,)), {"produce": src}),
                "dst": (DeviceMesh(("d",), (targets,)), {"consume": dst}),
            },
            dependencies=(
                DependencySpec(
                    OperatorKey("src", "produce"),
                    OperatorKey("dst", "consume"),
                    (
                        TensorTransfer(
                            OperatorPort("value"),
                            OperatorPort("value"),
                            WholeUnitRouting(assignment),
                        ),
                    ),
                ),
            ),
        )
        code = graph.dependencies[0].transfers[0].transfer_code
        mailboxes = []
        for rank in range(targets):
            config = spmd_config_from_graph(graph, "dst")
            manager = OperatorManager(
                config,
                rank=rank,
                spmd_id="dst",
                static_args={},
                instance_container=OperatorInstanceContainer(),
            )
            expected = manager.get_expected_transfer_shards("consume")
            assert expected == {code: 1}
            assert manager.resolve_tensor_producers("consume", code, rank) == [
                assignment[rank]
            ]
            mailbox = WorkerMailbox()
            mailbox.declare_step("consume", 0, expected, [])
            mailboxes.append(mailbox)
        arrays = [np.arange(7, dtype=np.int32) + rank * 100 for rank in range(sources)]
        for rank in range(sources):
            view = graph.spmd("src")
            plan = compile_send_plan(
                graph_view=view,
                source_op_id="produce",
                rank=rank,
                successor_contexts=(SuccessorRouteContext(0, view.outgoing, rank),),
                sender_spmd_code=1,
                interleave_unit_size=None,
            )
            frames = build_step_frame_groups(
                compiled_plan=plan,
                flat_results=[
                    TransportTensor(arrays[rank], global_shape=(7,), source_step=0)
                ],
                step_transfer_cache={},
                exec_step=0,
            )
            seen = []
            for _, (target, header, payloads) in frames:
                assert len(payloads) == 1 and np.shares_memory(
                    payloads[0], arrays[rank]
                )
                frame = header.to_frame(payloads)
                assert int(frame.entry_i32[0][ENTRY_EXPECTED_COUNT]) == 1
                await mailboxes[target].deposit_transport_frame("consume", "src", frame)
                seen.append(target)
            assert tuple(seen) == tuple(
                c for c, p in enumerate(assignment) if p == rank
            )
        for rank, mailbox in enumerate(mailboxes):
            assert mailbox.is_step_ready("consume", 0)
            _, shards, _ = await mailbox.get_tensor_shard_views_and_arrivals(
                "consume", code, 0
            )
            assert len(shards) == 1
            view = next(iter(shards.values()))[0]
            assert np.shares_memory(view.flat_values, arrays[assignment[rank]])
        print(
            f"WHOLE_UNIT_DATAPLANE_PASS ordinary_operator sources={sources} assignment={assignment}",
            flush=True,
        )
