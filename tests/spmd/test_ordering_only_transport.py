from types import SimpleNamespace

import pytest

import fluxway as fw
from fluxway.common.decorators import operator, spmd
from fluxway.common.compiled_graph import (
    CompiledDependency,
    DependencySpec,
    OperatorKey,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.coherence import ExecutionFenceState
from fluxway.common.operator_spec import OperatorSpec
from fluxway.spmd.worker.data_plane import (
    CompiledSendPlan,
    SuccessorRouteContext,
    build_completed_without_tensor_step_frame_groups,
    build_step_frame_groups,
    compile_send_plan,
)
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.rpc.protocol import TransportFrameEnvelope
from fluxway.transport.manifest import (
    ENTRY_EXPECTED_COUNT,
    ENTRY_ROUTE_CODE,
    FRAME_FLAGS,
    FrameFlag,
    TransportFrameBuilder,
)
from tests.compiled_graph_fixtures import compiled_spmd_spec


@spmd(
    resource_profile=fw.ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("data",),
    mesh_axis_shapes=(1,),
)
class _OrderingSPMD:
    @operator(name="before", is_source=True)
    def before(self) -> None:
        return None

    @operator(name="after")
    def after(self) -> None:
        return None


def _compiled_ordering_plan() -> tuple[CompiledSendPlan, CompiledDependency]:
    source_mesh = DeviceMesh(("data",), (3,))
    target_mesh = DeviceMesh(("data",), (2,))
    compiled_graph = compile_graph(
        name="ordering-only",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"before": OperatorSpec(name="before", is_source=True)},
            ),
            compiled_spmd_spec(
                "target",
                target_mesh,
                {"after": OperatorSpec(name="after")},
            ),
        ),
        dependencies=(
            DependencySpec(
                source=OperatorKey("source", "before"),
                target=OperatorKey("target", "after"),
            ),
        ),
    )
    source_plan = compiled_graph.spmd("source")
    dependency = source_plan.outgoing[0]
    send_plan = compile_send_plan(
        graph_view=source_plan,
        source_op_id="before",
        rank=1,
        successor_contexts=(
            SuccessorRouteContext(
                context_index=0,
                dependencies=(dependency,),
                effective_rank=1,
            ),
        ),
        sender_spmd_code=4,
        interleave_unit_size=None,
    )
    return send_plan, dependency


def _assert_ordering_frames(frame_groups, dependency_code: int):
    assert [frame[1][0] for frame in frame_groups] == [0, 1]
    for context_index, (_target_rank, header, payloads) in frame_groups:
        assert context_index == 0
        assert int(header.frame_i32[FRAME_FLAGS]) == int(FrameFlag.ORDERING_COMPLETION)
        assert int(header.entry_i32[0, ENTRY_ROUTE_CODE]) == dependency_code
        assert int(header.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 3
        assert header.payload_i32.shape[0] == 0
        assert payloads == ()


def test_transport_compiler_emits_header_only_frame_for_every_target_rank():
    plan, dependency = _compiled_ordering_plan()

    assert plan.successor_routes[0].tensor_routes == ()
    assert plan.successor_routes[0].ordering_routes[0].target_op == "after"
    _assert_ordering_frames(
        build_step_frame_groups(
            compiled_plan=plan,
            flat_results=[],
            step_transfer_cache={},
            exec_step=7,
        ),
        dependency.code,
    )


def test_skipped_step_still_emits_ordering_completion():
    plan, dependency = _compiled_ordering_plan()
    _assert_ordering_frames(
        build_completed_without_tensor_step_frame_groups(
            compiled_plan=plan,
            empty_values=None,
            exec_step=7,
        ),
        dependency.code,
    )


@pytest.mark.asyncio
async def test_ordering_arrivals_participate_in_the_existing_step_barrier():
    mailbox = WorkerMailbox()
    ready = []
    mailbox.set_ready_callback(lambda op_name, step: ready.append((op_name, step)))
    mailbox.declare_step(
        "after",
        7,
        expected_transfers={},
        expected_states=[],
        expected_orderings={17: 2},
    )

    assert not mailbox.is_step_ready("after", 7)
    await mailbox.deposit_ordering(
        "after", 17, 7, sender_id="source", rank=0, expected_count=2
    )
    assert not mailbox.is_step_ready("after", 7)
    await mailbox.deposit_ordering(
        "after", 17, 7, sender_id="source", rank=1, expected_count=2
    )

    assert mailbox.is_step_ready("after", 7)
    assert ready == [("after", 7)]
    with pytest.raises(ValueError, match="DUPLICATE ORDERING"):
        await mailbox.deposit_ordering(
            "after", 17, 7, sender_id="source", rank=1, expected_count=2
        )


def test_operator_manager_compiles_ordering_route_from_compiled_graph():
    builder = fw.Builder()
    builder.add_node("pipeline", _OrderingSPMD)
    builder.add_func_edge("pipeline", "before", "pipeline", "after", local_only=True)
    graph = builder.build("ordering-manager")
    config = graph.nodes["pipeline"].spmd_config
    manager = OperatorManager(config, rank=0, spmd_id="pipeline", static_args={})

    plan = graph.require_compiled_graph()
    assert plan is not None
    dependency = plan.dependencies[0]
    code = dependency.code
    assert manager.get_outgoing_ordering_dependencies("before") == (dependency,)
    assert manager.get_ordering_target_op(code) == "after"
    assert manager.get_expected_ordering_counts("after") == {code: 1}


@pytest.mark.asyncio
async def test_worker_receives_ordering_frames_through_existing_batch_rpc_abi():
    raw_worker = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(raw_worker)
    worker.spmd_id = "target"
    worker.rank = 0
    worker._spmd_id_by_code = {4: "source"}
    worker._fence_state = ExecutionFenceState.OPEN
    worker._profiler_runtime = SimpleNamespace(enabled=False)
    worker._barriers_declared = {}
    ready = []
    worker._mailbox = WorkerMailbox()
    worker._mailbox.set_ready_callback(
        lambda op_name, step: ready.append((op_name, step))
    )
    target_spec = OperatorSpec(name="after")
    worker._op_manager = SimpleNamespace(
        get_ordering_target_op=lambda code: "after" if code == 17 else None,
        get_expected_ordering_counts=lambda op_name: {17: 3},
        get_expected_transfer_shards=lambda op_name: {},
        get_expected_control_arg_names=lambda op_name: [],
        get_operator_spec=lambda op_name: target_spec,
    )

    def frame(sender_rank: int) -> TransportFrameEnvelope:
        builder = TransportFrameBuilder(
            step=7,
            sender_spmd_code=4,
            sender_rank=sender_rank,
            sender_world_size=3,
            flags=FrameFlag.ORDERING_COMPLETION,
        )
        builder.add_ordering_completion_entry(dependency_code=17, expected_count=3)
        built = builder.build()
        return TransportFrameEnvelope(
            header=built.header,
            payload_segments=built.payload_segments,
        )

    watermarks = await raw_worker.receive_transport_frames(
        worker,
        (frame(0), frame(1), frame(2)),
    )

    assert watermarks == (0, 0, 0)
    assert ready == [("after", 7)]
