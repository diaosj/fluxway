import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from fluxway.common.compiled_graph import (
    DependencySpec,
    OperatorKey,
    OperatorPort,
    TensorTransfer,
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.coherence import ExecutionFenceState
from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.operator_policy import PeriodicBarrierConfig
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.common.types import DrainBoundary
from fluxway.spmd.worker.data_plane import (
    CompiledDependencyCondition,
    CompiledDependencyDecisionRoute,
    CompiledSendPlan,
    CompiledSuccessorRoute,
    SuccessorRouteContext,
    build_dependency_decision_only_frame_groups,
    build_step_frame_groups,
    compile_send_plan,
    evaluate_dependency_decisions,
    skipped_dependency_decisions,
)
from fluxway.spmd.worker.mailbox.mailbox import StepBarrier, WorkerMailbox
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.rpc.protocol import TransportFrameEnvelope
from fluxway.transport.coherence import entry_layout_state, entry_tensor_kind
from fluxway.transport.manifest import (
    ENTRY_AUX,
    ENTRY_EXPECTED_COUNT,
    ENTRY_STATE_FLAGS,
    ENTRY_PAYLOAD_COUNT,
    ENTRY_ROUTE_CODE,
    FRAME_FLAGS,
    DependencyDecision,
    DependencyDecisionMismatchError,
    FrameFlag,
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
)
from tests.compiled_graph_fixtures import compiled_spmd_spec


def _compiled_condition_plan(
    condition: Callable[[ExecutionResult], bool | Awaitable[bool]] | None,
    *,
    rank: int = 0,
    sender_spmd_code: int = 3,
):
    replicated = ShardingSpec.chunk(PartitionSpec())
    source_mesh = DeviceMesh(("data",), (2,))
    target_mesh = DeviceMesh(("data",), (2,))
    source = OperatorSpec(
        name="produce",
        output_arguments={"data": ("value",)},
        output_sharding={"value": replicated},
    )
    target = OperatorSpec(
        name="consume",
        input_arguments={"data": ("value",)},
        input_sharding={"value": replicated},
    )
    dependency = DependencySpec(
        source=OperatorKey("source", "produce"),
        target=OperatorKey("target", "consume"),
        transfers=(TensorTransfer(OperatorPort("value"), OperatorPort("value")),),
        condition=condition,
    )
    graph = compile_graph(
        name="condition-completion",
        spmds=(
            compiled_spmd_spec("source", source_mesh, {"produce": source}),
            compiled_spmd_spec("target", target_mesh, {"consume": target}),
        ),
        dependencies=(dependency,),
    )
    source_view = graph.spmd("source")
    return compile_send_plan(
        graph_view=source_view,
        source_op_id="produce",
        rank=rank,
        successor_contexts=(
            SuccessorRouteContext(0, source_view.outgoing, effective_rank=rank),
        ),
        sender_spmd_code=sender_spmd_code,
        interleave_unit_size=None,
    )


def test_send_plan_accepts_zero_based_spmd_code():
    plan = _compiled_condition_plan(None, sender_spmd_code=0)

    assert plan.sender_spmd_code == 0


def _compiled_ordering_condition_plan(
    condition: Callable[[ExecutionResult], bool | Awaitable[bool]] | None,
):
    source_mesh = DeviceMesh(("data",), (2,))
    target_mesh = DeviceMesh(("data",), (2,))
    dependency = DependencySpec(
        source=OperatorKey("source", "before"),
        target=OperatorKey("target", "after"),
        condition=condition,
    )
    graph = compile_graph(
        name="conditional-ordering-completion",
        spmds=(
            compiled_spmd_spec(
                "source",
                source_mesh,
                {"before": OperatorSpec(name="before")},
            ),
            compiled_spmd_spec(
                "target",
                target_mesh,
                {"after": OperatorSpec(name="after")},
            ),
        ),
        dependencies=(dependency,),
    )
    compiled_dependency = graph.dependencies[0]
    source_view = graph.spmd("source")
    plan = compile_send_plan(
        graph_view=source_view,
        source_op_id="before",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(
                0,
                (compiled_dependency,),
                effective_rank=0,
            ),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    return plan, compiled_dependency


def _compiled_local_ordering_plan():
    mesh = DeviceMesh(("data",), (1,))
    dependency = DependencySpec(
        source=OperatorKey("pipeline", "before"),
        target=OperatorKey("pipeline", "after"),
        local_only=True,
    )
    graph = compile_graph(
        name="local-ordering-skip",
        spmds=(
            compiled_spmd_spec(
                "pipeline",
                mesh,
                {
                    "before": OperatorSpec(name="before"),
                    "after": OperatorSpec(name="after"),
                },
            ),
        ),
        dependencies=(dependency,),
    )
    compiled_dependency = graph.dependencies[0]
    view = graph.spmd("pipeline")
    plan = compile_send_plan(
        graph_view=view,
        source_op_id="before",
        rank=0,
        successor_contexts=(
            SuccessorRouteContext(
                0,
                (compiled_dependency,),
                effective_rank=0,
            ),
        ),
        sender_spmd_code=3,
        interleave_unit_size=None,
    )
    return plan, compiled_dependency


def _tensor_frame(*, transfer_code: int, step: int):
    builder = TransportFrameBuilder(
        step=step,
        sender_spmd_code=4,
        sender_rank=0,
        sender_world_size=1,
    )
    entry_index = builder.add_tensor_transfer_entry(
        transfer_code=transfer_code,
        layout_state=0,
        tensor_kind=TensorEntryKind.TENSOR,
        expected_count=1,
        coherence_epoch=step,
    )
    builder.add_payload(
        entry_index, PayloadRole.VALUES, np.array([1], dtype=np.int32), 1
    )
    return builder.build()


def test_condition_frame_is_header_only_dependency_outcome():
    builder = TransportFrameBuilder(
        step=7,
        sender_spmd_code=3,
        sender_rank=1,
        sender_world_size=2,
        flags=FrameFlag.DEPENDENCY_DECISION,
    )
    builder.add_dependency_decision_entry(
        dependency_code=17,
        decision=DependencyDecision.SKIP,
        expected_count=2,
    )
    frame = builder.build()

    assert int(frame.frame_i32[FRAME_FLAGS]) == int(FrameFlag.DEPENDENCY_DECISION)
    assert int(frame.entry_i32[0, ENTRY_ROUTE_CODE]) == 17
    assert int(frame.entry_i32[0, ENTRY_AUX]) == int(DependencyDecision.SKIP)
    assert (
        TensorEntryKind(entry_tensor_kind(int(frame.entry_i32[0, ENTRY_STATE_FLAGS])))
        is TensorEntryKind.NOT_APPLICABLE
    )
    assert int(frame.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 2
    assert entry_layout_state(int(frame.entry_i32[0, ENTRY_STATE_FLAGS])) == 0
    assert int(frame.entry_i32[0, ENTRY_PAYLOAD_COUNT]) == 0
    assert frame.payload_segments == ()

    with pytest.raises(ValueError, match="duplicate dependency_code=17"):
        builder.add_dependency_decision_entry(
            dependency_code=17,
            decision=DependencyDecision.SKIP,
            expected_count=2,
        )


def test_compiled_send_plan_rejects_condition_without_decision_route():
    with pytest.raises(ValueError, match="without decision routes"):
        CompiledSendPlan.create(
            op_name="produce",
            source_spmd_id="source",
            rank=0,
            sender_spmd_code=1,
            sender_world_size=1,
            successor_routes=(),
            output_count=0,
            transfer_code_to_name={},
            dependency_conditions=(
                CompiledDependencyCondition(
                    dependency_code=7,
                    condition=lambda _result: True,
                ),
            ),
        )


@pytest.mark.parametrize(
    "decisions",
    [None, {}, {999: DependencyDecision.PROCEED}],
)
def test_conditional_frame_build_requires_exact_condition_decisions(decisions):
    plan = _compiled_condition_plan(lambda _result: True)

    with pytest.raises(ValueError, match="requires"):
        build_step_frame_groups(
            compiled_plan=plan,
            flat_results=[],
            step_transfer_cache={},
            exec_step=1,
            dependency_decisions=decisions,
        )


def test_decision_only_frame_build_requires_every_dependency_route():
    plan, _dependency = _compiled_ordering_condition_plan(None)

    with pytest.raises(ValueError, match="require all"):
        build_dependency_decision_only_frame_groups(
            compiled_plan=plan,
            dependency_decisions={999: DependencyDecision.SKIP},
            exec_step=1,
        )


@pytest.mark.parametrize("invalid", [np.bool_(True), 1])
@pytest.mark.asyncio
async def test_condition_requires_exact_bool(invalid):
    plan = _compiled_condition_plan(lambda _result: invalid)
    with pytest.raises(TypeError, match="exact bool"):
        await evaluate_dependency_decisions(
            plan,
            ExecutionResult(status=ExecutionStatus.SUCCESS, data="value"),
        )


@pytest.mark.asyncio
async def test_async_condition_is_awaited_without_blocking_the_decision_path():
    entered = asyncio.Event()
    release = asyncio.Event()

    async def condition(_result):
        entered.set()
        await release.wait()
        return False

    plan = _compiled_condition_plan(condition)
    evaluation = asyncio.create_task(
        evaluate_dependency_decisions(
            plan,
            ExecutionResult(status=ExecutionStatus.SUCCESS, data="value"),
        )
    )
    await asyncio.wait_for(entered.wait(), timeout=1.0)
    assert not evaluation.done()
    release.set()
    assert await asyncio.wait_for(evaluation, timeout=1.0) == {
        plan.dependency_conditions[0].dependency_code: DependencyDecision.SKIP
    }


@pytest.mark.asyncio
async def test_independent_async_conditions_are_evaluated_concurrently():
    both_entered = asyncio.Event()
    release = asyncio.Event()
    entered = 0

    async def condition(_result):
        nonlocal entered
        entered += 1
        if entered == 2:
            both_entered.set()
        await release.wait()
        return True

    plan = CompiledSendPlan.create(
        op_name="source",
        source_spmd_id="source",
        rank=0,
        sender_spmd_code=0,
        sender_world_size=1,
        successor_routes=(
            CompiledSuccessorRoute(
                context_index=0,
                spmd_id="target",
                effective_rank=0,
                tensor_routes=(),
                dependency_decision_routes=(
                    CompiledDependencyDecisionRoute(1, "target", (0,), 1),
                    CompiledDependencyDecisionRoute(2, "target", (0,), 1),
                ),
            ),
        ),
        output_count=0,
        transfer_code_to_name={},
        dependency_conditions=(
            CompiledDependencyCondition(1, condition),
            CompiledDependencyCondition(2, condition),
        ),
    )

    evaluation = asyncio.create_task(evaluate_dependency_decisions(plan, object()))
    await asyncio.wait_for(both_entered.wait(), timeout=1.0)
    release.set()

    assert await asyncio.wait_for(evaluation, timeout=1.0) == {
        1: DependencyDecision.PROCEED,
        2: DependencyDecision.PROCEED,
    }


@pytest.mark.asyncio
async def test_condition_awaitable_result_is_supported():
    def condition(_result):
        async def result():
            return False

        return result()

    plan = _compiled_condition_plan(condition)
    assert await evaluate_dependency_decisions(
        plan,
        ExecutionResult(status=ExecutionStatus.SUCCESS, data="value"),
    ) == {plan.dependency_conditions[0].dependency_code: DependencyDecision.SKIP}


@pytest.mark.asyncio
async def test_sync_bool_condition_bypasses_awaitable_inspection():
    plan = _compiled_condition_plan(lambda _result: True)

    with patch(
        "fluxway.spmd.worker.data_plane.inspect.isawaitable",
        side_effect=AssertionError("exact bool must use the synchronous fast path"),
    ):
        decisions = await evaluate_dependency_decisions(
            plan,
            ExecutionResult(status=ExecutionStatus.SUCCESS, data="value"),
        )

    dependency_code = plan.dependency_conditions[0].dependency_code
    assert decisions == {dependency_code: DependencyDecision.PROCEED}


@pytest.mark.asyncio
async def test_condition_evaluation_does_not_reflect_on_function_type():
    plan = _compiled_condition_plan(lambda _result: True)

    with patch(
        "fluxway.spmd.worker.data_plane.inspect.iscoroutinefunction",
        side_effect=AssertionError("runtime reflection"),
    ):
        assert await evaluate_dependency_decisions(
            plan,
            ExecutionResult(status=ExecutionStatus.SUCCESS, data="value"),
        )


@pytest.mark.asyncio
async def test_condition_is_evaluated_once_per_dependency_per_source_rank():
    calls = []

    def condition(result):
        calls.append(result)
        return False

    plan = _compiled_condition_plan(condition)
    result = ExecutionResult(status=ExecutionStatus.SUCCESS, data="value")

    assert await evaluate_dependency_decisions(plan, result) == {
        plan.successor_routes[0].dependency_decision_routes[0].dependency_code: (
            DependencyDecision.SKIP
        )
    }
    assert calls == [result]


def test_false_condition_fans_out_header_only_outcome_to_every_target_rank():
    plan = _compiled_condition_plan(lambda _result: False)
    dependency_code = (
        plan.successor_routes[0].dependency_decision_routes[0].dependency_code
    )
    frame_groups = build_dependency_decision_only_frame_groups(
        compiled_plan=plan,
        dependency_decisions={dependency_code: DependencyDecision.SKIP},
        exec_step=4,
    )

    assert [frame_arg[0] for _, frame_arg in frame_groups] == [0, 1]
    for _, (_, header, payloads) in frame_groups:
        assert int(header.frame_i32[FRAME_FLAGS]) == int(FrameFlag.DEPENDENCY_DECISION)
        assert int(header.entry_i32[0, ENTRY_ROUTE_CODE]) == dependency_code
        assert int(header.entry_i32[0, ENTRY_EXPECTED_COUNT]) == 2
        assert payloads == ()


@pytest.mark.asyncio
async def test_unconditional_data_path_emits_no_condition_frame():
    plan = _compiled_condition_plan(None)
    assert not plan.dependency_conditions
    assert (
        await evaluate_dependency_decisions(
            plan,
            ExecutionResult(status=ExecutionStatus.SUCCESS, data="value"),
        )
        is None
    )

    frame_groups = build_step_frame_groups(
        compiled_plan=plan,
        flat_results=[np.array([1, 2], dtype=np.int32)],
        step_transfer_cache={},
        exec_step=2,
    )

    assert frame_groups
    assert all(
        int(header.frame_i32[FRAME_FLAGS]) != int(FrameFlag.DEPENDENCY_DECISION)
        for _, (_, header, _) in frame_groups
    )


@pytest.mark.asyncio
async def test_conditional_ordering_dependency_emits_only_dependency_decision():
    plan, dependency = _compiled_ordering_condition_plan(lambda _result: False)

    assert plan.successor_routes[0].ordering_routes == ()
    assert [
        route.dependency_code
        for route in plan.successor_routes[0].dependency_decision_routes
    ] == [dependency.code]

    decisions = await evaluate_dependency_decisions(
        plan,
        ExecutionResult(status=ExecutionStatus.SUCCESS, data=None),
    )
    frame_groups = build_step_frame_groups(
        compiled_plan=plan,
        flat_results=[],
        step_transfer_cache={},
        exec_step=4,
        dependency_decisions=decisions,
    )

    assert decisions == {dependency.code: DependencyDecision.SKIP}
    assert len(frame_groups) == 2
    assert all(
        FrameFlag(int(header.frame_i32[FRAME_FLAGS])) is FrameFlag.DEPENDENCY_DECISION
        for _, (_, header, _) in frame_groups
    )


def test_skipped_unconditional_ordering_emits_skip_without_ordering_completion():
    plan, dependency = _compiled_ordering_condition_plan(None)

    assert [
        route.dependency_code for route in plan.successor_routes[0].ordering_routes
    ] == [dependency.code]
    decisions = skipped_dependency_decisions(plan)
    frame_groups = build_dependency_decision_only_frame_groups(
        compiled_plan=plan,
        dependency_decisions=decisions,
        exec_step=5,
    )

    assert decisions == {dependency.code: DependencyDecision.SKIP}
    assert len(frame_groups) == 2
    assert all(
        FrameFlag(int(header.frame_i32[FRAME_FLAGS])) is FrameFlag.DEPENDENCY_DECISION
        for _, (_, header, _) in frame_groups
    )


@pytest.mark.parametrize(
    ("decision", "expected_skipped"),
    [
        (DependencyDecision.PROCEED, False),
        (DependencyDecision.SKIP, True),
    ],
)
def test_decision_only_barrier_completes_without_tensor(
    decision: DependencyDecision,
    expected_skipped: bool,
):
    barrier = StepBarrier(
        "after",
        6,
        expected_transfers={},
        expected_states=[],
        expected_dependency_decisions={101: 2},
    )

    assert not barrier.is_ready()
    assert not barrier.dependency_decision_arrived(101, decision, expected_count=2)
    assert barrier.dependency_decision_arrived(101, decision, expected_count=2)
    assert barrier.is_ready()
    assert barrier.is_skipped() is expected_skipped


@pytest.mark.asyncio
async def test_local_skipped_ordering_does_not_deposit_success_completion():
    plan, dependency = _compiled_local_ordering_plan()
    deposits = []

    async def deposit_ordering(*args, **kwargs):
        deposits.append((args, kwargs))

    worker_class = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(worker_class)
    worker.rank = 0
    worker.spmd_id = "pipeline"
    worker._mailbox = SimpleNamespace(deposit_ordering=deposit_ordering)
    worker._ensure_step_declared = lambda _op_name, _step: None

    await worker_class._deposit_local_only_orderings(
        worker,
        plan,
        7,
        {dependency.code: DependencyDecision.SKIP},
    )
    assert deposits == []

    await worker_class._deposit_local_only_orderings(worker, plan, 7, {})
    assert len(deposits) == 1


@pytest.mark.asyncio
async def test_false_condition_waits_for_peer_dependency_before_skipping_step():
    ready = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op_name, step: ready.append((op_name, step)))
    mailbox.declare_step(
        "consume",
        5,
        expected_transfers={11: 2, 12: 1},
        expected_states=["unused-when-skipped"],
        transfer_dependencies={11: 101, 12: 102},
        expected_dependency_decisions={101: 2},
    )

    await mailbox.deposit_dependency_decision(
        "consume",
        101,
        5,
        sender_id="source",
        rank=0,
        expected_count=2,
        decision=DependencyDecision.SKIP,
    )
    await mailbox.deposit_dependency_decision(
        "consume",
        101,
        5,
        sender_id="source",
        rank=1,
        expected_count=2,
        decision=DependencyDecision.SKIP,
    )
    assert not mailbox.is_step_ready("consume", 5)

    await mailbox.deposit_transport_frame(
        "consume",
        "peer",
        _tensor_frame(transfer_code=12, step=5),
    )

    assert mailbox.is_step_ready("consume", 5)
    assert mailbox.is_step_skipped("consume", 5)
    assert ready == [("consume", 5)]


@pytest.mark.asyncio
async def test_periodic_proceed_waits_for_tensor_before_completing_virtual_step():
    ready = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op_name, step: ready.append((op_name, step)))
    mailbox.declare_step(
        "consume",
        0,
        expected_transfers={11: 1},
        expected_states=[],
        barrier_config=PeriodicBarrierConfig(n=1),
        transfer_dependencies={11: 101},
        expected_dependency_decisions={101: 1},
    )

    await mailbox.deposit_dependency_decision(
        "consume",
        101,
        0,
        sender_id="source",
        rank=0,
        expected_count=1,
        decision=DependencyDecision.PROCEED,
    )

    assert ready == []
    assert not mailbox.is_step_ready("consume", 0)

    await mailbox.deposit_transport_frame(
        "consume",
        "source",
        _tensor_frame(transfer_code=11, step=0),
    )

    assert ready == [("consume", 0)]
    assert mailbox.is_step_ready("consume", 0)
    assert not mailbox.is_step_skipped("consume", 0)


@pytest.mark.asyncio
async def test_periodic_skip_completes_each_virtual_step_without_tensor():
    ready = []
    mailbox = WorkerMailbox()
    mailbox.set_ready_callback(lambda op_name, step: ready.append((op_name, step)))
    mailbox.declare_step(
        "consume",
        0,
        expected_transfers={11: 1},
        expected_states=[],
        barrier_config=PeriodicBarrierConfig(n=1),
        transfer_dependencies={11: 101},
        expected_dependency_decisions={101: 1},
    )

    for step in range(2):
        await mailbox.deposit_dependency_decision(
            "consume",
            101,
            step,
            sender_id="source",
            rank=0,
            expected_count=1,
            decision=DependencyDecision.SKIP,
        )

    assert ready == [("consume", 0), ("consume", 1)]
    assert mailbox.is_step_ready("consume", 0)
    assert mailbox.is_step_ready("consume", 1)
    assert mailbox.is_step_skipped("consume", 0)
    assert mailbox.is_step_skipped("consume", 1)


@pytest.mark.asyncio
async def test_dependency_decisions_require_consistency_and_unique_sender():
    mailbox = WorkerMailbox()
    mailbox.declare_step(
        "consume",
        8,
        expected_transfers={11: 2},
        expected_states=[],
        transfer_dependencies={11: 101},
        expected_dependency_decisions={101: 2},
    )
    await mailbox.deposit_dependency_decision(
        "consume",
        101,
        8,
        sender_id="source",
        rank=0,
        expected_count=2,
        decision=DependencyDecision.SKIP,
    )
    with pytest.raises(ValueError, match="DUPLICATE DEPENDENCY DECISION"):
        await mailbox.deposit_dependency_decision(
            "consume",
            101,
            8,
            sender_id="source",
            rank=0,
            expected_count=2,
            decision=DependencyDecision.SKIP,
        )
    with pytest.raises(DependencyDecisionMismatchError):
        await mailbox.deposit_dependency_decision(
            "consume",
            101,
            8,
            sender_id="source",
            rank=1,
            expected_count=2,
            decision=DependencyDecision.PROCEED,
        )


@pytest.mark.asyncio
async def test_propagated_skip_can_complete_unconditional_local_dependency():
    mailbox = WorkerMailbox()
    mailbox.declare_step(
        "consume",
        3,
        expected_transfers={11: 1},
        expected_states=[],
        transfer_dependencies={11: 101},
        expected_dependency_decisions={},
    )
    await mailbox.deposit_dependency_decision(
        "consume",
        101,
        3,
        sender_id="local",
        rank=0,
        expected_count=1,
        decision=DependencyDecision.SKIP,
    )

    assert mailbox.is_step_ready("consume", 3)
    assert mailbox.is_step_skipped("consume", 3)


@pytest.mark.asyncio
async def test_worker_receives_multi_sender_condition_completion_through_batch_abi():
    plan = _compiled_condition_plan(lambda _result: False)
    route = plan.successor_routes[0].dependency_decision_routes[0]
    transfer_code = plan.successor_routes[0].tensor_routes[0].transfer_code
    worker_class = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(worker_class)
    worker.spmd_id = "target"
    worker.rank = 0
    worker._spmd_id_by_code = {3: "source"}
    worker._fence_state = ExecutionFenceState.OPEN
    worker._profiler_runtime = SimpleNamespace(enabled=False)
    worker._barriers_declared = {}
    worker._mailbox = WorkerMailbox()
    worker._op_manager = SimpleNamespace(
        get_dependency_decision_expected_count=lambda code: 2,
        get_dependency_decision_target_op=lambda code: "consume",
        get_expected_transfer_shards=lambda op_name: {transfer_code: 2},
        get_expected_control_arg_names=lambda op_name: [],
        get_operator_spec=lambda op_name: OperatorSpec(name="consume"),
        get_expected_ordering_counts=lambda op_name: {},
        get_transfer_dependency_codes=lambda op_name: {
            transfer_code: route.dependency_code
        },
        get_expected_dependency_decisions=lambda op_name: {route.dependency_code: 2},
    )

    envelopes = []
    for sender_rank in range(2):
        builder = TransportFrameBuilder(
            step=6,
            sender_spmd_code=3,
            sender_rank=sender_rank,
            sender_world_size=2,
            flags=FrameFlag.DEPENDENCY_DECISION,
        )
        builder.add_dependency_decision_entry(
            dependency_code=route.dependency_code,
            decision=DependencyDecision.SKIP,
            expected_count=2,
        )
        frame = builder.build()
        envelopes.append(
            TransportFrameEnvelope(
                header=frame.header,
                payload_segments=frame.payload_segments,
            )
        )

    watermarks = await worker_class.receive_transport_frames(
        worker,
        tuple(envelopes),
    )

    assert watermarks == (0, 0)
    assert worker._mailbox.is_step_ready("consume", 6)
    assert worker._mailbox.is_step_skipped("consume", 6)


def test_skipped_result_uses_async_send_registration_path():
    worker_class = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(worker_class)
    skipped = ExecutionResult(status=ExecutionStatus.SKIPPED)

    assert worker_class._result_requires_send(worker, skipped)


@pytest.mark.asyncio
async def test_skipped_ready_step_does_not_call_user_function():
    worker_class = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(worker_class)
    worker.rank = 0
    worker._world_size = 1
    worker._completed_step_count = {"consume": 0}
    worker._drain_boundary = DrainBoundary()
    worker._mailbox = WorkerMailbox()
    worker._mailbox.declare_step(
        "consume",
        0,
        expected_transfers={11: 1},
        expected_states=[],
        transfer_dependencies={11: 101},
        expected_dependency_decisions={101: 1},
    )
    await worker._mailbox.deposit_dependency_decision(
        "consume",
        101,
        0,
        sender_id="source",
        rank=0,
        expected_count=1,
        decision=DependencyDecision.SKIP,
    )
    worker._op_manager = type(
        "Manager",
        (),
        {
            "is_periodic_trigger": staticmethod(lambda _name: False),
            "get_dispatcher": staticmethod(
                lambda _name: type(
                    "Dispatcher",
                    (),
                    {"should_execute": staticmethod(lambda *_args: True)},
                )()
            ),
            "get_operator_spec": staticmethod(
                lambda _name: type(
                    "OperatorSpec",
                    (),
                    {
                        "is_source": False,
                        "is_interleaved": False,
                        "has_source_descriptor": False,
                    },
                )()
            ),
        },
    )()
    worker._drain_cond = asyncio.Condition()
    worker._drain_step_count = None
    worker._barriers_declared = {("consume", 0): True}
    worker._uses_direct_local_input_fast_path = lambda _name: False
    worker._assemble_input = lambda *_args: (_ for _ in ()).throw(
        AssertionError("skipped step must not assemble user input")
    )

    result = await worker_class._execute_operator_prepare_input(worker, "consume", 0)

    assert result.status is ExecutionStatus.SKIPPED
