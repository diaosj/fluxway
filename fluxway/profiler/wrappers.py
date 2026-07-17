from __future__ import annotations

from collections.abc import Callable
from contextlib import AsyncExitStack
from typing import Any

from fluxway.common.operator_policy import BarrierConfig
from fluxway.common.types.tensor import TransportTensor
from fluxway.profiler.clock import AxonNativeHlcBridge, RayFluxwayHlcBridge
from fluxway.profiler.record import TraceCarrier
from fluxway.profiler.recorder import TraceRecorder
from fluxway.profiler.user_api import activate_user_profiler
from fluxway.rpc.protocol import AxonDataPlane, DataPlaneCallable, RayActorDataPlane
from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.transport.manifest import (
    FRAME_ENTRY_COUNT,
    FRAME_PAYLOAD_COUNT,
    FRAME_STEP,
    TransportFrame,
)


class TracingDataPlaneCallable(DataPlaneCallable):
    def __init__(
        self,
        delegate: DataPlaneCallable,
        recorder: TraceRecorder,
        *,
        remote_spmd_id: str,
        remote_rank: int,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._remote_spmd_id = remote_spmd_id
        self._remote_rank = remote_rank
        self._ray_clock = RayFluxwayHlcBridge()
        self._axon_clock = AxonNativeHlcBridge()

    @property
    def delegate(self) -> DataPlaneCallable:
        return self._delegate

    async def receive_transport_frames(
        self,
        frames: tuple[Any, ...],
    ) -> Any:
        if not frames:
            raise ValueError("receive_transport_frames requires at least one frame.")
        carrier = self._recorder.root_carrier.child(
            "transport_frame", self._remote_spmd_id, self._remote_rank
        )
        async with AsyncExitStack() as stack:
            for frame in frames:
                step = int(frame.header.frame_i32[FRAME_STEP])
                await stack.enter_async_context(
                    self._recorder.data_plane_receive_transport_frame(
                        carrier=carrier,
                        remote_rank=self._remote_rank,
                        step=step,
                    )
                )
                await stack.enter_async_context(
                    self._recorder.data_plane_receiver_wait(
                        remote_rank=self._remote_rank,
                        step=step,
                    )
                )
            if isinstance(self._delegate, RayActorDataPlane):
                result = await self._delegate.receive_transport_frames_profiled(
                    frames,
                    profiler_recorder=self._recorder,
                )
            else:
                result = await self._delegate.receive_transport_frames(frames)
        self._record_receive_clock(carrier, result)
        return result

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> Any:
        carrier = self._recorder.root_carrier.child(
            "control", self._remote_spmd_id, self._remote_rank, op_name, step
        )
        async with self._recorder.data_plane_receive_control(
            carrier=carrier,
            step=step,
            field_count=len(names),
        ):
            result = await self._delegate.receive_control(op_name, step, names, values)
        self._record_receive_clock(carrier, result)
        return result

    def _record_receive_clock(self, carrier: TraceCarrier, result: Any) -> None:
        if isinstance(self._delegate, AxonDataPlane):
            received = self._axon_clock.receive_result(carrier, result)
        else:
            received = self._ray_clock.receive(carrier)
        self._recorder.data_plane_hlc_receive(
            carrier=received,
            receiver_hlc_physical_ns=received.receiver_hlc_physical_ns,
            receiver_hlc_logical=received.receiver_hlc_logical,
            clock_source=received.clock_source,
        )

    def __repr__(self) -> str:
        return (
            "TracingDataPlaneCallable("
            f"delegate={self._delegate!r}, remote={self._remote_spmd_id}:{self._remote_rank})"
        )


class TracingDataPlaneBackendExecutor:
    def __init__(self, delegate: Any, recorder: TraceRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    async def send_tensor_frames(self, **kwargs: Any) -> Any:
        return await self._send(self._delegate.send_tensor_frames, kwargs)

    async def send_completed_without_tensor_frames(self, **kwargs: Any) -> Any:
        return await self._send(
            self._delegate.send_completed_without_tensor_frames,
            kwargs,
        )

    async def send_dependency_decision_frames(self, **kwargs: Any) -> Any:
        return await self._send(
            self._delegate.send_dependency_decision_frames,
            kwargs,
        )

    async def _send(self, send: Any, kwargs: dict[str, Any]) -> Any:
        async with self._recorder.data_plane_send_step_frames(
            rank=int(kwargs.get("rank", 0)),
            exec_step=int(kwargs.get("exec_step", 0)),
        ):
            kwargs["profiler_recorder"] = self._recorder
            return await send(**kwargs)


class TracingWorkerMailbox(WorkerMailbox):
    def __init__(self, recorder: TraceRecorder) -> None:
        super().__init__()
        self._recorder = recorder

    def set_ready_callback(self, callback: Callable[[str, int], None]) -> None:
        def traced_callback(op_name: str, step: int) -> None:
            self._recorder.mailbox_ready_fire(op_name=op_name, step=step)
            callback(op_name, step)

        super().set_ready_callback(traced_callback)

    def declare_step(
        self,
        op_name: str,
        step: int,
        expected_transfers: dict[int, int],
        expected_states: list[str],
        barrier_config: BarrierConfig | None = None,
        expected_orderings: dict[int, int] | None = None,
        transfer_dependencies: dict[int, int] | None = None,
        expected_dependency_decisions: dict[int, int] | None = None,
    ) -> None:
        self._recorder.mailbox_declare_step(
            op_name=op_name,
            step=step,
            transfer_count=len(expected_transfers),
            state_count=len(expected_states),
        )
        super().declare_step(
            op_name,
            step,
            expected_transfers,
            expected_states,
            barrier_config,
            expected_orderings,
            transfer_dependencies,
            expected_dependency_decisions,
        )

    async def deposit_control_arguments(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> None:
        async with self._recorder.mailbox_deposit_control(
            op_name=op_name,
            step=step,
            field_count=len(names),
        ):
            await super().deposit_control_arguments(op_name, step, names, values)

    async def deposit_transport_frame(
        self,
        op_name: str,
        sender_id: str,
        frame: TransportFrame,
    ) -> int:
        step = int(frame.frame_i32[FRAME_STEP])
        async with self._recorder.mailbox_deposit_frame(
            op_name=op_name,
            step=step,
            entry_count=int(frame.frame_i32[FRAME_ENTRY_COUNT]),
            payload_count=int(frame.frame_i32[FRAME_PAYLOAD_COUNT]),
        ):
            return await super().deposit_transport_frame(
                op_name,
                sender_id,
                frame,
            )

    async def deposit_local_tensor(
        self,
        op_name: str,
        transfer_code: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: Any,
    ) -> int:
        if not isinstance(tensor, TransportTensor):
            return await super().deposit_local_tensor(
                op_name,
                transfer_code,
                step,
                sender_id,
                rank,
                tensor,
            )
        expected_count = tensor.routing_entity_count
        async with self._recorder.mailbox_deposit_local_tensor(
            op_name=op_name,
            transfer_code=transfer_code,
            step=step,
            data_count=expected_count,
        ):
            return await super().deposit_local_tensor(
                op_name,
                transfer_code,
                step,
                sender_id,
                rank,
                tensor,
            )

    async def clear_step(self, op_name: str, step: int) -> None:
        async with self._recorder.mailbox_clear_step(op_name=op_name, step=step):
            await super().clear_step(op_name, step)

    async def wait_step_ready(self, op_name: str, step: int) -> None:
        async with self._recorder.mailbox_wait_step_ready(op_name=op_name, step=step):
            await super().wait_step_ready(op_name, step)


class TracingLocalStepPlanExecutor:
    def __init__(self, delegate: Any, recorder: TraceRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    async def execute(self, plan: Any, registry: Any, step_context: Any = None) -> Any:
        if step_context is None:
            raise ValueError("TracingLocalStepPlanExecutor requires a step context.")
        async with self._recorder.local_step_plan_execute(
            op_name=step_context.op_name,
            step=step_context.step,
            node_count=len(getattr(plan, "node_kinds", ())),
        ):
            return await self._delegate.execute(plan, registry, step_context)

    async def drain_user_callables(self) -> None:
        return await self._delegate.drain_user_callables()

    async def drain(self) -> None:
        return await self._delegate.drain()

    @property
    def user_callable_executor(self) -> Any:
        return self._delegate.user_callable_executor


class TracingUserCallableExecutor:
    def __init__(self, delegate: Any, recorder: TraceRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    async def execute(self, spec: Any) -> Any:
        selected = self._delegate.select(spec)
        execution_mode = selected.lane.value
        compiled_operator = spec.compiled_operator
        if compiled_operator is None:
            raise ValueError(
                "traced user callable execution requires compiled_operator."
            )
        graph_id = spec.graph_id
        if graph_id is None:
            raise ValueError("traced user callable execution requires graph_id.")
        op_name = compiled_operator.key.op_id
        step = self._operator_step(spec)
        rank = self._operator_rank(spec)
        async with self._recorder.user_callable_execute(
            graph_id=graph_id,
            compiled_operator=compiled_operator,
            step=step,
            execution_mode=execution_mode,
        ):
            with activate_user_profiler(
                recorder=self._recorder,
                op_name=op_name,
                step=step,
                rank=rank,
                execution_mode=execution_mode,
            ):
                return await self._delegate.execute_selected(selected)

    async def drain(self) -> None:
        return await self._delegate.drain()

    async def shutdown(self) -> None:
        return await self._delegate.shutdown()

    @staticmethod
    def _operator_step(spec: Any) -> int:
        step = getattr(spec, "operator_step", None)
        if step is None:
            raise ValueError("traced user callable execution requires operator_step.")
        return int(step)

    @staticmethod
    def _operator_rank(spec: Any) -> int:
        scope = getattr(spec, "object_scope", None)
        if scope is None:
            raise ValueError(
                "traced user callable execution requires an SPMDObjectScope."
            )
        return int(scope.rank)
