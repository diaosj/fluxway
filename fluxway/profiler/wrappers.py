from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any

from fluxway.common.types.tensor import TransportTensor
from fluxway.profiler.clock import AxonNativeHlcBridge, RayFluxwayHlcBridge
from fluxway.profiler.record import TraceCarrier
from fluxway.profiler.recorder import TraceRecorder
from fluxway.profiler.user_api import activate_user_profiler
from fluxway.rpc.protocol import AxonDataPlane, DataPlaneCallable, RayActorDataPlane
from fluxway.transport.manifest import FRAME_EDGE_COUNT, FRAME_PAYLOAD_COUNT, FRAME_STEP


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

    async def send_step_frames(self, **kwargs: Any) -> Any:
        async with self._recorder.data_plane_send_step_frames(
            rank=int(kwargs.get("rank", 0)),
            exec_step=int(kwargs.get("exec_step", 0)),
        ):
            kwargs["profiler_recorder"] = self._recorder
            return await self._delegate.send_step_frames(**kwargs)


class TracingWorkerMailbox:
    def __init__(self, delegate: Any, recorder: TraceRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    def set_ready_callback(self, callback: Any) -> None:
        def traced_callback(op_name: str, step: int) -> None:
            self._recorder.mailbox_ready_fire(op_name=op_name, step=step)
            callback(op_name, step)

        self._delegate.set_ready_callback(traced_callback)

    def declare_step(
        self,
        op_name: str,
        step: int,
        expected_tensors: dict[int, int],
        expected_states: list[str],
        barrier_config: Any = None,
    ) -> None:
        self._recorder.mailbox_declare_step(
            op_name=op_name,
            step=step,
            tensor_count=len(expected_tensors),
            state_count=len(expected_states),
        )
        self._delegate.declare_step(
            op_name,
            step,
            expected_tensors,
            expected_states,
            barrier_config,
        )

    def configure_transport_expected_counts(
        self, op_name: str, step: int, expected_counts: dict[int, int]
    ) -> None:
        self._recorder.mailbox_expected_count_update(
            op_name=op_name,
            step=step,
            tensor_count=len(expected_counts),
        )
        self._delegate.configure_transport_expected_counts(
            op_name,
            step,
            expected_counts,
        )

    async def deposit_control_arguments(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> None:
        async with self._recorder.mailbox_deposit_control(
            op_name=op_name,
            step=step,
            field_count=len(names),
        ):
            await self._delegate.deposit_control_arguments(op_name, step, names, values)

    async def deposit_transport_frame(
        self,
        op_name: str,
        sender_id: str,
        frame: Any,
    ) -> int:
        step = int(frame.frame_i32[FRAME_STEP])
        async with self._recorder.mailbox_deposit_frame(
            op_name=op_name,
            step=step,
            edge_count=int(frame.frame_i32[FRAME_EDGE_COUNT]),
            payload_count=int(frame.frame_i32[FRAME_PAYLOAD_COUNT]),
        ):
            return await self._delegate.deposit_transport_frame(
                op_name,
                sender_id,
                frame,
            )

    async def deposit_local_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        sender_id: str,
        rank: int,
        tensor: Any,
    ) -> int:
        if not isinstance(tensor, TransportTensor):
            raise TypeError(
                "TracingWorkerMailbox.deposit_local_tensor requires a TransportTensor, "
                f"got {type(tensor).__name__}."
            )
        expected_count = tensor.routing_entity_count
        async with self._recorder.mailbox_deposit_local_tensor(
            op_name=op_name,
            tuuid=tuuid,
            step=step,
            data_count=expected_count,
        ):
            data_count = await self._delegate.deposit_local_tensor(
                op_name,
                tuuid,
                step,
                sender_id,
                rank,
                tensor,
            )
        if data_count != expected_count:
            raise ValueError(
                "TracingWorkerMailbox local tensor data_count mismatch: "
                f"op={op_name} tuuid={tuuid} step={step} "
                f"expected={expected_count} actual={data_count}."
            )
        return data_count

    async def clear_step(self, op_name: str, step: int) -> None:
        async with self._recorder.mailbox_clear_step(op_name=op_name, step=step):
            await self._delegate.clear_step(op_name, step)

    async def wait_step_ready(self, op_name: str, step: int) -> None:
        async with self._recorder.mailbox_wait_step_ready(op_name=op_name, step=step):
            await self._delegate.wait_step_ready(op_name, step)

    def is_step_ready(self, op_name: str, step: int) -> bool:
        return self._delegate.is_step_ready(op_name, step)

    async def get_all_control_arguments(
        self, op_name: str, step: int
    ) -> dict[str, Any]:
        return await self._delegate.get_all_control_arguments(op_name, step)

    async def get_all_tensor_shard_views(
        self, op_name: str, step: int
    ) -> dict[int, Any]:
        return await self._delegate.get_all_tensor_shard_views(op_name, step)

    async def get_tensor_shard_views(
        self, op_name: str, tuuid: int, step: int
    ) -> dict[Any, Any]:
        return await self._delegate.get_tensor_shard_views(op_name, tuuid, step)

    async def get_tensor_arrivals(
        self, op_name: str, tuuid: int, step: int
    ) -> dict[Any, Any]:
        return await self._delegate.get_tensor_arrivals(op_name, tuuid, step)

    async def get_tensor_shard_views_and_arrivals(
        self, op_name: str, tuuid: int, step: int
    ) -> tuple[int, dict[Any, Any], dict[Any, Any]]:
        return await self._delegate.get_tensor_shard_views_and_arrivals(
            op_name, tuuid, step
        )

    async def get_tensor_shard_views_and_arrivals_many(
        self,
        op_name: str,
        tuuids: list[int] | tuple[int, ...],
        step: int,
    ) -> list[tuple[int, dict[Any, Any], dict[Any, Any]]]:
        return await self._delegate.get_tensor_shard_views_and_arrivals_many(
            op_name, tuuids, step
        )

    async def snapshot(self) -> dict[str, Any]:
        return await self._delegate.snapshot()

    async def get_size(self) -> int:
        return await self._delegate.get_size()

    async def get_capacity(self) -> int:
        return await self._delegate.get_capacity()


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
        lane = self._delegate._lane_for(spec)
        execution_mode = str(getattr(lane, "value", lane))
        operator_facts = getattr(spec, "operator_facts", None)
        if operator_facts is None:
            raise ValueError("traced user callable execution requires operator_facts.")
        canonical_facts = operator_facts.operator
        operator_name = getattr(spec, "operator_name", None)
        if operator_name is not None and operator_name != canonical_facts.op_id:
            raise ValueError("operator_name does not match graph_manifest.")
        op_name = canonical_facts.op_id
        step = self._operator_step(spec)
        rank = self._operator_rank(spec)
        async with self._recorder.user_callable_execute(
            manifest_id=operator_facts.manifest_id,
            operator_facts=canonical_facts,
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
                return await self._execute_selected_mode(spec, execution_mode)

    def actor_owned_operator_names(self) -> set[str | None]:
        return self._delegate.actor_owned_operator_names()

    async def drain(self) -> None:
        return await self._delegate.drain()

    async def export_profiler_snapshots(self) -> tuple[Any, ...]:
        return await self._delegate.export_profiler_snapshots()

    async def trigger_snapshot_begin(self, spmd_dir: str) -> None:
        async with self._recorder.user_callable_snapshot_hook(
            hook_name="on_snapshot_begin",
            operator_count=len(self._delegate._ray_actors),
        ):
            return await self._delegate.trigger_snapshot_begin(spmd_dir)

    async def trigger_snapshot_commit(
        self, spmd_dir: str, operator_states: dict[str, Any]
    ) -> None:
        async with self._recorder.user_callable_snapshot_hook(
            hook_name="on_snapshot_commit",
            operator_count=len(operator_states),
        ):
            return await self._delegate.trigger_snapshot_commit(
                spmd_dir,
                operator_states,
            )

    async def save_state(self, ckpt_dir: str) -> dict[str, Any]:
        async with self._recorder.user_callable_snapshot_hook(
            hook_name="save_state",
            operator_count=len(self._delegate._ray_actors),
        ):
            return await self._delegate.save_state(ckpt_dir)

    async def restore_state(self, states: dict[str, Any], ckpt_dir: str) -> None:
        async with self._recorder.user_callable_snapshot_hook(
            hook_name="restore_state",
            operator_count=len(states),
        ):
            return await self._delegate.restore_state(states, ckpt_dir)

    async def _execute_selected_mode(self, spec: Any, execution_mode: str) -> Any:
        if execution_mode == "inline_async":
            return await spec.fn()
        if execution_mode == "inline_sync":
            return self._delegate._execute_sync_sampled(spec)
        if execution_mode == "worker_process":
            return await self._delegate._execute_worker_process(spec)
        if execution_mode == "local_free_thread":
            return await self._delegate._execute_local_free_thread(spec)
        if execution_mode == "ray_execution_actor":
            return await self._delegate._execute_ray_actor(spec)
        if execution_mode == "native_backend":
            raise NotImplementedError(
                "native_backend user callable execution mode is not installed."
            )
        raise ValueError(
            f"unsupported user callable execution mode {execution_mode!r}."
        )

    @staticmethod
    def _operator_step(spec: Any) -> int:
        step = getattr(spec, "operator_step", None)
        if step is None:
            raise ValueError("traced user callable execution requires operator_step.")
        return int(step)

    @staticmethod
    def _operator_rank(spec: Any) -> int:
        rank = getattr(spec, "operator_rank", None)
        if rank is None:
            raise ValueError("traced user callable execution requires operator_rank.")
        return int(rank)
