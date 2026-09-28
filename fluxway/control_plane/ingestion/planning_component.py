"""Ordered source observations and canonical checkpoint planning."""

from __future__ import annotations

import asyncio
import copy
import functools
import multiprocessing
import threading
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, TypeVar

from fluxway.common.ingestion import LayoutUnitDescriptor, SourceLayoutActivation
from fluxway.common.io_lane import IOLane
from fluxway.common.ingestion.segment import _id
from fluxway.common.metadata import metadata_digest
from fluxway.common.types import CheckpointCut
from fluxway.control_plane.components import ComponentState
from fluxway.control_plane.ingestion.adapter import (
    CatalogViewObservation,
    IngestionSourceAdapter,
)
from fluxway.control_plane.ingestion.compiler import (
    AlgebraicLayoutCompiler,
    CompiledIngestionLayout,
    TailLayoutMode,
    LayoutCompileRequest,
    AssignmentPolicy,
    ReaderTopologyDescriptor,
    _request_from_dict,
    _request_to_dict,
    compile_layout_request,
)
from fluxway.common.ingestion import WorkerLayoutDescriptor
from fluxway.control_plane.logical_work_ledger import (
    LogicalWorkEpoch,
    LogicalWorkLedger,
)

_Result = TypeVar("_Result")


@dataclass(frozen=True, slots=True)
class IngestionPlacementContext:
    """Lazily binds attested physical reader placement to a logical request."""

    resolve_reader_node_ids: Callable[[], tuple[str, ...]]

    def __post_init__(self) -> None:
        if not callable(self.resolve_reader_node_ids):
            raise TypeError("ingestion placement resolver must be callable")

    def bind_request(self, request: LayoutCompileRequest) -> LayoutCompileRequest:
        if not isinstance(request, LayoutCompileRequest):
            raise TypeError("ingestion placement requires LayoutCompileRequest")
        if not request.assignment_policy.node_affinity_hint:
            return request
        node_ids = self.resolve_reader_node_ids()
        if not isinstance(node_ids, tuple):
            raise TypeError("resolved reader node ids must be a tuple")
        return replace(
            request,
            topology=request.topology.bind_node_ids(node_ids),
        )


class IngestionLayoutPlanningComponent:
    def __init__(
        self, component_id: str, *, topology: ReaderTopologyDescriptor | None = None
    ) -> None:
        self.component_id = _id(component_id, "layout planning component id")
        self._topology = topology
        self._windows: dict[int, CompiledIngestionLayout] = {}
        self._pending_units: tuple[LayoutUnitDescriptor, ...] = ()
        self._cursor: dict[str, Any] = {}
        self._observation_identity: tuple[str, str] | None = None
        self._accept_lock = threading.Lock()
        self._window_lock = asyncio.Lock()
        self._publication_lock = asyncio.Lock()
        self._adapter_lane = IOLane(
            f"fluxway-source-adapter-{self.component_id}", capacity=1
        )
        self._lifecycle_lane = IOLane(
            f"fluxway-source-lifecycle-{self.component_id}", capacity=1
        )
        self._compile_executor: ProcessPoolExecutor | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._operations: set[asyncio.Task[Any]] = set()
        self._operation_failures: dict[asyncio.Task[Any], BaseException] = {}

    @property
    def layout_digest(self) -> str | None:
        return (
            None
            if not self._windows
            else self._windows[max(self._windows)].layout_digest
        )

    @property
    def windows(self) -> tuple[CompiledIngestionLayout, ...]:
        return tuple(self._windows.values())

    def set_reader_topology(self, topology: ReaderTopologyDescriptor) -> None:
        if not isinstance(topology, ReaderTopologyDescriptor):
            raise TypeError("source planning requires ReaderTopologyDescriptor")
        if self._windows:
            raise RuntimeError(
                "reader topology must be set before compiling source windows"
            )
        self._topology = topology

    async def on_state(self, state: ComponentState) -> None:
        if state is None:
            raise TypeError("layout planning component state is required")

    def _compile_pool(self) -> ProcessPoolExecutor:
        if self._compile_executor is None:
            self._compile_executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._compile_executor

    def _submit_operation(
        self, operation: Callable[[], Coroutine[Any, Any, _Result]]
    ) -> asyncio.Task[_Result]:
        if self._closed:
            raise RuntimeError("layout planning component is closed")
        task = asyncio.create_task(operation())
        self._operations.add(task)
        task.add_done_callback(self._operation_finished)
        return task

    def _operation_finished(self, task: asyncio.Task[Any]) -> None:
        self._operations.discard(task)
        if task.cancelled():
            self._operation_failures[task] = asyncio.CancelledError()
            return
        error = task.exception()
        if error is not None:
            self._operation_failures[task] = error

    async def _await_operation(
        self, operation: Callable[[], Coroutine[Any, Any, _Result]]
    ) -> _Result:
        task = self._submit_operation(operation)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            raise
        except BaseException:
            self._operation_failures.pop(task)
            raise

    async def close(self) -> None:
        if self._close_task is None:
            self._closed = True
            self._close_task = asyncio.create_task(self._finish_close())
            self._close_task.add_done_callback(self._observe_close)
        await asyncio.shield(self._close_task)

    @staticmethod
    def _observe_close(task: asyncio.Task[None]) -> None:
        if not task.cancelled():
            task.exception()

    async def _finish_close(self) -> None:
        if self._operations:
            await asyncio.gather(*tuple(self._operations), return_exceptions=True)
        compile_executor = self._compile_executor
        self._compile_executor = None
        failures = list(self._operation_failures.values())
        try:
            if compile_executor is not None:
                await self._lifecycle_lane.call(
                    compile_executor.shutdown,
                    wait=True,
                    cancel_futures=True,
                )
        except BaseException as error:
            failures.append(error)
        try:
            await self._adapter_lane.close()
        except BaseException as error:
            failures.append(error)
        try:
            await self._lifecycle_lane.close()
        except BaseException as error:
            failures.append(error)
        if failures:
            raise BaseExceptionGroup("source planning close failed", failures)

    async def _compile_prepared(
        self, request: LayoutCompileRequest, *, epoch: int, start_position: int
    ) -> CompiledIngestionLayout:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._compile_pool(),
            functools.partial(compile_layout_request, request, epoch, start_position),
        )

    async def _build_ingestion_plan(
        self,
        source: Any,
        *,
        spmd_id: str,
        operator_name: str,
        config: Any,
        placement_context: IngestionPlacementContext | None = None,
    ) -> LayoutCompileRequest:
        build = getattr(source, "build_ingestion_plan", None)
        if not callable(build):
            raise TypeError("source descriptor must provide build_ingestion_plan")
        request = await self._adapter_lane.call(
            build,
            spmd_id=spmd_id,
            operator_name=operator_name,
            config=config,
        )
        if not isinstance(request, LayoutCompileRequest):
            raise TypeError("source descriptor must return LayoutCompileRequest")
        if placement_context is not None:
            request = placement_context.bind_request(request)
        return request

    async def build_ingestion_plan(
        self,
        source: Any,
        *,
        spmd_id: str,
        operator_name: str,
        config: Any,
        placement_context: IngestionPlacementContext | None = None,
    ) -> LayoutCompileRequest:
        return await self._await_operation(
            functools.partial(
                self._build_ingestion_plan,
                source,
                spmd_id=spmd_id,
                operator_name=operator_name,
                config=config,
                placement_context=placement_context,
            )
        )

    def accept(
        self, request: LayoutCompileRequest, *, epoch: int = 0, start_step: int = 0
    ) -> CompiledIngestionLayout:
        with self._accept_lock:
            prior = self._windows.get(epoch)
            if prior is not None:
                if (
                    prior.request != request
                    or prior.work_epoch.start_step != start_step
                ):
                    raise ValueError(
                        "layout planning component received conflicting request"
                    )
                return prior
            previous = self._windows[max(self._windows)] if self._windows else None
            if epoch != len(self._windows) or start_step != (
                0 if previous is None else previous.work_epoch.end_step
            ):
                raise ValueError("layout planning epochs and steps must be contiguous")
            if previous is not None:
                if previous.work_epoch.terminal:
                    raise ValueError(
                        "layout planning cannot append after terminal window"
                    )
                if (
                    previous.request.source_identity,
                    previous.request.schema_digest,
                    previous.request.topology.global_cardinality,
                ) != (
                    request.source_identity,
                    request.schema_digest,
                    request.topology.global_cardinality,
                ):
                    raise ValueError(
                        "layout planning source, schema or logical batch changed"
                    )
            compiled = AlgebraicLayoutCompiler().compile(
                request, epoch=epoch, start_step=start_step
            )
            self._windows[epoch] = compiled
            self._topology = request.topology
            return compiled

    def logical_work_epoch(self, *, epoch: int, start_step: int) -> LogicalWorkEpoch:
        plan = self._windows[epoch].work_epoch
        if plan.start_step != start_step:
            raise ValueError("logical work start step differs from planned window")
        return plan

    def _validate_prepared(
        self,
        request: LayoutCompileRequest,
        compiled: CompiledIngestionLayout,
        *,
        epoch: int,
        start_position: int,
    ) -> None:
        with self._accept_lock:
            prior = self._windows.get(epoch)
            if prior is not None:
                if prior != compiled or prior.request != request:
                    raise ValueError(
                        "layout publication conflicts with existing window"
                    )
                return
            previous = self._windows[max(self._windows)] if self._windows else None
            if epoch != len(self._windows) or start_position != (
                0 if previous is None else previous.work_epoch.end_step
            ):
                raise ValueError("layout planning epochs and steps must be contiguous")
            if previous is not None:
                if previous.work_epoch.terminal:
                    raise ValueError(
                        "layout planning cannot append after terminal window"
                    )
                if (
                    previous.request.source_identity,
                    previous.request.schema_digest,
                    previous.request.topology.global_cardinality,
                ) != (
                    request.source_identity,
                    request.schema_digest,
                    request.topology.global_cardinality,
                ):
                    raise ValueError(
                        "layout planning source, schema or logical batch changed"
                    )
            if (
                compiled.request != request
                or compiled.work_epoch.start_step != start_position
            ):
                raise ValueError("compiled layout does not match publication request")

    def _install_prepared(
        self,
        request: LayoutCompileRequest,
        compiled: CompiledIngestionLayout,
        *,
        pending_units: tuple[LayoutUnitDescriptor, ...] | None = None,
        cursor: Mapping[str, Any] | None = None,
        observation_identity: tuple[str, str] | None = None,
    ) -> None:
        with self._accept_lock:
            self._windows[compiled.work_epoch.epoch] = compiled
            self._topology = request.topology
            if pending_units is not None:
                self._pending_units = pending_units
            if cursor is not None:
                self._cursor = copy.deepcopy(dict(cursor))
            if observation_identity is not None:
                self._observation_identity = observation_identity

    async def _plan_and_publish(
        self,
        ledger: LogicalWorkLedger,
        request: LayoutCompileRequest,
        *,
        epoch: int,
        start_position: int,
        cursor: Mapping[str, Any] | None = None,
        observation_identity: tuple[str, str] | None = None,
        max_units: int | None = None,
        max_descriptor_bytes: int | None = None,
    ) -> SourceLayoutActivation:
        if not isinstance(ledger, LogicalWorkLedger):
            raise TypeError("layout publication requires LogicalWorkLedger")

        async with self._publication_lock:
            compiled = await self._compile_prepared(
                request, epoch=epoch, start_position=start_position
            )
            if max_units is not None and len(request.units) > max_units:
                raise ValueError("source planning unit bound exceeded")
            if max_descriptor_bytes is not None:
                encoded_size = sum(
                    len(value.encoded) for value in compiled.encoded_worker_layouts()
                )
                if encoded_size > max_descriptor_bytes:
                    raise ValueError("source planning descriptor bound exceeded")
            self._validate_prepared(
                request, compiled, epoch=epoch, start_position=start_position
            )
            activation = compiled.to_source_layout_activation(
                epoch=epoch,
                start_position=start_position,
                end_position=compiled.work_epoch.end_step,
                terminal=compiled.work_epoch.terminal,
            )
            prepared = ledger.prepare_publication(
                compiled.work_epoch,
                tuple(
                    WorkerLayoutDescriptor.decode(value)
                    for value in activation.descriptors
                ),
            )
            # All validation and compilation are complete. This owner-loop section
            # intentionally contains no await: planner and ledger become visible
            # as one publication image.
            self._install_prepared(
                request,
                compiled,
                pending_units=compiled.pending_units,
                cursor=cursor,
                observation_identity=observation_identity,
            )
            ledger.apply_publication(prepared)
            return activation

    async def plan_and_publish(
        self,
        ledger: LogicalWorkLedger,
        request: LayoutCompileRequest,
        *,
        epoch: int,
        start_position: int,
        cursor: Mapping[str, Any] | None = None,
        observation_identity: tuple[str, str] | None = None,
        max_units: int | None = None,
        max_descriptor_bytes: int | None = None,
    ) -> SourceLayoutActivation:
        return await self._await_operation(
            functools.partial(
                self._plan_and_publish,
                ledger,
                request,
                epoch=epoch,
                start_position=start_position,
                cursor=cursor,
                observation_identity=observation_identity,
                max_units=max_units,
                max_descriptor_bytes=max_descriptor_bytes,
            )
        )

    async def _plan_adapter_window(
        self,
        adapter: IngestionSourceAdapter,
        ledger: LogicalWorkLedger,
        cursor: Mapping[str, Any],
        *,
        cardinality_limit: int,
        topology: ReaderTopologyDescriptor,
        tail_layout_mode: TailLayoutMode | str,
        assignment_policy: AssignmentPolicy,
        epoch: int,
        start_position: int,
    ) -> SourceLayoutActivation | None:
        if not isinstance(adapter, IngestionSourceAdapter):
            raise TypeError("adapter does not implement IngestionSourceAdapter")
        if not isinstance(cursor, Mapping):
            raise TypeError("adapter cursor must be a mapping")
        if (
            not isinstance(cardinality_limit, int)
            or isinstance(cardinality_limit, bool)
            or cardinality_limit <= 0
        ):
            raise ValueError("adapter cardinality_limit must be positive")
        async with self._window_lock:
            if self._observation_identity is not None and dict(cursor) != dict(
                adapter.decode_cursor(self._cursor)
            ):
                raise ValueError("source observation cursor is stale")
            observation = await self._adapter_lane.call(
                adapter.capture_view,
                dict(cursor),
                cardinality_limit=cardinality_limit,
            )
            if not isinstance(observation, CatalogViewObservation):
                raise TypeError("adapter must return CatalogViewObservation")
            if sum(unit.cardinality for unit in observation.units) > cardinality_limit:
                raise ValueError("adapter capture_view exceeded its cardinality_limit")
            identity = (observation.source_identity, observation.schema_digest)
            if (
                self._observation_identity is not None
                and self._observation_identity != identity
            ):
                raise ValueError("source observation identity or schema changed")
            requested_mode = TailLayoutMode.from_value(tail_layout_mode)
            effective_mode = (
                TailLayoutMode.DISCARD_REMAINDER
                if observation.terminal
                and requested_mode is TailLayoutMode.RETAIN_PENDING
                else requested_mode
            )
            request = observation.layout_request(
                topology=topology,
                tail_layout_mode=effective_mode,
                assignment_policy=assignment_policy,
            )
            units = self._pending_units + observation.units
            request = replace(request, units=units)
            next_cursor = dict(adapter.encode_cursor(observation.cursor))
            if (
                not observation.terminal
                and request.cardinality < topology.global_cardinality
            ):
                self._pending_units = units
                self._cursor = next_cursor
                self._observation_identity = identity
                self._topology = topology
                return None
            activation = await self._plan_and_publish(
                ledger,
                request,
                epoch=epoch,
                start_position=start_position,
                cursor=next_cursor,
                observation_identity=identity,
            )
            return activation

    async def plan_adapter_window(
        self,
        adapter: IngestionSourceAdapter,
        ledger: LogicalWorkLedger,
        cursor: Mapping[str, Any],
        *,
        cardinality_limit: int,
        topology: ReaderTopologyDescriptor,
        tail_layout_mode: TailLayoutMode | str,
        assignment_policy: AssignmentPolicy,
        epoch: int,
        start_position: int,
    ) -> SourceLayoutActivation | None:
        return await self._await_operation(
            functools.partial(
                self._plan_adapter_window,
                adapter,
                ledger,
                cursor,
                cardinality_limit=cardinality_limit,
                topology=topology,
                tail_layout_mode=tail_layout_mode,
                assignment_policy=assignment_policy,
                epoch=epoch,
                start_position=start_position,
            )
        )

    def checkpoint_state(self, cut: CheckpointCut) -> Mapping[str, Any]:
        if not isinstance(cut, CheckpointCut) or cut.position is None:
            raise ValueError("layout checkpoint requires a positioned cut")
        with self._accept_lock:
            windows = []
            for compiled in self._windows.values():
                request = _request_to_dict(compiled.request)
                request.pop("topology")
                windows.append(
                    {
                        "epoch": compiled.work_epoch.epoch,
                        "start_step": compiled.work_epoch.start_step,
                        "global_cardinality": compiled.request.topology.global_cardinality,
                        "request": request,
                        "plan_digest": compiled.work_epoch.digest,
                    }
                )
            image = {
                "windows": windows,
                "pending_units": [asdict(unit) for unit in self._pending_units],
                "cursor": copy.deepcopy(self._cursor),
                "observation_identity": None
                if self._observation_identity is None
                else list(self._observation_identity),
            }
        return {"version": 2, "image": image, "digest": metadata_digest(image)}

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        if not isinstance(cut, CheckpointCut) or cut.position is None:
            raise ValueError("layout restore requires a positioned cut")
        if (
            not isinstance(state, Mapping)
            or set(state) != {"version", "image", "digest"}
            or state["version"] != 2
        ):
            raise ValueError("layout planning checkpoint fields do not match")
        image = state["image"]
        if metadata_digest(image) != state["digest"]:
            raise ValueError("layout planning checkpoint digest is invalid")
        if not isinstance(image, Mapping) or set(image) != {
            "windows",
            "pending_units",
            "cursor",
            "observation_identity",
        }:
            raise ValueError("layout planning checkpoint image fields do not match")
        windows: dict[int, CompiledIngestionLayout] = {}
        for value in image["windows"]:
            if self._topology is None:
                raise RuntimeError(
                    "source restore requires the current reader topology"
                )
            if value["global_cardinality"] != self._topology.global_cardinality:
                raise ValueError(
                    "source restore cannot change logical batch cardinality"
                )
            request = _request_from_dict(
                dict(value["request"], topology=asdict(self._topology))
            )
            compiled = AlgebraicLayoutCompiler().compile(
                request, epoch=value["epoch"], start_step=value["start_step"]
            )
            if compiled.work_epoch.digest != value["plan_digest"]:
                raise ValueError("restored source plan digest does not match")
            if value["epoch"] != len(windows) or value["start_step"] != (
                0 if not windows else windows[max(windows)].work_epoch.end_step
            ):
                raise ValueError("restored source windows are not contiguous")
            windows[value["epoch"]] = compiled
        pending = tuple(
            LayoutUnitDescriptor(**value) for value in image["pending_units"]
        )
        cursor = copy.deepcopy(dict(image["cursor"]))
        identity = (
            None
            if image["observation_identity"] is None
            else tuple(image["observation_identity"])
        )
        with self._accept_lock:
            self._windows = windows
            self._pending_units = pending
            self._cursor = cursor
            self._observation_identity = identity
