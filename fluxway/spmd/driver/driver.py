"""
SPMD Driver: Master controller for an SPMD instance.

Based on ARCHITECTURE.md Section 3.1

The SPMD Driver coordinates all workers within the SPMD:
- Starts and manages worker actors using placement group
- Handles worker heartbeats and monitors health
- Synchronizes worker steps within the SPMD
- Reports SPMD-level progress to Controller
- Handles control_argument broadcasting
- Manages op-level blocking
"""

import asyncio
from collections import defaultdict
from collections.abc import Iterable, Mapping
import copy
import threading
import time
import pathlib
from typing import Any

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fluxway.common.types import (
    DriverEvent,
    DriverHeartbeat,
    DriverState,
    DriverStatus,
    FlowControlState,
    HeartbeatConfig,
    ReaderState,
    SPMDConfig,
    WorkerHeartbeat,
)
from fluxway.common.utils import ensure_logging_configured
from fluxway.control_plane.checkpointer.checkpoint import _validate_manifest_pure_data
from fluxway.rpc.protocol import DataPlaneCallable
from fluxway.spmd.driver.credit import (
    CreditEdge,
    CreditUpdate,
    CreditWindow,
    UpstreamCreditState,
)
from fluxway.spmd.driver.state_machine import DriverStateMachine
from fluxway.spmd.driver.synchronizer import SPMDSynchronizer, MaxSkewSyncPolicy
from fluxway.spmd.driver.monitor import SPMDMonitor
from fluxway.spmd.worker import SPMDWorker
from fluxway.profiler import ProfilerRuntimeFactory
from fluxway.profiler.recorder import DRIVER_REGISTER_TOPOLOGY
from fluxway.placement import PlacementAttestationError
from fluxway.placement.worker_group import attest_node_ids
from fluxway.placement.worker_group import ray_actor_options

from fluxway.common.utils.logging import get_logger

logger = get_logger(__name__)


@ray.remote  # type: ignore[reportPossiblyUnboundVariable]
class SPMDDriver:  # pyright: ignore[reportRedeclaration]
    """
    SPMD Driver (Ray Actor).

    Coordinates workers within a single SPMD instance.
    Started by SPMDManager, manages workers using placement group.
    """

    def __init__(
        self,
        spmd_id: str,
        config: SPMDConfig,
        placement_group: Any | None,
        worker_bundle_indices: list[int],
        controller_handle: Any | None,
        graph_topology: dict[str, Any],
        worker_resource: Mapping[str, Any],
        expected_node_ids: list[str],
        checkpoint_backend: Any | None = None,
    ):
        """
        Initialize SPMD Driver.

        Args:
            spmd_id: SPMD identifier
            config: SPMD configuration
            placement_group: Ray PlacementGroup for worker allocation
            worker_bundle_indices: Bundle indices for workers [1, 2, ..., n]
            controller_handle: Reference to Controller (for callbacks)
            graph_topology: Dict with 'predecessors' and 'successors' lists
            checkpoint_backend: Per-worker checkpoint backend instance (WorkerCheckpointBackend)
        """
        # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
        ensure_logging_configured()

        self.spmd_id = spmd_id
        self.config = config
        self.placement_group = placement_group
        self.worker_bundle_indices = worker_bundle_indices
        self.controller_handle = controller_handle
        self.graph_topology = graph_topology
        self._checkpoint_backend = checkpoint_backend
        self._runtime_env = config.runtime_env
        self.worker_resource = worker_resource
        self.expected_node_ids = expected_node_ids
        self._attested_node_ids: tuple[str, ...] | None = None

        # State machine
        self.state_machine = DriverStateMachine(self)

        self.spmd_synchronizer = SPMDSynchronizer(
            spmd_id=spmd_id,
            num_workers=config.num_workers,
            sync_policy=MaxSkewSyncPolicy(config.max_worker_step_skew),
        )
        self.spmd_synchronizer.register_expected_ops(
            list(config.require_active_operator_names())
        )

        self.monitor = SPMDMonitor(spmd_id)

        # Worker handles
        self._worker_handles: dict[int, Any] = {}

        # Heartbeat configuration
        self._heartbeat_config = HeartbeatConfig()
        self._heartbeat_task: asyncio.Task | None = None
        self._heartbeat_running = False
        self._heartbeat_thread: threading.Thread | None = None

        # Worker heartbeat tracking
        self._worker_heartbeats: dict[int, WorkerHeartbeat] = {}
        self._worker_heartbeat_times: dict[int, float] = {}

        # Workers ready/running state tracking
        self._workers_ready: bool = False
        self._workers_running: bool = False
        self._stopping: bool = False

        # Init timeout (seconds) — if workers don't report within this time,
        # INIT_FAILED is triggered
        self._init_timeout: float = 120.0
        self._init_start_time: float = 0.0

        # Backpressure / Pause control
        self._is_paused = False
        self._source_state: ReaderState | None = ReaderState.INITIALIZING
        self._credit_epoch = 0
        self._upstream_credit_state = UpstreamCreditState((), self._credit_epoch)
        self._upstream_credit_edges_by_id: dict[str, CreditEdge] = {}
        self._downstream_credit_edges_by_target_op: dict[str, list[CreditEdge]] = (
            defaultdict(list)
        )
        self._upstream_driver_handles: dict[str, Any] = {}
        self._controller_credit_windows_by_op: dict[str, int] = {}
        self._peer_credit_hints_by_op: dict[str, int] = {}
        self._credit_sequence_by_edge_id: dict[str, int] = {}
        self._last_sent_allowed_by_edge_id: dict[str, int] = {}

        # Pending snapshot state
        self._pending_snapshot_id: str | None = None
        self._pending_target_step_count: int | None = None
        self._pending_ckpt_dir: str | None = None
        profiler_config = getattr(config, "profiler_config", None)
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name=f"driver/{self.spmd_id}",
        )
        self._profile_heartbeat = self._profiler_runtime.heartbeat_diagnostics_enabled()

        logger.debug(
            f"Driver initialized spmd_id={spmd_id} num_workers={config.num_workers}"
        )

    # =====================================================================
    # Initialization
    # =====================================================================

    async def initialize(self, checkpoint_id: str | None = None) -> None:
        """
        Initialize the driver and start all workers.

        Event-driven design:
        - Workers are started but INIT_COMPLETE is NOT triggered here.
        - INIT_COMPLETE is triggered when all workers report heartbeat
          (detected in receive_worker_heartbeat).
        - Heartbeat to controller starts immediately so controller
          can track this driver's state.
        - An init timeout is scheduled: if workers don't report within
          the timeout, INIT_FAILED is triggered.

        Args:
            checkpoint_id: Optional checkpoint to restore from

        Returns:
            True if workers were started successfully
        """
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None:
            span = profiler_recorder.driver_initialize(
                worker_count=self.config.num_workers
            )
            await span.__aenter__()
        else:
            span = None
        try:
            logger.debug(f"Initializing SPMD spmd_id={self.spmd_id}")

            # Set up worker synchronizer callbacks
            self.spmd_synchronizer.set_callbacks(
                on_spmd_op_completed=self._on_spmd_op_completed,
                block_worker_op=self._block_worker_op,
                unblock_worker_op=self._unblock_worker_op,
            )

            # Start all worker actors
            await self._start_workers()

            if checkpoint_id is not None:
                raise NotImplementedError(
                    "Checkpoint restore must go through Controller.restore_snapshot(). "
                    f"checkpoint_id={checkpoint_id} spmd_id={self.spmd_id}"
                )

            # Record init start time for timeout detection
            self._init_start_time = time.time()

            # Start heartbeat to controller immediately (state=INITIALIZING)
            # so controller can see this driver exists
            self._start_heartbeat()

            # NOTE: INIT_COMPLETE is NOT triggered here.
            # It will be triggered in receive_worker_heartbeat() when
            # all workers have reported their first heartbeat.

            logger.debug(
                f"Workers started waiting for heartbeats spmd_id={self.spmd_id}"
            )
        except Exception as e:
            logger.error(f"Initialization failed error={e} spmd_id={self.spmd_id}")
            self.state_machine.trigger(DriverEvent.INIT_FAILED, error=str(e))
            raise
        finally:
            if span is not None:
                await span.__aexit__(None, None, None)

    def _build_worker_opts(self, bundle_idx: int, name: str) -> dict[str, Any]:
        """Build .options() kwargs for a worker actor."""
        opts: dict[str, Any] = {"name": name}
        if self.placement_group is not None:
            opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(  # type: ignore[reportPossiblyUnboundVariable]
                placement_group=self.placement_group,
                placement_group_bundle_index=bundle_idx,
                placement_group_capture_child_tasks=False,
            )
        if self._runtime_env:
            opts["runtime_env"] = self._runtime_env

        opts.update(ray_actor_options(self.worker_resource))
        logger.debug(f"Worker opts for {name}: {opts}")
        return opts

    def get_attested_node_ids(self) -> tuple[str, ...]:
        if self._attested_node_ids is None:
            raise PlacementAttestationError(
                f"SPMD {self.spmd_id!r} has not attested worker node IDs"
            )
        return self._attested_node_ids

    async def _start_workers(self) -> None:
        """Start all worker actors."""
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None:
            span = profiler_recorder.driver_start_workers(
                worker_count=self.config.num_workers
            )
            await span.__aenter__()
        else:
            span = None

        # Get self handle for registering with workers
        try:
            self_handle = ray.get_runtime_context().current_actor  # type: ignore

            for rank, bundle_idx in enumerate(self.worker_bundle_indices):
                opts = self._build_worker_opts(
                    bundle_idx, f"spmd_worker_{self.spmd_id}_{rank}"
                )
                worker = SPMDWorker.options(**opts).remote(  # type: ignore[attr-defined]
                    config=self.config,
                    rank=rank,
                    spmd_id=self.spmd_id,
                    driver_handle=self_handle,
                    checkpoint_backend=self._checkpoint_backend,
                )

                self._worker_handles[rank] = worker
                logger.debug(f"Started worker rank={rank} spmd_id={self.spmd_id}")

            # Collect each worker's node_id concurrently without blocking the
            # event loop.  ray.get(  # type: ignore[reportPossiblyUnboundVariable]) on an async actor would freeze the actor
            # until all RPCs return, preventing heartbeat processing and
            # incoming RPC handling during the wait.
            node_id_refs = {
                rank: worker.get_node_id.remote()
                for rank, worker in self._worker_handles.items()
            }
            ranks = list(node_id_refs.keys())
            node_id_values = await asyncio.gather(*node_id_refs.values())
            node_ids = dict(zip(ranks, node_id_values, strict=True))
            actual = tuple(node_ids[rank] for rank in range(len(node_ids)))
            self._attested_node_ids = actual
            expected = tuple(self.expected_node_ids)
            attest_node_ids(self.spmd_id, expected, actual)
            logger.info(
                "placement.worker.attested spmd_id=%s node_ids=%s",
                self.spmd_id,
                actual,
            )
            rank_info = self._compute_rank_info(node_ids)

            # Gather IP of rank 0 to act as rendezvous address
            if 0 in self._worker_handles:
                rank0_ip = await self._worker_handles[0].get_node_ip.remote()
                if rank_info[0]["num_nodes"] == 1:
                    rank0_ip = "127.0.0.1"

                # All ranks must enter `setup_and_start` (and therefore the
                # operator `__init__` invoked by `_pre_instantiate_operators`)
                max_retries = 5
                for attempt in range(max_retries):
                    rank0_port = await self._worker_handles[0].find_free_port.remote()
                    rank0_address = f"{rank0_ip}:{rank0_port}"
                    try:
                        await asyncio.gather(
                            *[
                                worker.setup_and_start.remote(
                                    **rank_info[rank],
                                    rank0_address=rank0_address,
                                )
                                for rank, worker in self._worker_handles.items()
                            ]
                        )
                        break
                    except Exception as e:
                        if (
                            "already in use" in str(e).lower()
                            or "address already in use" in str(e).lower()
                            or "eaddrinuse" in str(e).lower()
                        ):
                            if attempt < max_retries - 1:
                                await asyncio.sleep(0.1 * (2**attempt))
                                continue
                        raise
            else:
                rank0_address = None
                # No rank 0 in this SPMD group (degenerate); still launch any
                # remaining workers concurrently.
                if self._worker_handles:
                    await asyncio.gather(
                        *[
                            worker.setup_and_start.remote(
                                **rank_info[rank],
                                rank0_address=rank0_address,
                            )
                            for rank, worker in self._worker_handles.items()
                        ]
                    )

            logger.debug(
                f"All workers started worker_count={len(self._worker_handles)} spmd_id={self.spmd_id}"
            )
        finally:
            if span is not None:
                await span.__aexit__(None, None, None)

    def _compute_rank_info(self, node_ids: dict[int, str]) -> dict[int, dict[str, int]]:
        node_groups: dict[str, list[int]] = defaultdict(list)
        for rank, node_id in node_ids.items():
            node_groups[node_id].append(rank)

        for ranks in node_groups.values():
            ranks.sort()

        sorted_nodes = sorted(node_groups, key=lambda node_id: node_groups[node_id][0])
        node_to_rank = {
            node_id: node_rank for node_rank, node_id in enumerate(sorted_nodes)
        }

        return {
            rank: {
                "local_rank": local_rank,
                "local_world_size": len(ranks),
                "node_rank": node_to_rank[node_id],
                "num_nodes": len(sorted_nodes),
            }
            for node_id, ranks in node_groups.items()
            for local_rank, rank in enumerate(ranks)
        }

    def get_worker_handles(self) -> dict[int, Any]:
        """Get all worker handles for topology registration."""
        return dict(self._worker_handles)

    async def inspect_operator_state(self) -> dict[int, Any]:
        tasks = {
            rank: worker.inspect_operator_state.remote()
            for rank, worker in self._worker_handles.items()
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

    async def export_profiler_snapshots(self) -> dict[Any, Any]:
        tasks = {
            rank: worker.export_profiler_snapshot.remote()
            for rank, worker in self._worker_handles.items()
        }
        snapshots: dict[Any, Any] = {}
        if self._profiler_runtime.enabled:
            snapshots["driver"] = self._profiler_runtime.snapshot()
        if not tasks:
            return {key: value for key, value in snapshots.items() if value is not None}
        results = await asyncio.gather(*tasks.values())
        snapshots.update(
            {
                rank: snapshot
                for rank, snapshot in zip(tasks.keys(), results, strict=True)
                if snapshot is not None
            }
        )
        return snapshots

    def register_topology(
        self,
        predecessor_handlers: dict[
            str, dict[int, "DataPlaneCallable"]
        ],  # {spmd_id: {rank: handle}}
        successor_handlers: dict[
            str, dict[int, "DataPlaneCallable"]
        ],  # {spmd_id: {rank: handle}}
        predecessor_count: int,
        successor_count: int,
        local_spmd_code: int,
        spmd_id_by_code: dict[int, str],
    ) -> None:
        """
        Register topology with provided worker handles.

        Called by Controller when all SPMDs are ready.
        Worker handles are provided directly to avoid actor re-entrancy deadlocks.

        Args:
            predecessor_handlers: {spmd_id: {rank: worker_handle}}
            successor_handlers: {spmd_id: {rank: worker_handle}}
            predecessor_count: expected predecessor count for workers
            successor_count: expected successor count for workers
        """
        logger.debug(
            f"Topology registration starting "
            f"predecessor_count={predecessor_count} successor_count={successor_count} "
            f"spmd_id={self.spmd_id}"
        )
        if self._profiler_runtime.enabled:
            self._profiler_runtime.recorder.worker_scheduler_instant(
                spec=DRIVER_REGISTER_TOPOLOGY,
                payload=(predecessor_count, successor_count),
            )

        # Step 1: Notify workers of expected topology counts
        for worker in self._worker_handles.values():
            worker.set_expected_topology.remote(predecessor_count, successor_count)
            worker.register_spmd_code_map.remote(local_spmd_code, spmd_id_by_code)

        # Step 2: Register predecessor worker handles
        for spmd_id, handlers in predecessor_handlers.items():
            for worker in self._worker_handles.values():
                worker.register_predecessor_handlers.remote(spmd_id, handlers)
            logger.debug(
                f"Registered predecessor predecessor_spmd_id={spmd_id} spmd_id={self.spmd_id}"
            )

        # Step 3: Register successor worker handles
        for spmd_id, handlers in successor_handlers.items():
            for worker in self._worker_handles.values():
                worker.register_successor_handlers.remote(spmd_id, handlers)
            logger.debug(
                f"Registered successor successor_spmd_id={spmd_id} spmd_id={self.spmd_id}"
            )

        logger.debug(f"Topology registration completed spmd_id={self.spmd_id}")

    def register_credit_topology(
        self,
        upstream_driver_handles: dict[str, Any],
        upstream_edges: tuple[CreditEdge, ...],
        downstream_edges: tuple[CreditEdge, ...],
        epoch: int = 0,
    ) -> None:
        self._credit_epoch = epoch
        self._upstream_driver_handles = dict(upstream_driver_handles)
        self._upstream_credit_state = UpstreamCreditState(
            tuple(upstream_edges), self._credit_epoch
        )
        self._upstream_credit_edges_by_id = {
            edge.edge_id: edge for edge in upstream_edges
        }
        self._downstream_credit_edges_by_target_op = defaultdict(list)
        self._controller_credit_windows_by_op = {}
        self._peer_credit_hints_by_op = {}
        self._credit_sequence_by_edge_id = {}
        self._last_sent_allowed_by_edge_id = {}

        for edge in downstream_edges:
            handle = self._upstream_driver_handles.get(edge.source_spmd_id)
            if handle is None:
                raise ValueError(
                    f"Missing upstream driver handle for credit edge {edge.edge_id!r} "
                    f"on driver {self.spmd_id!r}."
                )
            self._downstream_credit_edges_by_target_op[edge.target_op].append(edge)
            self._credit_sequence_by_edge_id[edge.edge_id] = 0
            self._last_sent_allowed_by_edge_id[edge.edge_id] = edge.max_inflight_steps

        initial_allowed_steps = self._upstream_credit_state.allowed_steps()
        self._peer_credit_hints_by_op = dict(initial_allowed_steps)
        self._controller_credit_windows_by_op = dict(initial_allowed_steps)
        self._apply_effective_credit_windows(initial_allowed_steps)

    def apply_credit_window(self, windows: tuple[CreditWindow, ...]) -> None:
        changed_ops: set[str] = set()
        for window in windows:
            if window.epoch != self._credit_epoch:
                raise ValueError(
                    f"Credit window epoch mismatch for {self.spmd_id}.{window.op_name}: "
                    f"expected={self._credit_epoch}, got={window.epoch}."
                )
            allowed_step = window.allowed_step
            if allowed_step < 0:
                raise ValueError(
                    f"Credit window cannot be negative for {self.spmd_id}.{window.op_name}: "
                    f"{allowed_step}."
                )
            if window.op_name not in self._peer_credit_hints_by_op:
                raise ValueError(
                    f"Driver {self.spmd_id!r} received controller credit window "
                    f"for unknown upstream op {window.op_name!r}."
                )
            current = self._controller_credit_windows_by_op.get(window.op_name)
            if current is not None and allowed_step < current:
                raise ValueError(
                    f"Controller credit window cannot decrease for "
                    f"{self.spmd_id}.{window.op_name}: current={current}, "
                    f"got={allowed_step}."
                )
            if current != allowed_step:
                self._controller_credit_windows_by_op[window.op_name] = allowed_step
                changed_ops.add(window.op_name)
        self._apply_effective_credit_windows(changed_ops)

    def receive_credit_update(
        self, downstream_id: str, edge_id: str, update: CreditUpdate
    ) -> None:
        edge = self._upstream_credit_edges_by_id.get(edge_id)
        if edge is None:
            raise ValueError(
                f"Driver {self.spmd_id!r} received unknown credit edge {edge_id!r}."
            )
        if edge.target_spmd_id != downstream_id:
            raise ValueError(
                f"Credit edge {edge_id!r} downstream mismatch: "
                f"expected={edge.target_spmd_id!r}, got={downstream_id!r}."
            )
        if self._upstream_credit_state.receive(edge_id, update):
            self._peer_credit_hints_by_op[edge.source_op] = (
                self._upstream_credit_state.allowed_step(edge.source_op)
            )
            self._apply_effective_credit_windows((edge.source_op,))

    def _apply_effective_credit_windows(self, op_names: Iterable[str]) -> None:
        for op_name in op_names:
            controller_allowed = self._controller_credit_windows_by_op.get(op_name)
            peer_allowed = self._peer_credit_hints_by_op.get(op_name)
            if controller_allowed is None:
                raise RuntimeError(
                    f"Driver {self.spmd_id!r} has no controller credit window "
                    f"for op {op_name!r}."
                )
            if peer_allowed is None:
                raise RuntimeError(
                    f"Driver {self.spmd_id!r} has no peer credit hint "
                    f"for op {op_name!r}."
                )
            self.spmd_synchronizer.update_allowed_step(
                op_name, min(controller_allowed, peer_allowed)
            )

    async def stop(self, save_checkpoint: bool = True) -> None:
        """Stop execution."""
        logger.debug(f"Stopping spmd_id={self.spmd_id}")
        self._stopping = True

        # Stop heartbeat
        await self._stop_heartbeat()

        # Stop all workers
        stop_refs: list[tuple[int, Any]] = []
        for rank, worker in self._worker_handles.items():
            try:
                stop_refs.append((rank, worker.stop.remote()))
            except Exception as e:
                logger.error(
                    f"Failed to stop worker rank={rank} error={e} spmd_id={self.spmd_id}"
                )

        worker_handles = list(self._worker_handles.items())

        if stop_refs:
            results = await asyncio.gather(
                *(ref for _, ref in stop_refs), return_exceptions=True
            )
            for (rank, _), result in zip(stop_refs, results, strict=True):
                if isinstance(result, Exception):
                    logger.error(
                        f"Worker stop RPC failed rank={rank} error={result} "
                        f"spmd_id={self.spmd_id}"
                    )

        detach_refs: list[tuple[int, Any]] = []
        for rank, worker in worker_handles:
            try:
                detach_refs.append((rank, worker.set_driver_handle.remote(None)))
            except Exception as e:
                logger.error(
                    f"Failed to detach worker rank={rank} error={e} spmd_id={self.spmd_id}"
                )

        if detach_refs:
            detach_results = await asyncio.gather(
                *(ref for _, ref in detach_refs), return_exceptions=True
            )
            for (rank, _), result in zip(detach_refs, detach_results, strict=True):
                if isinstance(result, Exception):
                    logger.error(
                        f"Worker detach RPC failed rank={rank} error={result} "
                        f"spmd_id={self.spmd_id}"
                    )

        self._worker_handles.clear()

    # =====================================================================
    # Snapshot Orchestration (Drain → Snapshot → Resume)
    # =====================================================================

    async def request_drain(self, target_count: int, reason: str = "") -> None:
        if self.state_machine.state != DriverState.RUNNING:
            raise RuntimeError(
                f"request_drain requires RUNNING state, "
                f"got {self.state_machine.state} spmd_id={self.spmd_id}"
            )
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_drain(
                target_step=target_count,
                worker_count=len(self._worker_handles),
            ):
                await self._request_drain_unprofiled(target_count)
            return
        await self._request_drain_unprofiled(target_count)

    async def _request_drain_unprofiled(self, target_count: int) -> None:
        self._pending_target_step_count = target_count
        self.state_machine.trigger(DriverEvent.DRAIN_REQUESTED)
        await asyncio.gather(
            *[
                worker.set_drain_target_count.remote(target_count)
                for worker in self._worker_handles.values()
            ]
        )
        self.spmd_synchronizer.drain_to_step_count(
            target_count=target_count, on_drained=self._on_drain_complete
        )

    async def request_snapshot(
        self, snapshot_id: str, target_count: int, ckpt_dir: str
    ) -> None:
        if self.state_machine.state != DriverState.DRAINED:
            raise RuntimeError(
                f"request_snapshot requires DRAINED state, "
                f"got {self.state_machine.state} spmd_id={self.spmd_id}"
            )
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_snapshot(
                snapshot_id=snapshot_id,
                target_step=target_count,
                worker_count=len(self._worker_handles),
            ):
                await self._request_snapshot_unprofiled(
                    snapshot_id,
                    target_count,
                    ckpt_dir,
                )
            return
        await self._request_snapshot_unprofiled(snapshot_id, target_count, ckpt_dir)

    async def _request_snapshot_unprofiled(
        self, snapshot_id: str, target_count: int, ckpt_dir: str
    ) -> None:
        self.state_machine.trigger(DriverEvent.SNAPSHOT_REQUESTED)
        self._pending_snapshot_id = snapshot_id
        self._pending_target_step_count = target_count
        self._pending_ckpt_dir = ckpt_dir
        await self._persist_worker_states()

    def _on_drain_complete(self) -> None:
        if self.state_machine.state != DriverState.DRAINING:
            raise RuntimeError(
                f"_on_drain_complete requires DRAINING state, "
                f"got {self.state_machine.state} spmd_id={self.spmd_id}"
            )
        self.state_machine.trigger(DriverEvent.DRAIN_COMPLETE)
        if self.controller_handle is not None:
            self.controller_handle.report_driver_drained.remote(self.spmd_id)

    async def _persist_worker_states(self) -> None:
        if self._pending_ckpt_dir is None or self._pending_target_step_count is None:
            logger.error(
                f"Failed to persist worker states: pending_ckpt_dir or pending_target_step_count is not set. spmd_id={self.spmd_id}"
            )
            self._clear_drain_and_fail()
            return
        spmd_dir = (
            pathlib.Path(self._pending_ckpt_dir)
            / str(self._pending_target_step_count)
            / self.spmd_id
        )
        try:
            force_write = getattr(self._checkpoint_backend, "_force_write", False)
            if spmd_dir.exists() and any(spmd_dir.iterdir()):
                if not force_write:
                    raise FileExistsError(
                        f"Checkpoint directory {spmd_dir} already exists and is not empty. "
                        "Set force_write=True in CheckpointConfig to overwrite."
                    )
                else:
                    import shutil

                    shutil.rmtree(spmd_dir, ignore_errors=True)

            spmd_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(
                f"Failed to create checkpoint directory {spmd_dir} error={e} spmd_id={self.spmd_id}"
            )
            self._clear_drain_and_fail()
            return

        if self.controller_handle is None:
            logger.error(
                f"controller_handle is None, cannot report snapshot metadata "
                f"spmd_id={self.spmd_id}"
            )
            self._clear_drain_and_fail()
            return

        try:
            if 0 in self._worker_handles:
                await self._worker_handles[0].trigger_snapshot_begin.remote(
                    self._pending_ckpt_dir, self._pending_target_step_count
                )

            ranks = list(self._worker_handles.keys())
            results = await asyncio.gather(
                *(
                    self._worker_handles[r].save_state_to_disk.remote(
                        self._pending_ckpt_dir, self._pending_target_step_count
                    )
                    for r in ranks
                )
            )

            # Pair every rank with its (spmd_state, rank_storage, operator_states)
            # triple returned by save_state_to_disk.  The `spmd_state` carries
            # the logical progress (`completed_step_count`, `static_args`) that
            # is required during restore_from_disk; without forwarding it to the
            # controller the persisted manifest is empty and any subsequent
            # resume raises `'completed_step_count' missing from spmd_state`.
            # Aggregate raw operator_states across ranks for
            # trigger_snapshot_commit.  Operators like sparse_spmd need
            # the gathered metadata (e.g. lance fragment metas from all
            # shards) to finalize the committed checkpoint.
            gathered_states: dict[str, dict[int, dict]] = defaultdict(dict)
            for rank, (_, _, op_states) in zip(ranks, results, strict=True):
                for op_name, op_state in op_states.items():
                    gathered_states[op_name][rank] = op_state

            if 0 in self._worker_handles:
                await self._worker_handles[0].trigger_snapshot_commit.remote(
                    self._pending_ckpt_dir,
                    self._pending_target_step_count,
                    dict(gathered_states),
                )

            canonical_spmd_state = results[0][0]
            shared_operator_states = {}
            for op_name, per_rank_states in gathered_states.items():
                if ranks[0] not in per_rank_states:
                    continue
                candidate_state = copy.deepcopy(per_rank_states[ranks[0]])
                try:
                    _validate_manifest_pure_data(
                        candidate_state,
                        path=(
                            f"manifest.{self.spmd_id}."
                            f"spmd_state.operator_states.{op_name}"
                        ),
                    )
                except TypeError:
                    continue
                shared_operator_states[op_name] = candidate_state
            canonical_spmd_state = dict(canonical_spmd_state)
            canonical_spmd_state["operator_states"] = shared_operator_states

            rank_metadata: dict[int, tuple[dict, dict]] = {
                rank: (canonical_spmd_state, rank_storage)
                for rank, (_, rank_storage, _) in zip(ranks, results, strict=True)
            }

        except Exception as e:
            logger.error(f"Snapshot lifecycle error for spmd_id={self.spmd_id}: {e}")
            self._clear_drain_and_fail()
            return

        self.controller_handle.receive_snapshot_metadata.remote(
            self.spmd_id, self._pending_snapshot_id, rank_metadata
        )

    def _clear_drain_and_fail(self) -> None:
        """Unblock all drained Workers, then mark this Driver as failed.

        Workers self-block in _execute_operator by await-ing _drain_resume_event.
        If SNAPSHOT_FAILED fires without first clearing the drain, all Workers
        remain permanently suspended because resume() requires SNAPSHOTTING state
        and will raise in ERROR state.
        """
        self.spmd_synchronizer.clear_drain()
        for worker in self._worker_handles.values():
            worker.clear_drain_target_count.remote()
        if self.controller_handle and self._pending_snapshot_id:
            self.controller_handle.report_snapshot_failed.remote(
                self.spmd_id, self._pending_snapshot_id
            )
        self.state_machine.trigger(DriverEvent.SNAPSHOT_FAILED)

    def resume(self) -> None:
        """
        Phase 3 of snapshot: clear drain ceiling and resume training.

        Transitions: SNAPSHOTTING → RUNNING.
        Must be called by the controller after snapshot data is consumed.

        Order:
          1. Synchronizer clears the ACK fence.
          2. Fan-out clear_drain_target_count to all Workers so they resume
             executing steps that were deferred behind the target.
        """
        if self.state_machine.state not in (
            DriverState.SNAPSHOTTING,
            DriverState.DRAINED,
        ):
            raise RuntimeError(
                f"resume requires SNAPSHOTTING or DRAINED state, "
                f"got {self.state_machine.state} spmd_id={self.spmd_id}"
            )
        self.spmd_synchronizer.clear_drain()
        for worker in self._worker_handles.values():
            worker.clear_drain_target_count.remote()
        self.state_machine.trigger(DriverEvent.SNAPSHOT_COMPLETE)

    async def restore_from_disk(
        self,
        spmd_state: dict[str, Any],
        rank_storage: dict[int, dict[str, Any]],
    ) -> None:
        """
        Fan out checkpoint restore to all workers in parallel.

        Every worker in ``_worker_handles`` receives a ``load_state_from_disk``
        call.  Workers whose rank is absent from ``rank_storage`` (e.g. newly
        added data-parallel ranks) receive ``rank_storage=None`` and will inherit
        the logical progress from ``spmd_state`` without loading on-disk tensors.

        Args:
            spmd_state:   SPMD-level logical state (``completed_step_count``,
                          ``static_args``) shared by all ranks.
            rank_storage: Per-rank physical storage paths keyed by integer rank.
        """
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_restore(
                snapshot_id="restore_from_disk",
                worker_count=len(self._worker_handles),
            ):
                await self._restore_from_disk_unprofiled(spmd_state, rank_storage)
            return
        await self._restore_from_disk_unprofiled(spmd_state, rank_storage)

    async def _restore_from_disk_unprofiled(
        self,
        spmd_state: dict[str, Any],
        rank_storage: dict[int, dict[str, Any]],
    ) -> None:
        self.spmd_synchronizer.load_spmd_state(spmd_state)

        tasks = [
            worker.load_state_from_disk.remote(spmd_state, rank_storage.get(rank))
            for rank, worker in self._worker_handles.items()
        ]
        await asyncio.gather(*tasks)

    # =====================================================================
    # Worker Heartbeat Handling
    # =====================================================================

    def receive_worker_heartbeat(self, rank: int, heartbeat: WorkerHeartbeat) -> None:
        """
        Receive heartbeat from a worker.

        Args:
            rank: Worker rank
            heartbeat: Worker heartbeat data
        """
        if self._stopping:
            logger.debug(
                f"Ignoring worker heartbeat during shutdown rank={rank} spmd_id={self.spmd_id}"
            )
            return

        self._worker_heartbeats[rank] = heartbeat
        self._worker_heartbeat_times[rank] = time.time()
        if self._profile_heartbeat:
            self._profiler_runtime.recorder.driver_worker_heartbeat(
                rank=rank,
                status=heartbeat.status.name,
            )
        logger.heartbeat(
            f"Received worker heartbeat rank={rank} "
            f"status={heartbeat.status.name} spmd_id={self.spmd_id}"
        )

        # Check for worker error
        if heartbeat.error_message:
            self._handle_worker_error(rank, heartbeat.error_message)

        # Check if all workers have reported heartbeat (workers_ready)
        if not self._workers_ready:
            if len(self._worker_heartbeat_times) >= self.config.num_workers:
                self._workers_ready = True
                logger.heartbeat(
                    f"All workers ready worker_count={self.config.num_workers} spmd_id={self.spmd_id}"
                )

                # Event-driven: trigger INIT_COMPLETE now that all workers are alive
                if self.state_machine.state == DriverState.INITIALIZING:
                    self.state_machine.trigger(DriverEvent.INIT_COMPLETE)
                    logger.heartbeat(f"INIT_COMPLETE triggered spmd_id={self.spmd_id}")

        # Check if all workers are running (workers_running)
        if not self._workers_running and self._workers_ready:
            from fluxway.common.types import WorkerStatus

            all_running = all(
                hb.status in (WorkerStatus.RUNNING, WorkerStatus.BLOCKED)
                for hb in self._worker_heartbeats.values()
            )
            if all_running and len(self._worker_heartbeats) >= self.config.num_workers:
                self._workers_running = True
                logger.heartbeat(
                    f"All workers running triggering START spmd_id={self.spmd_id}"
                )
                # Trigger READY → RUNNING transition
                if self.state_machine.state == DriverState.READY:
                    self.state_machine.trigger(DriverEvent.START)

    def _handle_worker_error(self, rank: int, error: str) -> None:
        """Handle worker error."""
        logger.error(f"Worker error rank={rank} error={error} spmd_id={self.spmd_id}")

        self.state_machine.trigger(DriverEvent.WORKER_ERROR, rank=rank, error=error)

    # =====================================================================
    # Worker Op Completion Handling
    # =====================================================================

    def report_worker_op_completed(self, rank: int, op_name: str, step: int) -> None:
        """
        Receive op completion report from a worker.

        Args:
            rank: Worker rank
            op_name: Operation name
            step: Completed step number
        """
        if self._stopping:
            logger.debug(
                f"Ignoring worker op completion during shutdown rank={rank} "
                f"op={op_name} step={step} spmd_id={self.spmd_id}"
            )
            return

        if self._profiler_runtime.enabled:
            self._profiler_runtime.recorder.driver_op_completion(
                rank=rank,
                op_name=op_name,
                step=step,
            )
        self.spmd_synchronizer.report_worker_op_completed(rank, op_name, step)

    def _on_spmd_op_completed(self, spmd_id: str, op_name: str, step: int) -> None:
        if self.controller_handle is None:
            raise RuntimeError(
                f"Driver {self.spmd_id!r} cannot report op completion without "
                "a controller handle."
            )
        self.controller_handle.report_op_completed.remote(spmd_id, op_name, step)
        self._send_credit_updates(op_name, step)

    def _send_credit_updates(self, op_name: str, completed_count: int) -> None:
        edges = self._downstream_credit_edges_by_target_op.get(op_name, ())
        for edge in edges:
            allowed_step = completed_count + edge.max_inflight_steps
            last_allowed = self._last_sent_allowed_by_edge_id[edge.edge_id]
            if allowed_step <= last_allowed:
                continue
            handle = self._upstream_driver_handles.get(edge.source_spmd_id)
            if handle is None:
                raise ValueError(
                    f"Missing upstream driver handle for credit edge {edge.edge_id!r} "
                    f"on driver {self.spmd_id!r}."
                )
            sequence = self._credit_sequence_by_edge_id[edge.edge_id] + 1
            self._credit_sequence_by_edge_id[edge.edge_id] = sequence
            self._last_sent_allowed_by_edge_id[edge.edge_id] = allowed_step
            handle.receive_credit_update.remote(
                self.spmd_id,
                edge.edge_id,
                CreditUpdate(
                    epoch=self._credit_epoch,
                    sequence=sequence,
                    allowed_step=allowed_step,
                ),
            )

    # =====================================================================
    # Op Blocking
    # =====================================================================

    def block_op(self, op_name: str) -> None:
        """Block an op for all workers (called by Controller)."""
        self.spmd_synchronizer.block_spmd_op(op_name)

    def unblock_op(self, op_name: str) -> None:
        """Unblock an op for all workers (called by Controller)."""
        self.spmd_synchronizer.unblock_spmd_op(op_name)

    def block_all_ops(self) -> None:
        """Block all ops for all workers."""
        self.spmd_synchronizer.block_all_ops()
        self.state_machine.trigger(DriverEvent.BLOCK_REQUESTED)

    def unblock_all_ops(self) -> None:
        """Unblock all ops for all workers."""
        self.spmd_synchronizer.unblock_all_ops()
        self.state_machine.trigger(DriverEvent.UNBLOCK_REQUESTED)

    def _block_worker_op(self, rank: int, op_name: str, op_runnable_epoch: int) -> None:
        """Block a specific worker's op."""
        worker = self._worker_handles.get(rank)
        if worker is not None:
            try:
                worker.set_op_runnable.remote(op_name, False, op_runnable_epoch)
            except Exception as e:
                logger.error(
                    f"Failed to block worker op rank={rank} error={e} spmd_id={self.spmd_id}"
                )

    def _unblock_worker_op(
        self, rank: int, op_name: str, op_runnable_epoch: int
    ) -> None:
        """Unblock a specific worker's op."""
        worker = self._worker_handles.get(rank)
        if worker is not None:
            try:
                worker.set_op_runnable.remote(op_name, True, op_runnable_epoch)
            except Exception as e:
                logger.error(
                    f"Failed to unblock worker op rank={rank} error={e} spmd_id={self.spmd_id}"
                )

    def _block_all_workers(self) -> None:
        """Block all workers (internal method)."""
        try:
            ops = list(self.spmd_synchronizer.get_spmd_steps().keys())
            for op_name in ops:
                self.spmd_synchronizer.block_spmd_op(op_name)
        except Exception as e:
            logger.error(f"Failed to block workers error={e} spmd_id={self.spmd_id}")

    # =====================================================================
    # State Argument Broadcasting
    # =====================================================================

    def broadcast_control_argument(
        self, target_op: str, names: list[str], step: int, values: list[Any]
    ) -> None:
        """
        Broadcast control_arguments to all workers via receive_control.

        Args:
            target_op: Target operator type on the receiving workers
            names: List of state variable names
            step: Current step number
            values: List of state values
        """
        for rank, worker in self._worker_handles.items():
            try:
                worker.receive_control.remote(target_op, step, names, values)
                logger.debug(
                    f"Broadcast control_arguments names={names} step={step} "
                    f"target_rank={rank} spmd_id={self.spmd_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to send control_arguments to worker rank={rank} error={e} "
                    f"spmd_id={self.spmd_id}"
                )

        logger.debug(
            f"Broadcast control_arguments to all workers names={names} step={step} "
            f"spmd_id={self.spmd_id}"
        )

    def routing_control_argument_to_controller(
        self,
        target_spmd_id: str,
        target_op: str,
        names: list[str],
        step: int,
        values: list[Any],
    ) -> None:
        """
        Route control_arguments to controller (called by rank0 worker).

        Args:
            target_spmd_id: Target SPMD ID
            target_op: Target operator type on the receiving side
            names: State variable names
            step: Current step number
            values: State values
        """
        if self._stopping:
            logger.debug(
                f"Ignoring control_argument routing during shutdown target_spmd_id={target_spmd_id} "
                f"step={step} spmd_id={self.spmd_id}"
            )
            return

        if self.controller_handle is not None:
            try:
                self.controller_handle.receive_control_argument.remote(
                    self.spmd_id, target_spmd_id, target_op, names, step, values
                )
                logger.debug(
                    f"Routed control_arguments to controller names={names} step={step} "
                    f"spmd_id={self.spmd_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to route control_arguments to controller error={e} "
                    f"spmd_id={self.spmd_id}"
                )

    # =====================================================================
    # Heartbeat Reporting to Controller
    # =====================================================================

    def _start_heartbeat(self) -> None:
        """Start heartbeat reporting to controller."""
        if self._heartbeat_running:
            return

        self._heartbeat_running = True
        # Use a thread for synchronous heartbeat loop
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    async def _stop_heartbeat(self) -> None:
        """Stop heartbeat reporting.

        The thread join is executed via asyncio.to_thread so that the
        driver's event loop remains responsive while waiting for the
        heartbeat thread to finish its current sleep interval.  Without
        this, a sequential shutdown of N drivers on the controller side
        would accumulate up to N × (heartbeat_interval + 1 s) of blocking
        time on the event loop.
        """
        self._heartbeat_running = False
        heartbeat_thread = self._heartbeat_thread
        if (
            heartbeat_thread is not None
            and heartbeat_thread.is_alive()
            and heartbeat_thread is not threading.current_thread()
        ):
            timeout = self._heartbeat_config.heartbeat_interval + 1.0
            if self._profile_heartbeat:
                async with self._profiler_runtime.recorder.driver_heartbeat_thread_join(
                    timeout_ms=int(timeout * 1000),
                ):
                    await asyncio.to_thread(heartbeat_thread.join, timeout)
            else:
                await asyncio.to_thread(heartbeat_thread.join, timeout)
            if heartbeat_thread.is_alive():
                logger.warning(
                    f"Heartbeat thread did not terminate within timeout spmd_id={self.spmd_id}"
                )
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """Heartbeat reporting loop.

        Also checks for init timeout: if the driver is still in
        INITIALIZING state after _init_timeout seconds, triggers INIT_FAILED.
        """
        while self._heartbeat_running:
            try:
                self._send_heartbeat()

                # Check init timeout
                if (
                    self.state_machine.state == DriverState.INITIALIZING
                    and self._init_start_time > 0
                ):
                    elapsed = time.time() - self._init_start_time
                    if elapsed > self._init_timeout:
                        logger.error(
                            f"Worker init timeout spmd_id={self.spmd_id} "
                            f"elapsed={elapsed:.1f}s "
                            f"reported_workers={len(self._worker_heartbeat_times)}/{self.config.num_workers}"
                        )
                        self.state_machine.trigger(
                            DriverEvent.INIT_FAILED,
                            error=f"Worker init timeout after {elapsed:.1f}s",
                        )
            except Exception as e:
                logger.error(
                    f"Error sending heartbeat error={e} spmd_id={self.spmd_id}"
                )

            time.sleep(self._heartbeat_config.heartbeat_interval)

    def _send_heartbeat(self) -> None:
        """Send heartbeat to controller."""
        if self.controller_handle is None:
            return

        # Count healthy workers
        healthy_count = 0
        current_time = time.time()
        for rank in range(self.config.num_workers):
            last_time = self._worker_heartbeat_times.get(rank, 0)
            if current_time - last_time < self._heartbeat_config.heartbeat_timeout:
                healthy_count += 1

        # Determine status
        if self.state_machine.is_error():
            status = DriverStatus.ERROR
        elif healthy_count < self.config.num_workers:
            status = DriverStatus.WARNING
        else:
            status = DriverStatus.HEALTHY

        heartbeat = DriverHeartbeat(
            spmd_id=self.spmd_id,
            timestamp=time.time(),
            status=status,
            state=self.state_machine.state,
            error_message=self.state_machine.get_error_message(),
            num_workers=self.config.num_workers,
            healthy_num_workers=healthy_count,
            workers_ready=self._workers_ready,
            workers_running=self._workers_running,
        )

        try:
            self.controller_handle.receive_driver_heartbeat.remote(heartbeat)
            logger.heartbeat(
                f"Driver heartbeat sent status={status.name} "
                f"state={heartbeat.state.name} "
                f"healthy_workers={healthy_count}/{self.config.num_workers} "
                f"spmd_id={self.spmd_id}"
            )
        except Exception as e:
            logger.error(f"Failed to send heartbeat error={e} spmd_id={self.spmd_id}")

    # =====================================================================
    # Controller Reporting
    # =====================================================================

    def _report_completion(self) -> None:
        """Report completion to controller."""
        if self.controller_handle is not None:
            try:
                self.controller_handle.report_spmd_completed.remote(self.spmd_id)
            except Exception as e:
                logger.error(
                    f"Failed to report completion error={e} spmd_id={self.spmd_id}"
                )

    def _report_error(self, error: str) -> None:
        """Report error to controller."""
        if self.controller_handle is not None:
            try:
                self.controller_handle.report_spmd_error.remote(self.spmd_id, error)
            except Exception as e:
                logger.error(f"Failed to report error error={e} spmd_id={self.spmd_id}")

    # =====================================================================
    # Topology Update
    # =====================================================================

    def update_topology(self, predecessors: list[str], successors: list[str]) -> None:
        """
        Update topology information.

        Args:
            predecessors: List of predecessor SPMD IDs
            successors: List of successor SPMD IDs
        """
        self.graph_topology["predecessors"] = predecessors
        self.graph_topology["successors"] = successors
        logger.debug(
            f"Topology updated predecessors={predecessors} successors={successors} "
            f"spmd_id={self.spmd_id}"
        )

    # =====================================================================
    # Status Query
    # =====================================================================

    def get_state(self) -> DriverState:
        """Get current driver state."""
        return self.state_machine.state

    def get_status(self) -> dict[str, Any]:
        """Get detailed driver status."""
        status = {
            "spmd_id": self.spmd_id,
            "state": self.state_machine.state.value,
            "node_type": "spmd",  # Generic node type
            "num_workers": self.config.num_workers,
            "completed_step_count": self.spmd_synchronizer.get_spmd_steps(),
            "locked_ops": self.spmd_synchronizer.get_locked_ops(),
            "error_workers": self.state_machine.get_error_workers(),
        }

        return status

    # =====================================================================
    # Source Node Control
    # =====================================================================

    def pause_sources(self) -> None:
        """
        Pause all source operators (called by Controller due to backpressure).
        """
        if self._is_paused:
            return

        self._is_paused = True
        self._source_state = ReaderState.PAUSED
        self._report_flow_control_state()
        logger.debug(f"Sources paused due to backpressure spmd_id={self.spmd_id}")

        for worker in self._worker_handles.values():
            try:
                worker.pause_sources.remote()
            except Exception as e:
                logger.error(f"Failed to pause worker error={e} spmd_id={self.spmd_id}")

    def resume_sources(self) -> None:
        """
        Resume all source operators (called by Controller when backpressure relieved).
        """
        if not self._is_paused:
            return

        self._is_paused = False
        self._source_state = ReaderState.READING
        self._report_flow_control_state()
        logger.debug(f"Sources resumed spmd_id={self.spmd_id}")

        for worker in self._worker_handles.values():
            try:
                worker.resume_sources.remote()
            except Exception as e:
                logger.error(
                    f"Failed to resume worker error={e} spmd_id={self.spmd_id}"
                )

    async def start_source_operators(self) -> None:
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.driver_source_start(
                worker_count=len(self._worker_handles),
            ):
                await self._start_source_operators_unprofiled()
            return
        await self._start_source_operators_unprofiled()

    async def _start_source_operators_unprofiled(self) -> None:
        self._source_state = ReaderState.READING
        self._report_flow_control_state()
        tasks = [
            worker.start_source_operators.remote()
            for worker in self._worker_handles.values()
        ]
        if tasks:
            await asyncio.gather(*tasks)

    def _report_flow_control_state(self) -> None:
        if self.controller_handle is None:
            raise RuntimeError(
                f"Driver {self.spmd_id!r} cannot report flow-control state "
                "without a controller handle."
            )
        self.controller_handle.report_flow_control_state.remote(
            FlowControlState(
                spmd_id=self.spmd_id,
                timestamp=time.time(),
                paused=self._is_paused,
                epoch=self._credit_epoch,
            )
        )

    def get_source_state(self) -> ReaderState | None:
        """Get current source state."""
        return self._source_state
