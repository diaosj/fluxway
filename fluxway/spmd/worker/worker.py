"""
SPMD Worker: Ray Actor for computation execution.

Based on ARCHITECTURE.md Section 3.2

Each SPMD Worker runs user-defined forward/backward/update logic
and implements dual-stream convergence with mailbox-based execution.

Key responsibilities:
1. Provide handler registration interface (handlers only, topology from config)
2. Provide data reception interface
3. Execute operators from priority queue
4. Send outputs to successors using computed routing
5. Provide op-level blocking (block one op without affecting others)
6. Report op completion to driver
7. Handle control_argument reception and propagate to successors

This implementation uses Ray's async actor API for concurrency.
See: https://docs.ray.io/en/latest/ray-core/actors/async_api.html
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass
import os
import socket
import time
import inspect
from pathlib import Path


import ray

from fluxway.common.tensor_backend import (
    get_backend,
    force_writable,
    ensure_cpu_serializable,
)
from fluxway.common.decorators import (
    operator_context,
    get_current_context,
    get_instance_storage,
    _resolve_owner_class,
    _inject_context_properties,
    _STATIC_CONTEXT_ATTR_NAMES,
    _is_worker_process_required,
    _validate_worker_process_marker,
)
from fluxway.common.execution import compile
from fluxway.common.types import (
    SPMDConfig,
    WorkerHeartbeat,
    WorkerStatus,
    WorkerTelemetry,
)
from fluxway.common.types.configs import MailboxStatus
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.common.types.sharding import (
    ShardingSpec,
    PhysicalLayout,
)
from fluxway.common.index_descriptor import IndexDescriptor
from fluxway.transport.manifest import (
    EDGE_EXPECTED_COUNT,
    EDGE_TUUID,
    EmptyKind,
    FRAME_EDGE_COUNT,
    FRAME_SENDER_SPMD_CODE,
    FRAME_STEP,
    PayloadRole,
    TransportFrame,
    TransportFrameBuilder,
    TransportFrameHeader,
)
from fluxway.transport.kernels import FrameTableCapacity
from fluxway.common.utils import ensure_logging_configured
from fluxway.common.utils.logging import get_logger
from fluxway.common.utils.tree import tree_flatten, tree_unflatten
from fluxway.rpc.protocol import DataPlaneCallable
from fluxway.spmd.worker.mailbox import WorkerMailbox
from fluxway.spmd.worker.data_plane import SuccessorRouteContext
from fluxway.spmd.worker.data_plane_executor import RayDataPlaneBackendExecutor
from fluxway.spmd.worker.local_plan import (
    LocalStepPlan,
    LocalStepPlanExecutor,
    MappingCallableRegistry,
)
from fluxway.spmd.worker.user_callable_executor import UserCallableSpec
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionStatus,
    ExecutionResult,
)
from typing import Any, Optional, get_type_hints, cast


@compile(static_argnums=())
def _jit_sort_fused(values, origin_indices, routing_indices, receipt_tags):
    backend = get_backend()
    sort_idx = backend.argsort(origin_indices)
    return (
        backend.take(values, sort_idx),
        backend.take(origin_indices, sort_idx),
        backend.take(routing_indices, sort_idx),
        backend.take(receipt_tags, sort_idx),
        sort_idx,
    )


def _pad_tensor_to_power_of_2(tensor, original_length: int, cap: int, pad_value=0):
    if cap == original_length:
        return tensor
    backend = get_backend()
    pad_shape = list(backend.shape(tensor))
    pad_shape[0] = cap - original_length
    if pad_value == 0:
        zeros = backend.zeros(tuple(pad_shape), dtype=backend.dtype(tensor))
    else:
        zeros = backend.full(tuple(pad_shape), pad_value, dtype=backend.dtype(tensor))
    return backend.cat([tensor, zeros], dim=0)


@compile(static_argnums=(1,))
def _jit_reduce_scalars(stacked, op: str):
    backend = get_backend()
    if op == "sum":
        return backend.sum(stacked)
    if op == "mean":
        return backend.mean(stacked)
    if op == "min":
        return backend.min(stacked)
    if op == "max":
        return backend.max(stacked)
    raise ValueError(f"Unsupported reduction op: {op}")


def _describe_layout_for_error(layout: ShardingSpec | PhysicalLayout | None) -> str:
    if layout is None:
        return "None"

    if isinstance(layout, ShardingSpec):
        strategy_name = type(layout.strategy).__name__
        return (
            f"ShardingSpec(strategy={strategy_name}, "
            f"content={layout.is_content_addressable}, "
            f"static={layout.is_static}, "
            f"partitions={layout.partitions}, "
            f"routing_dim={layout.routing_dim})"
        )

    if isinstance(layout, PhysicalLayout):
        strategy_name = type(layout.strategy).__name__
        return (
            f"PhysicalLayout(strategy={strategy_name}, "
            f"content={layout.is_content_addressable}, "
            f"static={layout.is_static}, "
            f"partitions={layout.partitions}, "
            f"routing_dim={layout.routing_dim}, "
            f"global_shape={layout.global_shape}, "
            f"mesh_shape={layout.mesh.shape}, "
            f"mesh_axes={layout.mesh.axis_names})"
        )

    raise TypeError(
        f"[Fluxway Protocol Error] Unsupported layout descriptor "
        f"{type(layout).__name__}; expected ShardingSpec, PhysicalLayout, or None."
    )


logger = get_logger(__name__)


@dataclass(frozen=True)
class _SuccessorSendContext:
    spmd_id: str
    info: Any
    handlers: dict[int, Any]
    effective_rank: int


@ray.remote
class SPMDWorker:
    """
    SPMD Worker (Ray Async Actor).

    Implements dual-stream convergence with priority queue execution:
    - Data is received by WorkerMailbox (async-safe)
    - Executes operators based on priority metadata
    - Supports op-level blocking without affecting other ops
    - Reports op completion to driver
    - Handles control_argument for cross-SPMD state transfer

    Uses asyncio for concurrency instead of threading.
    Ray will automatically detect this as an async actor due to async method definitions.
    """

    _step_tuuid_cache: dict[int, dict[int, Any]]
    _completed_step_count: dict[str, int]
    _pending_future_steps: dict[str, list[int]]
    _successor_handlers: dict[str, dict[int, Any]]
    _send_semaphore: "asyncio.Semaphore"
    _heartbeat_interval: float
    _last_error: str | None
    _op_runnable: dict[str, "asyncio.Event"]
    _mailbox: WorkerMailbox
    _drain_cond: "asyncio.Condition"

    @property
    def op_manager(self) -> OperatorManager:
        if self._op_manager is None:
            raise RuntimeError("OperatorManager is not initialized yet.")
        return self._op_manager

    def __init__(
        self,
        config: SPMDConfig,
        rank: int,
        spmd_id: str = "",
        driver_handle: Any | None = None,
        checkpoint_backend: Any | None = None,
    ):
        """
        Initialize worker.

        Args:
            config: SPMD configuration with user functions
            rank: This worker's rank in the SPMD
            spmd_id: ID of this SPMD (for metadata tracking)
            driver_handle: Reference to the SPMD driver for callbacks (Native Ray)
            node_type: Type of node (SPMD or READER)
            reader_task: File read task (for READER nodes)
            reader_config: Reader configuration (for READER nodes)
        """
        # Ray actors run in separate worker processes; ensure they respect
        # FLUXWAY_LOG_LEVEL / FLUXWAY_LOG_FORCE so DEBUG logs are not dropped.
        ensure_logging_configured()
        logger.debug("phase_one_init_begin spmd_id=%s rank=%s", spmd_id, rank)
        self.config = config
        self.rank = rank
        self.spmd_id = spmd_id or ""
        self._running = False

        # Pluggable per-worker checkpoint serialization. Defaults to OrbaxBackend.
        self._checkpoint_backend = checkpoint_backend

        # Driver handle for callbacks (Control Plane - Native Ray)
        self._driver_handle: Any | None = driver_handle

        # Mesh from config
        self._mesh = config.mesh
        self._world_size = config.num_workers
        self.local_rank: int | None = None
        self.local_world_size: int | None = None
        self.node_rank: int | None = None
        self.num_nodes: int | None = None
        self.rank0_address: str | None = None

        self._op_manager: OperatorManager | None = None

        # Completed step counters {op_name: count} - tracks how many steps were executed
        self._completed_step_count: dict[str, int] = dict.fromkeys(
            self.config.operators if self.config is not None else {}, 0
        )

        # Op-level blocking: asyncio.Event (set=runnable, clear=blocked)
        self._op_runnable = {
            op_name: asyncio.Event()
            for op_name in (self.config.operators if self.config is not None else {})
        }
        for event in self._op_runnable.values():
            event.set()  # Initially all ops are runnable

        # Pending tasks when blocked (to avoid priority queue reentry loop)
        self._pending_blocked: dict[str, list[int]] = {
            op_name: []
            for op_name in (self.config.operators if self.config is not None else {})
        }

        # Pending tasks for future steps (arrived early, waiting for current step to complete)
        # Structure: {op_name: List[step]} - steps waiting to be executed
        self._pending_future_steps: dict[str, list[int]] = {
            op_name: []
            for op_name in (self.config.operators if self.config is not None else {})
        }

        # Priority queue: asyncio.PriorityQueue with (priority, step, op_name)
        # Can extract priority from OperatorMeta
        self._execution_queue: asyncio.PriorityQueue[tuple[int, int, str]] = (
            asyncio.PriorityQueue()
        )

        # --- Backpressure components ---
        self._max_in_flight = 50
        self._send_semaphore = asyncio.Semaphore(self._max_in_flight)
        self._outbound_semaphore = asyncio.Semaphore(self._max_in_flight)
        self._mailbox_capacity = 100  # Default value
        self._high_watermark = int(self._mailbox_capacity * 0.8)

        # Internal Mailbox (async-safe), wired to auto-enqueue when a step is ready
        self._mailbox = WorkerMailbox()
        self._mailbox.set_ready_callback(self._on_step_ready)

        # Handler storage (Data Plane - wrapped in DataPlaneCallable)
        # {spmd_id: {rank: DataPlaneCallable}}
        self._predecessor_handlers: dict[str, dict[int, DataPlaneCallable]] = {}
        self._successor_handlers = {}
        self._local_spmd_code: int | None = None
        self._spmd_id_by_code: dict[int, str] = {}

        # Heartbeat configuration
        self._heartbeat_interval: float = 5.0
        self._heartbeat_running: bool = False
        self._heartbeat_task: asyncio.Task | None = None

        # Tracks tasks currently sitting in _execution_queue to prevent double-scheduling.
        # Removed when dequeued (whether to execute, pending_blocked, or pending_future).
        self._in_op_step_queue: set = set()

        # State tracking
        self._last_error: str | None = None

        # Run loop synchronization
        self._start_event = asyncio.Event()  # Signals when to start execution
        self._run_task: asyncio.Task | None = None
        self._background_tasks: set[asyncio.Task] = set()
        self._outbound_tasks: set[asyncio.Task] = set()
        self._data_plane_executor = RayDataPlaneBackendExecutor()
        self._local_step_plan_executor = LocalStepPlanExecutor()

        # Source operator pause control
        self._source_pause_events: dict[str, asyncio.Event] = {}

        # Drain-to-step state
        self._drain_target_count: int | None = None
        self._drain_cond: asyncio.Condition = asyncio.Condition()
        if self.config is not None and getattr(
            self.config, "requires_drain_at_startup", False
        ):
            # Block at step 0 until Controller issues the initial drain target!
            self._drain_target_count = 0

        # Expected topology counts for auto-start detection
        # Initialize to None to strictly wait for registration from driver
        self._expected_predecessor_count: int | None = None
        self._expected_successor_count: int | None = None

        # Cached assembled input tensors: {op_name: {tuuid: TransportTensor}}
        # Updated by _assemble_input, used by _send_output for pass-through metadata
        self._last_assembled_tensors: dict[str, dict[str, TransportTensor | None]] = {
            op_name: {}
            for op_name in (self.config.operators if self.config is not None else {})
        }

        # Metadata for "Mirror Sharding" (Symmetry Routing)
        # Structure: {step_id: {op_name.tuuid: TransportTensor}}
        self._step_tuuid_cache = {}

        # Tracks which (op_name, step) barriers have been registered to avoid double-declaring.
        self._barriers_declared: dict[tuple[str, int], bool] = {}

        logger.debug(
            "phase_one_init_complete spmd_id=%s rank=%s world_size=%s",
            self.spmd_id,
            self.rank,
            self._world_size,
        )

    def __del__(self):
        # Ensure background I/O is complete (e.g. Orbax AsyncCheckpointer)
        if hasattr(self, "_checkpoint_backend") and self._checkpoint_backend:
            self._checkpoint_backend.finalize()

    def _get_checkpoint_backend(self):
        if self._checkpoint_backend is None:
            from fluxway.control_plane.checkpointer import OrbaxBackend

            self._checkpoint_backend = OrbaxBackend()
        return self._checkpoint_backend

    # =========================================================================
    # Operator Pre-Instantiation
    # =========================================================================

    @staticmethod
    def _pre_instantiate_operators(
        config: "SPMDConfig",
        rank: int,
        *,
        local_rank: int | None = None,
        local_world_size: int | None = None,
        node_rank: int | None = None,
        num_nodes: int | None = None,
        rank0_address: str | None = None,
    ) -> None:
        """Eagerly instantiate all SPMD operator class instances.

        This runs inside SPMDWorker.__init__, before OperatorManager is created
        and before the Ray RPC event loop starts. Operator classes get their
        __init__ called while JAX's XLA backend is still uninitialised, which
        allows users to call jax.distributed.initialize (or any other
        per-process setup) safely from __init__ via self.rank / self.world_size.

        The static_args declared for this SPMD node are filtered by each class's
        __init__ signature and forwarded as keyword arguments, so users can
        declare custom init parameters and have them populated from static_args.
        """
        import inspect as _inspect

        world_size = config.num_workers
        static_args = config.static_args or {}

        seen: set = set()
        with operator_context(
            static_args=static_args,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            node_rank=0 if node_rank is None else node_rank,
            num_nodes=1 if num_nodes is None else num_nodes,
            rank0_address=rank0_address,
        ):
            storage, lock = get_instance_storage()
            for op_meta in (config.operator_metas or {}).values():
                if not op_meta.owner_class_qualname:
                    continue
                cls = _resolve_owner_class(
                    op_meta.owner_class_module, op_meta.owner_class_qualname
                )
                _validate_worker_process_marker(cls)
                if cls is None or id(cls) in seen:
                    continue
                seen.add(id(cls))
                if id(cls) in storage:
                    continue

                try:
                    init_sig = _inspect.signature(cls.__init__)
                    init_params = set(init_sig.parameters.keys()) - {"self"}
                    has_var_keyword = any(
                        p.kind == _inspect.Parameter.VAR_KEYWORD
                        for p in init_sig.parameters.values()
                    )
                    init_kwargs = (
                        dict(static_args)
                        if has_var_keyword
                        else {k: v for k, v in static_args.items() if k in init_params}
                    )
                except (ValueError, TypeError):
                    init_kwargs = {}

                instance = object.__new__(cls)
                _inject_context_properties(cls)
                ctx = get_current_context()
                for _attr in _STATIC_CONTEXT_ATTR_NAMES:
                    instance.__dict__[_attr] = getattr(ctx, _attr)
                logger.debug(
                    "operator_init_begin owner_class=%s rank=%s",
                    cls.__qualname__,
                    rank,
                )
                try:
                    instance.__init__(**init_kwargs)
                except Exception:
                    logger.exception(
                        "operator_init_failed owner_class=%s rank=%s world_size=%s "
                        "local_rank=%s local_world_size=%s node_rank=%s num_nodes=%s",
                        cls.__qualname__,
                        rank,
                        world_size,
                        local_rank,
                        local_world_size,
                        0 if node_rank is None else node_rank,
                        1 if num_nodes is None else num_nodes,
                    )
                    raise
                logger.debug(
                    "operator_init_complete owner_class=%s rank=%s",
                    cls.__qualname__,
                    rank,
                )
                storage[id(cls)] = instance

    # =========================================================================
    # Driver Handle Management
    # =========================================================================

    def get_node_id(self) -> str:
        """Return the current Ray node identifier for topology-aware startup."""
        return ray.get_runtime_context().get_node_id()

    def get_node_ip(self) -> str:
        """Return the current Ray node IP address for distributed rendezvous."""
        if "POD_IP" in os.environ:
            return os.environ["POD_IP"]

        try:
            # Connect to Ray's GCS server to determine the cluster-routable IP
            gcs_address = ray.get_runtime_context().gcs_address
            if gcs_address:
                head_ip, head_port = gcs_address.split(":")
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                    # Doesn't actually connect, just uses OS routing table
                    s.connect((head_ip, int(head_port)))
                    return s.getsockname()[0]
        except Exception:
            pass

        return ray.util.get_node_ip_address()

    def find_free_port(self) -> int:
        """Find an available port on the current node."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def setup_and_start(
        self,
        local_rank: int,
        local_world_size: int,
        node_rank: int,
        num_nodes: int,
        rank0_address: str | None = None,
    ) -> None:
        """Finish topology-aware startup after the driver has all worker node IDs."""
        logger.debug(
            "setup_and_start_begin spmd_id=%s rank=%s local_rank=%s "
            "local_world_size=%s node_rank=%s num_nodes=%s",
            self.spmd_id,
            self.rank,
            local_rank,
            local_world_size,
            node_rank,
            num_nodes,
        )
        try:
            self.local_rank = local_rank
            self.local_world_size = local_world_size
            self.node_rank = node_rank
            self.num_nodes = num_nodes
            self.rank0_address = rank0_address

            self._pre_instantiate_operators(
                self.config,
                self.rank,
                local_rank=local_rank,
                local_world_size=local_world_size,
                node_rank=node_rank,
                num_nodes=num_nodes,
                rank0_address=rank0_address,
            )
            logger.debug(
                "setup_and_start_preinstantiate_complete spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
            self._op_manager = OperatorManager(
                self.config,
                self.rank,
                self.spmd_id,
                local_rank=local_rank,
                local_world_size=local_world_size,
                node_rank=node_rank,
                num_nodes=num_nodes,
            )
            logger.debug(
                "setup_and_start_operator_manager_complete spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
            self._source_pause_events = {
                op_name: asyncio.Event()
                for op_name in (
                    self.config.operators if self.config is not None else {}
                )
                if self.op_manager.is_source_operator(op_name)
            }
            for event in self._source_pause_events.values():
                event.set()

            if self._driver_handle is not None:
                self._start_heartbeat()
            self._start_run_loop()
            logger.debug(
                "setup_and_start_runtime_started spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
        except Exception:
            logger.exception(
                "setup_and_start_failed spmd_id=%s rank=%s local_rank=%s "
                "local_world_size=%s node_rank=%s num_nodes=%s",
                self.spmd_id,
                self.rank,
                local_rank,
                local_world_size,
                node_rank,
                num_nodes,
            )
            raise

    def set_driver_handle(self, driver_handle: Any | None) -> None:
        """
        Set the driver handle for callbacks.

        Args:
            driver_handle: Reference to the SPMD driver
        """
        self._driver_handle = driver_handle

    def get_driver_handle(self) -> Any | None:
        """Get the driver handle."""
        return self._driver_handle

    def register_spmd_code_map(
        self, local_spmd_code: int, spmd_id_by_code: dict[int, str]
    ) -> None:
        if not isinstance(local_spmd_code, int) or local_spmd_code < 0:
            raise ValueError(
                f"local_spmd_code must be a non-negative int, got {local_spmd_code!r}."
            )
        if local_spmd_code not in spmd_id_by_code:
            raise ValueError(
                f"local_spmd_code={local_spmd_code} is missing from spmd_id_by_code."
            )
        self._local_spmd_code = local_spmd_code
        self._spmd_id_by_code = dict(spmd_id_by_code)

    # =========================================================================
    # Handler Registration (handlers only, topology from config)
    # =========================================================================

    def register_predecessor_handlers(
        self, spmd_id: str, handlers: dict[int, DataPlaneCallable]
    ) -> None:
        """
        Register handlers for a predecessor SPMD.

        Note: Topology info is already in config/OperatorMeta.
        Only handlers need runtime registration.

        Args:
            spmd_id: Predecessor SPMD ID
            handlers: {rank: DataPlaneCallable}
        """
        self._predecessor_handlers[spmd_id] = handlers
        logger.debug(
            f"Registered predecessor handlers predecessor_spmd_id={spmd_id} "
            f"ranks={list(handlers.keys())} spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Check if all handlers are registered for auto-start
        self._check_and_auto_start()

    def register_successor_handlers(
        self, spmd_id: str, handlers: dict[int, DataPlaneCallable]
    ) -> None:
        """
        Register handlers for a successor SPMD.

        Note: Topology info is already in config/OperatorMeta.
        Only handlers need runtime registration.

        Args:
            spmd_id: Successor SPMD ID
            handlers: {rank: DataPlaneCallable}
        """
        self._successor_handlers[spmd_id] = handlers
        logger.debug(
            f"Registered successor handlers successor_spmd_id={spmd_id} "
            f"ranks={list(handlers.keys())} spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Check if all handlers are registered for auto-start
        self._check_and_auto_start()

    def set_expected_topology(
        self, predecessor_count: int, successor_count: int
    ) -> None:
        """
        Set expected predecessor/successor counts for auto-start detection.

        Called by driver before topology registration begins.

        Args:
            predecessor_count: Expected number of predecessor SPMDs
            successor_count: Expected number of successor SPMDs
        """
        self._expected_predecessor_count = predecessor_count
        self._expected_successor_count = successor_count
        logger.debug(
            f"Topology expectation set predecessor_count={predecessor_count} "
            f"successor_count={successor_count} spmd_id={self.spmd_id} rank={self.rank}"
        )

        # If both are 0 (source node with no successors), auto-start immediately
        if predecessor_count == 0 and successor_count == 0:
            self._auto_start()
        else:
            # Check if handlers are already registered
            self._check_and_auto_start()

    def _check_and_auto_start(self) -> None:
        """
        Check if all expected handlers are registered and auto-start if ready.
        """
        if self._running:
            return  # Already running

        if (
            self._expected_predecessor_count is None
            or self._expected_successor_count is None
        ):
            logger.debug(
                f"Waiting for topology counts spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        # Check if all expected handlers are registered
        pred_registered = (
            len(self._predecessor_handlers) >= self._expected_predecessor_count
        )
        succ_registered = (
            len(self._successor_handlers) >= self._expected_successor_count
        )

        logger.debug(
            f"Handler registration status "
            f"pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count} "
            f"succ={len(self._successor_handlers)}/{self._expected_successor_count} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

        if pred_registered and succ_registered:
            logger.debug(
                f"All handlers registered auto-starting "
                f"pred={len(self._predecessor_handlers)}/{self._expected_predecessor_count} "
                f"succ={len(self._successor_handlers)}/{self._expected_successor_count} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            self._auto_start()

    def _auto_start(self) -> None:
        """Internal auto-start method."""
        logger.debug(
            f"_auto_start called running={self._running} spmd_id={self.spmd_id} rank={self.rank}"
        )
        if self._running:
            return

        self._running = True
        self._start_event.set()
        logger.debug(f"Worker auto-started spmd_id={self.spmd_id} rank={self.rank}")

        if self.config.auto_start_source_operators:
            self._spawn_background_task(self._trigger_source_operators())

    def _spawn_background_task(self, coro: Any) -> asyncio.Task:
        """Track detached tasks so stop() can await their shutdown."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._on_background_task_done)
        return task

    def _on_background_task_done(self, task: asyncio.Task) -> None:
        self._background_tasks.discard(task)
        if task.cancelled():
            return
        exc = task.exception()
        if exc is None:
            return
        self._last_error = f"{type(exc).__name__}: {exc}"
        logger.error(
            "Background task failed spmd_id=%s rank=%s error=%s",
            self.spmd_id,
            self.rank,
            exc,
            exc_info=(type(exc), exc, exc.__traceback__),
        )

    def _spawn_outbound_task(self, coro: Any) -> asyncio.Task:
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        self._outbound_tasks.add(task)

        def _done(t: asyncio.Task) -> None:
            self._outbound_tasks.discard(t)
            self._outbound_semaphore.release()
            self._on_background_task_done(t)

        task.add_done_callback(_done)
        return task

    async def _drain_outbound_tasks(self) -> None:
        await self._local_step_plan_executor.drain()
        await self._local_step_plan_executor.drain_user_callables()
        while True:
            tasks = [task for task in self._outbound_tasks if not task.done()]
            if not tasks:
                break
            await asyncio.gather(*tasks)
        await self._local_step_plan_executor.drain()
        await self._local_step_plan_executor.drain_user_callables()
        if self._last_error is not None:
            raise RuntimeError(f"Worker has failed background task: {self._last_error}")

    async def _trigger_source_operators(self) -> None:
        """
        Trigger initial steps for source operators (those with 0 input tensors).
        """
        source_ops = self.op_manager.get_source_operators()
        logger.debug(
            f"Source operators identified ops={source_ops} spmd_id={self.spmd_id} rank={self.rank}"
        )
        for op_name in source_ops:
            target_step = self._completed_step_count.get(op_name, 0)
            logger.info(
                f"Auto-triggering source op for step {target_step} op={op_name} spmd_id={self.spmd_id} rank={self.rank}"
            )
            await self._enqueue(op_name, target_step)

    async def start_source_operators(self) -> None:
        await self._trigger_source_operators()

    # =========================================================================
    # Op-Level Blocking
    # =========================================================================

    def block_op(self, op_name: str) -> None:
        """
        Block an operator's execution.

        When blocked, tasks of this op_name will be held in pending queue
        instead of being executed. Other ops continue normally.
        """
        logger.debug(f"Op blocked op={op_name} spmd_id={self.spmd_id} rank={self.rank}")
        self._op_runnable[op_name].clear()

    async def unblock_op(self, op_name: str) -> None:
        """
        Unblock an operator and re-enqueue pending tasks.
        """
        self._op_runnable[op_name].set()

        pending = self._pending_blocked[op_name]
        self._pending_blocked[op_name] = []
        for step in pending:
            await self._enqueue(op_name, step)

        logger.debug(
            f"Op unblocked op={op_name} spmd_id={self.spmd_id} rank={self.rank}"
        )

    def is_op_blocked(self, op_name: str) -> bool:
        """Check if an operator is blocked."""
        return not self._op_runnable[op_name].is_set()

    # =========================================================================
    # Eager Mode Support
    # =========================================================================

    async def prepare_eager_op(
        self,
        op_name: str,
        step: int,
        tuuid_sharding: dict[int, "ShardingSpec"],
        expected_counts: dict[int, int] | None = None,
    ) -> None:
        """
        Prepare for an Eager operator execution by registering its input TUUIDs.
        Called by the Driver BEFORE the task is triggered on all workers.
        """
        if self._op_manager:
            self.op_manager.register_eager_inputs(
                op_name, tuuid_sharding, expected_counts
            )

        # Ensure the step is declared in the mailbox so it can start receiving data
        self._ensure_step_declared(op_name, step)

        logger.debug(
            f"Prepared eager op op={op_name} step={step} input_count={len(tuuid_sharding)} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

    # =========================================================================
    # Data Reception
    # =========================================================================

    async def receive_control(
        self, op_name: str, step: int, names: list[str], values: list[Any]
    ) -> None:
        """
        Receive state data from Driver (control plane).

        Args:
            op_name: Operator name
            step: Step number
            names: State variable names
            values: State data values
        """
        # Normal mode: step-ordered execution
        completed_step_count = self._completed_step_count.get(op_name, 0)

        # Drop stale data (data for steps already executed)
        # Note: periodic ops don't have stale steps as they use virtual sequence
        is_periodic = (
            self.op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )
        if not is_periodic and step < completed_step_count:
            logger.warning(
                f"Dropped stale state data op={op_name} names={names} "
                f"step={step} completed_step_count={completed_step_count} spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        # Lazily declare the step barrier on first data arrival
        self._ensure_step_declared(op_name, step)

        logger.debug(
            f"Received states op={op_name} step={step} names={names} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Cache in mailbox — barrier fires on_ready automatically
        await self._mailbox.deposit_control_arguments(
            op_name, step=step, names=names, values=values
        )

    async def receive_transport_frame(
        self,
        header: TransportFrameHeader,
        *payload_iovs: Any,
    ) -> int:
        if not isinstance(header, TransportFrameHeader):
            raise TypeError(
                f"header must be TransportFrameHeader, got {type(header).__name__}."
            )
        frame = header.to_frame(payload_iovs)
        sender_spmd_code = int(frame.frame_i32[FRAME_SENDER_SPMD_CODE])
        try:
            sender_id = self._spmd_id_by_code[sender_spmd_code]
        except KeyError as exc:
            raise ValueError(
                f"[SPMDWorker] Unknown sender_spmd_code={sender_spmd_code} "
                f"spmd_id={self.spmd_id} rank={self.rank}."
            ) from exc

        watermark = 0
        op_to_edge_rows: dict[str, list[int]] = {}
        for row_idx, edge in enumerate(frame.edge_i32):
            tuuid = int(edge[EDGE_TUUID])
            for op_name in self.op_manager.get_operators_for_tuuid(tuuid):
                op_to_edge_rows.setdefault(op_name, []).append(row_idx)

        for op_name, rows in op_to_edge_rows.items():
            self._ensure_step_declared(op_name, int(frame.frame_i32[FRAME_STEP]))
            op_frame = TransportFrame(
                frame_i32=frame.frame_i32.copy(),
                edge_i32=frame.edge_i32[rows],
                payload_i32=frame.payload_i32,
                shape_i32=frame.shape_i32,
                global_shape_i64=frame.global_shape_i64,
                compact_i64=frame.compact_i64,
                payload_segments=frame.payload_segments,
            )
            op_frame.frame_i32[FRAME_EDGE_COUNT] = len(rows)
            expected_counts = {}
            for edge in op_frame.edge_i32:
                tuuid = int(edge[EDGE_TUUID])
                expected_count = int(edge[EDGE_EXPECTED_COUNT])
                if expected_count <= 0:
                    raise ValueError(
                        f"[SPMDWorker] Transport frame for op={op_name} "
                        f"step={int(frame.frame_i32[FRAME_STEP])} tuuid={tuuid} "
                        f"has invalid EDGE_EXPECTED_COUNT={expected_count}."
                    )
                previous = expected_counts.get(tuuid)
                if previous is not None and previous != expected_count:
                    raise ValueError(
                        f"[SPMDWorker] Transport frame for op={op_name} "
                        f"step={int(frame.frame_i32[FRAME_STEP])} tuuid={tuuid} "
                        f"contains conflicting EDGE_EXPECTED_COUNT values: "
                        f"{previous} and {expected_count}."
                    )
                expected_counts[tuuid] = expected_count
            self._mailbox.configure_transport_expected_counts(
                op_name, int(frame.frame_i32[FRAME_STEP]), expected_counts
            )
            watermark = await self._mailbox.deposit_transport_frame(
                op_name,
                sender_id,
                op_frame,
            )

        return watermark

    def _ensure_step_declared(self, op_name: str, step: int) -> None:
        """
        Lazily register a StepBarrier for (op_name, step) on first data arrival.

        Expected counts are read from OperatorManager so the barrier knows
        exactly how many shards/states to wait for before firing.
        """
        if self._mailbox.is_step_ready(op_name, step):
            return  # Already done (or re-fired from a previous run)
        if self._barriers_declared.get((op_name, step)):
            return  # Already declared

        expected_tensors = (
            self.op_manager.get_expected_tensor_shards(op_name)
            if self._op_manager
            else {}
        )
        expected_states = (
            self.op_manager.get_expected_control_arg_names(op_name)
            if self._op_manager
            else []
        )
        meta = (
            self.op_manager.get_operator_metadata(op_name) if self._op_manager else None
        )
        self._mailbox.declare_step(
            op_name,
            step,
            expected_tensors,
            expected_states,
            barrier_config=meta.barrier_config if meta is not None else None,
        )
        self._barriers_declared[(op_name, step)] = True

    def _on_step_ready(self, op_name: str, step: int) -> None:
        """
        Called by WorkerMailbox when a step's StepBarrier fires.

        Schedules _enqueue as an asyncio task so the lock in deposit_shards/
        deposit_states can return immediately without blocking on queue insertion.
        """
        logger.debug(
            f"Step ready op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
        )
        self._spawn_background_task(self._enqueue(op_name, step))

    async def _enqueue(self, op_name: str, step: int) -> None:
        """Enqueue a task for execution."""
        from fluxway.common.types.configs import OperatorMeta

        if not self._running:
            logger.debug(
                f"Skipping enqueue after stop op={op_name} step={step} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        key = (op_name, step)
        if key in self._in_op_step_queue:
            logger.debug(
                f"Idempotency guard: already in queue op={op_name} step={step}"
            )
            return
        self._in_op_step_queue.add(key)

        logger.debug(
            f"Enqueue called op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
        )
        meta = self.op_manager._get_operator_metadata(op_name)
        priority = meta.priority if isinstance(meta, OperatorMeta) else 0
        await self._execution_queue.put((-priority, step, op_name))
        logger.debug(
            f"Task enqueued op={op_name} step={step} priority={priority} spmd_id={self.spmd_id} rank={self.rank}"
        )

    # =========================================================================
    # Execution Logic
    # =========================================================================

    def _start_run_loop(self) -> None:
        """
        Start the run loop as an asyncio task.

        The loop will block until start() is called.
        """
        # Create and schedule the run task
        # Note: In Ray async actors, we schedule the task to run in the actor's event loop
        self._run_task = asyncio.get_event_loop().create_task(self._run())
        logger.debug(
            f"Run loop task started waiting for start signal spmd_id={self.spmd_id} rank={self.rank}"
        )

    async def _run(self) -> None:
        """
        Internal main execution loop (async).

        Blocks on PriorityQueue until tasks are available.
        Respects op-level blocking.
        """
        # Wait for start signal from driver
        logger.debug(
            f"Waiting for start signal spmd_id={self.spmd_id} rank={self.rank}"
        )
        await self._start_event.wait()

        logger.debug(f"Execution loop started spmd_id={self.spmd_id} rank={self.rank}")

        # Normal SPMD mode: data-driven execution loop
        while self._running:
            try:
                # Blocking wait for next task (async)
                logger.debug(
                    f"Waiting for next task in execution queue spmd_id={self.spmd_id} rank={self.rank}"
                )
                priority, step, op_name = await self._execution_queue.get()
                self._in_op_step_queue.discard((op_name, step))
                logger.debug(
                    f"Execution loop picked up task op={op_name} step={step} priority={priority} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )

                # Check if we should stop
                if not self._running:
                    self._execution_queue.task_done()
                    break

                if not self._op_runnable[op_name].is_set():
                    # Hold in pending queue instead of re-enqueueing
                    self._pending_blocked[op_name].append(step)
                    logger.debug(
                        f"Op blocked in queue op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
                    )
                    continue

                logger.debug(
                    f"Processing task op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
                )
                # Check if this op is in periodic_trigger mode
                is_periodic = (
                    self.op_manager.is_periodic_trigger(op_name)
                    if self._op_manager
                    else False
                )

                if not is_periodic:
                    # Normal mode: verify step matches current expected execution step
                    completed_count = self._completed_step_count[op_name]
                    if step < completed_count:
                        # Truly stale task (already executed), drop it
                        logger.warning(
                            f"Stale task dropped op={op_name} "
                            f"task_step={step} completed_count={completed_count} spmd_id={self.spmd_id} rank={self.rank}"
                        )
                        continue
                    elif step > completed_count:
                        # Future task (arrived early), hold for later execution
                        self._pending_future_steps[op_name].append(step)
                        logger.debug(
                            f"Future task held op={op_name} "
                            f"task_step={step} completed_count={completed_count} spmd_id={self.spmd_id} rank={self.rank}"
                        )
                        continue

                # Execute (for periodic mode, execute any task that comes)
                if self._last_error is not None:
                    raise RuntimeError(
                        f"Worker has failed background task: {self._last_error}"
                    )
                result = await self._execute_operator(op_name, step)

                # Guard: if a restore happened during execution (e.g. while
                # the worker was blocked in the drain self-block), the step
                # is now stale.  Skip advancement to preserve the watermark.
                if result.status is ExecutionStatus.STALE_AFTER_RESTORE:
                    continue

                # Advance execution step counter
                await self._advance_step(op_name, result=result)

            except asyncio.CancelledError:
                logger.info(
                    f"Run loop cancelled spmd_id={self.spmd_id} rank={self.rank}"
                )
                break
            except Exception as e:
                logger.error(
                    f"Error in execution loop error={e} spmd_id={self.spmd_id} rank={self.rank}"
                )

    async def stop(self) -> None:
        """Stop the execution loop gracefully."""
        self._running = False
        self._driver_handle = None
        # Wake any worker blocked on drain condition
        async with self._drain_cond:
            self._drain_target_count = None
            self._drain_cond.notify_all()
        # Put a sentinel to unblock the queue
        await self._execution_queue.put((999, 0, "__stop__"))

        pending_tasks = []
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            pending_tasks.append(self._run_task)
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            pending_tasks.append(self._heartbeat_task)
        outbound_tasks = [task for task in self._outbound_tasks if not task.done()]
        if outbound_tasks:
            await asyncio.gather(*outbound_tasks, return_exceptions=True)
        background_tasks = [task for task in self._background_tasks if not task.done()]
        for task in background_tasks:
            task.cancel()
        pending_tasks.extend(background_tasks)

        if pending_tasks:
            await asyncio.gather(*pending_tasks, return_exceptions=True)
        self._background_tasks.clear()

        logger.debug(f"Worker stopping spmd_id={self.spmd_id} rank={self.rank}")

    async def _check_all_empty(self, op_name: str, step: int) -> bool:
        """Check if all received shards for a step are empty packets."""
        expected_tuuids = self.op_manager.get_expected_tuuids(op_name)
        if not expected_tuuids:
            return False

        all_tensor_shards = await self._mailbox.get_all_tensor_shard_views(
            op_name, step
        )

        for tuuid in expected_tuuids:
            if tuuid not in all_tensor_shards:
                return False
            shards = all_tensor_shards[tuuid]
            if not shards:
                return False
            for key, tensor_list in shards.items():
                backend = get_backend()
                for tensor in tensor_list:
                    shape = backend.shape(tensor.flat_values)
                    if shape and shape[0] > 0:
                        return False
        return True

    async def _propagate_empty_packets(self, op_name: str, step: int) -> None:
        """Propagate empty packets to successor SPMDs for skipped/empty steps."""
        if self._op_manager is None:
            return

        successor_ids = self.op_manager.get_successor_ids(op_name)
        if not successor_ids:
            return

        if self._local_spmd_code is None:
            raise ValueError(
                f"[SPMDWorker] local_spmd_code is not registered "
                f"spmd_id={self.spmd_id} rank={self.rank}."
            )
        backend = get_backend()
        empty_values = backend.zeros((0,), dtype=backend.int32())

        for spmd_id in successor_ids:
            succ_info = self.op_manager.get_successor_info(op_name, spmd_id)
            if not succ_info:
                continue

            output_sharding = self.op_manager.get_output_sharding(op_name)
            effective_rank = self.rank
            dispatcher = self.op_manager.get_dispatcher(op_name)
            if getattr(dispatcher, "interleave_unit_size", None) is not None:
                effective_rank = self.rank % dispatcher.interleave_unit_size
            frame_edge_counts: dict[int, int] = defaultdict(int)
            for _target_name, tuuid in succ_info.tuuid_map.items():
                source_logical = output_sharding.get(tuuid)
                target_logical = succ_info.tuuid_sharding.get(tuuid)

                if source_logical is None or target_logical is None:
                    raise ValueError(
                        f"[SPMDWorker] Cannot resolve empty propagation routing for "
                        f"TUUID {tuuid} from {op_name} to {spmd_id}: "
                        f"source_sharding={bool(source_logical)}, "
                        f"target_sharding={bool(target_logical)}"
                    )

                source_spec = source_logical.bind(self._mesh)
                target_spec = target_logical.bind(succ_info.mesh or self._mesh)
                for target_rank in source_spec.resolve_consumers(
                    target_spec, effective_rank
                ):
                    frame_edge_counts[target_rank] += 1

            frames_by_target: dict[int, TransportFrameBuilder] = {
                target_rank: TransportFrameBuilder(
                    step=step,
                    sender_spmd_code=self._local_spmd_code,
                    sender_rank=self.rank,
                    sender_world_size=self._world_size,
                    capacity=FrameTableCapacity(
                        edge_count=edge_count,
                        payload_count=edge_count,
                        shape_count=edge_count,
                        segment_count=edge_count,
                    ),
                )
                for target_rank, edge_count in frame_edge_counts.items()
            }
            for target_name, tuuid in succ_info.tuuid_map.items():
                source_logical = output_sharding.get(tuuid)
                target_logical = succ_info.tuuid_sharding.get(tuuid)

                if source_logical is None or target_logical is None:
                    raise ValueError(
                        f"[SPMDWorker] Cannot resolve empty propagation routing for "
                        f"TUUID {tuuid} from {op_name} to {spmd_id}: "
                        f"source_sharding={bool(source_logical)}, "
                        f"target_sharding={bool(target_logical)}"
                    )

                source_spec = source_logical.bind(self._mesh)
                target_spec = target_logical.bind(succ_info.mesh or self._mesh)
                consumers = source_spec.resolve_consumers(target_spec, effective_rank)
                for target_rank in consumers:
                    expected_count = len(
                        source_spec.resolve_producers(target_spec, target_rank)
                    )
                    if expected_count <= 0:
                        raise ValueError(
                            f"[SPMDWorker] Empty propagation resolved invalid "
                            f"expected_count={expected_count} target_rank={target_rank} "
                            f"spmd_id={self.spmd_id} successor={spmd_id} tuuid={tuuid}."
                        )
                    builder = frames_by_target[target_rank]
                    edge = builder.add_edge(
                        tuuid=tuuid,
                        meta_state=0,
                        empty_kind=EmptyKind.PROTOCOL_EMPTY,
                        min_rows=0,
                        expected_count=expected_count,
                    )
                    builder.add_payload(edge, PayloadRole.VALUES, empty_values, 0)

            handlers = self._successor_handlers.get(spmd_id, {})
            frame_args = []
            for target_rank, builder in sorted(frames_by_target.items()):
                frame = builder.build()
                frame_args.append(
                    (
                        target_rank,
                        frame.header,
                        frame.payload_segments,
                    )
                )
            await self._outbound_semaphore.acquire()
            self._spawn_outbound_task(
                self._send_protocol_empty_frames_background(handlers, frame_args, step)
            )

    async def _send_protocol_empty_frames_background(
        self,
        handlers: dict[int, Any],
        frame_args: list[tuple[int, TransportFrameHeader, tuple[Any, ...]]],
        step: int,
    ) -> None:
        if frame_args:
            await asyncio.gather(
                *(
                    self._data_plane_executor.send_transport_frame(
                        handlers,
                        frame_arg,
                        source_spmd_id=self.spmd_id,
                        source_rank=self.rank,
                        output_step=step,
                        send_semaphore=self._send_semaphore,
                    )
                    for frame_arg in frame_args
                )
            )

    async def _execute_operator(self, op_name: str, step: int) -> ExecutionResult:
        """Execute an operator: assemble data, execute, send outputs."""
        try:
            # Drain self-block: `step` is the 0-indexed task sequence number that is
            # about to execute.  When step N is about to run, exactly N steps have
            # already been completed (completed_step_count == N).  Therefore blocking
            # at step >= target_count guarantees that workers halt only after completing
            # exactly target_count steps, which is when the Synchronizer's ACK fires.
            async with self._drain_cond:
                while (
                    self._drain_target_count is not None
                    and step >= self._drain_target_count
                ):
                    logger.debug(
                        f"Self-blocking due to drain target={self._drain_target_count} "
                        f"task_step={step} spmd_id={self.spmd_id} rank={self.rank}"
                    )
                    await self._drain_cond.wait()

            # Re-validate after drain wait: a checkpoint restore may have
            # happened while we were blocked, advancing completed_step_count
            # past this step.  Executing a stale step would corrupt the
            # watermark and cause downstream deadlocks.
            current_completed = self._completed_step_count.get(op_name, 0)
            if step < current_completed:
                logger.warning(
                    f"Step became stale during drain wait (likely restore) "
                    f"op={op_name} step={step} completed_count={current_completed} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
                await self._mailbox.clear_step(op_name, step)
                self._barriers_declared.pop((op_name, step), None)
                return ExecutionResult(status=ExecutionStatus.STALE_AFTER_RESTORE)

            # --- Dispatcher Check: Should this rank execute this step? ---
            dispatcher = self.op_manager.get_dispatcher(op_name)
            if not dispatcher.should_execute(self.rank, self._world_size, step):
                logger.debug(
                    f"Dispatcher skipped execution op={op_name} step={step} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
                await self._mailbox.clear_step(op_name, step)
                self._barriers_declared.pop((op_name, step), None)
                return ExecutionResult(status=ExecutionStatus.SKIPPED)

            # --- Optional fastpath: all received shards are empty packets ---
            # When enable_empty_fastpath is True, short-circuit by directly
            # propagating empty packets to downstream and skipping user operator
            # dispatch. When False (default), all steps — including all-empty
            # ones — are dispatched to the user-defined operator, letting user
            # code decide how to handle empty inputs.
            if self.config.enable_empty_fastpath:
                all_empty = await self._check_all_empty(op_name, step)
                if all_empty:
                    logger.debug(
                        f"All shards empty propagating empty packets op={op_name} step={step} "
                        f"spmd_id={self.spmd_id} rank={self.rank}"
                    )
                    await self._propagate_empty_packets(op_name, step)
                    await self._mailbox.clear_step(op_name, step)
                    self._barriers_declared.pop((op_name, step), None)
                    return ExecutionResult(status=ExecutionStatus.EMPTY_PROPAGATION)

            # Assemble input data
            try:
                input_data = await self._assemble_input(op_name, step)
            except Exception as e:
                logger.error(
                    f"Failed to assemble input op={op_name} step={step} error={e} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
                raise e

            op_result = await self._execute_local_step_plan(op_name, step, input_data)
            logger.debug(
                f"Execution finished op={op_name} step={step} spmd_id={self.spmd_id} rank={self.rank}"
            )

            result = ExecutionResult(status=ExecutionStatus.SUCCESS, data=op_result)

            if op_result is None:
                logger.debug(
                    f"Operator returned None continuing op={op_name} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )

            # Send output
            logger.debug(
                f"Sending output op={op_name} step={step} result_type={type(op_result).__name__} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            await self._send_output(op_name, result, exec_step=step)

            # Clear mailbox
            await self._mailbox.clear_step(op_name, step)
            self._barriers_declared.pop((op_name, step), None)

            return result

        except Exception as e:
            import traceback

            logger.error(
                f"Operator execution failed op={op_name} error={e} "
                f"traceback={traceback.format_exc()} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            raise

    async def _execute_local_step_plan(
        self, op_name: str, step: int, input_data: Any
    ) -> Any:
        plan = LocalStepPlan.single_call(op_name, resource_id=0)

        meta = self.op_manager.get_operator_metadata(op_name)
        is_stateful = bool(getattr(meta, "owner_class_qualname", ""))
        owner_classes: tuple[tuple[str, str, type], ...] = ()
        process_affine = False
        if is_stateful:
            owner_cls = _resolve_owner_class(
                meta.owner_class_module, meta.owner_class_qualname
            )
            owner_classes = (
                (
                    meta.owner_class_module,
                    meta.owner_class_qualname,
                    owner_cls,
                ),
            )
            process_affine = _is_worker_process_required(owner_cls)
        worker_self = self
        operator_args: tuple[Any, ...] = ()
        operator_kwargs: dict[str, Any] = {}
        if isinstance(input_data, dict):
            operator_kwargs = input_data
        else:
            operator_args = (input_data,)

        async def execute_operator_callable() -> Any:
            if operator_kwargs:
                return await self.op_manager.execute_async(
                    op_name, step, **operator_kwargs
                )
            if operator_args:
                return await self.op_manager.execute_async(
                    op_name, step, *operator_args
                )
            return await self.op_manager.execute_async(op_name, step)

        class OperatorCallableRegistry(MappingCallableRegistry):
            def spec_for(self, callable_id: str, resource_id: int) -> UserCallableSpec:
                spec = super().spec_for(callable_id, resource_id)

                return UserCallableSpec(
                    callable_id=spec.callable_id,
                    fn=spec.fn,
                    is_stateful=is_stateful,
                    resource_id=spec.resource_id,
                    operator_config=worker_self.config if is_stateful else None,
                    operator_name=callable_id if is_stateful else None,
                    operator_step=step if is_stateful else None,
                    operator_args=operator_args if is_stateful else (),
                    operator_kwargs=operator_kwargs if is_stateful else None,
                    operator_rank=worker_self.rank if is_stateful else None,
                    operator_spmd_id=worker_self.spmd_id if is_stateful else "",
                    local_rank=worker_self.local_rank if is_stateful else None,
                    local_world_size=(
                        worker_self.local_world_size if is_stateful else None
                    ),
                    node_rank=(
                        0 if worker_self.node_rank is None else worker_self.node_rank
                    ),
                    num_nodes=(
                        1 if worker_self.num_nodes is None else worker_self.num_nodes
                    ),
                    rank0_address=worker_self.rank0_address,
                    operator_owner_classes=owner_classes if is_stateful else (),
                    process_affine=process_affine,
                )

        registry = OperatorCallableRegistry({op_name: execute_operator_callable})
        return await self._local_step_plan_executor.execute(plan, registry)

    async def _assemble_input(self, op_name: str, step: int) -> Any:
        static_args = self.op_manager.get_static_args()
        expected_state_names = self.op_manager.get_expected_control_arg_names(op_name)
        all_states = await self._mailbox.get_all_control_arguments(op_name, step)
        state_args = {
            name: all_states[name]
            for name in expected_state_names
            if name in all_states
        }

        meta = self.op_manager.get_operator_metadata(op_name)
        input_types = self.op_manager.get_operator_input_types(op_name)
        pre_topo = self.op_manager.get_predecessor_topology(op_name)

        all_tuuids = []
        p_to_tuuids = {}
        for p_name in input_types:
            if p_name in static_args:
                continue

            p_paths = meta.input_param_map.get(p_name, [p_name])
            p_tuuids = []
            for path in p_paths:
                tuuid = None
                for info in pre_topo.values():
                    if path in info.tuuid_map:
                        tuuid = info.tuuid_map[path]
                        break
                p_tuuids.append(tuuid)

            p_to_tuuids[p_name] = p_tuuids
            all_tuuids.extend([u for u in p_tuuids if u is not None])

        unique_tuuids = list(set(all_tuuids))

        shard_tasks = [
            self._mailbox.get_tensor_shard_views(op_name, u, step)
            for u in unique_tuuids
        ]
        arrival_tasks = [
            self._mailbox.get_tensor_arrivals(op_name, u, step) for u in unique_tuuids
        ]
        all_shards_list = await asyncio.gather(*shard_tasks)
        all_arrivals_list = await asyncio.gather(*arrival_tasks)
        tuuid_to_shards = dict(zip(unique_tuuids, all_shards_list, strict=True))  # type: ignore
        tuuid_to_arrivals = dict(zip(unique_tuuids, all_arrivals_list, strict=True))  # type: ignore

        assemble_tasks = [
            self._assemble_tensor(
                op_name, u, step, tuuid_to_shards[u], tuuid_to_arrivals[u]
            )
            for u in unique_tuuids
        ]
        assembled_list = await asyncio.gather(*assemble_tasks)
        tuuid_to_assembled = dict(zip(unique_tuuids, assembled_list, strict=True))  # type: ignore

        if step not in self._step_tuuid_cache:
            self._step_tuuid_cache[step] = {}
        self._step_tuuid_cache[step].update(tuuid_to_assembled)

        input_kwargs = {}
        for p_name, p_tuuids in p_to_tuuids.items():
            if not p_tuuids or p_tuuids[0] is None:
                continue

            p_leaves = [tuuid_to_assembled[u] for u in p_tuuids]
            tdef = meta.input_treedefs.get(p_name)
            if tdef:
                input_kwargs[p_name] = tree_unflatten(tdef, p_leaves)
            else:
                input_kwargs[p_name] = p_leaves[0]

        final_args = {}
        for p_name in input_types:
            if p_name in static_args:
                final_args[p_name] = static_args[p_name]
        final_args.update(state_args)
        final_args.update(input_kwargs)
        return final_args

    def _get_values(self, obj: Any) -> Any:
        if hasattr(obj, "flat_values"):
            return obj.flat_values
        return obj

    def _get_routing_indices(self, obj: Any) -> Any | None:
        if hasattr(obj, "routing_indices_materialized"):
            return obj.routing_indices_materialized
        return None

    def _flatten_shards(
        self, shards: dict[tuple[str, int], list[Any]], index: int = 0
    ) -> list[Any]:
        """
        Extract the N-th tensor from each multi-tensor packet.
        """
        tensors = []
        for sender_res in shards.values():
            if index < len(sender_res):
                tensors.append(sender_res[index])
        return tensors

    def _flatten_shards_with_ranks(
        self, shards: dict[Any, list[Any]], index: int = 0
    ) -> list[tuple[Any, Any]]:
        """Flatten shards and include their sender keys (ranks)."""
        flat = []
        for key in sorted(shards.keys()):
            tensor_list = shards[key]
            if index < len(tensor_list):
                flat.append((key, tensor_list[index]))
        return flat

    def _normalize_and_cat_shards(self, tensor_list: list[Any], dim: int = 0) -> Any:
        backend = get_backend()
        count = len(tensor_list)
        if count == 0:
            raise ValueError("Cannot concatenate an empty shard list.")

        # Fast Path: Single tensor is already the result (Zero-copy)
        if count == 1:
            t = tensor_list[0]
            # Ray deserialization may produce readonly NumPy arrays.
            # force_writable patches __array_interface__ / __cuda_array_interface__
            # to grant write access without copying.
            return force_writable(t)

        # Find the archetype (first non-empty tensor with proper rank)
        prototype = None
        for t in tensor_list:
            if backend.size(t) > 0:
                prototype = t
                break

        if prototype is None:
            proto_shape = tuple(backend.shape(tensor_list[0]))
            proto_dtype = backend.dtype(tensor_list[0])
            for t in tensor_list[1:]:
                if tuple(backend.shape(t)) != proto_shape:
                    raise ValueError(
                        "Cannot concatenate all-empty shards with different shapes."
                    )
                if backend.dtype(t) != proto_dtype:
                    raise ValueError(
                        "Cannot concatenate all-empty shards with different dtypes."
                    )
            return backend.cat(tensor_list, dim=dim)

        # Check for mismatch against prototype to decide if we need the slow-path
        needs_normalization = False
        proto_shape = backend.shape(prototype)
        proto_dtype = backend.dtype(prototype)
        for t in tensor_list:
            if backend.shape(t) != proto_shape or backend.dtype(t) != proto_dtype:
                needs_normalization = True
                break

        # Homogeneous Fast Path: If no mismatches found, skip alignment loop
        if not needs_normalization:
            return backend.cat(tensor_list, dim=dim)

        proto_shape = backend.shape(prototype)
        proto_dtype = backend.dtype(prototype)
        proto_ndims = len(proto_shape)

        normalized = []
        for t in tensor_list:
            t_shape = backend.shape(t)
            if len(t_shape) != proto_ndims:
                new_shape = list(proto_shape)
                new_shape[dim] = 0
                normalized.append(backend.zeros(tuple(new_shape), dtype=proto_dtype))
            elif backend.dtype(t) != proto_dtype:
                normalized.append(backend.cast(t, proto_dtype))
            else:
                normalized.append(t)

        return backend.cat(normalized, dim=dim)

    # =========================================================================
    # Shard Assembly Helpers (shared by all _assemble_* functions)
    # =========================================================================

    def _merge_index_descriptors(
        self, shard_list: list[Any], property: str = "origin_indices"
    ) -> Optional["IndexDescriptor"]:
        """Merge IndexDescriptors from shards only when ALL shards carry one."""
        descriptors: list[IndexDescriptor] = [
            getattr(s, property) for s in shard_list if getattr(s, property) is not None
        ]
        if len(descriptors) == len(shard_list) and shard_list:
            return IndexDescriptor.concatenate(descriptors)
        return None

    def _merge_materialized_index_descriptors(
        self,
        shards: list[Any],
        property: str = "origin_indices_materialized",
    ) -> Any | None:
        """Merge materialized indices from shards."""
        if not any(
            hasattr(s, property) and getattr(s, property) is not None for s in shards
        ):
            return None
        index_list = []
        for s in shards:
            idx = getattr(s, property, None)
            if idx is None and len(s) > 0:
                raise ValueError(
                    f"Shard {s} is missing materialized index descriptor {property}"
                )
            if idx is None:
                raise ValueError(
                    f"Shard {s} is missing {property}; partial metadata cannot be merged."
                )
            index_list.append(idx)
        return self._normalize_and_cat_shards(index_list, dim=0)

    def _maybe_merge_receipts(self, shards: list[Any]) -> Optional["SymmetryReceipt"]:
        """Merge SymmetryReceipts from input shards into a single receipt."""
        receipts = [s.receipt for s in shards]
        if all(r is not None for r in receipts) and receipts:
            lengths = [self._routing_entity_count(s) for s in shards]
            return SymmetryReceipt.concatenate(receipts, lengths)  # type: ignore[arg-type]
        return None

    def _routing_entity_count(self, shard: Any) -> int:
        backend = get_backend()
        if getattr(shard, "is_ragged", False):
            splits = shard.nested_splits
            if not splits:
                raise ValueError("Ragged shard is missing nested_splits.")
            return int(backend.shape(splits[0])[0]) - 1
        values = self._get_values(shard)
        shape = backend.shape(values)
        if not shape:
            return 1
        return int(shape[0])

    def _merge_nested_splits(self, shards: list[Any]) -> list[Any]:
        if not any(getattr(s, "is_ragged", False) for s in shards):
            return []
        if not all(getattr(s, "is_ragged", False) for s in shards):
            raise ValueError("Cannot merge mixed dense and ragged shards.")

        backend = get_backend()
        level_count = len(shards[0].nested_splits)
        if level_count == 0:
            raise ValueError("Ragged shard is missing nested_splits.")
        for shard in shards[1:]:
            if len(shard.nested_splits) != level_count:
                raise ValueError(
                    "Cannot merge ragged shards with different nested level counts."
                )

        merged_splits = []
        for level in range(level_count):
            lengths = []
            for shard in shards:
                splits = shard.nested_splits[level]
                lengths.append(splits[1:] - splits[:-1])
            merged_lengths = self._normalize_and_cat_shards(lengths, dim=0)
            zero = backend.zeros((1,), dtype=backend.dtype(merged_lengths))
            merged_splits.append(
                backend.cat([zero, backend.cumsum(merged_lengths, dim=0)], dim=0)
            )
        return merged_splits

    def _wrap_shard_for_operator(self, shard: Any) -> TransportTensor:
        if isinstance(shard, TransportTensor):
            return shard
        return TransportTensor(
            self._get_values(shard),
            nested_splits=shard.nested_splits,
            routing_indices=shard.routing_indices,
            origin_indices=shard.origin_indices,
            global_shape=shard.global_shape,
            receipt=shard.receipt,
            routing_axes=shard.routing_axes,
        )

    def _describe_shard_flow_for_error(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards_dict: dict[Any, list[Any]],
        filtered_shards: list[Any],
        input_layout: "PhysicalLayout | None",
        predecessor_topology: dict[str, Any],
    ) -> str:
        backend = get_backend()
        producer_path = getattr(self.op_manager, "_tuuid_to_op_outputs", {}).get(tuuid)
        tuuid_name = ".".join(producer_path) if producer_path else f"TUUID:{tuuid}"

        predecessor_lines = []
        for pre_spmd_id, pre_info in predecessor_topology.items():
            if tuuid not in pre_info.tuuid_sharding:
                continue
            producer_layout = self.op_manager._resolve_producer_spec(
                tuuid, provider_map=pre_info.tuuid_sharding
            )
            rank_keys = [
                key
                for key in shards_dict.keys()
                if isinstance(key, tuple) and len(key) == 2 and key[0] == pre_spmd_id
            ]
            predecessor_lines.append(
                f"{pre_spmd_id}: producer={_describe_layout_for_error(producer_layout)}, "
                f"mailbox_keys={rank_keys}"
            )

        shard_lines = []
        for i, shard in enumerate(filtered_shards[:16]):
            values = self._get_values(shard)
            shape = tuple(backend.shape(values))
            origin = shard.origin_indices_materialized
            routing = self._get_routing_indices(shard)
            if origin is not None and int(backend.shape(origin)[0]) > 0:
                origin_np = backend.to_numpy(origin)
                origin_text = (
                    f"origin_len={len(origin_np)}, "
                    f"origin_min={int(origin_np.min())}, "
                    f"origin_max={int(origin_np.max())}"
                )
            else:
                origin_text = "origin_len=0"
            if routing is not None and int(backend.shape(routing)[0]) > 0:
                routing_np = backend.to_numpy(routing)
                routing_text = (
                    f"routing_len={len(routing_np)}, "
                    f"routing_min={int(routing_np.min())}, "
                    f"routing_max={int(routing_np.max())}"
                )
            else:
                routing_text = "routing_len=0"
            shard_lines.append(
                f"shard[{i}]: shape={shape}, global_shape={shard.global_shape}, "
                f"{origin_text}, {routing_text}, ragged_levels={len(shard.nested_splits)}"
            )

        shard_summary = "; ".join(shard_lines) if shard_lines else "none"
        predecessor_summary = (
            "; ".join(predecessor_lines) if predecessor_lines else "none"
        )
        return (
            f"name={tuuid_name}, op={op_name}, tuuid={tuuid}, step={step}, "
            f"consumer_rank={self.rank}, consumer={_describe_layout_for_error(input_layout)}, "
            f"mailbox_keys={list(shards_dict.keys())}, "
            f"filtered_shards={len(filtered_shards)}, "
            f"predecessors=[{predecessor_summary}], "
            f"filtered_detail=[{shard_summary}]"
        )

    # =========================================================================
    # Tensor Assembly
    # =========================================================================

    async def _assemble_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        return self._assemble_tensor_sync(op_name, tuuid, step, shards, arrivals)

    def _assemble_tensor_sync(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        """Assemble tensor shards into a complete TransportTensor."""
        # Note: Input layout is now retrieved as a PhysicalLayout bound to current mesh
        input_layout = self.op_manager._resolve_input_spec(op_name, tuuid)
        predecessor_topology = self.op_manager.get_predecessor_topology(op_name)

        if input_layout is None or input_layout.is_content_addressable:
            # Fallback or dynamic: merge all shards
            return self._assemble_to_dynamic_tensor(
                op_name, tuuid, step, shards, arrivals, predecessor_topology
            )

        # Check for dynamic predecessors
        has_dynamic_predecessor = any(
            (spec := p.tuuid_sharding.get(tuuid)) is not None
            and spec.is_content_addressable
            for p in predecessor_topology.values()
        )

        if has_dynamic_predecessor:
            return self._assemble_dynamic_to_static_tensor(
                op_name, tuuid, step, shards, input_layout, predecessor_topology
            )
        return self._assemble_static_to_static_tensor(
            op_name, tuuid, step, shards, input_layout, predecessor_topology
        )

    def _filter_shards_by_routing(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards_dict: dict[Any, list[Any]],
        predecessor_topology: dict[str, Any],
    ) -> list[Any]:
        """
        Filters shards from mailbox based on analytical or fallback routing.
        """
        filtered_shards = []
        for pre_spmd_id, pre_info in predecessor_topology.items():
            # Only query predecessors that actually provide this TUUID.
            if tuuid not in pre_info.tuuid_sharding:
                continue

            input_layout = self.op_manager._resolve_input_spec(op_name, tuuid)
            producer_layout = self.op_manager._resolve_producer_spec(
                tuuid, provider_map=pre_info.tuuid_sharding
            )
            if (
                producer_layout is not None
                and producer_layout.is_content_addressable
                and input_layout is not None
                and not input_layout.is_content_addressable
            ):
                for key, rank_shards in shards_dict.items():
                    if not isinstance(key, tuple) or len(key) != 2:
                        continue
                    shard_spmd_id, _rank = key
                    if shard_spmd_id != pre_spmd_id:
                        continue
                    found = self._flatten_shards({key: rank_shards}, index=0)
                    filtered_shards.extend(found)
                continue

            # Unified resolution (handles interleave, etc.)
            producer_ranks = self.op_manager.resolve_tensor_producers(
                op_name, tuuid, pre_spmd_id, self.rank, step=step
            )

            for rank in producer_ranks:
                key = (pre_spmd_id, rank)
                if key in shards_dict:
                    found = self._flatten_shards({key: shards_dict[key]}, index=0)
                    filtered_shards.extend(found)
        return filtered_shards

    def _collect_arrived_shards_for_tuuid(
        self,
        tuuid: int,
        shards_dict: dict[Any, list[Any]],
        predecessor_topology: dict[str, Any],
    ) -> list[Any]:
        collected: list[Any] = []
        for pre_spmd_id, pre_info in predecessor_topology.items():
            if tuuid not in pre_info.tuuid_sharding:
                continue
            for key, rank_shards in shards_dict.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                shard_spmd_id, _rank = key
                if shard_spmd_id != pre_spmd_id:
                    continue
                collected.extend(self._flatten_shards({key: rank_shards}, index=0))
        return collected

    def _collect_arrivals_for_tuuid(
        self,
        tuuid: int,
        arrivals_dict: dict[Any, dict[str, Any]],
        predecessor_topology: dict[str, Any],
    ) -> dict[Any, dict[str, Any]]:
        arrivals: dict[Any, dict[str, Any]] = {}
        for pre_spmd_id, pre_info in predecessor_topology.items():
            if tuuid not in pre_info.tuuid_sharding:
                continue
            for key, meta in arrivals_dict.items():
                if not isinstance(key, tuple) or len(key) != 2:
                    continue
                shard_spmd_id, _rank = key
                if shard_spmd_id == pre_spmd_id:
                    arrivals[key] = meta
        return arrivals

    def _protocol_empty_dynamic_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        arrivals: dict[Any, dict[str, Any]],
    ) -> "TransportTensor":
        expected_values = {
            int(meta.get("expected_count", 0)) for meta in arrivals.values()
        }
        if len(expected_values) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Conflicting expected_count metadata "
                f"for all-empty dynamic tensor op={op_name} tuuid={tuuid} "
                f"step={step}: {sorted(expected_values)}."
            )
        expected = next(iter(expected_values), 0)
        if expected <= 0:
            raise ValueError(
                f"[Fluxway Protocol Error] Missing expected_count for all-empty "
                f"dynamic tensor op={op_name} tuuid={tuuid} step={step}."
            )
        if len(arrivals) != expected:
            raise ValueError(
                f"[Fluxway Protocol Error] Incomplete all-empty dynamic tensor "
                f"arrival metadata op={op_name} tuuid={tuuid} step={step}: "
                f"arrived={len(arrivals)} expected={expected}."
            )

        first_meta = next(iter(arrivals.values()))
        first_schema = tuple(first_meta.get("payload_schema") or ())
        if not first_schema:
            raise ValueError(
                f"[Fluxway Protocol Error] Protocol-empty tensor lacks payload schema "
                f"op={op_name} tuuid={tuuid} step={step}."
            )
        first_empty_kind = first_meta.get("empty_kind")
        if first_empty_kind is None:
            raise ValueError(
                f"[Fluxway Protocol Error] Protocol-empty tensor lacks empty_kind "
                f"op={op_name} tuuid={tuuid} step={step}."
            )
        if EmptyKind(int(first_empty_kind)) is not EmptyKind.PROTOCOL_EMPTY:
            raise ValueError(
                f"[Fluxway Protocol Error] All-empty dynamic tensor must be composed "
                f"only of PROTOCOL_EMPTY arrivals op={op_name} tuuid={tuuid} step={step}."
            )

        for key, meta in arrivals.items():
            raw_empty_kind = meta.get("empty_kind")
            if raw_empty_kind is None:
                raise ValueError(
                    f"[Fluxway Protocol Error] Protocol-empty tensor lacks empty_kind "
                    f"op={op_name} tuuid={tuuid} step={step} sender={key}."
                )
            empty_kind = EmptyKind(int(raw_empty_kind))
            if empty_kind is not EmptyKind.PROTOCOL_EMPTY:
                raise ValueError(
                    f"[Fluxway Protocol Error] Mixed empty semantics for dynamic tensor "
                    f"op={op_name} tuuid={tuuid} step={step} sender={key}: "
                    f"{empty_kind.name}."
                )
            schema = tuple(meta.get("payload_schema") or ())
            if schema != first_schema:
                raise ValueError(
                    f"[Fluxway Protocol Error] Protocol-empty schema mismatch "
                    f"op={op_name} tuuid={tuuid} step={step} sender={key}."
                )

        value_schema = [
            s for s in first_schema if int(s["role"]) == int(PayloadRole.VALUES)
        ]
        if len(value_schema) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Protocol-empty tensor requires exactly "
                f"one VALUES payload schema op={op_name} tuuid={tuuid} step={step}; "
                f"got {len(value_schema)}."
            )
        shape = list(value_schema[0]["shape"])
        if not shape:
            shape = [0]
        shape[0] = 0
        from fluxway.common.types.tensor import SymmetryReceipt

        backend = get_backend()
        values = backend.zeros(tuple(shape), dtype=value_schema[0]["dtype"])
        global_shape = first_meta.get("global_shape")
        sender_world_sizes = {
            int(meta.get("sender_world_size", 0)) for meta in arrivals.values()
        }
        if len(sender_world_sizes) != 1:
            raise ValueError(
                f"[Fluxway Protocol Error] Conflicting sender_world_size metadata "
                f"for all-empty dynamic tensor op={op_name} tuuid={tuuid} "
                f"step={step}: {sorted(sender_world_sizes)}."
            )
        sender_world_size = next(iter(sender_world_sizes), 0)
        if sender_world_size <= 0:
            raise ValueError(
                f"[Fluxway Protocol Error] Missing sender_world_size for all-empty "
                f"dynamic tensor op={op_name} tuuid={tuuid} step={step}."
            )
        receipt = SymmetryReceipt(
            sender_world_size,
            backend.zeros((0,), dtype=backend.int32()),
        )
        return TransportTensor(values, global_shape=global_shape, receipt=receipt)

    def _assemble_static_to_static_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: dict[Any, list[Any]],
        input_layout: "PhysicalLayout",
        predecessor_topology: dict[str, Any],
    ) -> "TransportTensor":
        """Assemble static-to-static tensor using exact physical shards."""
        shard_list = self._filter_shards_by_routing(
            op_name, tuuid, step, shards, predecessor_topology
        )

        if not shard_list:
            flow_detail = self._describe_shard_flow_for_error(
                op_name,
                tuuid,
                step,
                shards,
                shard_list,
                input_layout,
                predecessor_topology,
            )
            raise ValueError(
                f"[Fluxway Assembly Error] Static-to-static assembly for op "
                f"'{op_name}' (TUUID {tuuid}, rank {self.rank}, step {step}) "
                f"received no routed shards. Flow detail: {flow_detail}"
            )

        if len(shard_list) == 1:
            return self._wrap_shard_for_operator(shard_list[0])

        flat_values = [self._get_values(s) for s in shard_list]
        merged_values = self._normalize_and_cat_shards(flat_values, dim=0)
        merged_nested_splits = self._merge_nested_splits(shard_list)
        merged_routing_indices = self._merge_index_descriptors(
            shard_list, "routing_indices"
        )
        merged_origin_indices = self._merge_index_descriptors(
            shard_list, "origin_indices"
        )

        first_shard = shard_list[0]
        global_shape = (
            first_shard.global_shape if hasattr(first_shard, "global_shape") else None
        )

        merged_receipt = self._maybe_merge_receipts(shard_list)

        return TransportTensor(
            merged_values,
            nested_splits=merged_nested_splits,
            routing_indices=merged_routing_indices,
            origin_indices=merged_origin_indices,
            global_shape=global_shape,
            receipt=merged_receipt,
        )

    def _assemble_scalar_reduction(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: dict[Any, list[Any]],
        input_layout: "PhysicalLayout",
    ) -> TransportTensor:
        backend = get_backend()

        all_values = []
        for rank_shards in shards.values():
            for s in rank_shards:
                val = self._get_values(s)
                if backend.shape(val) != ():
                    val = backend.reshape(val, ())
                all_values.append(val)

        if not all_values:
            raise ValueError(
                f"No scalar shards found for TUUID {tuuid} at step {step} for reduction."
            )

        stacked = backend.stack(all_values)
        op = (
            cast(Any, input_layout.reduce_op).value
            if hasattr(input_layout.reduce_op, "value")
            else str(input_layout.reduce_op)
        )
        reduced = _jit_reduce_scalars(stacked, op.lower())
        return TransportTensor.from_tensor(reduced, global_shape=())

    def _assemble_dynamic_to_static_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: dict[Any, list[Any]],
        input_layout: "PhysicalLayout",
        predecessor_topology: dict[str, Any],
    ) -> "TransportTensor":
        """Assemble dynamic-to-static (Sparse-to-Dense) tensor using origin_indices."""
        backend = get_backend()

        filtered_shards = self._filter_shards_by_routing(
            op_name, tuuid, step, shards, predecessor_topology
        )

        # Metadata should be extracted from ALL expected shards (even empty ones)
        # to ensure global_shape and receipt provenance is preserved.
        shard_global_shape = None
        for s in filtered_shards:
            if hasattr(s, "global_shape") and s.global_shape is not None:
                shard_global_shape = s.global_shape
                break

        # Align global_shape: use shard metadata, or fallback to layout, or infer.
        if shard_global_shape is not None:
            global_shape = shard_global_shape
        elif input_layout.global_shape is not None:
            global_shape = input_layout.global_shape
        else:
            global_shape = None  # Will infer later if data arrives

        valid_shards = [
            s for s in filtered_shards if backend.shape(self._get_values(s))[0] > 0
        ]

        if not valid_shards:
            flow_detail = self._describe_shard_flow_for_error(
                op_name,
                tuuid,
                step,
                shards,
                filtered_shards,
                input_layout,
                predecessor_topology,
            )
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-static assembly for op "
                f"'{op_name}' (TUUID {tuuid}, rank {self.rank}, step {step}) "
                "received no non-empty data shards. Dense SPMD inputs must be "
                "slot-addressed by real payload rows; constructing a zero local "
                "buffer would hide upstream routing data loss. "
                f"Flow detail: {flow_detail}"
            )

        if any(getattr(s, "is_ragged", False) for s in valid_shards):
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-static assembly for op "
                f"'{op_name}' (TUUID {tuuid}, rank {self.rank}, step {step}) "
                "received ragged payload shards. Fluxway transport keeps dynamic "
                "ragged streams compact; consumer-specific padding/pooling/masking "
                "must be implemented in the SPMD operator input adapter."
            )

        all_receipt = self._maybe_merge_receipts(valid_shards)

        routing_dim = getattr(input_layout, "routing_dim", 0)
        if isinstance(routing_dim, list | tuple):
            routing_dim = routing_dim[0]

        all_values = backend.cat(
            [self._get_values(s) for s in valid_shards], dim=routing_dim
        )

        # Materialize indices (origin_indices preferred for logical ordering).
        # Both origin_indices and routing_indices encode *global* per-row
        # positions; without them we have no information-theoretic basis to
        # decide which destination slot each row belongs to.  The previous
        # arange(0, num) fallback silently produced sender-local positions
        # that collide across senders (every sender claiming positions 0..N),
        # which is wrong in any multi-sender scenario.  Fail loudly instead
        # of generating undefined scatter behaviour.
        all_indices_list = []
        for s in valid_shards:
            idx = s.origin_indices_materialized
            if idx is None:
                idx = self._get_routing_indices(s)
            if idx is None:
                raise ValueError(
                    f"[Fluxway Protocol Error] Dynamic-to-static assembly for "
                    f"op '{op_name}' (TUUID {tuuid}) received a shard with "
                    f"neither origin_indices nor routing_indices. The producer "
                    f"must attach per-row global position metadata so each row "
                    f"can be deterministically scattered into the destination "
                    f"buffer; arange-style fallbacks would yield colliding "
                    f"sender-local positions across multiple senders and "
                    f"silently corrupt the result."
                )
            all_indices_list.append(idx)
        all_indices = backend.cat(all_indices_list, dim=0)

        # global_shape resolution hierarchy (Strict)
        if shard_global_shape is not None:
            global_shape = shard_global_shape
        elif input_layout.global_shape is not None:
            global_shape = input_layout.global_shape
        else:
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-Static conversion failed for TUUID {tuuid}. "
                "Global shape information is missing. Inference from raw indices is forbidden. "
                "Please Ensure the producer TransportTensor has global_shape set."
            )

        local_shape = input_layout.get_shard_shape(self.rank, data_shape=global_shape)
        slice_start, slice_end = input_layout.get_shard_bounds(
            self.rank, routing_dim, data_shape=global_shape
        )

        routing_extent = (
            global_shape[routing_dim] if routing_dim < len(global_shape) else None
        )
        num_shards = input_layout.shard_size(routing_dim)
        if (
            routing_extent is not None
            and num_shards > 1
            and routing_extent % num_shards != 0
        ):
            raise ValueError(
                f"[Fluxway Assembly Error] global_shape[{routing_dim}]={routing_extent} "
                f"is not divisible by shard_count={num_shards} "
                f"in op '{op_name}' (TUUID {tuuid}). "
                f"This causes partition boundary misalignment between producers and consumers. "
                f"Ensure the upstream batch_size is divisible by the DP world_size."
            )

        mask = (all_indices >= slice_start) & (all_indices < slice_end)
        local_indices = all_indices[mask] - slice_start

        logical_len_early = int(backend.sum(mask))
        buffer_capacity = local_shape[routing_dim]

        # When origin_indices have collisions (multiple source rows map to the
        # same target slot via scatter-add), logical_len_early can exceed
        # buffer_capacity even though the actual number of unique target slots
        # fits.  Compute unique slots first and validate against those.
        n_unique_slots = logical_len_early  # default: 1:1 mapping
        if logical_len_early > buffer_capacity and logical_len_early > 0:
            unique_local = backend.unique(local_indices)
            n_unique_slots = int(backend.shape(unique_local)[0])

        if n_unique_slots > buffer_capacity:
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic-to-static buffer overflow in op '{op_name}' "
                f"(TUUID {tuuid}, rank {self.rank}, step {step}). "
                f"Received {logical_len_early} data elements mapping to {n_unique_slots} unique "
                f"buffer positions, but buffer capacity is {buffer_capacity} "
                f"(global_shape={global_shape}, local_shape={local_shape}). "
                f"The producer's global_shape is too small for the actual data volume. "
                f"This typically means the upstream unique/sparse operation produced more elements "
                f"than global_shape[0] accounts for (e.g. multi-column features after ravel+unique)."
            )

        if logical_len_early > 0 and n_unique_slots < logical_len_early:
            logger.warning(
                f"[Fluxway Assembly Warning] origin_indices collision detected in op '{op_name}' "
                f"(TUUID {tuuid}, rank {self.rank}, step {step}). "
                f"{logical_len_early} data elements map to only {n_unique_slots} unique buffer positions. "
                f"{logical_len_early - n_unique_slots} elements will be silently overwritten by index_update. "
                f"This usually indicates the producer uses row-level origin_indices for multi-column features, "
                f"causing multiple unique keys from the same row to collide."
            )

        if routing_dim == 0:
            local_values = all_values[mask]
        else:
            indices_to_take = backend.nonzero(
                mask, size=int(backend.shape(all_values)[routing_dim])
            ).flatten()
            idx_tuple = [slice(None)] * len(local_shape)
            idx_tuple[routing_dim] = indices_to_take
            local_values = all_values[tuple(idx_tuple)]

        assembled_data = backend.zeros(local_shape, dtype=all_values.dtype)
        indices_tuple = [slice(None)] * len(local_shape)
        indices_tuple[routing_dim] = local_indices
        assembled_data = backend.index_update(
            assembled_data, tuple(indices_tuple), local_values
        )

        # Assemble mirror receipt into the static layout if present
        assembled_receipt = all_receipt
        if all_receipt is not None:
            final_ranks = backend.full(
                (local_shape[routing_dim],),
                all_receipt.world_size,
                dtype=backend.int32(),
            )
            sliced_receipt = all_receipt[mask]
            final_ranks = backend.index_update(
                final_ranks, local_indices, sliced_receipt.sender_rank_tags
            )
            assembled_receipt = SymmetryReceipt(all_receipt.world_size, final_ranks)

        return TransportTensor(
            assembled_data,
            routing_indices=None,
            global_shape=global_shape,
            receipt=assembled_receipt,
        )

    def _assemble_to_dynamic_tensor(
        self,
        op_name: str,
        tuuid: int,
        step: int,
        shards: dict[Any, list[Any]],
        arrivals: dict[Any, dict[str, Any]],
        predecessor_topology: dict[str, Any],
    ) -> "TransportTensor":
        """
        Assemble to-dynamic (Sparse/Dense -> Sparse) tensor by merging shards.
        Handles both static_to_dynamic and dynamic_to_dynamic scenarios.
        """
        backend = get_backend()

        filtered_shards = self._collect_arrived_shards_for_tuuid(
            tuuid, shards, predecessor_topology
        )

        if not filtered_shards:
            filtered_arrivals = self._collect_arrivals_for_tuuid(
                tuuid, arrivals, predecessor_topology
            )
            if filtered_arrivals:
                return self._protocol_empty_dynamic_tensor(
                    op_name, tuuid, step, filtered_arrivals
                )
            flow_detail = self._describe_shard_flow_for_error(
                op_name,
                tuuid,
                step,
                shards,
                filtered_shards,
                None,
                predecessor_topology,
            )
            raise ValueError(
                f"[Fluxway Assembly Error] Dynamic assembly for op '{op_name}' "
                f"(TUUID {tuuid}, rank {self.rank}, step {step}) received no "
                "arrived shards. Dynamic inputs require an explicit empty or "
                "non-empty packet from every expected sender; synthesizing an "
                f"empty TransportTensor would hide a transport/barrier bug. "
                f"Flow detail: {flow_detail}"
            )

        all_values_list = [self._get_values(s) for s in filtered_shards]
        merged_values = self._normalize_and_cat_shards(all_values_list, dim=0)
        merged_nested_splits = self._merge_nested_splits(filtered_shards)

        # Spatial Indices (global keys).  If *any* shard carries
        # routing_indices, *every* shard must — they label per-row global
        # positions and a partial set is meaningless (an arange-style
        # fallback for the missing shards would silently invent sender-local
        # positions that collide across producers).  Reject the heterogeneous
        # case explicitly so the contract violation surfaces at the boundary.
        merged_routing_indices = None
        if any(self._get_routing_indices(s) is not None for s in filtered_shards):
            indices_list = []
            for s in filtered_shards:
                idx = self._get_routing_indices(s)
                if idx is None:
                    raise ValueError(
                        f"[Fluxway Protocol Error] Dynamic assembly for op "
                        f"'{op_name}' (TUUID {tuuid}) received an inconsistent "
                        f"set of shards: some carry routing_indices and some "
                        f"do not.  routing_indices encode per-row *global* "
                        f"positions; producers must agree on whether to emit "
                        f"them.  An arange fallback would generate "
                        f"sender-local positions that collide across senders "
                        f"and silently corrupt downstream routing."
                    )
                indices_list.append(idx)
            merged_routing_indices = self._normalize_and_cat_shards(indices_list, dim=0)

        # Origin Indices (logical offsets for deterministic sorting)
        merged_origin_indices = self._merge_materialized_index_descriptors(
            filtered_shards
        )

        merged_receipt = self._maybe_merge_receipts(filtered_shards)

        original_global_shape = None
        if filtered_shards:
            original_global_shape = filtered_shards[0].global_shape

        # DETERMINISTIC SORTING:
        if merged_origin_indices is not None:
            # Evaluate receipt: Only materialized receipts need sorting
            receipt_tags = None
            if merged_receipt is not None and not merged_receipt.is_scalar():
                receipt_tags = merged_receipt.sender_rank_tags

            # Optimization Trick: Dummy Substitution
            # Bypass JAX's inability to dynamically trace `None` by substituting missing
            # components with a valid tensor. Computations on dummies will be eliminated by CSE.
            dummy_routing = False
            if merged_routing_indices is None:
                merged_routing_indices = merged_origin_indices
                dummy_routing = True

            dummy_receipt = False
            if receipt_tags is None:
                receipt_tags = merged_origin_indices
                dummy_receipt = True

            # -------- CAPACITY PADDING TO REDUCE RECOMPILATION --------
            # By bucketizing Ragged/variable sequence lengths up to the nearest powers of 2,
            # we restrict JAX JIT from undergoing severe compilation storms.
            backend = get_backend()
            shape_merged = backend.shape(merged_values)
            actual_len: int = int(shape_merged[0]) if shape_merged else 0

            cap = 1 if actual_len <= 1 else 1 << (actual_len - 1).bit_length()

            padded_values = _pad_tensor_to_power_of_2(
                merged_values, actual_len, cap, pad_value=0
            )
            # Padded origin indices to positive infinity so they sort strictly to the end
            padded_origin = _pad_tensor_to_power_of_2(
                merged_origin_indices, actual_len, cap, pad_value=2147483647
            )
            padded_routing = _pad_tensor_to_power_of_2(
                merged_routing_indices, actual_len, cap, pad_value=0
            )
            padded_tags = _pad_tensor_to_power_of_2(
                receipt_tags, actual_len, cap, pad_value=0
            )

            sorted_values, sorted_origin, sorted_routing, sorted_tags, sort_idx = (
                _jit_sort_fused(
                    padded_values, padded_origin, padded_routing, padded_tags
                )
            )

            # --- Unpad Outputs back to dynamic length ---
            merged_values = sorted_values[:actual_len]
            merged_origin_indices = sorted_origin[:actual_len]
            merged_routing_indices = (
                sorted_routing[:actual_len] if not dummy_routing else None
            )

            if merged_receipt is not None and not dummy_receipt:
                merged_receipt = SymmetryReceipt(
                    merged_receipt.world_size, sorted_tags[:actual_len]
                )

        return TransportTensor(
            merged_values,
            nested_splits=merged_nested_splits,
            routing_indices=merged_routing_indices,
            global_shape=(
                original_global_shape
                if original_global_shape is not None
                else backend.shape(merged_values)
            ),
            receipt=merged_receipt,
            origin_indices=merged_origin_indices,
        )

    async def _advance_step(self, op_name: str, result: ExecutionResult) -> None:
        """
        Advance session progress after completing a step.
        Explicitly differentiates between Count (Progress Watermark) and Index (Task Sequence).
        """
        self._completed_step_count[op_name] += 1
        completed_step_count = self._completed_step_count[op_name]

        logger.debug(
            f"Step completed op={op_name} total_count={completed_step_count} "
            f"spmd_id={self.spmd_id} rank={self.rank}"
        )

        # Schedule next task: The index of the next task equals the number of already finished tasks.
        # e.g., Finished Step Index 0 -> Completed Step Count 1 -> Next Task Index 1.
        next_step_index = completed_step_count

        # If it's a source operator, automatically trigger the next step
        if self.op_manager.is_source_operator(op_name):
            # Check for global step limit (max_steps)
            max_steps = self.config.max_steps
            if max_steps is not None and completed_step_count >= max_steps:
                logger.debug(
                    f"Source op reached max_steps stopping op={op_name} max_steps={max_steps} "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
                # Max steps reached, do NOT enqueue next task.
            elif (
                result.status is not ExecutionStatus.SUCCESS or result.data is not None
            ):
                # Wait if paused by backpressure
                if op_name in self._source_pause_events:
                    await self._source_pause_events[op_name].wait()

                self._spawn_background_task(
                    self._enqueue(op_name, next_step_index)
                )  # Use index for the next task
            else:
                logger.debug(
                    f"Source op finished op={op_name} spmd_id={self.spmd_id} rank={self.rank}"
                )

        # Report definitive completed count to Driver for synchronization/drain.
        self._report_op_completed(op_name, completed_step_count)

        # Check if this op is in periodic_trigger mode
        is_periodic = (
            self.op_manager.is_periodic_trigger(op_name) if self._op_manager else False
        )

        if not is_periodic:
            # Normal mode: re-enqueue pending future tasks that match the new step
            pending = self._pending_future_steps[op_name]
            to_enqueue = [s for s in pending if s == next_step_index]
            self._pending_future_steps[op_name] = [
                s for s in pending if s != next_step_index
            ]

            for s in to_enqueue:
                await self._enqueue(op_name, s)

    def _report_op_completed(self, op_name: str, step: int) -> None:
        """
        Report op completion to driver.

        Args:
            op_name: Operation name
            step: Completed step number
        """
        if self._driver_handle is not None:
            logger.debug(
                f"Reporting op completion op={op_name} step={step} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            self._driver_handle.report_worker_op_completed.remote(
                self.rank, op_name, step
            )

        # Cleanup Mirror Sharding Cache for very old steps
        old_steps = [
            s
            for s in self._step_tuuid_cache.keys()
            if s < step - self.config.operator_metas[op_name].max_inflight_steps + 1
        ]
        for s in old_steps:
            del self._step_tuuid_cache[s]

    # =========================================================================
    # Data Sending
    # =========================================================================

    async def _send_output(
        self, op_name: str, result: ExecutionResult, *, exec_step: int
    ) -> None:
        """Send output to successor SPMDs using TransportTensor."""
        if result.status in (
            ExecutionStatus.SKIPPED,
            ExecutionStatus.EMPTY_PROPAGATION,
        ) or (result.status is ExecutionStatus.SUCCESS and result.data is None):
            # If the result is None (e.g., end of stream or no data Produced),
            # we must propagate empty packets to all successors to prevent deadlocks.
            await self._propagate_empty_packets(op_name, exec_step)
            return

        output_uuids = self.op_manager.get_output_tuuids(op_name)
        # Flatten the result using guided tree_flatten
        meta = self.op_manager.get_operator_metadata(op_name)
        flat_results, treedef = tree_flatten(
            result.data,
            treedef_template=meta.output_treedef if meta is not None else None,
        )

        if exec_step not in self._step_tuuid_cache:
            self._step_tuuid_cache[exec_step] = {}

        successor_ids = sorted(set(self.op_manager.get_successor_ids(op_name)))
        output_sharding = self.op_manager.get_output_sharding(op_name)
        tuuid_to_name = self.op_manager.get_output_tuuid_to_name(op_name)
        step_tuuid_cache = dict(self._step_tuuid_cache.get(exec_step, {}))

        # Extract state arguments
        state_data = {}
        expected_state_names = self.op_manager.get_output_state_names(op_name)
        if expected_state_names:
            # Use tree_unflatten to reconstruct the original structure for state extraction
            # This is a temporary measure until state is properly integrated into the graph
            reconstructed_result = tree_unflatten(treedef, flat_results)
            for name in expected_state_names:
                val = getattr(reconstructed_result, name, None)
                if val is not None:
                    state_data[name] = val

        successor_contexts = []
        for spmd_id in successor_ids:
            succ_info = self.op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                raise ValueError(
                    f"[SPMDWorker] Missing successor topology for op={op_name} "
                    f"successor={spmd_id} source={self.spmd_id}:{self.rank}."
                )

            handlers = self._successor_handlers.get(spmd_id, {})
            if not handlers:
                raise ValueError(
                    f"[SPMDWorker] Missing successor handlers for op={op_name} "
                    f"successor={spmd_id} source={self.spmd_id}:{self.rank}."
                )
            if self._local_spmd_code is None:
                raise ValueError(
                    f"[SPMDWorker] local_spmd_code is not registered "
                    f"spmd_id={self.spmd_id} rank={self.rank}."
                )

            effective_rank = self.rank
            if succ_info.is_interleaved:
                effective_rank = self.rank % succ_info.interleave_unit_size

            successor_contexts.append(
                _SuccessorSendContext(
                    spmd_id=spmd_id,
                    info=succ_info,
                    handlers=handlers,
                    effective_rank=effective_rank,
                )
            )

        if successor_contexts:
            local_spmd_code = self._local_spmd_code
            assert local_spmd_code is not None
            await self._outbound_semaphore.acquire()
            self._spawn_outbound_task(
                self._send_step_data_plane_frames_background(
                    op_name,
                    flat_results,
                    output_uuids,
                    output_sharding,
                    successor_contexts,
                    tuuid_to_name,
                    step_tuuid_cache,
                    exec_step,
                    local_spmd_code,
                    self._world_size,
                )
            )

        for spmd_id in successor_ids:
            succ_info = self.op_manager.get_successor_info(op_name, spmd_id)
            if succ_info is None:
                raise ValueError(
                    f"[SPMDWorker] Missing successor topology for op={op_name} "
                    f"successor={spmd_id} source={self.spmd_id}:{self.rank}."
                )
            # --- Send control_argument via control plane ---
            if state_data:
                # Check condition
                if succ_info.condition_fn is not None:
                    if not succ_info.condition_fn(result):
                        continue

                # Collect all state names and values for this target
                names_to_send = []
                values_to_send = []
                # Leader of the active group/SPMD sends to avoid duplicates
                is_leader = self.rank % succ_info.interleave_unit_size == 0

                for state_name, state_value in state_data.items():
                    # Control states are mapped by name (str), not TUUID (int).
                    # TODO(He Jia): Currently we assume names match across the edge unless a state-specific map is added.
                    target_state_name = state_name
                    if is_leader:
                        names_to_send.append(target_state_name)
                        # Materialize device tensors to host numpy at the
                        # control-plane (Ray pickle) boundary so the
                        # backend-free driver can deserialize safely.
                        values_to_send.append(ensure_cpu_serializable(state_value))

                if names_to_send:
                    logger.debug(
                        f"Sending control_arguments names={names_to_send} "
                        f"step={exec_step} target_spmd_id={spmd_id} "
                        f"spmd_id={self.spmd_id} rank={self.rank}"
                    )
                    if self._driver_handle is not None:
                        try:
                            self._driver_handle.routing_control_argument_to_controller.remote(
                                spmd_id,
                                op_name,
                                names_to_send,
                                exec_step,
                                values_to_send,
                            )
                        except Exception as e:
                            logger.error(
                                f"Failed to send control_arguments via driver error={e} "
                                f"target_spmd_id={spmd_id} spmd_id={self.spmd_id} rank={self.rank}"
                            )

    async def _send_step_data_plane_frames_background(
        self,
        op_name: str,
        flat_results: list[Any],
        output_uuids: list[int],
        output_sharding: dict[int, ShardingSpec],
        successor_contexts: list[_SuccessorSendContext],
        tuuid_to_name: dict[int, str] | None,
        step_tuuid_cache: dict[int, Any],
        exec_step: int,
        sender_spmd_code: int,
        sender_world_size: int,
    ) -> None:
        dispatcher = self.op_manager.get_dispatcher(op_name)
        interleave_unit_size = getattr(dispatcher, "interleave_unit_size", None)
        route_contexts = [
            SuccessorRouteContext(
                context_index=index,
                spmd_id=context.spmd_id,
                info=context.info,
                effective_rank=context.effective_rank,
            )
            for index, context in enumerate(successor_contexts)
        ]
        await self._data_plane_executor.send_step_frames(
            op_name=op_name,
            rank=self.rank,
            mesh=self._mesh,
            flat_results=flat_results,
            output_uuids=output_uuids,
            output_sharding=output_sharding,
            successor_contexts=route_contexts,
            tuuid_to_name=tuuid_to_name,
            step_tuuid_cache=step_tuuid_cache,
            exec_step=exec_step,
            sender_spmd_code=sender_spmd_code,
            sender_world_size=sender_world_size,
            interleave_unit_size=interleave_unit_size,
            successor_handlers=[context.handlers for context in successor_contexts],
            source_spmd_id=self.spmd_id,
            max_in_flight_frames=self._max_in_flight,
        )

    def _reconstruct_data_type_from_dict(self, cls: type, data: dict[str, Any]) -> Any:
        """Reconstruct a @data_type or dataclass from a nested dictionary of fields."""
        from fluxway.common.decorators import _is_fluxway_data_type

        if not _is_fluxway_data_type(cls):
            # If not a data_type, it might be a simple type or a raw dict
            return data

        # Get type hints for constructor
        try:
            hints = get_type_hints(cls)
        except Exception:
            hints = {}

        kwargs = {}
        for name, hint in hints.items():
            if name in data:
                val = data[name]
                if isinstance(val, dict) and _is_fluxway_data_type(hint):
                    kwargs[name] = self._reconstruct_data_type_from_dict(hint, val)
                else:
                    kwargs[name] = val

        return cls(**kwargs)

    def _extract_control_arguments(
        self, data: Any, state_names: list[str]
    ) -> dict[str, Any]:
        """Extract named control argument values from operator output data."""
        args: dict[str, Any] = {}

        for name in state_names:
            val = getattr(data, name, None)
            if val is not None:
                args[name] = val

        if len(args) < len(state_names) and isinstance(data, dict):
            for name in state_names:
                if name not in args and name in data:
                    args[name] = data[name]

        return args

    # =========================================================================
    # Static Arguments
    # =========================================================================

    def set_static_arg(self, name: str, value: Any) -> None:
        """Set a static argument value."""
        self.op_manager.set_static_arg(name, value)

    def get_static_arg(self, name: str) -> Any | None:
        """Get a static argument value."""
        return self.op_manager.get_static_arg(name)

    # =========================================================================
    # State Argument Handling
    # =========================================================================

    def receive_control_argument(self, state_name: str, state_value: Any) -> None:
        """
        Receive control_argument from driver (broadcast from Controller).

        State arguments are set as static args for operator execution.

        Args:
            state_name: State variable name
            state_value: State value
        """
        self.set_static_arg(state_name, state_value)
        logger.debug(
            f"Received control_argument name={state_name} spmd_id={self.spmd_id} rank={self.rank}"
        )

    def send_control_argument(
        self,
        target_spmd_id: str,
        target_op_name: str,
        state_name: str,
        step: int,
        state_value: Any,
    ) -> None:
        """
        Send control_argument to target SPMD.

        Only rank 0 worker should call this method.
        Path: rank0-worker -> driver -> controller -> target-driver -> all-workers

        Args:
            target_spmd_id: Target SPMD ID
            target_op_name: Target operator type on the receiving side
            state_name: State variable name
            step: Current step number
            state_value: State value

        Raises:
            RuntimeError: If called from non-rank-0 worker
        """
        if self.rank != 0:
            raise RuntimeError("Only rank 0 worker can send control_argument")

        if self._driver_handle is None:
            logger.error(
                f"No driver handle cannot send control_argument "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
            return

        try:
            self._driver_handle.routing_control_argument_to_controller.remote(
                target_spmd_id,
                target_op_name,
                state_name,
                step,
                ensure_cpu_serializable(state_value),
            )
            logger.debug(
                f"Sent control_argument name={state_name} target_spmd_id={target_spmd_id} "
                f"target_op={target_op_name} step={step} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send control_argument error={e} spmd_id={self.spmd_id} rank={self.rank}"
            )

    # =========================================================================
    # Heartbeat
    # =========================================================================

    def _start_heartbeat(self) -> None:
        """Start heartbeat reporting to driver as an asyncio task."""
        if self._heartbeat_running:
            return

        self._heartbeat_running = True
        self._heartbeat_task = asyncio.get_event_loop().create_task(
            self._heartbeat_loop()
        )
        logger.debug(
            f"Worker heartbeat loop started spmd_id={self.spmd_id} rank={self.rank}"
        )

    async def stop_heartbeat(self) -> None:
        """Stop heartbeat reporting."""
        self._heartbeat_running = False
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

    async def _heartbeat_loop(self) -> None:
        """Heartbeat reporting loop (async)."""
        while self._heartbeat_running:
            try:
                await self._send_heartbeat()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"[spmd_id={self.spmd_id} rank={self.rank}] Error sending heartbeat: {e}"
                )

            await asyncio.sleep(self._heartbeat_interval)

    async def _send_heartbeat(self) -> None:
        """Send heartbeat to driver."""
        if self._driver_handle is None:
            return

        # Determine status
        if self._last_error:
            status = WorkerStatus.ERROR
        elif any(not event.is_set() for event in self._op_runnable.values()):
            status = WorkerStatus.BLOCKED
        elif self._running:
            status = WorkerStatus.RUNNING
        else:
            status = WorkerStatus.INITIALIZING

        # Get mailbox status for backpressure monitoring
        mailbox_status = await self._get_mailbox_status()

        heartbeat = WorkerHeartbeat(
            rank=self.rank,
            timestamp=time.time(),
            completed_step_count=dict(self._completed_step_count),
            status=status,
            error_message=self._last_error,
            telemetry=self._collect_telemetry(),
            mailbox_status=mailbox_status,
        )

        try:
            self._driver_handle.receive_worker_heartbeat.remote(self.rank, heartbeat)
            logger.heartbeat(
                f"Worker heartbeat sent status={status.name} "
                f"completed_step_count={heartbeat.completed_step_count} mailbox_full={mailbox_status.is_full} "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        except Exception as e:
            logger.error(
                f"Failed to send heartbeat error={e} spmd_id={self.spmd_id} rank={self.rank}"
            )

    async def _get_mailbox_status(self) -> MailboxStatus:
        """Get mailbox status for backpressure monitoring."""
        current_size = await self._mailbox.get_size()
        capacity = await self._mailbox.get_capacity()
        full_ratio = 0.8  # Consider full when 80% capacity reached
        is_full = current_size >= capacity * full_ratio

        return MailboxStatus(
            is_full=is_full,
            capacity=capacity,
            current_size=current_size,
            full_ratio=full_ratio,
        )

    def _collect_telemetry(self) -> WorkerTelemetry | None:
        """Collect worker telemetry data."""
        # Placeholder - implement actual telemetry collection
        return WorkerTelemetry(worker_id=f"{self.spmd_id}_{self.rank}")

    def _actor_owned_operator_names(self) -> set[str | None]:
        executor = self._local_step_plan_executor.user_callable_executor
        return executor.actor_owned_operator_names()

    # =========================================================================
    # State Persistence (checkpoint I/O)
    # =========================================================================

    async def trigger_snapshot_begin(self, ckpt_dir: str, step: int) -> None:
        """
        Trigger the snapshot begin hook for all operators on this worker.
        Called on Rank 0 by the Driver before parallel saving starts.
        """
        await self._drain_outbound_tasks()
        spmd_dir = str(Path(ckpt_dir) / str(step) / self.spmd_id)
        await self._local_step_plan_executor.user_callable_executor.trigger_snapshot_begin(
            spmd_dir
        )
        if self._op_manager:
            actor_owned_ops = self._actor_owned_operator_names()
            seen_cls_ids: set = set()
            storage, lock = get_instance_storage()
            for op_name, op_meta in (self.config.operator_metas or {}).items():
                if op_name in actor_owned_ops:
                    continue
                if not op_meta.owner_class_qualname:
                    continue
                cls = _resolve_owner_class(
                    op_meta.owner_class_module, op_meta.owner_class_qualname
                )
                if cls is None or id(cls) in seen_cls_ids:
                    continue
                seen_cls_ids.add(id(cls))
                with lock:
                    instance = storage.get(id(cls))
                if instance is None:
                    continue
                if hasattr(instance, "on_snapshot_begin") and callable(
                    instance.on_snapshot_begin
                ):
                    instance.on_snapshot_begin(spmd_dir)

    async def trigger_snapshot_commit(
        self, ckpt_dir: str, step: int, gathered_states: dict[str, dict[int, dict]]
    ) -> None:
        """
        Trigger the snapshot commit hook for all operators on this worker.
        Called on Rank 0 by the Driver after all ranks have saved successfully.
        """
        await self._drain_outbound_tasks()
        spmd_dir = str(Path(ckpt_dir) / str(step) / self.spmd_id)
        await self._local_step_plan_executor.user_callable_executor.trigger_snapshot_commit(
            spmd_dir,
            {
                op_name: gathered_states.get(op_name, {})
                for op_name in (self.config.operator_metas or {})
            },
        )
        if self._op_manager:
            actor_owned_ops = self._actor_owned_operator_names()
            seen_cls_ids: set = set()
            storage, lock = get_instance_storage()
            for op_name, op_meta in (self.config.operator_metas or {}).items():
                if op_name in actor_owned_ops:
                    continue
                if not op_meta.owner_class_qualname:
                    continue
                cls = _resolve_owner_class(
                    op_meta.owner_class_module, op_meta.owner_class_qualname
                )
                if cls is None or id(cls) in seen_cls_ids:
                    continue
                seen_cls_ids.add(id(cls))
                with lock:
                    instance = storage.get(id(cls))
                if instance is None:
                    continue
                if hasattr(instance, "on_snapshot_commit") and callable(
                    instance.on_snapshot_commit
                ):
                    op_states = gathered_states.get(op_name, {})
                    instance.on_snapshot_commit(spmd_dir, op_states)

    async def save_state_to_disk(
        self, ckpt_dir: str, step: int
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """
        Persist this worker's state to disk.

        Returns:
            A ``(spmd_state, rank_storage, operator_states)`` triple:

            - ``spmd_state``: SPMD-level logical state that is topology-independent
              (``completed_step_count``).  All ranks in the same SPMD share
              identical values; the Driver stores this once per SPMD.
            - ``rank_storage``: rank-specific physical storage paths returned by the
              checkpoint backend (``ckpt_dir``, ``orbax_final_path``,
              ``local_state_dir``, …).  Empty dict when this rank has no on-disk
              tensors.
            - ``operator_states``: raw per-operator save_state() return values.
              The Driver aggregates these across all ranks and passes them to
              ``trigger_snapshot_commit`` so that operators requiring cross-rank
              coordination (e.g. lance commit_fragments) can finalize.
        """
        await self._drain_outbound_tasks()
        completed_step_count_snapshot = dict(self._completed_step_count)

        spmd_dir = str(Path(ckpt_dir) / str(step) / self.spmd_id)
        operator_states = await self._collect_operator_save_state(spmd_dir)

        if operator_states:
            rank_storage = await self._get_checkpoint_backend().save(
                {"operator_states": operator_states},
                ckpt_dir,
                self.spmd_id,
                self.rank,
                step,
            )
        else:
            rank_storage = {}

        spmd_state = {
            "completed_step_count": completed_step_count_snapshot,
        }

        logger.debug(
            "State saved to disk spmd_id=%s rank=%s step=%s",
            self.spmd_id,
            self.rank,
            step,
        )
        return spmd_state, rank_storage, operator_states

    async def load_state_from_disk(
        self,
        spmd_state: dict[str, Any],
        rank_storage: dict[str, Any] | None,
    ) -> None:
        """
        Restore this worker's state from the manifest.

        Args:
            spmd_state:   SPMD-level logical state shared by all ranks
                          (``completed_step_count``).
            rank_storage: Rank-specific physical storage paths.  ``None`` when
                          this rank has no on-disk checkpoint (e.g. scale-up to
                          a larger worker count).
        """
        if "completed_step_count" not in spmd_state:
            raise RuntimeError(
                f"load_state_from_disk: 'completed_step_count' missing from spmd_state "
                f"for spmd_id={self.spmd_id} rank={self.rank}. "
                f"The manifest may be corrupt or was produced by an incompatible version."
            )
        restored_source_steps: list[tuple[str, int]] = []
        for op_name, count in spmd_state["completed_step_count"].items():
            self._completed_step_count[op_name] = count
            if self.op_manager.is_source_operator(op_name):
                max_steps = self.config.max_steps
                if max_steps is None or count < max_steps:
                    restored_source_steps.append((op_name, count))

        shared_operator_states = spmd_state.get("operator_states")
        if rank_storage is None:
            if shared_operator_states:
                await self._restore_operator_state(shared_operator_states, "")
                logger.debug(
                    "State loaded from shared operator metadata spmd_id=%s rank=%s",
                    self.spmd_id,
                    self.rank,
                )
                self._trigger_restored_source_steps(restored_source_steps)
                return
            logger.debug(
                "No on-disk or shared operator state for spmd_id=%s rank=%s",
                self.spmd_id,
                self.rank,
            )
            self._trigger_restored_source_steps(restored_source_steps)
            return

        target_state: dict[str, Any] = {
            "completed_step_count": self._completed_step_count,
        }
        if "orbax_final_path" in rank_storage:
            target_state["operator_states"] = self._call_operator_restore_target_state()
        elif shared_operator_states:
            target_state["operator_states"] = shared_operator_states
        state = await self._get_checkpoint_backend().load(
            rank_storage, target_state=target_state
        )

        ckpt_dir = rank_storage.get("ckpt_dir", "")
        operator_states = state.get("operator_states", shared_operator_states)
        if operator_states:
            await self._restore_operator_state(operator_states, ckpt_dir)
        logger.debug(
            "State loaded from disk spmd_id=%s rank=%s", self.spmd_id, self.rank
        )
        self._trigger_restored_source_steps(restored_source_steps)

    def _trigger_restored_source_steps(
        self, restored_source_steps: list[tuple[str, int]]
    ) -> None:
        for op_name, count in restored_source_steps:
            logger.info(
                "Auto-triggering source op for restored step %s op=%s "
                "spmd_id=%s rank=%s",
                count,
                op_name,
                self.spmd_id,
                self.rank,
            )
            self._spawn_background_task(self._enqueue(op_name, count))

    async def _collect_operator_save_state(self, ckpt_dir: str) -> dict[str, Any]:
        states = self._call_operator_save_state(ckpt_dir)
        actor_states = (
            await self._local_step_plan_executor.user_callable_executor.save_state(
                ckpt_dir
            )
        )
        states.update(actor_states)
        return states

    async def _restore_operator_state(
        self, operator_states: dict[str, Any], ckpt_dir: str
    ) -> None:
        local_states = self._filter_local_operator_states(operator_states)
        if local_states:
            self._call_operator_restore_state(local_states, ckpt_dir)
        await self._local_step_plan_executor.user_callable_executor.restore_state(
            operator_states, ckpt_dir
        )

    def _filter_local_operator_states(
        self, operator_states: dict[str, Any]
    ) -> dict[str, Any]:
        actor_owned_ops = self._actor_owned_operator_names()
        storage, lock = get_instance_storage()
        local_states: dict[str, Any] = {}
        for op_name, op_state in operator_states.items():
            if op_name in actor_owned_ops:
                continue
            op_meta = (self.config.operator_metas or {}).get(op_name)
            if op_meta is None:
                raise RuntimeError(
                    f"Operator {op_name} not found in config during restore "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
            if not op_meta.owner_class_qualname:
                continue
            cls = _resolve_owner_class(
                op_meta.owner_class_module, op_meta.owner_class_qualname
            )
            if cls is None:
                raise RuntimeError(
                    f"Operator '{op_name}' failed to resolve class "
                    f"spmd_id={self.spmd_id} rank={self.rank}"
                )
            with lock:
                instance = storage.get(id(cls))
            if instance is not None:
                local_states[op_name] = op_state
        return local_states

    def _call_operator_save_state(self, ckpt_dir: str) -> dict[str, Any]:
        """Call save_state() on each operator instance and collect results."""
        actor_owned_ops = self._actor_owned_operator_names()
        seen_cls_ids: set = set()
        states: dict[str, Any] = {}
        for op_name, op_meta in (self.config.operator_metas or {}).items():
            if op_name in actor_owned_ops:
                continue
            if not op_meta.owner_class_qualname:
                continue
            cls = _resolve_owner_class(
                op_meta.owner_class_module, op_meta.owner_class_qualname
            )
            if cls is None or id(cls) in seen_cls_ids:
                continue
            seen_cls_ids.add(id(cls))
            state = self._call_single_operator_save_state(op_name, ckpt_dir)
            if state is not None:
                states[op_name] = state
        return states

    def _call_single_operator_save_state(self, op_name: str, ckpt_dir: str) -> Any:
        op_meta = (self.config.operator_metas or {}).get(op_name)
        if op_meta is None or not op_meta.owner_class_qualname:
            return None
        cls = _resolve_owner_class(
            op_meta.owner_class_module, op_meta.owner_class_qualname
        )
        if cls is None:
            return None
        storage, lock = get_instance_storage()
        with lock:
            instance = storage.get(id(cls))
        if instance is None:
            return None
        if not (hasattr(instance, "save_state") and callable(instance.save_state)):
            return None
        if "ckpt_dir" in inspect.signature(instance.save_state).parameters:
            return instance.save_state(ckpt_dir=ckpt_dir)
        return instance.save_state()

    def _call_operator_restore_target_state(self) -> dict[str, Any]:
        actor_owned_ops = self._actor_owned_operator_names()
        seen_cls_ids: set = set()
        states: dict[str, Any] = {}
        storage, lock = get_instance_storage()
        for op_name, op_meta in (self.config.operator_metas or {}).items():
            if op_name in actor_owned_ops:
                continue
            if not op_meta.owner_class_qualname:
                continue
            cls = _resolve_owner_class(
                op_meta.owner_class_module, op_meta.owner_class_qualname
            )
            if cls is None or id(cls) in seen_cls_ids:
                continue
            seen_cls_ids.add(id(cls))
            with lock:
                instance = storage.get(id(cls))
            if instance is None:
                continue
            target_fn = getattr(instance, "restore_target_state", None)
            if callable(target_fn):
                states[op_name] = target_fn()
        if not states:
            raise RuntimeError(
                f"Global checkpoint restore requires at least one operator to "
                f"implement restore_target_state() spmd_id={self.spmd_id} "
                f"rank={self.rank}"
            )
        return states

    def _call_operator_restore_state(
        self, operator_states: dict[str, Any], ckpt_dir: str
    ) -> None:
        """Call restore_state() on each operator instance with its persisted state."""
        for op_name, op_state in operator_states.items():
            self._call_single_operator_restore_state(op_name, op_state, ckpt_dir)

    def _call_single_operator_restore_state(
        self, op_name: str, op_state: Any, ckpt_dir: str
    ) -> None:
        storage, lock = get_instance_storage()
        op_meta = (self.config.operator_metas or {}).get(op_name)
        if op_meta is None:
            raise RuntimeError(
                f"Operator {op_name} not found in config during restore "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        cls = _resolve_owner_class(
            op_meta.owner_class_module, op_meta.owner_class_qualname
        )
        if cls is None:
            raise RuntimeError(
                f"Operator '{op_name}' failed to resolve class "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        with lock:
            instance = storage.get(id(cls))
        if instance is None:
            raise RuntimeError(
                f"Operator {op_name} instance not found in _instance_storage during restore "
                f"spmd_id={self.spmd_id} rank={self.rank}"
            )
        if hasattr(instance, "restore_state") and callable(instance.restore_state):
            if "ckpt_dir" in inspect.signature(instance.restore_state).parameters:
                instance.restore_state(op_state, ckpt_dir=ckpt_dir)
            else:
                instance.restore_state(op_state)

    async def inspect_operator_state(self) -> dict[str, Any]:
        seen_cls_ids: set[int] = set()
        snapshots: dict[str, Any] = {}
        storage, lock = get_instance_storage()
        for op_name, op_meta in (self.config.operator_metas or {}).items():
            if not op_meta.owner_class_qualname:
                continue
            cls = _resolve_owner_class(
                op_meta.owner_class_module, op_meta.owner_class_qualname
            )
            if cls is None or id(cls) in seen_cls_ids:
                continue
            seen_cls_ids.add(id(cls))
            with lock:
                instance = storage.get(id(cls))
            if instance is None:
                continue
            snapshot_fn = getattr(instance, "audit_state_snapshot", None)
            if callable(snapshot_fn):
                snapshots[op_name] = snapshot_fn()
        operator_inputs: dict[str, Any] = {}
        tuuid_consumers: dict[str, list[str]] = {}
        if self._op_manager is not None:
            for op_name in sorted(self.config.operator_metas or {}):
                input_sharding = self.op_manager.get_input_sharding(op_name)
                expected_tuuids = self.op_manager.get_expected_tuuids(op_name)
                predecessor_topology = self.op_manager.get_predecessor_topology(op_name)
                successor_topology = self.op_manager.get_successor_topology(op_name)
                operator_inputs[op_name] = {
                    "input_sharding_tuuids": sorted(u for u in input_sharding),
                    "expected_tuuids": sorted(u for u in expected_tuuids),
                    "output_tuuids": sorted(
                        u for u in self.op_manager.get_output_tuuids(op_name)
                    ),
                    "predecessors": {
                        spmd_id: {
                            "world_size": info.world_size,
                            "tuuid_map": dict(sorted(info.tuuid_map.items())),
                            "tuuid_sharding": sorted(
                                tuuid for tuuid in info.tuuid_sharding
                            ),
                        }
                        for spmd_id, info in sorted(predecessor_topology.items())
                    },
                    "successors": {
                        spmd_id: {
                            "world_size": info.world_size,
                            "tuuid_map": dict(sorted(info.tuuid_map.items())),
                            "tuuid_sharding": sorted(
                                tuuid for tuuid in info.tuuid_sharding
                            ),
                        }
                        for spmd_id, info in sorted(successor_topology.items())
                    },
                }
                for tuuid in input_sharding:
                    tuuid_consumers[str(tuuid)] = (
                        self.op_manager.get_operators_for_tuuid(tuuid)
                    )
        return {
            "spmd_id": self.spmd_id,
            "rank": self.rank,
            "world_size": self._world_size,
            "completed_step_count": dict(self._completed_step_count),
            "operators": snapshots,
            "runtime": {
                "drain_target_count": self._drain_target_count,
                "barriers_declared": [
                    [op_name, step] for op_name, step in sorted(self._barriers_declared)
                ],
                "blocked_ops": [
                    op_name
                    for op_name, event in sorted(self._op_runnable.items())
                    if not event.is_set()
                ],
                "pending_blocked": {
                    op_name: list(steps)
                    for op_name, steps in sorted(self._pending_blocked.items())
                    if steps
                },
                "pending_future_steps": {
                    op_name: list(steps)
                    for op_name, steps in sorted(self._pending_future_steps.items())
                    if steps
                },
                "execution_queue_size": self._execution_queue.qsize(),
                "in_op_step_queue": [
                    [op_name, step] for op_name, step in sorted(self._in_op_step_queue)
                ],
                "mailbox": await self._mailbox.snapshot(),
                "operator_inputs": operator_inputs,
                "tuuid_consumers": tuuid_consumers,
            },
        }

    # =========================================================================
    # Utility
    # =========================================================================

    def get_current_step(self, op_name: str) -> int:
        """Get current step for an operator name."""
        return self._completed_step_count[op_name]

    def get_rank(self) -> int:
        """Get this worker's rank."""
        return self.rank

    def is_running(self) -> bool:
        """Check if worker is running."""
        return self._running

    # =========================================================================
    # Drain Control (called by Driver before drain_to_step_count)
    # =========================================================================

    async def set_drain_target_count(self, target_count: int) -> None:
        """
        Request worker to self-block once 'target_count' steps are finished.
        """
        async with self._drain_cond:
            self._drain_target_count = target_count
            self._drain_cond.notify_all()
            logger.debug(
                f"Drain target set to {target_count} spmd_id={self.spmd_id} rank={self.rank}"
            )

    async def clear_drain_target_count(self) -> None:
        """Remove the drain target and let deferred executions proceed."""
        async with self._drain_cond:
            self._drain_target_count = None
            self._drain_cond.notify_all()
            logger.debug(
                f"Drain target cleared spmd_id={self.spmd_id} rank={self.rank}"
            )

    # =========================================================================
    # Source Node Control
    # =========================================================================

    def pause_sources(self) -> None:
        """
        Pause all source operators (called by driver when downstream is busy).
        """
        for event in self._source_pause_events.values():
            event.clear()
        logger.debug(
            f"Sources paused on worker spmd_id={self.spmd_id} rank={self.rank}"
        )

    def resume_sources(self) -> None:
        """
        Resume all source operators (called by driver when backpressure relieved).
        """
        for event in self._source_pause_events.values():
            event.set()
        logger.debug(
            f"Sources resumed on worker spmd_id={self.spmd_id} rank={self.rank}"
        )
