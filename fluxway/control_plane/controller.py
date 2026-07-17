"""
Controller - Central coordination node for the Control Plane (Ray Actor).

Based on ARCHITECTURE.md Section 2.1

The Controller orchestrates the training job by coordinating:
- Monitor: Cluster state monitoring and telemetry
- Scheduler: Resource allocation via Ray Placement Groups
- SPMDManager: SPMD lifecycle management
- StateManager: Global state and checkpointing
- HeartbeatManager: Driver heartbeat management
- ControllerStateMachine: Lifecycle state management
- AsyncStepManager: SPMD async step difference control

Implemented as a Ray Actor for async operation.
"""

import asyncio
import functools
import inspect
import os
from pathlib import Path
from typing import Any, TypeVar
from collections.abc import Callable

import numpy as np
import ray

from fluxway.common.types import (
    AsyncStepConfig,
    ControllerEvent,
    ControllerState,
    DriverHeartbeat,
    ExecutionFailure,
    FlowControlState,
    Graph,
    HeartbeatConfig,
)
from fluxway.common.utils import ensure_logging_configured
from fluxway.common.utils.logging import get_logger
from fluxway.control_plane.monitor import Monitor
from fluxway.control_plane.scheduler import Scheduler, build_topology
from fluxway.control_plane.spmd_manager import SPMDManager, SPMDHandle
from fluxway.control_plane.state_manager import StateManager
from fluxway.control_plane.heartbeat import HeartbeatManager
from fluxway.control_plane.flow_control import FlowControlManager
from fluxway.control_plane.state_machine import ControllerStateMachine
from fluxway.control_plane.async_manager import AsyncStepManager
from fluxway.control_plane.errors import ExecutionFailedError
from fluxway.common.types import CheckpointConfig
from fluxway.common.types.drain import DrainStrategy, GraphModeDrainStrategy
from fluxway.rpc.protocol import DataPlaneRegistry
from fluxway.profiler import ProfilerRuntimeFactory
from fluxway.profiler.record import ProfilerSnapshot
from fluxway.placement import (
    PlacementContractError,
    PlacementError,
    PlacementRecipe,
    PlacementRuntimeConfig,
)

logger = get_logger(__name__)


_F = TypeVar("_F", bound=Callable[..., Any])


def require_manager(manager_attr: str) -> Callable[[_F], _F]:
    """
    Decorator to ensure that a manager component is initialized before calling
    the method.

    Two cases when the manager attribute is None:
    - Controller not yet initialized (_initialized=False): transient startup race,
      the caller (driver) will retry on the next heartbeat cycle. Log a warning
      and return without raising so the fire-and-forget ObjectRef is not lost silently.
    - Controller already initialized (_initialized=True): this is a post-init bug;
      raise RuntimeError unconditionally.
    """

    def decorator(func: _F) -> _F:
        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                self = args[0]
                if getattr(self, manager_attr) is None:
                    if not self._initialized:
                        logger.warning(
                            f"{func.__name__} called before controller initialized, dropping"
                        )
                        return None
                    raise RuntimeError(
                        f"{manager_attr.strip('_')} is None after initialization "
                        f"when calling {func.__name__}"
                    )
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                self = args[0]
                if getattr(self, manager_attr) is None:
                    if not self._initialized:
                        logger.warning(
                            f"{func.__name__} called before controller initialized, dropping"
                        )
                        return None
                    raise RuntimeError(
                        f"{manager_attr.strip('_')} is None after initialization "
                        f"when calling {func.__name__}"
                    )
                return func(*args, **kwargs)

            return sync_wrapper  # type: ignore[return-value]

    return decorator


def require_spmd_manager(func):
    return require_manager("_spmd_manager")(func)


def require_heartbeat_manager(func):
    return require_manager("_heartbeat_manager")(func)


@ray.remote
class ControllerActor:
    """
    Central Controller for the Fluxway Control Plane (Ray Actor).

    Coordinates all components for distributed training orchestration:
    - Initializes and manages the training graph
    - Allocates resources via Scheduler
    - Manages SPMD instances via SPMDManager
    - Tracks state via StateManager
    - Monitors cluster health via Monitor
    - Manages driver heartbeats via HeartbeatManager
    - Controls lifecycle via ControllerStateMachine
    - Enforces async step limits via AsyncStepManager

    Note: Resource-related operations are proxied through Scheduler.
    Controller can get SPMD view from spmd_manager but without resource info.
    """

    # ------------------------------------------------------------------
    # Properties with cloudpickle-safe guards
    #
    # Ray's metaclass calls getattr(object.__new__(cls), attr) during
    # export_actor_class.  __init__ has not run, so backing attrs are
    # absent from __dict__.  Raising AttributeError here is intentional:
    # cloudpickle's _lookup_module_and_qualname wraps the traversal in
    # `except AttributeError` and silently falls back to by-value
    # pickling, which is harmless.  RuntimeError (not a subclass of
    # AttributeError) would propagate and crash serialization.
    # ------------------------------------------------------------------

    @property
    def spmd_manager(self) -> "SPMDManager":
        if "_spmd_manager" not in self.__dict__:
            raise AttributeError("spmd_manager")
        value = self.__dict__["_spmd_manager"]
        if value is None:
            raise RuntimeError("SPMDManager is not initialized")
        return value

    @property
    def scheduler(self) -> "Scheduler":
        if "_scheduler" not in self.__dict__:
            raise AttributeError("scheduler")
        return self.__dict__["_scheduler"]

    @property
    def heartbeat_manager(self) -> "HeartbeatManager":
        if "_heartbeat_manager" not in self.__dict__:
            raise AttributeError("heartbeat_manager")
        value = self.__dict__["_heartbeat_manager"]
        if value is None:
            raise RuntimeError("HeartbeatManager is not initialized")
        return value

    @property
    def async_step_manager(self) -> "AsyncStepManager":
        if "_async_step_manager" not in self.__dict__:
            raise AttributeError("async_step_manager")
        return self.__dict__["_async_step_manager"]

    @property
    def state_manager(self) -> "StateManager":
        if "_state_manager" not in self.__dict__:
            raise AttributeError("state_manager")
        return self.__dict__["_state_manager"]

    @property
    def state_machine(self) -> "ControllerStateMachine":
        if "_state_machine" not in self.__dict__:
            raise AttributeError("state_machine")
        return self.__dict__["_state_machine"]

    @property
    def monitor(self) -> "Monitor":
        if "_monitor" not in self.__dict__:
            raise AttributeError("monitor")
        return self.__dict__["_monitor"]

    def __init__(
        self,
        graph: Graph,
        placement_recipe: PlacementRecipe,
        heartbeat_config: HeartbeatConfig | None = None,
        async_step_config: AsyncStepConfig | None = None,
        checkpoint_config: CheckpointConfig | None = None,
        transport_config: dict[str, str] | None = None,
        profiler_config: Any | None = None,
        placement_runtime: PlacementRuntimeConfig | None = None,
    ):
        """
        Initialize Controller with configuration.

        Args:
            graph: The task graph defining SPMD nodes and edges
            heartbeat_config: Heartbeat configuration
            async_step_config: Async step configuration
            checkpoint_config: Configuration for checkpointer backend
            transport_config: Data plane transport backend per SPMD (e.g. {"*": "ray"})
            profiler_config: Optional profiler runtime configuration
        """
        # Ray actors run in separate worker processes; ensure INFO logs are not dropped.
        ensure_logging_configured()

        self._graph = graph
        self._placement_recipe = placement_recipe
        self._heartbeat_config = heartbeat_config or HeartbeatConfig()
        self._async_step_config = async_step_config or AsyncStepConfig()
        self._checkpoint_config = checkpoint_config
        self._transport_config = transport_config or {}
        self._profiler_config = profiler_config
        self._profiler_runtime = ProfilerRuntimeFactory.from_config(
            profiler_config,
            process_name="controller",
        )
        self._heartbeat_observer = self._profiler_runtime.heartbeat_observer()

        self._monitor = Monitor()
        self._state_manager = StateManager()
        self._scheduler = Scheduler(
            spmd_manager=None,
            graph=graph,
            placement_recipe=placement_recipe,
            placement_runtime=placement_runtime or PlacementRuntimeConfig(),
        )
        self._state_machine = ControllerStateMachine(self)
        self._async_step_manager = AsyncStepManager(
            self._graph, self._async_step_config
        )
        self._flow_control_manager = FlowControlManager(
            self._graph, self._async_step_config
        )
        # snapshot_id → {expected, received, metadata, event}
        self._pending_snapshots: dict[str, Any] = {}

        self._checkpoint_task: asyncio.Task | None = None
        self._last_checkpointed_step: int = 0

        # Track source nodes (entry points)
        self._source_nodes = (
            self._graph.source_nodes
            if self._graph.source_nodes
            else self._find_source_nodes()
        )

        self._initialized = False
        self._spmd_manager: SPMDManager | None = None
        self._heartbeat_manager: HeartbeatManager | None = None
        self._checkpointer = None
        self._drain_strategy: DrainStrategy | None = None
        self._last_requested_drain_target: int = 0
        self._last_completed_drain_target: int = 0
        self._drained_drivers: set = set()
        self._ready_event = asyncio.Event()

        logger.debug(f"ControllerActor created with {len(self._graph.nodes)} nodes")

    def _find_source_nodes(self) -> list[str]:
        """Find nodes with no predecessors (source nodes)."""
        return list(self._graph.require_compiled_graph().source_spmd_ids)

    def get_resource_allocation_summary(self) -> dict[str, Any]:
        """Expose scheduler resource summaries for debugging and inspection."""
        if self.scheduler.allocation_report is not None:
            return self.scheduler.allocation_report.model_dump(exclude={"recipe"})
        if self.scheduler.resolved_plan is not None:
            return self.scheduler.resolved_plan.model_dump(exclude={"recipe"})
        raise PlacementContractError("resource allocation has not been resolved")

    async def initialize(self) -> bool:
        """
        Initialize the Controller (event-driven).

        Performs:
        1. Set controller handle on spmd_manager
        2. Register SPMDs for heartbeat monitoring
        3. Allocate resources
        4. Build topology data structure (no remote calls)
        5. Start SPMDs (fire-and-forget)
        6. Transition to PENDING state

        Subsequent phases are driven by heartbeat events:
        - PENDING → TOPOLOGY_REGISTERING: when all drivers report workers_ready
        - TOPOLOGY_REGISTERING → RUNNING: when all workers are running

        Note: Heartbeat timeout monitoring starts in RUNNING state entry,
        not here. Heartbeat *reception* works from the start.

        Returns:
            True if initialization successful
        """
        profiler_recorder = (
            self._profiler_runtime.recorder if self._profiler_runtime.enabled else None
        )
        if profiler_recorder is not None:
            span = profiler_recorder.controller_initialize(
                spmd_count=len(self._graph.nodes)
            )
            await span.__aenter__()
        else:
            span = None
        try:
            # Create sub-components locally in the controller process
            self_handle = ray.get_runtime_context().current_actor
            self._spmd_manager = SPMDManager(
                controller_handle=self_handle,
                profiler_recorder=profiler_recorder,
            )
            self.scheduler.set_spmd_manager(
                self.spmd_manager
            )  # Set SPMDManager on scheduler

            self._heartbeat_manager = HeartbeatManager(
                config=self._heartbeat_config,
                on_all_ready=self._on_all_drivers_ready,
                on_all_workers_running=self._on_all_workers_running,
                on_driver_error=self._on_driver_error,
                observer=self._heartbeat_observer,
            )

            if self._checkpoint_config is not None:
                from fluxway.control_plane.checkpointer import (
                    FluxwayCheckpointer,
                    WorkerCheckpointBackend,
                )

                self._checkpointer = FluxwayCheckpointer(
                    self,
                    ckpt_dir=self._checkpoint_config.ckpt_dir,
                    interval_steps=self._checkpoint_config.interval_steps,
                    force_write=self._checkpoint_config.force_write,
                )
                self.spmd_manager.set_checkpoint_backend(
                    WorkerCheckpointBackend.create(
                        self._checkpoint_config.backend,
                        force_write=self._checkpoint_config.force_write,
                    )
                )
                self._drain_strategy = GraphModeDrainStrategy(
                    interval_steps=self._checkpoint_config.interval_steps
                )

            # Assign specific max_steps to each node based on global config and overrides
            for node_id, node in self._graph.nodes.items():
                if node.spmd_config is not None:
                    node.spmd_config.profiler_config = self._profiler_config
                    limit = self._async_step_config.node_max_steps.get(
                        node_id, self._async_step_config.max_steps
                    )
                    if limit is not None:
                        node.spmd_config.max_steps = limit
                        logger.debug(f"Assigned max_steps={limit} to node {node_id}")

                    node.spmd_config.auto_start_source_operators = False
                    if self._drain_strategy is not None:
                        node.spmd_config.requires_drain_at_startup = True

            # Register SPMDs for heartbeat monitoring
            for node_name in self._graph.nodes:
                self.heartbeat_manager.register_spmd(node_name)

            # Allocate resources
            if profiler_recorder is None:
                await self.scheduler.allocate_resources()
            else:
                async with profiler_recorder.spmd_manager_placement_group_create(
                    spmd_id="*",
                    worker_count=sum(
                        node.spmd_config.num_workers
                        for node in self._graph.nodes.values()
                        if node.spmd_config is not None
                    ),
                ):
                    await self.scheduler.allocate_resources()

            # Build and establish topology data structure
            # (adjacency lists only, no remote calls to drivers)
            if profiler_recorder is None:
                self.scheduler.establish_topology(
                    build_topology(self._graph.require_compiled_graph())
                )
            else:
                async with profiler_recorder.controller_topology_registration(
                    spmd_count=len(self._graph.nodes),
                    edge_count=len(self._graph.require_compiled_graph().dependencies),
                ):
                    self.scheduler.establish_topology(
                        build_topology(self._graph.require_compiled_graph())
                    )

            # Start SPMDs (fire-and-forget, readiness tracked via heartbeat)
            if profiler_recorder is None:
                await self.scheduler.init_spmds()
            else:
                async with profiler_recorder.spmd_manager_driver_start(
                    spmd_id="*",
                    worker_count=len(self._graph.nodes),
                ):
                    await self.scheduler.init_spmds()

            # Transition state machine: INITIALIZING → PENDING
            await self.state_machine.trigger_event(ControllerEvent.INIT_COMPLETE)

            self._initialized = True
            logger.debug(
                f"Controller initialized with {len(self._graph.nodes)} SPMDs, "
                f"waiting for drivers to report ready via heartbeat"
            )
            return True

        except PlacementError as e:
            self.scheduler.abort_initialization(e)
            logger.error(f"Controller placement initialization failed: {e}")
            raise
        except Exception as e:
            self.scheduler.abort_initialization(e)
            logger.error(f"Controller initialization failed: {e}")
            raise
        finally:
            if span is not None:
                await span.__aexit__(None, None, None)

    async def start(self, resume: str | int | None = None) -> int | None:
        """
        Main execution entry point.
        Sequence: Initialize -> wait READY -> restore -> START -> drain -> start source.
        """
        if self._initialized:
            raise RuntimeError("Controller is already initialized or started")

        try:
            # 1. Initialize
            await self.initialize()

            # 2. Wait for READY
            await self._ready_event.wait()
            if self.state_machine.state != ControllerState.READY:
                raise RuntimeError(
                    f"Failed to reach READY state, current state: {self.state_machine.state}"
                )

            # 3. Restore
            restored_step = None
            if resume is not None:
                if self._checkpointer is None:
                    raise ValueError("Cannot resume without a checkpoint_config")
                restored_step = await self._checkpointer.restore(resume)

            # 4. Trigger START -> RUNNING
            success = await self.state_machine.trigger_event(ControllerEvent.START)
            if not success:
                raise RuntimeError("Failed to transition to RUNNING from READY")

            # 5. Issue initial drain
            if getattr(self, "_issue_initial_drain", None):
                await self._issue_initial_drain()

            # 6. Start source operators
            start_sources = getattr(self, "start_source_operators", None)
            if start_sources:
                await start_sources()

            return restored_step

        except Exception as e:
            logger.error(f"Failed during controller start: {e}")
            from fluxway.common.types import ExecutionFailure

            await self.state_machine.trigger_event(
                ControllerEvent.DRIVER_ERROR,
                spmd_id="controller",
                failure=ExecutionFailure.from_exception(
                    spmd_id="controller", rank=None, phase="STARTUP", exception=e
                ),
            )
            raise

    # =====================================================================
    # Properties (as methods for Ray Actor)
    # =====================================================================

    def get_graph(self) -> Graph:
        """Get the task graph."""
        return self._graph

    def get_state(self) -> ControllerState:
        """Get current controller state."""
        return self.state_machine.state

    async def get_status(self) -> dict[str, Any]:
        """Get detailed controller status."""
        return await self.get_cluster_status()

    @require_heartbeat_manager
    async def get_all_heartbeats(self) -> dict[str, Any]:
        """Get all latest heartbeats from all SPMDs."""
        return await self.heartbeat_manager.get_all_heartbeats()

    async def get_async_step_progress(self) -> dict[str, dict[str, int]]:
        steps = self.async_step_manager.get_all_spmd_steps()
        return {
            spmd_id: dict(spmd_step.completed_step_count)
            for spmd_id, spmd_step in steps.items()
        }

    def is_initialized(self) -> bool:
        """Check if controller is initialized."""
        return self._initialized

    # =====================================================================
    # SPMD Handle Access
    # =====================================================================

    @require_spmd_manager
    def get_spmd_handle(self, spmd_id: str) -> SPMDHandle | None:
        """
        Get handle to an SPMD instance from spmd_manager.

        Args:
            spmd_id: ID of the SPMD

        Returns:
            SPMDHandle or None if not found
        """
        return self.spmd_manager.get_spmd_handle(spmd_id)

    # =====================================================================
    # Heartbeat Interface (called by drivers)
    # =====================================================================

    @require_heartbeat_manager
    async def receive_driver_heartbeat(self, heartbeat: DriverHeartbeat) -> None:
        """
        Receive heartbeat from a driver.

        Args:
            heartbeat: Driver heartbeat data
        """
        if self._heartbeat_observer is not None:
            async with self._heartbeat_observer.receive_driver_heartbeat(heartbeat):
                await self._receive_driver_heartbeat_unprofiled(heartbeat)
            return
        await self._receive_driver_heartbeat_unprofiled(heartbeat)

    async def _receive_driver_heartbeat_unprofiled(
        self, heartbeat: DriverHeartbeat
    ) -> None:
        await self.heartbeat_manager.receive_heartbeat(heartbeat)

    # =====================================================================
    # Op Completion Interface (called by drivers)
    # =====================================================================

    @require_spmd_manager
    async def report_op_completed(
        self, spmd_id: str, op_name: str, completed_count: int
    ) -> None:
        """
        Receive op completion report from a driver.

        Args:
            spmd_id: SPMD identifier
            op_name: Operation name
            completed_count: Completed count (number of steps finished)
        """
        self.async_step_manager.report_op_completed(spmd_id, op_name, completed_count)
        await self._apply_flow_control_windows(
            self._flow_control_manager.observe_op_completed(
                spmd_id, op_name, completed_count
            )
        )
        if (
            self._drain_strategy is None
            and self.async_step_manager.is_all_completed()
            and self.state_machine.is_running()
        ):
            await self.state_machine.trigger_event(
                ControllerEvent.ALL_DRIVERS_COMPLETED
            )

    async def report_flow_control_state(self, state: FlowControlState) -> None:
        self._flow_control_manager.report_state(state)

    @require_spmd_manager
    async def _apply_flow_control_windows(
        self, windows_by_spmd_id: dict[str, tuple[Any, ...]]
    ) -> None:
        apply_tasks = []
        for spmd_id, windows in windows_by_spmd_id.items():
            if not windows:
                continue
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"Missing driver handle for SPMD {spmd_id!r} "
                    "while applying flow-control windows."
                )
            apply_tasks.append(driver.apply_credit_window.remote(windows))
        if apply_tasks:
            await asyncio.gather(*apply_tasks)

    async def report_driver_drained(self, spmd_id: str) -> None:
        """Called by a driver when it has successfully drained to the target."""
        self._drained_drivers.add(spmd_id)
        if len(self._drained_drivers) == len(self._graph.nodes):
            logger.debug("All drivers have reached the drain target")
            await self._on_all_drivers_drained()

    async def _issue_initial_drain(self) -> None:
        """Issues the very first drain target to unlock workers at step 0."""
        if self._drain_strategy is not None:
            initial_target = self._drain_strategy.get_initial_drain_target(
                self._last_requested_drain_target
            )
            if initial_target is not None:
                max_steps = getattr(self.async_step_manager._config, "max_steps", None)
                if max_steps is not None:
                    initial_target = min(initial_target, max_steps)
                logger.info(f"Issuing initial drain target: {initial_target}")
                await self._schedule_drain(initial_target)

    async def _schedule_drain(self, target: int) -> None:
        """Broadcast a new drain target to all drivers."""
        max_steps = getattr(self.async_step_manager._config, "max_steps", None)
        if max_steps is not None:
            target = min(target, max_steps)

        self._last_requested_drain_target = target
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_drain(
                target_step=target,
                spmd_count=len(self._graph.nodes),
            ):
                await self._schedule_drain_unprofiled(target)
            return
        await self._schedule_drain_unprofiled(target)

    async def _schedule_drain_unprofiled(self, target: int) -> None:
        driver_tasks = []
        for spmd_id in self._graph.nodes.keys():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver:
                driver_tasks.append(driver.request_drain.remote(target))
        if driver_tasks:
            await asyncio.gather(*driver_tasks)

    async def _on_all_drivers_drained(self) -> None:
        """Triggered when all drivers are blocked at the designated drain target."""
        self._drained_drivers.clear()
        if self._checkpointer is not None:
            boundary = self._last_requested_drain_target

            # Skip redundant saves: if the checkpoint at this boundary
            # was already persisted (e.g. we just restored to this step
            # and no new training has occurred).
            if boundary == self._last_checkpointed_step and boundary > 0:
                logger.info(
                    f"Skipping checkpoint at step {boundary} "
                    f"(already persisted at this step)"
                )
                await self._resume_and_schedule_next_drain()
                return

            # Issue snapshot
            async def do_ckpt(boundary: int) -> None:
                logger.info(f"Checkpointer executing snapshot at count {boundary}")
                try:
                    if self._checkpointer is not None:
                        if self._profiler_runtime.enabled:
                            async with (
                                self._profiler_runtime.recorder.controller_checkpoint(
                                    target_step=boundary,
                                )
                            ):
                                await self._checkpointer.save(boundary)
                        else:
                            await self._checkpointer.save(boundary)
                except Exception as e:
                    logger.error(f"Checkpoint save failed: {e}")
                finally:
                    # Always release drain and schedule the next iteration
                    await self._resume_and_schedule_next_drain()

            self._checkpoint_task = asyncio.create_task(do_ckpt(boundary))
        else:
            # No checkpointer configured.
            # In future 'eager' mode, this would notify the caller.
            await self._resume_and_schedule_next_drain()

    async def _resume_and_schedule_next_drain(self) -> None:
        self._last_completed_drain_target = self._last_requested_drain_target
        max_steps = getattr(self.async_step_manager._config, "max_steps", None)

        if max_steps is not None and self._last_completed_drain_target >= max_steps:
            logger.debug(
                "Final drain completed (reached max_steps), triggering completion"
            )
            await self.state_machine.trigger_event(
                ControllerEvent.ALL_DRIVERS_COMPLETED
            )
            return

        # 1. Clear drain / Snapshot completed
        futures = []
        for spmd_id in self._graph.nodes.keys():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                futures.append(driver.resume.remote())
        if futures:
            await asyncio.gather(*futures, return_exceptions=True)

        # 2. Schedule next drain
        if self._drain_strategy is not None:
            next_target = self._drain_strategy.get_next_drain_target(
                self._last_requested_drain_target
            )
            if next_target is not None:
                await self._schedule_drain(next_target)

    @require_spmd_manager
    async def _lock_spmd_op(self, spmd_id: str, op_name: str) -> None:
        """Lock an SPMD's operation."""
        driver = self.spmd_manager.get_driver_handle(spmd_id)
        if driver is not None:
            try:
                await driver.block_op.remote(op_name)
                # logger.debug(f"Locked {spmd_id} {op_name}")
            except Exception as e:
                logger.error(f"Failed to lock {spmd_id} {op_name}: {e}")

    @require_spmd_manager
    async def _unlock_spmd_op(self, spmd_id: str, op_name: str) -> None:
        """Unlock an SPMD's operation."""
        driver = self.spmd_manager.get_driver_handle(spmd_id)
        if driver is not None:
            try:
                await driver.unblock_op.remote(op_name)
                # logger.debug(f"Unlocked {spmd_id} {op_name}")
            except Exception as e:
                logger.error(f"Failed to unlock {spmd_id} {op_name}: {e}")

    # =====================================================================
    # State Argument Transport
    # =====================================================================

    @require_spmd_manager
    async def receive_control_argument(
        self,
        source_spmd_id: str,
        target_spmd_id: str,
        target_op: str,
        names: list[str],
        step: int,
        values: list[Any],
    ) -> None:
        """
        Receive control_arguments and forward to target SPMD.

        Transport path: rank0-worker -> source-driver -> controller
                       -> target-driver -> all-workers

        Args:
            source_spmd_id: Source SPMD ID
            target_spmd_id: Target SPMD ID
            target_op: Target operator type on the receiving side
            names: State variable names
            step: Current step number
            values: State values
        """
        target_driver = self.spmd_manager.get_driver_handle(target_spmd_id)
        if target_driver is None:
            logger.error(
                f"Target SPMD {target_spmd_id} not found for control_arguments"
            )
            return

        try:
            await target_driver.broadcast_control_argument.remote(
                target_op, names, step, values
            )
            logger.debug(
                f"Forwarded control_arguments {names} from {source_spmd_id} "
                f"to {target_spmd_id}, target_op={target_op}, step={step}"
            )
        except Exception as e:
            logger.error(f"Failed to forward control_arguments: {e}")

    # =====================================================================
    # Global Snapshotting
    # =====================================================================

    @require_spmd_manager
    async def request_snapshot(
        self, snapshot_id: str, target_count: int, ckpt_dir: str
    ) -> dict[str, Any]:
        """
        Trigger a global snapshot and block until all Workers have persisted
        their state and all Drivers have reported metadata back.

        Returns the assembled manifest directly — no polling required.
        """
        logger.info(
            f"Requesting global snapshot '{snapshot_id}' at step {target_count}"
        )
        all_spmd_ids = self.spmd_manager.get_all_spmd_ids()
        if not all_spmd_ids:
            return {}

        # validate all drivers exist BEFORE registering the snapshot entry
        # and before dispatching any RPCs.
        drivers: dict[str, Any] = {}
        for spmd_id in all_spmd_ids:
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is None:
                raise RuntimeError(
                    f"request_snapshot: driver for spmd_id '{spmd_id}' not found. "
                    f"Cannot perform partial snapshot — aborting to prevent corrupt manifest."
                )
            drivers[spmd_id] = driver

        # Register the accounting entry, then dispatch — all drivers confirmed.
        done_event = asyncio.Event()
        self._pending_snapshots[snapshot_id] = {
            "expected": len(all_spmd_ids),
            "received": 0,
            "reported": set(),  # idempotent guard: each spmd_id counted at most once
            "metadata": {},
            "errors": [],
            "ckpt_dir": ckpt_dir,
            "event": done_event,
        }
        for spmd_id, driver in drivers.items():
            driver.request_snapshot.remote(snapshot_id, target_count, ckpt_dir)

        snapshot_timeout = int(os.environ.get("FLUXWAY_SNAPSHOT_TIMEOUT_SECS", "600"))
        try:
            await asyncio.wait_for(done_event.wait(), timeout=snapshot_timeout)
        except TimeoutError as err:
            for d in drivers.values():
                if d:
                    d._clear_drain_and_fail.remote()
            self._pending_snapshots.pop(snapshot_id, None)
            raise RuntimeError(
                f"Snapshot '{snapshot_id}' timed out after {snapshot_timeout}s. "
                f"A Driver may have crashed without calling report_snapshot_failed. "
                f"Set FLUXWAY_SNAPSHOT_TIMEOUT_SECS to adjust."
            ) from err

        snap_info = self._pending_snapshots.pop(snapshot_id)
        if snap_info["errors"]:
            for d in drivers.values():
                if d:
                    d._clear_drain_and_fail.remote()

            raise RuntimeError(
                f"Snapshot '{snapshot_id}' failed for SPMDs: {snap_info['errors']}. "
                f"Refusing to persist incomplete manifest."
            )
        manifest = {
            sid: {
                "spmd_state": per_rank_or_state["spmd_state"],
                "rank_storage": {
                    str(rank): storage
                    for rank, storage in per_rank_or_state["rank_storage"].items()
                },
            }
            for sid, per_rank_or_state in snap_info["metadata"].items()
        }
        return manifest

    @require_spmd_manager
    async def receive_snapshot_metadata(
        self,
        spmd_id: str,
        snapshot_id: str,
        rank_metadata: dict[int, tuple[dict, dict]],
    ) -> None:
        snap_info = self._pending_snapshots.get(snapshot_id)
        if snap_info is None:
            logger.warning(f"Received metadata for unknown snapshot '{snapshot_id}'")
            return

        # Idempotency guard: a Driver may call both receive_snapshot_metadata
        # (which fails partway) and report_snapshot_failed.  Only the first
        # call for a given spmd_id is counted to prevent received > expected.
        if spmd_id in snap_info["reported"]:
            logger.warning(
                f"receive_snapshot_metadata: duplicate report from '{spmd_id}' "
                f"for snapshot '{snapshot_id}' \u2014 ignoring"
            )
            return
        snap_info["reported"].add(spmd_id)

        # Each entry is a (spmd_state, rank_storage) tuple already resolved
        # by the Driver via save_state_to_disk.  spmd_state is identical
        # across all ranks (topology-independent); we use the first
        # rank's as the canonical value.
        rank_storage_map: dict[int, dict] = {}
        canonical_spmd_state: dict | None = None
        try:
            for rank, (spmd_state, rank_storage) in rank_metadata.items():
                if canonical_spmd_state is None:
                    canonical_spmd_state = spmd_state
                if rank_storage:
                    rank_storage_map[rank] = rank_storage
        except Exception as e:
            logger.error(f"Failed to resolve snapshot metadata for {spmd_id}: {e}")
            snap_info["errors"].append(spmd_id)
            snap_info["received"] += 1
            if snap_info["received"] == snap_info["expected"]:
                snap_info["event"].set()
            return

        snap_info["metadata"][spmd_id] = {
            "spmd_state": canonical_spmd_state or {},
            "rank_storage": rank_storage_map,
        }
        snap_info["received"] += 1

        if snap_info["received"] == snap_info["expected"]:
            logger.info(f"Snapshot '{snapshot_id}' complete")
            snap_info["event"].set()

    async def report_snapshot_failed(self, spmd_id: str, snapshot_id: str) -> None:
        """Unblock request_snapshot when a Driver triggers SNAPSHOT_FAILED.

        Called by a Driver that encountered an unrecoverable error during
        persistence and could not (or failed to) deliver metadata via
        receive_snapshot_metadata.  Advances the received counter so the
        Controller's done_event is eventually set rather than waiting forever.
        """
        snap_info = self._pending_snapshots.get(snapshot_id)
        if snap_info is None:
            logger.warning(
                f"report_snapshot_failed: unknown snapshot '{snapshot_id}' from {spmd_id}"
            )
            return

        # Same idempotency guard as receive_snapshot_metadata.
        if spmd_id in snap_info["reported"]:
            logger.warning(
                f"report_snapshot_failed: duplicate report from '{spmd_id}' "
                f"for snapshot '{snapshot_id}' — ignoring"
            )
            return
        snap_info["reported"].add(spmd_id)

        logger.error(
            f"Snapshot '{snapshot_id}' failed for SPMD '{spmd_id}' — marking as error"
        )
        snap_info["errors"].append(spmd_id)
        snap_info["received"] += 1
        if snap_info["received"] == snap_info["expected"]:
            snap_info["event"].set()

    @require_spmd_manager
    async def restore_snapshot(
        self, snapshot_id: str, step: int, manifest: dict[str, Any]
    ) -> None:
        """
        Restore a snapshot: each Driver loads its Workers' tensors from disk directly.

        Args:
            snapshot_id: Snapshot identifier for bookkeeping.
            step:        The logical step count of this snapshot.
            manifest:    {spmd_id: {"spmd_state": {...}, "rank_storage": {str(rank): {...}}}}
        """
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_restore(
                checkpoint_id=snapshot_id,
            ):
                await self._restore_snapshot_unprofiled(
                    snapshot_id,
                    step,
                    manifest,
                )
            return
        await self._restore_snapshot_unprofiled(snapshot_id, step, manifest)

    async def _restore_snapshot_unprofiled(
        self, snapshot_id: str, step: int, manifest: dict[str, Any]
    ) -> None:
        logger.info(f"Controller restoring snapshot {snapshot_id} (step {step})")

        if self.heartbeat_manager is not None:
            logger.debug(
                "Waiting for all drivers to report workers_ready before restore"
            )
            restore_timeout = int(
                os.environ.get("FLUXWAY_SNAPSHOT_TIMEOUT_SECS", "600")
            )
            ready = await self.heartbeat_manager.wait_until_all_drivers_ready(
                timeout=restore_timeout
            )
            if not ready:
                raise RuntimeError(
                    f"Timed out after {restore_timeout}s waiting for drivers to "
                    f"become restore-ready. A Driver may have failed to start. "
                    f"Set FLUXWAY_SNAPSHOT_TIMEOUT_SECS to adjust."
                )

        self._last_checkpointed_step = step
        self._last_requested_drain_target = step

        self.state_manager.save_snapshot(snapshot_id, manifest)

        # Hydrate Control Plane step watermarks from spmd_state before unblocking.
        flow_control_updates: dict[str, tuple[Any, ...]] = {}
        for spmd_id, spmd_entry in manifest.items():
            spmd_state = spmd_entry.get("spmd_state", {})
            for op_name, count in spmd_state.get("completed_step_count", {}).items():
                self.async_step_manager.report_op_completed(spmd_id, op_name, count)
                flow_control_updates.update(
                    self._flow_control_manager.observe_op_completed(
                        spmd_id, op_name, count
                    )
                )
        await self._apply_flow_control_windows(flow_control_updates)

        await self.block_all_spmds()

        try:
            restore_tasks = []
            for spmd_id, spmd_entry in manifest.items():
                driver = self.spmd_manager.get_driver_handle(spmd_id)
                if driver is None:
                    raise RuntimeError(
                        f"restore_snapshot: driver for spmd_id '{spmd_id}' not found. "
                        f"Cannot perform partial restore — aborting to prevent split-brain."
                    )
                spmd_state = spmd_entry["spmd_state"]
                rank_storage = {
                    int(k): v for k, v in spmd_entry.get("rank_storage", {}).items()
                }
                restore_tasks.append(
                    driver.restore_from_disk.remote(spmd_state, rank_storage)
                )
            if restore_tasks:
                await asyncio.gather(*restore_tasks)
        finally:
            await self.unblock_all_spmds()

        logger.info("Snapshot restored successfully")

    async def restore_checkpoint(self, step: int | None = None) -> int | None:
        """
        User-facing API to trigger checkpoint restoration.
        """
        if self._checkpointer is None:
            raise RuntimeError(
                "Checkpoint config was not provided during controller initialization"
            )

        restored_step = await self._checkpointer.restore(step)
        if restored_step is None:
            self._start_sources_when_running = True
            if self.state_machine.is_running():
                await self.start_source_operators()
        return restored_step

    @require_spmd_manager
    async def start_source_operators(self) -> None:
        tasks = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks.append(driver.start_source_operators.remote())
        if tasks:
            await asyncio.gather(*tasks)

    async def stop_checkpointer(self) -> None:
        """Await any in-flight checkpoint to completion before shutdown.

        Waiting (rather than cancelling) guarantees the last auto-checkpoint
        boundary is persisted.  do_ckpt() catches all exceptions internally,
        so this await never raises.  The upper wait bound is
        FLUXWAY_SNAPSHOT_TIMEOUT_SECS (default 600 s).
        """
        task = self._checkpoint_task
        if task is not None and not task.done():
            current_task = asyncio.current_task()
            if task is not current_task:
                await task
        self._checkpoint_task = None

    @require_spmd_manager
    async def inspect_operator_state(self) -> dict[str, Any]:
        tasks: dict[str, Any] = {}
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks[spmd_id] = driver.inspect_operator_state.remote()
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

    @require_spmd_manager
    async def capture_profiler_snapshots(
        self,
    ) -> dict[str, dict[int, ProfilerSnapshot | None]]:
        tasks: dict[str, Any] = {}
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks[spmd_id] = driver.export_profiler_snapshots.remote()
        if not tasks:
            raise RuntimeError("No SPMD drivers are available for profiler export.")

        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

    async def capture_profiler_snapshots_if_enabled(
        self,
    ) -> dict[str, dict[int, ProfilerSnapshot | None]] | None:
        if self._profiler_config is None:
            return None
        return await self.capture_profiler_snapshots()

    @require_spmd_manager
    async def export_profiler_trace(self, path: str) -> str:
        snapshots_by_spmd = await self.capture_profiler_snapshots()
        return await self.write_profiler_trace(path, snapshots_by_spmd)

    async def write_profiler_trace(
        self,
        path: str,
        snapshots_by_spmd: dict[str, dict[int, ProfilerSnapshot | None]],
    ) -> str:
        records = []
        names_by_id: dict[int, str] = {}
        payload_fields_by_name_id: dict[int, tuple[str, ...]] = {}

        for rank_snapshots in snapshots_by_spmd.values():
            for snapshot in rank_snapshots.values():
                if snapshot is None:
                    continue
                records_array = snapshot["records"].copy()
                records_array[:, 1] += snapshot["monotonic_to_epoch_offset_ns"]
                if len(records_array) > 0:
                    records.append(records_array)
                metadata = snapshot["metadata"]
                names_by_id.update(metadata.names_by_id)
                payload_fields_by_name_id.update(metadata.payload_fields_by_name_id)

        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_profiler_export(
                spmd_count=len(snapshots_by_spmd),
                record_count=sum(len(record_array) for record_array in records),
            ):
                pass
            controller_snapshot = self._profiler_runtime.snapshot()
            if controller_snapshot is not None:
                from fluxway.profiler.record import TraceMetadata

                records_obj = controller_snapshot["records"]
                assert isinstance(records_obj, np.ndarray)
                records_array = records_obj.copy()

                offset_obj = controller_snapshot["monotonic_to_epoch_offset_ns"]
                assert isinstance(offset_obj, int)
                records_array[:, 1] += offset_obj

                if len(records_array) > 0:
                    records.append(records_array)

                metadata = controller_snapshot["metadata"]
                assert isinstance(metadata, TraceMetadata)
                names_by_id.update(metadata.names_by_id)
                payload_fields_by_name_id.update(metadata.payload_fields_by_name_id)

        if not records:
            raise RuntimeError(
                "Profiler is enabled but no trace records were captured."
            )

        from fluxway.profiler.perfetto_writer import PerfettoTraceWriter
        from fluxway.profiler.record import TraceMetadata
        from fluxway.profiler.filtering import filter_steady_state_records
        from fluxway.profiler.graph_symbols import GraphTraceSymbols

        merged = np.vstack(records)
        metadata = TraceMetadata(
            names_by_id=names_by_id,
            payload_fields_by_name_id=payload_fields_by_name_id,
        )
        if bool(getattr(self._profiler_config, "steady_state_only", False)):
            merged = filter_steady_state_records(merged, metadata)
            if len(merged) == 0:
                raise RuntimeError(
                    "Profiler steady-state filtering removed all trace records."
                )
        output = Path(path)
        PerfettoTraceWriter().write(
            merged,
            output,
            metadata=metadata,
            graph_symbols=GraphTraceSymbols.from_compiled_graph(
                self._graph.require_compiled_graph()
            ),
        )
        return str(output)

    async def write_profiler_trace_if_enabled(
        self, snapshots_by_spmd: dict[str, Any] | None
    ) -> str | None:
        if self._profiler_config is None:
            return None
        if snapshots_by_spmd is None:
            snapshots_by_spmd = await self.capture_profiler_snapshots()
        output_path = getattr(self._profiler_config, "output_path", None)
        if output_path is None:
            output_path = (
                Path(self._profiler_config.trace_dir)
                / str(self._profiler_config.run_id)
                / "trace.pftrace"
            )
        return await self.write_profiler_trace(str(output_path), snapshots_by_spmd)

    async def export_profiler_trace_if_enabled(self) -> str | None:
        if self._profiler_config is None:
            return None
        output_path = getattr(self._profiler_config, "output_path", None)
        if output_path is None:
            output_path = (
                Path(self._profiler_config.trace_dir)
                / str(self._profiler_config.run_id)
                / "trace.pftrace"
            )
        return await self.export_profiler_trace(str(output_path))

    # =====================================================================
    # SPMD Error Handling
    # =====================================================================

    async def report_spmd_error(self, spmd_id: str, failure: ExecutionFailure) -> None:
        """
        Receive error report from a driver.

        Args:
            spmd_id: SPMD identifier
            failure: Structured execution failure
        """
        logger.error(f"SPMD {spmd_id} reported error: {failure}")
        await self.state_machine.trigger_event(
            ControllerEvent.DRIVER_ERROR, spmd_id=spmd_id, failure=failure
        )

    @require_spmd_manager
    @require_heartbeat_manager
    async def report_spmd_completed(self, spmd_id: str) -> None:
        """
        Receive completion report from a driver.

        Args:
            spmd_id: SPMD identifier
        """
        logger.debug(f"SPMD {spmd_id} reported completion")

        # Check if all SPMDs are completed
        all_completed = True
        for sid in self.spmd_manager.get_all_spmd_ids():
            heartbeat = await self.heartbeat_manager.get_driver_heartbeat(sid)
            if heartbeat is None:
                all_completed = False
                break

        if all_completed:
            await self.state_machine.trigger_event(
                ControllerEvent.ALL_DRIVERS_COMPLETED
            )

    # =====================================================================
    # Blocking Operations
    # =====================================================================

    @require_spmd_manager
    async def block_all_spmds(self) -> None:
        """Block all SPMD operations (called during error handling)."""
        logger.warning("Blocking all SPMDs")

        tasks = []
        block_errors: list[Exception] = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                try:
                    tasks.append(driver.block_all_ops.remote())
                except Exception as error:
                    block_errors.append(error)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            block_errors.extend(
                result for result in results if isinstance(result, Exception)
            )
        if len(block_errors) == 1:
            raise block_errors[0]
        if block_errors:
            raise ExceptionGroup("block SPMDs failed", block_errors)

    @require_spmd_manager
    async def unblock_all_spmds(self) -> None:
        """Unblock all SPMD operations (called during recovery)."""
        logger.debug("Unblocking all SPMDs")

        tasks = []
        for spmd_id in self.spmd_manager.get_all_spmd_ids():
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                tasks.append(driver.unblock_all_ops.remote())

        if tasks:
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                logger.error(f"Error unblocking SPMDs: {e}")

    # =====================================================================
    # Heartbeat Callbacks (async)
    # =====================================================================

    async def _on_all_drivers_ready(self) -> None:
        """Callback when all drivers report workers_ready=True.

        Triggers: PENDING → TOPOLOGY_REGISTERING → _initiate_topology_registration()
        """
        logger.debug(
            "All drivers are ready (workers_ready=True), triggering topology registration"
        )
        await self.state_machine.trigger_event(ControllerEvent.ALL_DRIVERS_READY)

    async def _on_all_workers_running(self) -> None:
        """Callback when all drivers report workers_running=True.

        Triggers: TOPOLOGY_REGISTERING → RUNNING → start heartbeat monitoring
        """
        logger.debug("All workers are running after topology registration")

        # Workers auto-start as soon as topology registration completes, which
        # can be well before the controller transitions to RUNNING.  Any steps
        # executed in that window are invisible to _maybe_trigger_checkpoint
        # (blocked by the is_running() guard), so _last_checkpointed_step would
        # remain 0 while actual progress is far ahead.  When the first post-RUNNING
        # report_op_completed arrives it would fire a checkpoint at boundary=interval
        # even though workers are already at step 200+, producing a checkpoint
        # directory named "50" that actually contains step-200 model state.
        #
        # Fix: align the watermark to the floor-division of the minimum observed
        # progress so the first real checkpoint fires at the next clean boundary
        # after where workers actually are.
        if self._checkpointer is not None:
            interval = self._checkpointer.interval_steps
            all_steps = self.async_step_manager.get_all_spmd_steps()
            if all_steps:
                min_count = (
                    min(
                        count
                        for spmd_step in all_steps.values()
                        for count in spmd_step.completed_step_count.values()
                        if count > 0
                    )
                    if any(
                        spmd_step.completed_step_count
                        for spmd_step in all_steps.values()
                    )
                    else 0
                )
                aligned = (min_count // interval) * interval
                if aligned > self._last_checkpointed_step:
                    logger.info(
                        f"Aligning checkpoint watermark from {self._last_checkpointed_step} "
                        f"to {aligned} (workers already at min_count={min_count} "
                        f"during startup window)"
                    )
                    self._last_checkpointed_step = aligned

        await self.state_machine.trigger_event(ControllerEvent.ALL_SPMDS_TOPOLOGY_READY)

    async def _on_driver_error(self, spmd_id: str, failure: ExecutionFailure) -> None:
        """Callback when a driver has an error."""
        await self.report_spmd_error(spmd_id, failure)

    # =====================================================================
    # Internal Methods for State Machine
    # =====================================================================

    @require_spmd_manager
    async def _initiate_topology_registration(self) -> None:
        """
        Initiate topology registration on all SPMDs.

        This method:
        1. Collects ALL worker handles from ALL drivers upfront to avoid actor deadlock.
        2. For each SPMD, wraps handles with DataPlaneCallable and calls register_topology.
        """
        all_spmd_ids = self.spmd_manager.get_all_spmd_ids()
        if not all_spmd_ids:
            logger.warning("No SPMDs found, cannot initiate topology registration")
            return

        # Collect all worker handles upfront to avoid actor re-entrancy deadlocks during registration
        # {spmd_id: {rank: worker_handle}}
        all_worker_handles: dict[str, dict[int, Any]] = {}

        fetch_tasks = []
        spmd_ids_to_fetch = []

        for spmd_id in all_spmd_ids:
            driver = self.spmd_manager.get_driver_handle(spmd_id)
            if driver is not None:
                fetch_tasks.append(driver.get_worker_handles.remote())
                spmd_ids_to_fetch.append(spmd_id)

        if fetch_tasks:
            try:
                results = await asyncio.gather(*fetch_tasks)
                for spmd_id, worker_map in zip(spmd_ids_to_fetch, results, strict=True):
                    all_worker_handles[spmd_id] = worker_map
                logger.debug(
                    f"Collected worker handles for: {list(all_worker_handles.keys())}"
                )
                for sid, handles in all_worker_handles.items():
                    logger.debug(f"  - {sid}: {len(handles)} workers")
            except Exception as e:
                logger.error(f"Failed to collect worker handles for topology: {e}")
                return

        def _get_transport_type(target_spmd_id: str) -> str:
            if target_spmd_id in self._transport_config:
                return self._transport_config[target_spmd_id]
            if "*" in self._transport_config:
                return self._transport_config["*"]
            node = self._graph.nodes.get(target_spmd_id)
            if node and node.spmd_config:
                return getattr(node.spmd_config, "transport_type", "ray")
            return "ray"

        # Wrap raw handles with DataPlaneCallable after predecessor/successor
        # direction is known, so each sender-side adapter can bind sender_id
        # once instead of carrying it in every hot-path data frame.

        # Initiate topology registration on each driver
        spmd_code_by_id = {
            spmd_id: code for code, spmd_id in enumerate(sorted(all_spmd_ids))
        }
        spmd_id_by_code = {code: spmd_id for spmd_id, code in spmd_code_by_id.items()}
        driver_handles_by_id: dict[str, Any] = {}
        for driver_spmd_id in all_spmd_ids:
            driver_handle = self.spmd_manager.get_driver_handle(driver_spmd_id)
            if driver_handle is None:
                raise RuntimeError(
                    f"Missing driver handle for SPMD {driver_spmd_id!r} "
                    "during credit topology registration."
                )
            driver_handles_by_id[driver_spmd_id] = driver_handle

        for spmd_id in all_spmd_ids:
            driver = driver_handles_by_id[spmd_id]
            if driver is None:
                continue

            predecessors = self.spmd_manager.get_predecessors(spmd_id)
            successors = self.spmd_manager.get_successors(spmd_id)

            logger.debug(
                f"Driver {spmd_id}: predecessors={predecessors}, successors={successors}"
            )

            # Build predecessor/successor worker handle maps: {spmd_id: {rank: handle}}
            succ_handlers = {
                s: {
                    rank: DataPlaneRegistry.create(
                        _get_transport_type(s),
                        handle,
                        sender_spmd_code=spmd_code_by_id[spmd_id],
                    )
                    for rank, handle in all_worker_handles[s].items()
                }
                for s in successors
                if s in all_worker_handles
            }
            pred_handlers = {
                p: {
                    rank: DataPlaneRegistry.create(
                        _get_transport_type(p),
                        handle,
                        sender_spmd_code=spmd_code_by_id[p],
                    )
                    for rank, handle in all_worker_handles[p].items()
                }
                for p in predecessors
                if p in all_worker_handles
            }

            logger.debug(f"  - pred_handlers keys: {list(pred_handlers.keys())}")
            logger.debug(f"  - succ_handlers keys: {list(succ_handlers.keys())}")

            try:
                upstream_edges, downstream_edges = (
                    self._flow_control_manager.topology_for_driver(spmd_id)
                )
                upstream_driver_handles = {
                    edge.source_spmd_id: driver_handles_by_id[edge.source_spmd_id]
                    for edge in downstream_edges
                }
                await driver.register_credit_topology.remote(
                    upstream_driver_handles,
                    upstream_edges,
                    downstream_edges,
                    0,
                )
                initial_windows = self._flow_control_manager.initial_windows_for_driver(
                    spmd_id
                )
                if initial_windows:
                    await driver.apply_credit_window.remote(initial_windows)
                # Now call register_topology with ACTUAL DataPlaneCallable objects
                driver.register_topology.remote(
                    pred_handlers,
                    succ_handlers,
                    len(predecessors),
                    len(successors),
                    spmd_code_by_id[spmd_id],
                    spmd_id_by_code,
                )
                logger.debug(f"Initiated topology registration for SPMD {spmd_id}")
            except Exception as e:
                logger.error(
                    f"Failed to initiate topology registration for {spmd_id}: {e}"
                )

    @require_spmd_manager
    @require_heartbeat_manager
    async def _cleanup_all_spmds(self) -> None:
        """
        Cleanup all SPMD resources.

        Called when entering COMPLETING or FAILING state.
        """
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_cleanup(
                spmd_count=len(self.spmd_manager.get_all_spmd_ids()),
            ):
                await self._cleanup_all_spmds_unprofiled()
            return
        await self._cleanup_all_spmds_unprofiled()

    async def _cleanup_all_spmds_unprofiled(self) -> None:
        logger.debug("Cleaning up all SPMDs")

        cleanup_errors: list[Exception] = []
        try:
            await self.heartbeat_manager.stop_monitoring_async()
        except Exception as error:
            cleanup_errors.append(error)

        spmd_cleanup_errors = await asyncio.gather(
            *(
                self._stop_and_unregister_spmd(spmd_id)
                for spmd_id in self.spmd_manager.get_all_spmd_ids()
            )
        )
        for errors in spmd_cleanup_errors:
            cleanup_errors.extend(errors)

        if self.scheduler.pg_manager.has_job_pg:
            try:
                self.scheduler.release_resources()
            except Exception as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise ExceptionGroup("SPMD cleanup failed", cleanup_errors)

    async def _stop_and_unregister_spmd(self, spmd_id: str) -> tuple[Exception, ...]:
        errors: list[Exception] = []
        try:
            stop_spmd = functools.partial(
                self.spmd_manager.stop_spmd,
                spmd_id,
                save_checkpoint=False,
                release_driver_handle=False,
            )
            if self._profiler_runtime.enabled:
                async with self._profiler_runtime.recorder.spmd_manager_driver_stop(
                    spmd_id=spmd_id,
                ):
                    await stop_spmd()
            else:
                await stop_spmd()
        except Exception as error:
            logger.error(f"Error stopping SPMD {spmd_id}: {error}")
            errors.append(error)
        finally:
            try:
                self.heartbeat_manager.unregister_spmd(spmd_id)
            except Exception as error:
                errors.append(error)
        return tuple(errors)

    async def _finalize_runtime(self, *, export_profiler_trace: bool) -> None:
        cleanup_errors: list[Exception] = []
        profiler_snapshots = None
        profiler_capture_succeeded = False
        try:
            await self.stop_checkpointer()
        except Exception as error:
            cleanup_errors.append(error)
        if export_profiler_trace:
            try:
                profiler_snapshots = await self.capture_profiler_snapshots_if_enabled()
                profiler_capture_succeeded = True
            except Exception as error:
                cleanup_errors.append(error)
        try:
            await self._cleanup_all_spmds()
        except Exception as error:
            cleanup_errors.append(error)
        if export_profiler_trace and profiler_capture_succeeded:
            try:
                await self.write_profiler_trace_if_enabled(profiler_snapshots)
            except Exception as error:
                cleanup_errors.append(error)
        try:
            self._release_stopped_spmd_handles()
        except Exception as error:
            cleanup_errors.append(error)
        if len(cleanup_errors) == 1:
            raise cleanup_errors[0]
        if cleanup_errors:
            raise ExceptionGroup("terminal cleanup failed", cleanup_errors)

    def _release_stopped_spmd_handles(self) -> None:
        """Release driver handles after terminal cleanup has fully completed."""
        self.spmd_manager.release_all_driver_handles()

    # =====================================================================
    # Lifecycle Control
    # =====================================================================

    async def stop(self, save_checkpoint: bool = True) -> None:
        """
        Stop training execution.

        Args:
            save_checkpoint: Whether to save checkpoint before stopping
        """
        logger.debug("Stopping controller")

        stop_errors: list[Exception] = []
        try:
            await self.block_all_spmds()
        except Exception as error:
            stop_errors.append(error)

        # Save checkpoint if requested (placeholder)
        if save_checkpoint:
            logger.warning("Checkpoint save not implemented")

        try:
            await self._finalize_runtime(export_profiler_trace=False)
        except Exception as error:
            stop_errors.append(error)
        if len(stop_errors) == 1:
            raise stop_errors[0]
        if stop_errors:
            raise ExceptionGroup("controller stop failed", stop_errors)

        logger.debug("Controller stopped")

    @require_heartbeat_manager
    async def get_cluster_status(self) -> dict[str, str]:
        """
        Get status of all SPMD nodes.

        Returns:
            Mapping of node name to status
        """
        status = await self.heartbeat_manager.get_all_driver_status()
        return {spmd_id: s.value for spmd_id, s in status.items()}

    async def wait_for_completion(self, timeout: float | None = None) -> bool:
        """
        Wait for training to complete.

        Args:
            timeout: Maximum time to wait (None for infinite)

        Returns:
            True if completed successfully

        Raises:
            ExecutionFailedError: The execution reached FAILED.
            TimeoutError: The timeout elapsed before a terminal state.
        """
        if self._profiler_runtime.enabled:
            async with self._profiler_runtime.recorder.controller_completion(
                spmd_count=len(self._graph.nodes),
            ):
                return await self._wait_for_completion_unprofiled(timeout)
        return await self._wait_for_completion_unprofiled(timeout)

    async def _wait_for_completion_unprofiled(
        self, timeout: float | None = None
    ) -> bool:
        state = await self.state_machine.wait_until_terminal(timeout)
        if state == ControllerState.COMPLETED:
            return True
        if state == ControllerState.FAILED:
            raise ExecutionFailedError(self.state_machine.get_failures())
        raise RuntimeError(f"Unexpected terminal state: {state}")


async def create_controller(
    graph: Graph,
    *,
    placement_recipe: PlacementRecipe,
    placement_runtime: PlacementRuntimeConfig | None = None,
    heartbeat_config: HeartbeatConfig | None = None,
    async_step_config: AsyncStepConfig | None = None,
    checkpoint_config: CheckpointConfig | None = None,
    runtime_env: dict[str, Any] | None = None,
    transport_config: dict[str, str] | None = None,
    profiler_config: Any | None = None,
) -> Any:
    """
    Create and initialize a Controller actor.

    This is the main entry point for creating a Controller.

    Args:
        graph: The static execution Graph.
        placement_recipe: Resource placement constraints and objectives.
        heartbeat_config: Optional heartbeat configuration
        async_step_config: Optional async step configuration
        checkpoint_config: Optional checkpoint configuration
        runtime_env: Optional Ray runtime environment
        transport_config: Optional data plane transport backend per SPMD (e.g. {"*": "ray"})
        profiler_config: Optional profiler runtime configuration

    Returns:
        Controller actor handle

    Raises:
        RuntimeError: If initialization fails
    """
    if not isinstance(graph, Graph):
        raise TypeError("create_controller requires a Graph")
    if not isinstance(placement_recipe, PlacementRecipe):
        raise TypeError("create_controller requires a PlacementRecipe")
    if placement_runtime is None:
        placement_runtime = PlacementRuntimeConfig()

    # Create the actor
    controller = ControllerActor.options(runtime_env=runtime_env).remote(  # type: ignore[attr-defined]
        graph=graph,
        placement_recipe=placement_recipe,
        heartbeat_config=heartbeat_config,
        async_step_config=async_step_config,
        checkpoint_config=checkpoint_config,
        transport_config=transport_config,
        profiler_config=profiler_config,
        placement_runtime=placement_runtime,
    )

    # Initialize the actor
    success = await controller.initialize.remote()  # type: ignore[attr-defined]
    if not success:
        raise RuntimeError("Failed to initialize controller")

    return controller


Controller = ControllerActor
