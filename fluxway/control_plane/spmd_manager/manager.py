"""
SPMD Manager for global SPMD lifecycle management.

Based on ARCHITECTURE.md Section 2.1.3

Responsibilities:
- SPMD lifecycle: Start SPMD Driver based on scheduling results
- SPMD topology: Establish connections between SPMDs
- SPMD initialization: Initialize SPMD state and internal FSM
- State recovery: Direct or FSM-triggered checkpoint recovery
- Node changes: Manage node creation, eviction, and rebuilding

Note: Controller layer only starts drivers. Workers are started by drivers.
"""

import asyncio
from collections.abc import Callable
from typing import Any

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fluxway.common.types import SPMDConfig
from fluxway.common.utils.logging import get_logger
from fluxway.placement import SPMDWorkerGroup
from fluxway.placement.worker_group import ray_actor_options

logger = get_logger(__name__)


class SPMDHandle:
    """
    Handle to an SPMD instance.

    Contains reference to the driver actor only.
    Worker handles are managed internally by the driver.
    """

    def __init__(self, spmd_id: str):
        self.spmd_id = spmd_id
        self._driver_handle: Any | None = None  # ray.ActorHandle
        self._config: SPMDConfig | None = None
        self._attested_node_ids: tuple[str, ...] | None = None

    @property
    def attested_node_ids(self) -> tuple[str, ...] | None:
        return self._attested_node_ids

    @property
    def driver_handle(self) -> Any | None:
        """Get the driver actor handle."""
        return self._driver_handle

    def set_driver_handle(self, handle: Any | None) -> None:
        """Set the driver actor handle."""
        self._driver_handle = handle

    def resolve_driver_method(self, method: str) -> Callable[..., Any]:
        if self._driver_handle is None:
            raise RuntimeError(f"Driver handle not set for SPMD {self.spmd_id}")
        remote = getattr(self._driver_handle, method).remote
        if not callable(remote):
            raise TypeError(f"Driver method {method!r} has no callable remote endpoint")
        return remote

    def call_driver(self, method: str, *args, **kwargs) -> Any:
        return self.resolve_driver_method(method)(*args, **kwargs)


class SPMDManager:  # pyright: ignore[reportRedeclaration]
    """
    Global SPMD Manager (Orchestrator).

    Manages the lifecycle of all SPMD drivers in the training job.
    Workers are managed by their respective drivers.
    """

    def __init__(
        self,
        controller_handle: Any | None = None,
    ):
        """
        Initialize SPMDManager.

        Args:
            controller_handle: Reference to the Controller (for driver callbacks)
        """
        self._controller_handle = controller_handle
        self._spmd_handles: dict[str, SPMDHandle] = {}
        self._pending_driver_calls: set[asyncio.Future] = set()
        self._topology: dict[str, list[str]] = {}  # spmd_id -> downstream spmd_ids
        self._reverse_topology: dict[
            str, list[str]
        ] = {}  # spmd_id -> upstream spmd_ids

    def set_controller_handle(self, controller_handle: Any) -> None:
        """Set the controller handle for driver callbacks."""
        self._controller_handle = controller_handle

    def register_spmd(self, spmd_id: str, handle: SPMDHandle) -> None:
        """
        Register an SPMD instance.

        Args:
            spmd_id: SPMD identifier
            handle: SPMD handle
        """
        if spmd_id in self._spmd_handles:
            logger.warning(f"SPMD {spmd_id} already registered, overwriting")

        self._spmd_handles[spmd_id] = handle
        logger.info(f"Registered SPMD: {spmd_id}")

    async def start_spmd(
        self,
        spmd_id: str,
        config: SPMDConfig,
        worker_group: SPMDWorkerGroup,
        checkpoint_id: str | None = None,
    ) -> SPMDHandle:
        """
        Start an SPMD instance by creating its driver.

        The driver will start workers internally using the provided
        placement group and bundle indices.

        Args:
            spmd_id: Unique identifier for the SPMD
            config: SPMD configuration
            placement_group: Ray PlacementGroup for resource allocation
            worker_bundle_indices: Bundle indices for workers [1, 2, ..., n]
            driver_bundle_index: Bundle index for driver (default 0)
            checkpoint_id: Optional checkpoint to restore from
            node_type: Type of node (SPMD or READER)
            reader_config: Reader configuration (required for READER nodes)

        Returns:
            Handle to the started SPMD, or None on failure
        """
        # Create handle
        handle = SPMDHandle(spmd_id)
        handle._config = config

        try:
            # Import driver here to avoid circular imports
            from fluxway.spmd.driver import SPMDDriver

            # Case to Any to satisfy static type checkers for Ray-specific calls (.remote, .options)
            spmd_driver_cls: Any = SPMDDriver

            # Build graph topology for this SPMD (self-loops are already filtered)
            graph_topology = {
                "predecessors": self.get_predecessors(spmd_id),
                "successors": self.get_successors(spmd_id),
            }

            # Create driver actor with placement group
            driver_name = f"spmd_driver_{spmd_id}"
            driver_opts: dict[str, Any] = {"name": driver_name}
            driver_opts.update(ray_actor_options(worker_group.driver_resources))
            driver_opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=worker_group.placement_group,
                placement_group_bundle_index=worker_group.driver_bundle_index,
                placement_group_capture_child_tasks=False,
            )
            # Driver is a control-plane actor: it must NOT inherit GPU-related
            # JAX env_vars (JAX_PLATFORMS=cuda,...) from the worker side,
            # otherwise pickle.loads of jax.Array on the driver eagerly
            # initializes a CUDA backend in a process that holds no GPU
            # bundle and crashes the SPMD pipeline.  Resolution order:
            #   1) explicit user override via config.driver_runtime_env
            #   2) safe default that pins jax to CPU on the driver actor.
            # Note: we deliberately do NOT clear CUDA_VISIBLE_DEVICES here
            # because Ray uses the driver's visible accelerator list to
            # propagate accelerator ids to child worker actors -- masking
            # it triggers `IndexError: list index out of range` inside
            # `ray._private.worker.get_accelerator_ids_for_accelerator_resource`.
            if config.driver_runtime_env is not None:
                driver_opts["runtime_env"] = config.driver_runtime_env

            driver = spmd_driver_cls.options(**driver_opts).remote(  # type: ignore
                spmd_id=spmd_id,
                config=config,
                placement_group=worker_group.placement_group,
                worker_bundle_indices=list(worker_group.rank_to_bundle),
                controller_handle=self._controller_handle,
                graph_topology=graph_topology,
                worker_resource=worker_group.worker_resources,
                expected_node_ids=list(worker_group.expected_node_ids),
            )

            handle.set_driver_handle(driver)

            initialization = driver.initialize.remote(checkpoint_id)
            await initialization
            handle._attested_node_ids = tuple(
                await driver.get_attested_node_ids.remote()
            )

            # Register handle
            self._spmd_handles[spmd_id] = handle
            return handle

        except Exception as e:
            logger.error(f"Failed to start SPMD {spmd_id}: {e}")
            raise

    async def stop_spmd(
        self,
        spmd_id: str,
        save_checkpoint: bool = True,
        release_driver_handle: bool = True,
    ) -> None:
        """
        Stop an SPMD instance.

        Args:
            spmd_id: ID of SPMD to stop
            save_checkpoint: Whether to save checkpoint before stopping
            release_driver_handle: Whether to clear the controller-side driver
                reference immediately after stop returns. Terminal reclaim retentions
                handles until the controller reaches its terminal state so Ray
                bookretentioning can settle before actor references are dropped.
        """
        handle = self._spmd_handles.get(spmd_id)
        if handle is None:
            logger.warning(f"SPMD {spmd_id} not found")
            return

        if handle.driver_handle is None:
            logger.warning(f"SPMD {spmd_id} has no driver handle")
            return

        try:
            # Stop the driver and wait for its worker shutdown path.
            await handle.call_driver("stop", save_checkpoint)
        finally:
            if release_driver_handle:
                handle.set_driver_handle(None)
            logger.info(f"Stopped SPMD {spmd_id}")

    def release_driver_handle(self, spmd_id: str) -> None:
        """Release the controller-side handle to a stopped driver actor."""
        handle = self._spmd_handles.get(spmd_id)
        if handle is None:
            return
        handle.set_driver_handle(None)

    def release_all_driver_handles(self) -> None:
        """Release controller-side handles for all registered SPMDs."""
        for handle in self._spmd_handles.values():
            handle.set_driver_handle(None)

    def get_spmd_handle(self, spmd_id: str) -> SPMDHandle | None:
        """
        Get handle to an SPMD instance.

        Args:
            spmd_id: SPMD identifier

        Returns:
            SPMDHandle or None if not found
        """
        return self._spmd_handles.get(spmd_id)

    def get_driver_handle(self, spmd_id: str) -> Any | None:
        """
        Get the driver actor handle for an SPMD.

        Args:
            spmd_id: SPMD identifier

        Returns:
            Driver ActorHandle or None
        """
        handle = self._spmd_handles.get(spmd_id)
        if handle is None:
            return None
        return handle.driver_handle

    def get_all_spmd_ids(self) -> list[str]:
        """Get all registered SPMD IDs."""
        return list(self._spmd_handles.keys())

    async def restore_spmd(
        self,
        spmd_id: str,
        config: SPMDConfig,
        worker_group: SPMDWorkerGroup,
        checkpoint_id: str | None = None,
    ) -> SPMDHandle:
        """
        Restore an SPMD from checkpoint.

        Args:
            spmd_id: ID of SPMD to restore
            config: SPMD configuration
            placement_group: Placement group for resources
            worker_bundle_indices: Worker bundle indices
            checkpoint_id: Checkpoint to restore from
            node_type: Type of node (SPMD or READER)
            reader_config: Reader configuration (for READER nodes)

        Returns:
            Handle to restored SPMD

        TODO: Implement checkpoint loading
        """
        logger.warning(
            f"restore_spmd called for {spmd_id} - checkpoint not implemented"
        )
        return await self.start_spmd(
            spmd_id=spmd_id,
            config=config,
            worker_group=worker_group,
            checkpoint_id=checkpoint_id,
        )

    def establish_topology(self, topology: dict[str, list[str]]) -> None:
        """
        Establish SPMD topology connections.

        Args:
            topology: Mapping of SPMD ID to downstream SPMD IDs
        """
        if self._spmd_handles:
            raise RuntimeError(
                "Topology must be established before Driver registration"
            )

        self._topology = {source: list(targets) for source, targets in topology.items()}

        # Build reverse topology
        self._reverse_topology = {spmd_id: [] for spmd_id in topology}
        for source, targets in topology.items():
            for target in targets:
                if target not in self._reverse_topology:
                    self._reverse_topology[target] = []
                if source not in self._reverse_topology[target]:
                    self._reverse_topology[target].append(source)

        logger.debug(f"Established topology with {len(topology)} SPMDs")

    def get_predecessors(self, spmd_id: str) -> list[str]:
        """
        Get predecessor SPMD IDs.
        """
        return self._reverse_topology.get(spmd_id, [])

    def get_successors(self, spmd_id: str) -> list[str]:
        """
        Get successor SPMD IDs.
        """
        return self._topology.get(spmd_id, [])

    def enqueue_driver_calls(
        self, method: str, *args, **kwargs
    ) -> asyncio.Future[list[Any]]:
        if not self._spmd_handles:
            raise RuntimeError("Driver fan-out requires registered participants")
        remotes = []
        for handle in self._spmd_handles.values():
            remotes.append(handle.resolve_driver_method(method))
        refs = []
        for remote in remotes:
            try:
                ref = asyncio.ensure_future(remote(*args, **kwargs))
                self._pending_driver_calls.add(ref)
                ref.add_done_callback(self._driver_call_finished)
                refs.append(ref)
            except Exception as error:
                logger.exception("Driver fan-out submission failed")
                failure = asyncio.get_running_loop().create_future()
                failure.set_exception(error)
                refs.append(failure)
                break
        return asyncio.gather(*refs)

    def _driver_call_finished(self, operation: asyncio.Future) -> None:
        self._pending_driver_calls.remove(operation)
        if not operation.cancelled():
            operation.exception()

    async def close_driver_calls(self) -> None:
        pending = tuple(self._pending_driver_calls)
        for operation in pending:
            operation.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
