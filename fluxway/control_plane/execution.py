from typing import Any, Literal, TYPE_CHECKING
from fluxway.common.types import (
    Graph,
    HeartbeatConfig,
    AsyncStepConfig,
    CheckpointConfig,
)
from fluxway.placement import (
    PlacementRecipe,
    PlacementRuntimeConfig,
)
from fluxway.control_plane.controller import ControllerActor

if TYPE_CHECKING:
    from fluxway.profiler.config import ProfilerConfig


class GraphExecution:
    """
    Facade for interacting with a running Graph execution.
    """

    def __init__(
        self,
        controller_handle: Any,
        allocation: Any,
        restored_step: int | None,
    ):
        self._controller = controller_handle
        self._allocation = allocation
        self._restored_step = restored_step

    @property
    def allocation(self) -> Any:
        return self._allocation

    @property
    def restored_step(self) -> int | None:
        return self._restored_step

    async def progress(self) -> dict[str, dict[str, int]]:
        return await self._controller.get_async_step_progress.remote()

    async def wait(self, timeout: float | None = None) -> None:
        """
        Wait for execution to complete.
        Returns None on success.
        Raises ExecutionFailedError on failure.
        Raises TimeoutError on timeout.
        """
        await self._controller.wait_for_completion.remote(timeout)

    async def cancel(self) -> None:
        """
        Cancel execution and clean up resources.
        """
        await self._controller.stop.remote(save_checkpoint=False)


async def start(
    graph: Graph,
    /,
    *,
    placement_recipe: PlacementRecipe,
    placement_runtime: PlacementRuntimeConfig | None = None,
    heartbeat_config: HeartbeatConfig | None = None,
    async_step_config: AsyncStepConfig | None = None,
    checkpoint_config: CheckpointConfig | None = None,
    resume: Literal["latest"] | int | None = None,
    runtime_env: dict[str, Any] | None = None,
    transport_config: dict[str, str] | None = None,
    profiler_config: "ProfilerConfig | None" = None,
) -> GraphExecution:
    """
    Start a Graph execution.
    """
    if resume is not None and checkpoint_config is None:
        raise ValueError("Cannot specify resume without a checkpoint_config")

    if placement_runtime is None:
        placement_runtime = PlacementRuntimeConfig()

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

    restored_step = await controller.start.remote(resume)

    # Get allocation synchronously after initialization is done
    allocation = await controller.get_resource_allocation_summary.remote()

    return GraphExecution(
        controller_handle=controller,
        allocation=allocation,
        restored_step=restored_step,
    )
