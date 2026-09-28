import asyncio
from collections.abc import AsyncIterator
from typing import Any, Literal, TYPE_CHECKING
from fluxway.common.types import (
    Graph,
    HeartbeatConfig,
    AsyncStepConfig,
    SourceControlConfig,
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
        self._completion_task: asyncio.Task[bool] | None = None

    @property
    def allocation(self) -> Any:
        return self._allocation

    @property
    def restored_step(self) -> int | None:
        return self._restored_step

    def _ensure_completion_task(self) -> asyncio.Task[bool]:
        if self._completion_task is None:
            ref = self._controller.wait_for_completion.remote(None)
            self._completion_task = asyncio.create_task(self._await_object_ref(ref))
        return self._completion_task

    @staticmethod
    async def _await_object_ref(ref: Any) -> bool:
        return await ref

    async def progress(self) -> dict[str, dict[str, int]]:
        return await self._controller.get_async_step_progress.remote()

    async def wait(self, timeout: float | None = None) -> None:
        """
        Wait for execution to complete.
        Returns None on success.
        Raises ExecutionFailedError on failure.
        Raises TimeoutError on timeout.
        """
        task = self._ensure_completion_task()
        if timeout is None:
            await task
            return
        await asyncio.wait_for(asyncio.shield(task), timeout=timeout)

    async def monitor_progress(
        self, interval_s: float
    ) -> AsyncIterator[dict[str, dict[str, int]]]:
        """
        Periodically stream progress updates until completion without raising TimeoutError.
        """
        if interval_s <= 0:
            raise ValueError(f"progress interval must be positive, got {interval_s}")
        while True:
            try:
                await self.wait(timeout=interval_s)
                return
            except TimeoutError:
                yield await self.progress()

    async def cancel(self) -> None:
        """
        Cancel execution and clean up resources.
        """
        try:
            await self._controller.stop.remote(save_checkpoint=False)
        finally:
            if self._completion_task is not None and not self._completion_task.done():
                self._completion_task.cancel()


async def start(
    graph: Graph,
    /,
    *,
    placement_recipe: PlacementRecipe,
    placement_runtime: PlacementRuntimeConfig | None = None,
    heartbeat_config: HeartbeatConfig | None = None,
    async_step_config: AsyncStepConfig | None = None,
    source_control: SourceControlConfig | None = None,
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
        source_control=source_control,
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
