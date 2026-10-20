from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

from fluxway.common.execution_layout import ExecutionGroup, ExecutionLayout
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import OperatorContext
from fluxway.common.types.sharding import PhysicalLayout, ShardingSpec


class ExecutionStatus(Enum):
    SUCCESS = auto()
    SKIPPED = auto()
    INACTIVE = auto()
    STALE_AFTER_RESTORE = auto()
    COMPLETED_WITHOUT_TENSOR = auto()
    SOURCE_EXHAUSTED = auto()


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    data: Any = None
    control_requests: tuple[Any, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutionStatus):
            raise TypeError("execution result status must be an ExecutionStatus")
        if not isinstance(self.control_requests, tuple):
            raise TypeError("execution result control_requests must be a tuple")
        if any(request is None for request in self.control_requests):
            raise ValueError("execution result control_requests cannot contain None")


class ExecutionDispatcher:
    def __init__(self, layout: ExecutionLayout) -> None:
        self.layout = layout

    def _validate_world_size(self, world_size: int) -> None:
        if world_size != self.layout.world_size:
            raise ValueError(
                "runtime world size disagrees with compiled execution layout"
            )

    def execution_cycle(self, physical_world_size: int) -> tuple[tuple[int, ...], ...]:
        self._validate_world_size(physical_world_size)
        return self.layout.cycle

    def bind_execution_group(
        self, mesh: DeviceMesh, physical_rank: int
    ) -> ExecutionGroup:
        self._validate_world_size(mesh.size)
        return self.layout.group_for_rank(physical_rank)

    def should_execute(
        self, physical_rank: int, physical_world_size: int, step: int
    ) -> bool:
        self._validate_world_size(physical_world_size)
        return self.layout.is_active(physical_rank, step)

    def get_effective_source_layout(
        self, mesh: DeviceMesh, routing: ShardingSpec
    ) -> PhysicalLayout:
        self._validate_world_size(mesh.size)
        return routing.bind(self.layout.logical_mesh)

    def resolve_active_producers(
        self,
        effective_layout: PhysicalLayout,
        target_layout: PhysicalLayout,
        target_rank: int,
        step: int,
        physical_world_size: int,
    ) -> list[int]:
        self._validate_world_size(physical_world_size)
        group = self.layout.group_for_step(step)
        return [
            group.physical_rank(rank)
            for rank in effective_layout.resolve_producers(target_layout, target_rank)
        ]

    def get_logical_context(
        self,
        physical_rank: int,
        physical_world_size: int,
        step: int,
        batch_size: int = 1,
        source_work_items: Any | None = None,
    ) -> OperatorContext:
        self._validate_world_size(physical_world_size)
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        group = self.layout.group_for_rank(physical_rank)
        return OperatorContext(
            physical_rank=physical_rank,
            physical_world_size=physical_world_size,
            logical_rank=group.logical_rank(physical_rank),
            logical_world_size=group.logical_mesh.size,
            step=step,
            global_source_offset=None,
            is_interleaved=len(self.layout.groups) > 1,
            source_work_items=source_work_items,
        )
