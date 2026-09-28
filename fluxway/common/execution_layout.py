from __future__ import annotations

from dataclasses import dataclass, field
from math import prod

from fluxway.common.mesh import DeviceMesh
from fluxway.common.frozen_mapping import freeze_mapping


class ExecutionGroup:
    __slots__ = ("_physical_ranks", "_logical_mesh", "_logical_by_physical")

    def __init__(
        self,
        physical_ranks: tuple[int, ...],
        logical_mesh: DeviceMesh,
    ) -> None:
        if not isinstance(physical_ranks, tuple):
            raise TypeError("execution group physical_ranks must be a tuple")
        if not isinstance(logical_mesh, DeviceMesh):
            raise TypeError("execution group logical_mesh must be DeviceMesh")
        frozen_mesh = logical_mesh.frozen_copy()
        ranks = physical_ranks
        if len(ranks) != frozen_mesh.size or len(set(ranks)) != len(ranks):
            raise ValueError(
                "execution group must contain one physical rank per logical lane"
            )
        if any(type(rank) is not int or rank < 0 for rank in ranks):
            raise ValueError("execution group requires nonnegative integer ranks")
        if tuple(frozen_mesh.flatten()) != tuple(range(len(ranks))):
            raise ValueError("execution group mesh must use logical rank coordinates")

        self._physical_ranks = ranks
        self._logical_mesh = frozen_mesh
        self._logical_by_physical = freeze_mapping(
            {rank: i for i, rank in enumerate(ranks)}
        )

    @property
    def physical_ranks(self) -> tuple[int, ...]:
        return self._physical_ranks

    @property
    def logical_mesh(self) -> DeviceMesh:
        return self._logical_mesh

    @classmethod
    def create(
        cls, physical_ranks: tuple[int, ...], logical_mesh: DeviceMesh
    ) -> ExecutionGroup:
        return cls(physical_ranks, logical_mesh)

    def logical_rank(self, physical_rank: int) -> int:
        try:
            return self._logical_by_physical[physical_rank]
        except KeyError as error:
            raise ValueError(
                f"physical rank {physical_rank} is outside execution group"
            ) from error

    def physical_rank(self, logical_rank: int) -> int:
        if not 0 <= logical_rank < len(self._physical_ranks):
            raise ValueError(f"logical rank {logical_rank} is outside execution group")
        return self._physical_ranks[logical_rank]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ExecutionGroup):
            return False
        return (
            self._physical_ranks == other._physical_ranks
            and self._logical_mesh == other._logical_mesh
        )

    def __hash__(self) -> int:
        return hash((self._physical_ranks, self._logical_mesh))

    def __repr__(self) -> str:
        return (
            f"ExecutionGroup(physical_ranks={self._physical_ranks!r}, "
            f"logical_mesh={self._logical_mesh!r})"
        )

    def __getstate__(self) -> tuple[tuple[int, ...], DeviceMesh]:
        return (self._physical_ranks, self._logical_mesh)

    def __setstate__(self, state: tuple[tuple[int, ...], DeviceMesh]) -> None:
        physical_ranks, logical_mesh = state
        self._physical_ranks = physical_ranks
        self._logical_mesh = logical_mesh
        self._logical_by_physical = freeze_mapping(
            {rank: i for i, rank in enumerate(physical_ranks)}
        )


@dataclass(frozen=True, slots=True)
class ExecutionLayout:
    groups: tuple[ExecutionGroup, ...]
    _group_by_rank: tuple[int, ...] = field(repr=False)
    cycle: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.groups, tuple):
            raise TypeError("execution layout groups must be a tuple")
        if not isinstance(self._group_by_rank, tuple):
            raise TypeError("execution layout rank index must be a tuple")
        if not isinstance(self.cycle, tuple):
            raise TypeError("execution layout cycle must be a tuple")
        groups = self.groups
        if not groups:
            raise ValueError("execution layout requires groups")
        mesh = groups[0].logical_mesh
        if any(group.logical_mesh != mesh for group in groups):
            raise ValueError("execution groups must share a logical mesh")
        ranks = tuple(rank for group in groups for rank in group.physical_ranks)
        if sorted(ranks) != list(range(len(ranks))):
            raise ValueError("execution groups must cover physical ranks exactly once")
        if len(self._group_by_rank) != len(ranks):
            raise ValueError("execution layout rank index length mismatch")
        if self.cycle != tuple(group.physical_ranks for group in groups):
            raise ValueError("execution layout cycle does not match groups")

    @property
    def world_size(self) -> int:
        return len(self._group_by_rank)

    @property
    def logical_mesh(self) -> DeviceMesh:
        return self.groups[0].logical_mesh

    def group_for_rank(self, physical_rank: int) -> ExecutionGroup:
        if not 0 <= physical_rank < self.world_size:
            raise ValueError(
                f"physical rank {physical_rank} is outside execution layout"
            )
        return self.groups[self._group_by_rank[physical_rank]]

    def group_for_step(self, step: int) -> ExecutionGroup:
        if type(step) is not int or step < 0:
            raise ValueError("execution step must be a nonnegative integer")
        return self.groups[step % len(self.groups)]

    def is_active(self, physical_rank: int, step: int) -> bool:
        return self.group_for_rank(physical_rank) is self.group_for_step(step)


def compile_execution_layout(
    mesh: DeviceMesh,
    *,
    unit_size: int | None = None,
    groups: tuple[tuple[int, ...], ...] | None = None,
) -> ExecutionLayout:
    size = mesh.size if unit_size is None else unit_size
    if type(size) is not int or size <= 0 or mesh.size % size:
        raise ValueError(
            f"mesh size {mesh.size} must be divisible by interleave_unit_size {size}"
        )
    trailing = prod(mesh.shape[1:])
    if size % trailing:
        raise ValueError(
            f"interleave_unit_size {size} must be divisible by trailing mesh size {trailing}"
        )
    ranks = (
        tuple(tuple(range(start, start + size)) for start in range(0, mesh.size, size))
        if groups is None
        else groups
    )
    logical_mesh = DeviceMesh(mesh.axis_names, (size // trailing,) + mesh.shape[1:])
    canonical_groups = tuple(
        ExecutionGroup.create(tuple(group), logical_mesh) for group in ranks
    )
    flat_ranks = tuple(
        rank for group in canonical_groups for rank in group.physical_ranks
    )
    if sorted(flat_ranks) != list(range(mesh.size)):
        raise ValueError("execution groups must cover physical ranks exactly once")
    inverse = [0] * mesh.size
    for index, group in enumerate(canonical_groups):
        for rank in group.physical_ranks:
            inverse[rank] = index
    layout = ExecutionLayout(
        groups=canonical_groups,
        _group_by_rank=tuple(inverse),
        cycle=tuple(group.physical_ranks for group in canonical_groups),
    )
    if layout.world_size != mesh.size:
        raise ValueError("execution groups must cover the declared physical mesh")
    return layout
