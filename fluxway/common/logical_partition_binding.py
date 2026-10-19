from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class LogicalPartitionBinding:
    """Immutable worker-local value selection over an execution logical rank."""

    values: Mapping[int, Any]
    axes: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.values, Mapping):
            raise TypeError("logical partition binding values must be a mapping")
        normalized: dict[int, Any] = {}
        for partition_id, value in self.values.items():
            if type(partition_id) is not int or partition_id < 0:
                raise TypeError(
                    "logical partition binding ids must be non-negative integers"
                )
            normalized[partition_id] = value
        if not normalized:
            raise ValueError("logical partition binding requires at least one value")
        object.__setattr__(self, "values", MappingProxyType(normalized))
        if self.axes is not None:
            axes = tuple(self.axes)
            if (
                not axes
                or len(axes) != len(set(axes))
                or any(not isinstance(axis, str) or not axis for axis in axes)
            ):
                raise ValueError("logical partition binding axes must be unique names")
            object.__setattr__(self, "axes", axes)

    def resolve(self, logical_partition_id: int, *, logical_mesh: Any = None) -> Any:
        if self.axes is not None:
            if logical_mesh is None or not hasattr(logical_mesh, "axis_names"):
                raise TypeError("logical mesh is required for axis-projected binding")
            coordinates = logical_mesh.get_coordinate(logical_partition_id)
            positions = {
                name: index for index, name in enumerate(logical_mesh.axis_names)
            }
            try:
                partition_id = 0
                for axis in self.axes:
                    partition_id = (
                        partition_id * logical_mesh.get_axis_size(axis)
                        + coordinates[positions[axis]]
                    )
            except KeyError as error:
                raise ValueError(
                    f"binding axis {error.args[0]!r} is absent from logical mesh"
                ) from error
            logical_partition_id = partition_id
        try:
            return self.values[logical_partition_id]
        except KeyError as error:
            raise KeyError(
                "logical partition binding has no value for partition "
                f"{logical_partition_id}"
            ) from error


__all__ = ["LogicalPartitionBinding"]
