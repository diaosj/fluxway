"""Base class for all Reader SPMDs."""

from typing import Any

from fluxway.common.decorators import operator, spmd
from fluxway.common.index_descriptor import ContinuousIndex, IndexDescriptor
from fluxway.common.tensor_backend import get_backend
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.ingestion import IngestionLayoutActivation


@spmd()
class BaseReader:
    """Base implementation for data-producing SPMD readers."""

    rank: int
    world_size: int
    static_args: dict[str, Any]

    def __init__(
        self,
        name: str = "",
        mesh: Any = None,
        runtime_env: Any = None,
        resource: Any = None,
        **kwargs: Any,
    ):
        pass

    async def activate_ingestion_layout(
        self, activation: IngestionLayoutActivation
    ) -> None:
        if not isinstance(activation, IngestionLayoutActivation):
            raise TypeError(
                "reader ingestion activation requires IngestionLayoutActivation"
            )

    @operator("read")
    def read(self) -> Any:
        """Produce a batch as a data-type object or tensor pytree."""
        raise NotImplementedError("Subclasses must implement read()")

    def create_tensor(
        self, data: Any, dtype: Any = None, uniform_dense_partition: bool = False
    ) -> "TransportTensor":
        """Create a dense or structurally inferred tensor for the active source batch."""
        context = self.ctx
        origin_indices, global_rows = self._source_batch_coordinates(
            _outer_length(data)
        )
        if uniform_dense_partition:
            backend = get_backend()
            tensor = backend.tensor(data, dtype=dtype)
            return TransportTensor.from_tensor(
                tensor,
                global_shape=(global_rows, *tensor.shape[1:]),
                origin_indices=origin_indices,
            )
        if isinstance(data, list):
            result = TransportTensor.from_lists(
                data,
                dtype=dtype,
                logical_world_size=context.logical_world_size,
                origin_indices=origin_indices,
                is_uniform_dense_partition=uniform_dense_partition,
            )
        else:
            backend = get_backend()
            tensor = backend.tensor(data, dtype=dtype)
            result = TransportTensor.from_tensor(
                tensor,
                logical_world_size=context.logical_world_size,
                origin_indices=origin_indices,
                is_uniform_dense_partition=uniform_dense_partition,
            )
        return result

    def create_row_sparse_tensor(
        self, rows: list[list[Any]], *, width: int, dtype: Any = None
    ) -> "TransportTensor":
        """Create a row-partitioned sparse occurrence stream."""
        origin_indices, global_rows = self._source_batch_coordinates(len(rows))
        return TransportTensor.from_row_partitioned(
            rows,
            width=width,
            dtype=dtype,
            global_shape=(global_rows, width),
            origin_indices=origin_indices,
        )

    def _source_batch_coordinates(self, row_count: int) -> tuple[IndexDescriptor, int]:
        context = self.ctx
        if context is None:
            raise RuntimeError(
                "reader tensor creation requires an active OperatorContext"
            )
        items = context.source_work_items
        if items is None:
            if context.global_source_offset is None:
                raise ValueError(
                    "reader requires source work or explicit global provenance"
                )
            return ContinuousIndex.chunk(
                context.global_source_offset, row_count
            ), row_count * context.logical_world_size
        if not items or sum(item.cardinality for item in items) != row_count:
            raise ValueError("reader output rows do not match acquired source work")
        batches = {item.assignment.global_batch_size for item in items}
        if len(batches) != 1 or any(item.key.step != context.step for item in items):
            raise ValueError("reader work items do not belong to one logical batch")
        return IndexDescriptor.concatenate(
            [
                ContinuousIndex.chunk(item.global_start, item.cardinality)
                for item in items
            ]
        ), batches.pop()


def _outer_length(data: Any) -> int:
    if isinstance(data, list):
        return len(data)
    shape = getattr(data, "shape", None)
    if shape is not None and len(shape) > 0:
        return int(shape[0])
    raise TypeError("reader tensor data must have an outer batch dimension")
