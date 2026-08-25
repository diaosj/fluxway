"""
Standardized Reader implementation leveraging Google Grain.
"""

from typing import Any

from fluxway.common.decorators import operator, spmd
from fluxway.common.types import SaveSession, RestoreSession
from fluxway.common.utils.logging import get_logger
from fluxway.reader.base_reader import BaseReader

logger = get_logger(__name__)

try:
    import grain.python as grain
except ImportError:
    grain = None


if grain is not None:

    class _CallableMapTransform(grain.MapTransform):
        def __init__(self, transform: Any) -> None:
            self._transform = transform

        def map(self, element: Any) -> Any:
            return self._transform(element)


class StridedIndexSampler:
    def __init__(
        self,
        base_sampler: Any,
        *,
        shard_index: int,
        shard_count: int,
        block_size: int = 1,
        drop_remainder: bool,
    ) -> None:
        if shard_count <= 0:
            raise ValueError(f"shard_count must be positive, got {shard_count}.")
        if shard_index < 0 or shard_index >= shard_count:
            raise ValueError(
                f"shard_index {shard_index} must be in [0, {shard_count})."
            )
        if isinstance(block_size, bool) or not isinstance(block_size, int):
            raise TypeError("block_size must be an integer.")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}.")
        self._base_sampler = base_sampler
        self._shard_index = shard_index
        self._shard_count = shard_count
        self._block_size = block_size
        base_length = len(base_sampler)
        if drop_remainder:
            base_length -= base_length % (shard_count * block_size)
        self._base_length = base_length

    def __len__(self) -> int:
        cycle_size = self._shard_count * self._block_size
        completed_cycles, remainder = divmod(self._base_length, cycle_size)
        block_start = self._shard_index * self._block_size
        return completed_cycles * self._block_size + min(
            max(remainder - block_start, 0),
            self._block_size,
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(base_sampler={self._base_sampler!r}, "
            f"shard_index={self._shard_index}, "
            f"shard_count={self._shard_count}, "
            f"block_size={self._block_size}, "
            f"base_length={self._base_length})"
        )

    def __getitem__(self, index: int) -> Any:
        if index < 0 or index >= len(self):
            raise IndexError(f"sampler index {index} must be in [0, {len(self)}).")
        block_index, block_offset = divmod(index, self._block_size)
        global_index = (
            block_index * self._shard_count * self._block_size
            + self._shard_index * self._block_size
            + block_offset
        )
        return self._base_sampler[global_index]


@spmd()
class GrainReader(BaseReader):
    """
    Standard Reader implementation using Google Grain.

    The DataLoader and its iterator are created eagerly at instantiation time
    (not lazily on the first ``read`` call).  This guarantees that
    ``save_state`` / ``restore_state`` always operate on a live iterator, which is
    required for correct Grain-style checkpointing where state is a JSON-
    serialisable index dict (``{"next_index": N}``) managed by the iterator.

    Supports Elastic Sharding via Logical Shard Pools.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._dataloader = None
        self._iterator = None
        self._output_type = None

        # The framework sets rank / world_size / static_args on `self` before
        # calling __init__, so we can eagerly initialise the data pipeline here.
        static_args = getattr(self, "static_args", None)
        rank = getattr(self, "rank", None)
        world_size = getattr(self, "world_size", 1)
        if static_args is not None and rank is not None:
            self._create_data_loader(static_args, rank, world_size)

    def _create_data_loader(
        self,
        static_args: dict[str, Any],
        rank: int,
        world_size: int,
        shard_index: int | None = None,
        shard_count: int | None = None,
    ):
        """
        Construct the Grain DataLoader pipeline and create the iterator.

        Sub-classes that override this method must call ``super()`` **or**
        ensure ``self._iterator`` is set to a Grain ``DatasetIterator`` by the
        time the method returns, so that ``save_state`` / ``restore_state`` work.

        Args:
            shard_index: If provided, overrides rank for data sharding.
            shard_count: If provided, overrides world_size for data sharding.
        """
        if grain is None:
            raise ImportError("google-grain not found. Install it to use GrainReader.")

        data_source = static_args.get("data_source")
        if data_source is None:
            raise ValueError("GrainReader requires 'data_source' in static_args.")

        transformations = []
        for tx in static_args.get("transformations", []):
            if isinstance(tx, grain.MapTransform):
                transformations.append(tx)
            elif callable(tx):
                transformations.append(_CallableMapTransform(tx))
            else:
                raise TypeError(
                    "GrainReader transformations must be callables or "
                    f"grain.MapTransform instances, got {type(tx)!r}."
                )

        num_logical_shards = static_args.get("num_logical_shards")
        if shard_index is not None and shard_count is not None:
            s_idx, s_cnt = shard_index, shard_count
        elif num_logical_shards:
            s_cnt = num_logical_shards
            s_idx = rank % num_logical_shards
        else:
            s_idx, s_cnt = rank, world_size

        sampler = static_args.get("sampler")
        data_loader_shard_options = None
        if sampler is None:
            base_sampler = grain.IndexSampler(
                num_records=len(data_source),
                shard_options=grain.NoSharding(),
                shuffle=static_args.get("shuffle", True),
                seed=static_args.get("seed", 42),
                num_epochs=static_args.get("num_epochs"),
            )
            sampler = StridedIndexSampler(
                base_sampler,
                shard_index=s_idx,
                shard_count=s_cnt,
                block_size=self.records_per_position(),
                drop_remainder=static_args.get("drop_remainder", False),
            )
            data_loader_shard_options = grain.NoSharding()

        self._output_type = static_args.get("output_data_type")
        self._dataloader = grain.DataLoader(
            data_source=data_source,
            sampler=sampler,
            operations=transformations,
            worker_count=static_args.get("num_threads", 0),
            shard_options=data_loader_shard_options,
        )
        self._iterator = iter(self._dataloader)

        logger.info(
            f"[spmd_id={getattr(self, 'spmd_id', 'unknown')}] "
            f"Rank {rank} mapped to Logical Shard {s_idx}/{s_cnt}. "
            f"Serving {len(sampler)} records."
        )

    @operator("read", is_source=True)
    def read(self) -> Any:
        return self.next_item()

    def next_item(self) -> Any:
        if self._iterator is None:
            raise RuntimeError(
                f"GrainReader (rank={getattr(self, 'rank', '?')}): iterator is not "
                f"initialised.  This means _create_data_loader was not called during "
                f"__init__.  Check that 'static_args', 'rank', and 'world_size' are "
                f"injected by the framework before __init__ is called."
            )
        batch = next(self._iterator, None)
        if batch is None:
            return None
        if self._output_type and not isinstance(batch, self._output_type):
            if isinstance(batch, dict):
                return self._output_type(**batch)
            return self._output_type(batch)
        return batch

    def records_per_position(self) -> int:
        return 1

    def save_state(self, context: SaveSession) -> dict[str, Any]:
        position = context.cut.position
        if position is None:
            raise ValueError("GrainReader checkpoint requires a checkpoint position")
        return {
            "position": position,
            "records_per_position": self.records_per_position(),
        }

    def restore_checkpoint_state(
        self, state: dict[str, Any], context: RestoreSession
    ) -> None:
        """
        Restore the iterator to a previously saved position.

        Raises ``TypeError`` when the iterator does not support ``set_state``
        so that sub-classes that use a non-Grain iterator get an explicit
        error instead of silently continuing from the wrong position.
        """
        if set(state) != {"position", "records_per_position"}:
            raise ValueError(
                "GrainReader checkpoint state fields do not match protocol"
            )
        position = state.get("position")
        records_per_position = state.get("records_per_position")
        if (
            isinstance(position, bool)
            or not isinstance(position, int)
            or position < 0
            or isinstance(records_per_position, bool)
            or not isinstance(records_per_position, int)
            or records_per_position <= 0
        ):
            raise ValueError("GrainReader checkpoint state has invalid position")
        if records_per_position != self.records_per_position():
            raise ValueError(
                "GrainReader checkpoint record width does not match source configuration"
            )
        if context.cut.position != position:
            raise ValueError(
                "GrainReader checkpoint state position does not match checkpoint cut"
            )
        self.restore_position(position)

    def restore_position(self, position: int) -> None:
        if position < 0:
            raise ValueError("GrainReader restore position must be non-negative")
        static_args = getattr(self, "static_args", None)
        rank = getattr(self, "rank", None)
        world_size = getattr(self, "world_size", None)
        if static_args is None or rank is None or world_size is None:
            raise RuntimeError("GrainReader restore requires initialized topology")
        self._create_data_loader(static_args, rank, world_size)
        local_count = self._local_items_before(position)
        assert self._iterator is not None
        for _ in range(local_count):
            if next(self._iterator, None) is None:
                raise RuntimeError(
                    f"GrainReader source ended before restore position {position}"
                )

    def _local_items_before(self, position: int) -> int:
        config_factory = getattr(type(self), "get_spmd_config", None)
        if config_factory is None:
            raise RuntimeError("GrainReader class has no SPMD configuration")
        config = config_factory()
        spec = config.operator_specs.get("read")
        if spec is None:
            raise RuntimeError("GrainReader SPMD configuration has no read operator")
        if not spec.is_interleaved:
            local_positions = position
        else:
            unit_size = spec.interleave_unit_size
            stride = self.world_size // unit_size
            unit_index = self.rank // unit_size
            if position <= unit_index:
                local_positions = 0
            else:
                local_positions = (position - 1 - unit_index) // stride + 1
        return local_positions * self.records_per_position()
