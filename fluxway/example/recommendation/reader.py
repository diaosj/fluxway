import json
import pathlib
import numpy as np
from typing import Any

from fluxway.common.decorators import spmd_source, operator
from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import GrainReader
from .data_types import UserBatch, ActiveFeatures, PassiveFeatures, LabelFeatures


@spmd_source(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("worker",),
    interleaved=True,
    interleave_unit_size=1,
)
class RecommendationReader(GrainReader):
    rank: int
    world_size: int
    """
    Produces batches of UserBatch in col-store layout (Active / Passive / Label).
    When no data_source is provided, generates a 1000-sample synthetic dataset.
    """

    def _create_data_loader(
        self,
        static_args: dict[str, Any],
        rank: int,
        world_size: int,
        shard_index: int | None = None,
        shard_count: int | None = None,
    ):
        if "data_source" not in static_args:
            import numpy as np

            np.random.seed(42)

            vocab_size_user = 100
            vocab_size_item = 5000

            num_samples = 1000
            user_ids = np.random.randint(0, vocab_size_user, size=num_samples)
            target_item_ids = np.random.randint(0, vocab_size_item, size=num_samples)

            data = []
            for i in range(num_samples):
                seq_len = np.random.randint(1, 6)
                hist_seq = np.random.randint(0, vocab_size_item, size=seq_len).tolist()

                data.append(
                    {
                        "user_id": [int(user_ids[i])],
                        "user_age_level": [np.random.randint(0, 5)],
                        "request_hour": [np.random.randint(0, 24)],
                        "hist_click_item_seq": [int(x) for x in hist_seq],
                        "hist_seq_length": [seq_len],
                        "target_item_id": [int(target_item_ids[i])],
                        "target_item_category": [np.random.randint(0, 10)],
                        "target_item_stats": [np.random.rand(), np.random.rand() * 20],
                        "is_click": [np.random.randint(0, 2)],
                    }
                )

            static_args["data_source"] = data
            static_args.setdefault("batch_size", 24)
            static_args.setdefault("num_epochs", 1)
            static_args.setdefault("shuffle", False)

        super()._create_data_loader(
            static_args, rank, world_size, shard_index, shard_count
        )
        self._completed_steps = 0

    def __init__(self):
        super().__init__()

    @operator(
        "read",
        output_sharding={
            "active": {"type": "static", "axis": ["worker"]},
            "passive": {"type": "static", "axis": ["worker"]},
            "labels": {"type": "static", "axis": ["worker"]},
        },
    )
    def read(self) -> UserBatch | None:
        static_args = getattr(self, "static_args", {})
        batch_size = static_args.get("batch_size", 24)

        # -- Invariants on configuration --
        assert isinstance(batch_size, int) and batch_size > 0, (
            f"[Reader] batch_size must be a positive int, got {batch_size!r}"
        )

        samples = []
        for _ in range(batch_size):
            item = self.next_item()
            if item is None:
                break
            samples.append(item)

        if not samples:
            return None

        self._completed_steps += 1

        logical_world = self.world_size if hasattr(self, "world_size") else 1
        global_batch_size = batch_size * logical_world
        g_shape = (global_batch_size,)
        origins = ContinuousIndex.chunk(0, global_batch_size)

        # -- Invariants on per-sample structure --
        required_keys = (
            "user_id",
            "user_age_level",
            "request_hour",
            "hist_click_item_seq",
            "hist_seq_length",
            "target_item_id",
            "target_item_category",
            "target_item_stats",
            "is_click",
        )
        for k in required_keys:
            assert k in samples[0], f"[Reader] sample missing required key '{k}'"

        # Determine feature dimensions from first sample
        stats_dim = len(samples[0]["target_item_stats"])
        assert stats_dim > 0, (
            f"[Reader] target_item_stats must be non-empty, got dim={stats_dim}"
        )
        # All samples must agree on stats dimensionality.
        for i, s in enumerate(samples):
            assert len(s["target_item_stats"]) == stats_dim, (
                f"[Reader] sample {i} target_item_stats dim "
                f"{len(s['target_item_stats'])} != expected {stats_dim}"
            )
            assert s["is_click"][0] in (0, 1), (
                f"[Reader] sample {i} is_click must be 0 or 1, got {s['is_click'][0]}"
            )
        g_shape_stats = (global_batch_size, stats_dim)

        active = ActiveFeatures(
            user_id=TransportTensor.from_lists(
                [s["user_id"][0] for s in samples],
                dtype=np.int64,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            user_age_level=TransportTensor.from_lists(
                [s["user_age_level"][0] for s in samples],
                dtype=np.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            request_hour=TransportTensor.from_lists(
                [s["request_hour"][0] for s in samples],
                dtype=np.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            hist_click_item_seq=TransportTensor.from_lists(
                [s["hist_click_item_seq"] for s in samples],
                dtype=np.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            hist_seq_length=TransportTensor.from_lists(
                [s["hist_seq_length"][0] for s in samples],
                dtype=np.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
        )

        passive = PassiveFeatures(
            target_item_id=TransportTensor.from_lists(
                [s["target_item_id"][0] for s in samples],
                dtype=np.int64,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            target_item_category=TransportTensor.from_lists(
                [s["target_item_category"][0] for s in samples],
                dtype=np.int32,
                global_shape=g_shape,
                origin_indices=origins,
            ),
            target_item_stats=TransportTensor.from_lists(
                [s["target_item_stats"] for s in samples],
                dtype=np.float32,
                global_shape=g_shape_stats,
                origin_indices=origins,
            ),
        )

        labels = LabelFeatures(
            is_click=TransportTensor.from_lists(
                [s["is_click"][0] for s in samples],
                dtype=np.int32,
                global_shape=g_shape,
                origin_indices=origins,
            )
        )

        return UserBatch(active=active, passive=passive, labels=labels)

    def save_state(self, ckpt_dir: str) -> dict[str, Any]:  # type: ignore
        shard_count = getattr(self, "world_size", 1)
        static_args = getattr(self, "static_args", {})
        path = pathlib.Path(ckpt_dir) / f"reader_rank_{self.rank}_of_{shard_count}.json"
        with open(path, "w") as f:
            json.dump(
                {
                    "completed_steps": getattr(self, "_completed_steps", 0),
                    "batch_size": static_args.get("batch_size", 24),
                },
                f,
            )
        return {}

    def restore_state(self, state: dict[str, Any], ckpt_dir: str) -> None:  # type: ignore
        # completed_steps is identical across all ranks, so take the max.
        # Pair it with the saved batch_size to get the exact items-per-rank to skip,
        # regardless of the new world_size or batch_size.
        completed_steps = 0
        saved_batch_size = None
        for p in sorted(pathlib.Path(ckpt_dir).glob("reader_rank_*_of_*.json")):
            with open(p) as f:
                data = json.load(f)
            steps = data.get("completed_steps", 0)
            if steps > completed_steps:
                completed_steps = steps
                saved_batch_size = data.get("batch_size")

        new_shard_count = getattr(self, "world_size", 1)
        static_args = getattr(self, "static_args", {})
        if saved_batch_size is None:
            saved_batch_size = static_args.get("batch_size", 24)

        per_rank_skip = completed_steps * saved_batch_size

        self._create_data_loader(static_args, getattr(self, "rank", 0), new_shard_count)
        for _ in range(per_rank_skip):
            self.next_item()

        self._completed_steps = completed_steps
