import json
import pathlib
import numpy as np
from typing import Any

from fluxway.common.decorators import spmd_source, operator, requires_worker_process
from fluxway.common.index_descriptor import ContinuousIndex
from fluxway.common.operator_policy import AsyncPolicy
from fluxway.common.types.tensor import TransportTensor
from fluxway.reader import GrainReader
from fluxway import ResourceProfile
from .data_types import UserBatch, ActiveFeatures, PassiveFeatures, LabelFeatures


def _row_partitioned_int_tensor(
    rows: list[list[int]],
    *,
    global_shape: tuple[int, int],
    origin_indices: Any,
) -> TransportTensor:
    flat = [value for row in rows for value in row]
    splits = [0]
    offset = 0
    for row in rows:
        offset += len(row)
        splits.append(offset)
    return TransportTensor(
        flat_values=np.asarray(flat, dtype=np.int64),
        nested_splits=[np.asarray(splits, dtype=np.int32)],
        global_shape=global_shape,
        origin_indices=origin_indices,
    )


@requires_worker_process
@spmd_source(
    resource_profile=ResourceProfile(base={"CPU": 1, "memory": 1024**3}),
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
            # Synthesize a recommendation corpus whose item / user popularity
            # follows a power-law (Zipf-like) distribution, mirroring real
            # traffic where a small set of "hot" entities dominates impressions.
            # This guarantees that different DP ranks observe overlapping sparse
            # IDs and thereby exercise the cross-DP embedding routing path
            # (e.g. the same item shows up on multiple input workers, gets routed to a single
            # sparse worker, and its embedding fans out to multiple dense
            # workers).
            seed = static_args.get("seed", 42)
            rng = np.random.default_rng(seed)

            vocab_size_user = static_args.get("vocab_size_user", 100)
            vocab_size_item = static_args.get("vocab_size_item", 2000)
            num_samples = static_args.get("num_samples", 1000)
            item_alpha = static_args.get("item_zipf_alpha", 1.2)
            user_alpha = static_args.get("user_zipf_alpha", 1.0)
            max_seq_len = static_args.get("max_seq_len", 5)

            item_probs = self._zipf_probs(vocab_size_item, item_alpha, rng)
            user_probs = self._zipf_probs(vocab_size_user, user_alpha, rng)

            # Heavier-tailed sequence length: most users have short history,
            # a few have longer ones.
            seq_len_choices = np.arange(1, max_seq_len + 1)
            seq_len_probs = 1.0 / seq_len_choices.astype(np.float64)
            seq_len_probs /= seq_len_probs.sum()

            user_ids = rng.choice(vocab_size_user, size=num_samples, p=user_probs)
            target_item_ids = rng.choice(
                vocab_size_item, size=num_samples, p=item_probs
            )
            seq_lens = rng.choice(seq_len_choices, size=num_samples, p=seq_len_probs)

            data = []
            for i in range(num_samples):
                sl = int(seq_lens[i])
                hist_seq = rng.choice(vocab_size_item, size=sl, p=item_probs).tolist()

                data.append(
                    {
                        "user_id": [int(user_ids[i])],
                        "user_age_level": [int(rng.integers(0, 5))],
                        "request_hour": [int(rng.integers(0, 24))],
                        "hist_click_item_seq": [int(x) for x in hist_seq],
                        "hist_seq_length": [sl],
                        "target_item_id": [int(target_item_ids[i])],
                        "target_item_category": [int(rng.integers(0, 10))],
                        "target_item_stats": [
                            rng.random(),
                            (rng.random() * 20),
                        ],
                        "is_click": [int(rng.integers(0, 2))],
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

    @staticmethod
    def _zipf_probs(
        vocab_size: int, alpha: float, rng: np.random.Generator
    ) -> np.ndarray:
        """Return a permuted Zipf-like probability vector over [0, vocab_size).

        The raw 1 / (rank + 1) ** alpha mass is then permuted so that "hot"
        IDs are scattered across the vocabulary instead of being clustered at
        small numeric values.  This decouples ID popularity from ID encoding,
        which is closer to real-world scenarios and prevents the downstream
        ``ModuloPartitioner`` from skewing all hot IDs onto a single shard.
        """
        ranks = np.arange(1, vocab_size + 1, dtype=np.float64)
        p = 1.0 / (ranks**alpha)
        p /= p.sum()
        perm = rng.permutation(vocab_size)
        out = np.empty_like(p)
        out[perm] = p
        return out

    @operator(
        "read",
        async_policy=AsyncPolicy.pipelined(depth=4),
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
        g_shape_one_sparse = (global_batch_size, 1)
        g_shape_item_sparse = (
            global_batch_size,
            int(static_args.get("max_seq_len", 5)) + 1,
        )
        g_shape_hist_sparse = (
            global_batch_size,
            int(static_args.get("max_seq_len", 5)),
        )
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
            user_id=_row_partitioned_int_tensor(
                [[s["user_id"][0]] for s in samples],
                global_shape=g_shape_one_sparse,
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
                global_shape=g_shape_hist_sparse,
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
            target_item_id=_row_partitioned_int_tensor(
                [[s["target_item_id"][0], *s["hist_click_item_seq"]] for s in samples],
                global_shape=g_shape_item_sparse,
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
