import os

import numpy as np
import sparse_embedding as se
import sparse_embedding.checkpoint

from fluxway.common.tensor_backend import get_backend
from fluxway.common.decorators import spmd, operator
from fluxway.common.utils.logging import get_logger
from fluxway.common.types.tensor import TransportTensor
from fluxway.example.recommendation.data_types import LookupOutput

logger = get_logger(__name__)


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("shard",),
)
class SparseEmbeddingSPMD:
    """Sharded embedding table backed by sparse_embedding.EmbeddingStore.

    Each shard owns the embeddings for keys hash-routed to it.  Forward lookup
    uses EmbeddingStore.lookup_or_insert (lazy initialisation on first access),
    and the backward pass accumulates gradients with Adam then calls
    optimizer_step once per training step.
    """

    def __init__(
        self,
        vocab_size: int = 1000,
        embed_dim: int = 3,
        num_threads: int | None = None,
    ):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.num_threads = num_threads or os.cpu_count() or 4

        self.store = se.EmbeddingStore(
            embedding_config=se.EmbeddingConfig(
                dims=[embed_dim],
                num_threads=self.num_threads,
                dtype="float32",
            ),
            initializer_config=se.Uniform(min_val=-0.01, max_val=0.01),
            optimizer_config=se.Adam(lr=0.01),
        )
        self._forward_keys_cache: dict[int, np.ndarray] = {}

        logger.debug(
            "SparseEmbeddingSPMD_init rank=%s world_size=%s vocab_size=%s embed_dim=%s",
            getattr(self, "rank", None),
            getattr(self, "world_size", None),
            vocab_size,
            embed_dim,
        )

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def on_snapshot_begin(self, ckpt_dir: str) -> None:
        uri = f"{ckpt_dir}/sparse_lance"
        backend = se.checkpoint.get_backend("lance")
        backend.init_dataset(self.store, uri)

    def save_state(self, ckpt_dir: str) -> dict:
        uri = f"{ckpt_dir}/sparse_lance"
        backend = se.checkpoint.get_backend("lance")
        metas = backend.export_fragments(self.store, uri)
        return {"uri": uri, "lance_meta": metas}

    def on_snapshot_commit(
        self, ckpt_dir: str, gathered_states: dict[int, dict]
    ) -> None:
        uri = f"{ckpt_dir}/sparse_lance"
        backend = se.checkpoint.get_backend("lance")
        all_metas = []
        for state in gathered_states.values():
            if "lance_meta" in state:
                all_metas.extend(state["lance_meta"])
        if all_metas:
            backend.commit_fragments(uri, all_metas)

    def restore_state(self, state: dict, ckpt_dir: str) -> None:
        uri = state.get("uri", f"{ckpt_dir}/sparse_lance")
        backend = se.checkpoint.get_backend("lance")
        partitioner = se.ModuloPartitioner(num_shards=self.world_size)
        backend.load(
            self.store,
            uri,
            filter_fn=partitioner,
            target_shard_id=self.rank,
        )

    # ------------------------------------------------------------------
    # Forward: lookup
    # ------------------------------------------------------------------

    @operator(
        name="lookup",
        data_argument=["unique_ids"],
        input_sharding={
            "unique_ids": {"type": "dynamic", "axis": ["shard"], "fn": "hash_routing"}
        },
        output_sharding={"sparse_features": {"type": "dynamic", "axis": ["shard"]}},
    )
    def lookup(self, unique_ids: TransportTensor) -> LookupOutput:
        step = self.ctx.step
        ids_np = unique_ids.flat_values.ravel().astype(np.int64)

        # Lazy-insert new keys, return embeddings for all keys in this shard.
        vals_dlm, _, _ = self.store.lookup_or_insert(ids_np, dim=self.embed_dim)
        backend = get_backend()
        emb_np = backend.from_dlpack(vals_dlm).reshape(ids_np.size, self.embed_dim)

        self._forward_keys_cache[step] = ids_np

        # -- 数值正确性验证: lookup keys 与 received unique_ids 一致 --
        np.testing.assert_array_equal(
            ids_np,
            unique_ids.flat_values.ravel().astype(np.int64),
            err_msg=f"[SparseSPMD] rank={self.rank} lookup keys 与 unique_ids 不匹配",
        )

        g_shape = None
        if unique_ids.global_shape and len(unique_ids.global_shape) >= 1:
            g_shape = (unique_ids.global_shape[0], self.embed_dim)

        output = TransportTensor(
            emb_np,
            origin_indices=unique_ids.origin_indices,
            global_shape=g_shape,
            receipt=unique_ids.receipt,
        )
        print(
            f"[SparseEmbeddingSPMD.lookup] rank={self.rank} step={step} "
            f"keys={ids_np} emb_norm={float(np.linalg.norm(emb_np)):.4f}"
        )
        return LookupOutput(sparse_features=output)

    # ------------------------------------------------------------------
    # Backward: accumulate gradients and take an optimizer step
    # ------------------------------------------------------------------

    @operator(
        name="update",
        data_argument=["grads"],
        input_sharding={
            "grads": {"type": "mirror", "reference": "lookup.sparse_features"}
        },
    )
    def update(self, grads: TransportTensor) -> None:
        step = self.ctx.step
        keys_np = self._forward_keys_cache.pop(step, None)

        if keys_np is None:
            raise ValueError(
                f"SparseEmbeddingSPMD rank={self.rank}: "
                f"missing forward cache for step {step}"
            )

        grads_np = grads.flat_values.reshape(keys_np.size, self.embed_dim)

        # -- 数值正确性验证: 梯度维度与 keys 一致 --
        assert grads_np.shape == (keys_np.size, self.embed_dim), (
            f"[SparseSPMD] update grads shape {grads_np.shape} "
            f"!= expected ({keys_np.size}, {self.embed_dim})"
        )

        self.store.accumulate_gradients(keys_np, grads_np)
        self.store.optimizer_step()

        print(
            f"[SparseEmbeddingSPMD.update] rank={self.rank} step={step} "
            f"grad_norm={float(np.linalg.norm(grads_np)):.4f}"
        )


def hash_routing(tensor, mesh):
    if isinstance(tensor, TransportTensor) and tensor.routing_indices is not None:
        val = tensor.routing_indices
    else:
        val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    partitioner = se.ModuloPartitioner(num_shards=mesh.size)
    shard_ids_dlm = partitioner.partition(val)
    return get_backend().from_dlpack(shard_ids_dlm)
