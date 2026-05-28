from typing import Any
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from numba import njit

from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import DPFeatures

# Select array backend for JIT-external operations based on GPU availability.
# When JAX uses GPU, CuPy provides zero-copy DLPack interop with JAX.
# When JAX falls back to CPU, NumPy is the natural choice.
_GPU_AVAILABLE = False
cupyx = None  # type: ignore[assignment]
try:
    import cupy as cp
    import cupyx  # type: ignore[no-redef]

    # Verify CUDA is actually usable (cupy may be installed without a GPU)
    try:
        _cc = cp.cuda.Device(0).compute_capability
        _GPU_AVAILABLE = True
    except Exception:
        _GPU_AVAILABLE = False
except ImportError:
    _GPU_AVAILABLE = False
    cp = None


def _xp():
    """Return CuPy if GPU is available, otherwise NumPy."""
    return cp if _GPU_AVAILABLE else np


# ---------------------------------------------------------------------------
# Numba JIT kernels for CPU acceleration
# ---------------------------------------------------------------------------


@njit(cache=True)
def _gather_2d_kernel(src, indices, out):
    """out[i, :] = src[indices[i], :] — 2D gather by row index.

    Replaces NumPy fancy indexing ``src[indices]`` with a compiled loop
    for consistent CPU throughput on large batches.
    """
    for i in range(indices.shape[0]):
        idx = indices[i]
        for d in range(src.shape[1]):
            out[i, d] = src[idx, d]


@njit(cache=True)
def _scatter_add_2d_kernel(out, idx, vals):
    """In-place ``out[idx] += vals`` for 2D ``out``/``vals`` and 1D ``idx``.

    Serial implementation to preserve np.add.at semantics under duplicate indices.
    """
    n = idx.shape[0]
    d = vals.shape[1]
    for i in range(n):
        k = idx[i]
        for j in range(d):
            out[k, j] += vals[i, j]


def _gather_sparse_input(emb_flat, inv_flat, batch_size, emb_dim, gname=""):
    """Reconstruct batch tensor from unique embeddings via inverse indices.

    Equivalent to ``emb_2d[inv_1d].reshape(batch_size, -1, emb_dim)``
    but uses Numba on CPU and CuPy fancy indexing on GPU for zero-copy
    data paths.
    """
    xp = _xp()

    emb_2d = xp.asarray(emb_flat).reshape(-1, emb_dim)
    inv_1d = xp.asarray(inv_flat).reshape(-1)

    if emb_2d.shape[0] == 0:
        raise ValueError(
            f"Sparse embedding unique payload 'emb_flat' for group {gname} cannot be empty! "
            f"emb_flat.shape={emb_flat.shape}, inv_flat.shape={inv_flat.shape}, "
            f"batch_size={batch_size}, emb_dim={emb_dim}. "
            "This indicates a bug in the fluxway routing layer."
        )

    if _GPU_AVAILABLE:
        rec = emb_2d[inv_1d]
    else:
        rec = np.empty((inv_1d.shape[0], emb_dim), dtype=emb_2d.dtype)
        _gather_2d_kernel(np.asarray(emb_2d), np.asarray(inv_1d), rec)
    return rec.reshape(batch_size, -1, emb_dim)


def _scatter_add_grad(emb_2d, inv_1d, grad_local):
    """Scatter-add per-sample gradients back to unique embedding rows.

    Equivalent to ``jnp.zeros(shape).at[inv].add(grad)`` but in-place.
    Uses cupyx.scatter_add on GPU and Numba _scatter_add_2d_kernel on CPU.
    """
    xp = _xp()
    if emb_2d.shape[0] == 0:
        raise ValueError(
            f"_scatter_add_grad received empty embedding tensor: "
            f"emb_2d.shape={emb_2d.shape}, inv_1d.shape={inv_1d.shape}, "
            f"grad_local.shape={grad_local.shape}. "
            "This indicates a bug in the fluxway routing layer."
        )

    g_unique = xp.zeros_like(emb_2d)

    if _GPU_AVAILABLE:
        cupyx.scatter_add(  # type: ignore[union-attr]
            g_unique,
            xp.asarray(inv_1d),
            grad_local,
        )
    else:
        _scatter_add_2d_kernel(
            np.asarray(g_unique),
            np.asarray(inv_1d),
            np.asarray(grad_local),
        )
    return g_unique


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp",),
)
class DenseSPMD:
    rank: int
    world_size: int
    ctx: "Any"
    rank0_address: str
    """Final computation node: forward pass, loss, and dense/embedding gradient computation."""

    def __init__(
        self,
        embed_dim: int = 3,
        dense_stats_dim: int = 2,
        grad_accum_steps: int = 2,
        name: str = "",
        mesh: "Any" = None,
        runtime_env: "Any" = None,
        resource: "Any" = None,
        **kwargs: "Any",
    ):
        self.embed_dim = embed_dim
        self.dense_stats_dim = dense_stats_dim
        self.grad_accum_steps = grad_accum_steps

        # -- Invariants on configuration --
        assert isinstance(embed_dim, int) and embed_dim > 0, (
            f"[DenseSPMD] embed_dim must be a positive int, got {embed_dim!r}"
        )
        assert isinstance(dense_stats_dim, int) and dense_stats_dim > 0, (
            f"[DenseSPMD] dense_stats_dim must be a positive int, got {dense_stats_dim!r}"
        )
        assert isinstance(grad_accum_steps, int) and grad_accum_steps >= 1, (
            f"[DenseSPMD] grad_accum_steps must be >= 1, got {grad_accum_steps!r}"
        )

        print(
            f"[DenseSPMD] init jax.distributed rank={self.rank} "
            f"world_size={self.world_size} coordinator={self.rank0_address}",
            flush=True,
        )
        jax.distributed.initialize(
            coordinator_address=self.rank0_address,
            num_processes=self.world_size,
            process_id=self.rank,
        )

        n_global = len(jax.devices())
        print(
            f"[DenseSPMD] jax.distributed ok rank={self.rank} "
            f"global_devices={n_global} local_devices={len(jax.local_devices())}"
        )
        self.mesh = Mesh(jax.devices(), ("dp",))

        input_dim = self.embed_dim * 2 + self.dense_stats_dim

        class DenseLayer(nnx.Module):
            def __init__(self, in_dim):
                self.linear = nnx.Linear(in_dim, 1, rngs=nnx.Rngs(0))

            def __call__(self, x):
                return self.linear(x).squeeze(-1)

        self.model = DenseLayer(input_dim)

        # Model params must live on the distributed mesh from creation so that
        # checkpoint save/restore sees a consistent pytree structure (all leaves
        # are globally-sharded JAX arrays, never SingleDeviceSharding).
        replicated = NamedSharding(self.mesh, P())
        state = nnx.state(self.model)
        nnx.update(
            self.model,
            jtu.tree_map(lambda p: jax.device_put(p, replicated), state),
        )
        self.grad_accum = jtu.tree_map(
            lambda p: jax.device_put(jnp.zeros_like(p), replicated),
            nnx.state(self.model),
        )

    def save_state(self) -> dict:
        return {"model_state": nnx.state(self.model), "grad_accum": self.grad_accum}

    def restore_state(self, state: dict) -> None:
        nnx.update(self.model, state["model_state"])
        self.grad_accum = state["grad_accum"]

    @operator(
        name="step",
        input_sharding={
            "emb1": {"type": "dynamic", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "dynamic", "axis": ["dp"]},
            "inv2": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "grad1": {"type": "mirror", "reference": "step.emb1"},
            "grad2": {"type": "mirror", "reference": "step.emb2"},
            "dense_grad": {"type": "static", "axis": []},
        },
    )
    def step(
        self,
        emb1: TransportTensor,
        inv1: TransportTensor,
        emb2: TransportTensor,
        inv2: TransportTensor,
        dp_data: DPFeatures,
    ) -> dict:
        # -- Outside JIT: GPU(CuPy) / CPU(NumPy) operations --
        xp = _xp()
        emb1_2d = xp.asarray(emb1.flat_values).reshape(-1, emb1.flat_values.shape[-1])  # type: ignore
        emb2_2d = xp.asarray(emb2.flat_values).reshape(-1, emb2.flat_values.shape[-1])  # type: ignore
        inv1_1d = xp.asarray(inv1.flat_values).reshape(-1)  # type: ignore
        inv2_1d = xp.asarray(inv2.flat_values).reshape(-1)  # type: ignore

        # -- Invariants on embedding dimension consistency --
        assert emb1_2d.shape[1] == self.embed_dim, (
            f"[DenseSPMD] emb1 last dim {emb1_2d.shape[1]} != configured embed_dim {self.embed_dim}"
        )
        assert emb2_2d.shape[1] == self.embed_dim, (
            f"[DenseSPMD] emb2 last dim {emb2_2d.shape[1]} != configured embed_dim {self.embed_dim}"
        )
        assert inv1_1d.shape[0] == inv2_1d.shape[0], (
            f"[DenseSPMD] inv1 length {inv1_1d.shape[0]} != inv2 length {inv2_1d.shape[0]}"
        )

        # -- Numerical correctness check: inv index range --
        assert inv1_1d.min() >= 0 and inv1_1d.max() < emb1_2d.shape[0], (
            f"[DenseSPMD] inv1 out of range: max={inv1_1d.max()}, emb1_unique_count={emb1_2d.shape[0]}"
        )
        assert inv2_1d.min() >= 0 and inv2_1d.max() < emb2_2d.shape[0], (
            f"[DenseSPMD] inv2 out of range: max={inv2_1d.max()}, emb2_unique_count={emb2_2d.shape[0]}"
        )

        # -- Embedding routing order/count invariants --
        # The cross-mesh router (input rank d -> sparse ranks {0..S-1} ->
        # dense rank d via Mirror back) must deliver exactly the unique
        # embeddings referenced by this rank's local inv, in the same
        # order as input rank d's local unique sequence. Two layers
        # cooperate to make this true:
        #   * input_spmd._sparse_unique stamps unique_tt with a per-row
        #     ExplicitIndex(arange(K)) — a local position within this
        #     rank's own unique sequence.  The label is orthogonal to
        #     hash routing (which is value-based) and only needs to be
        #     unique within the producer rank, because Mirror-back routes
        #     each row back to the dense rank that originally emitted it.
        #   * sparse_spmd.lookup propagates that origin_indices verbatim
        #     onto the embedding output. The framework's
        #     `_assemble_to_dynamic_tensor` then JIT-argsorts incoming
        #     shards by origin_indices, so any out-of-order arrival from
        #     multiple sparse ranks is stably reordered before this op
        #     sees it.
        # The two checks below catch any breakage of that contract:
        #   (1) emb_rows == inv.max()+1   -> no missing or extra rows
        #   (2) emb is numerically finite -> no transport corruption
        if inv1_1d.size > 0:
            expected_rows_1 = int(inv1_1d.max()) + 1
            assert emb1_2d.shape[0] == expected_rows_1, (
                f"[DenseSPMD] emb1 row count {emb1_2d.shape[0]} != "
                f"inv1.max+1 {expected_rows_1}; cross-mesh routing delivered "
                "the wrong number of unique embedding rows for emb1"
            )
        if inv2_1d.size > 0:
            expected_rows_2 = int(inv2_1d.max()) + 1
            assert emb2_2d.shape[0] == expected_rows_2, (
                f"[DenseSPMD] emb2 row count {emb2_2d.shape[0]} != "
                f"inv2.max+1 {expected_rows_2}; cross-mesh routing delivered "
                "the wrong number of unique embedding rows for emb2"
            )
        assert bool(xp.all(xp.isfinite(emb1_2d))), (  # type: ignore
            "[DenseSPMD] emb1 contains NaN/Inf -- possible cross-mesh "
            "transport corruption between SparseEmbeddingSPMD and DenseSPMD"
        )
        assert bool(xp.all(xp.isfinite(emb2_2d))), (  # type: ignore
            "[DenseSPMD] emb2 contains NaN/Inf -- possible cross-mesh "
            "transport corruption between SparseEmbeddingSPMD and DenseSPMD"
        )

        local_batch = int(inv1_1d.shape[0])

        # Without pooling, sparse embeddings are naturally (B, S, D). For now
        # each sample carries a single id (S=1), so we reshape directly back
        # to 2D for the jit forward. Switch to sum/mean pooling here when
        # sequence features are introduced.
        f1_rec_3d = _gather_sparse_input(
            emb1.flat_values, inv1.flat_values, local_batch, self.embed_dim, "f1"
        )
        f2_rec_3d = _gather_sparse_input(
            emb2.flat_values, inv2.flat_values, local_batch, self.embed_dim, "f2"
        )
        f1_rec = f1_rec_3d.reshape(local_batch, self.embed_dim)
        f2_rec = f2_rec_3d.reshape(local_batch, self.embed_dim)
        dense_stats = xp.asarray(dp_data.dense_stats.flat_values).reshape(  # type: ignore
            -1, dp_data.dense_stats.flat_values.shape[-1]
        )
        labels_local = (
            xp.asarray(dp_data.labels.is_click.flat_values)  # type: ignore
            .reshape(-1)
            .astype(xp.float32)  # type: ignore
        )

        # -- Invariants on DP-aligned features --
        assert dense_stats.shape[1] == self.dense_stats_dim, (
            f"[DenseSPMD] dense_stats last dim {dense_stats.shape[1]} "
            f"!= configured dense_stats_dim {self.dense_stats_dim}"
        )
        assert dense_stats.shape[0] == local_batch, (
            f"[DenseSPMD] dense_stats batch {dense_stats.shape[0]} "
            f"!= local_batch {local_batch}"
        )
        assert labels_local.shape[0] == local_batch, (
            f"[DenseSPMD] labels batch {labels_local.shape[0]} "
            f"!= local_batch {local_batch}"
        )

        # -- Numerical correctness check: gather restoration --
        # f1_rec[i] == emb1_2d[inv1_1d[i]], i.e. reconstructed per-sample
        # embedding aligns with the inverse mapping.
        assert f1_rec.shape[0] == inv1_1d.shape[0], (
            f"[DenseSPMD] f1_rec batch size {f1_rec.shape[0]} != inv1 size {inv1_1d.shape[0]}"
        )
        assert f2_rec.shape[0] == inv2_1d.shape[0], (
            f"[DenseSPMD] f2_rec batch size {f2_rec.shape[0]} != inv2 size {inv2_1d.shape[0]}"
        )

        dp_sharding = NamedSharding(self.mesh, P("dp"))
        replicated = NamedSharding(self.mesh, P())

        # -- JIT entry bridge: CuPy -> JAX (DLPack zero-copy) --
        f1_jax = jnp.asarray(f1_rec)
        f2_jax = jnp.asarray(f2_rec)
        stats_jax = jnp.asarray(dense_stats)
        labels_jax = jnp.asarray(labels_local)

        # All world_size JAX workers form one distributed session (via
        # jax.distributed.initialize in __init__), so the global shape is
        # world_size * local_batch regardless of CPU or GPU backend.
        jax_world_size = self.world_size

        global_f1 = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch, f1_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[f1_jax],
        )
        global_f2 = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch, f2_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[f2_jax],
        )
        global_stats = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch, stats_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[stats_jax],
        )
        global_labels = jax.make_array_from_single_device_arrays(
            shape=(jax_world_size * local_batch,),
            sharding=dp_sharding,
            arrays=[labels_jax],
        )

        # -- Inside JIT: JAX training core --
        @nnx.jit(
            in_shardings=(None, dp_sharding, dp_sharding, dp_sharding, dp_sharding),
            out_shardings=(None, dp_sharding, dp_sharding, replicated),
        )
        def forward_and_grad(model, f1, f2, stats, labels):
            x = jnp.concatenate([f1, f2, stats], axis=1)

            def loss_fn(m, input_x, y):
                pred = m(input_x)
                return jnp.mean((pred - y) ** 2)

            # dense_param_grads: replicated (same on all ranks after jit+allreduce)
            # x_grad: dp-sharded (each rank gets its local slice)
            dense_param_grads, x_grad = nnx.grad(loss_fn, argnums=(0, 1))(
                model, x, labels
            )
            loss_val = loss_fn(model, x, labels)
            emb1_dim = f1.shape[1]
            emb2_dim = f2.shape[1]
            return (
                dense_param_grads,
                x_grad[:, :emb1_dim],
                x_grad[:, emb1_dim : emb1_dim + emb2_dim],
                loss_val,
            )

        dense_param_grads, grad_f1_global, grad_f2_global, loss_val = forward_and_grad(
            self.model, global_f1, global_f2, global_stats, global_labels
        )

        # -- Post JIT: JAX -> xp --
        grad_f1_local = xp.asarray(grad_f1_global.addressable_shards[0].data)  # type: ignore
        grad_f2_local = xp.asarray(grad_f2_global.addressable_shards[0].data)  # type: ignore

        # scatter-add: (local_batch, emb_dim) → (K_u, emb_dim)
        g1_unique = _scatter_add_grad(emb1_2d, inv1_1d, grad_f1_local)
        g2_unique = _scatter_add_grad(emb2_2d, inv2_1d, grad_f2_local)

        # -- Numerical correctness check: scatter-add gradients --
        # The number of non-zero rows after scatter-add must not exceed the
        # number of unique keys.
        if _GPU_AVAILABLE:
            g1_nonzero_rows = int(cp.count_nonzero(cp.any(g1_unique != 0, axis=1)))  # type: ignore
            g2_nonzero_rows = int(cp.count_nonzero(cp.any(g2_unique != 0, axis=1)))  # type: ignore
        else:
            g1_nonzero_rows = int(np.count_nonzero(np.any(g1_unique != 0, axis=1)))
            g2_nonzero_rows = int(np.count_nonzero(np.any(g2_unique != 0, axis=1)))
        assert g1_nonzero_rows <= emb1_2d.shape[0], (
            f"[DenseSPMD] grad1 scatter-add non-zero rows {g1_nonzero_rows} "
            f"exceeds unique keys count {emb1_2d.shape[0]}"
        )
        assert g2_nonzero_rows <= emb2_2d.shape[0], (
            f"[DenseSPMD] grad2 scatter-add non-zero rows {g2_nonzero_rows} "
            f"exceeds unique keys count {emb2_2d.shape[0]}"
        )

        dense_grad_leaves = jtu.tree_leaves(dense_param_grads)
        dense_grad_flat = xp.concatenate(  # type: ignore
            [xp.asarray(g).ravel() for g in dense_grad_leaves]  # type: ignore
        )

        print(
            f"[DenseSPMD.step] rank={self.rank} "
            f"f1_rec={f1_rec.shape} f2_rec={f2_rec.shape} "
            f"loss={float(loss_val):.4f} "
            f"dense_grad_norms={[float(xp.linalg.norm(xp.asarray(g))) for g in dense_grad_leaves]}"  # type: ignore
        )

        return {
            "grad1": g1_unique,
            "grad2": g2_unique,
            "dense_grad": TransportTensor(
                dense_grad_flat, global_shape=(int(dense_grad_flat.shape[0]),)
            ),
        }

    @operator(
        name="update", input_sharding={"dense_grad": {"type": "static", "axis": []}}
    )
    def update(self, dense_grad: TransportTensor) -> None:
        step = self.ctx.step
        treedef = jtu.tree_structure(self.grad_accum)
        param_leaves = jtu.tree_leaves(nnx.state(self.model))
        flat = jnp.asarray(dense_grad.flat_values)

        # -- Invariants on dense gradient flat layout --
        expected_size = sum(int(leaf.size) for leaf in param_leaves)
        assert int(flat.size) == expected_size, (
            f"[DenseSPMD.update] dense_grad flat size {int(flat.size)} "
            f"!= sum of param leaf sizes {expected_size}"
        )
        assert flat.ndim == 1, (
            f"[DenseSPMD.update] dense_grad must be 1D after ravel, got shape={flat.shape}"
        )

        cursor = 0
        split_leaves = []
        for leaf in param_leaves:
            split_leaves.append(flat[cursor : cursor + leaf.size].reshape(leaf.shape))
            cursor += leaf.size

        self.grad_accum = jtu.tree_map(
            lambda acc, g: acc + g,
            self.grad_accum,
            jtu.tree_unflatten(treedef, split_leaves),
        )

        if step % self.grad_accum_steps == 0:
            params = nnx.state(self.model)
            updated = jtu.tree_map(lambda p, g: p - 0.01 * g, params, self.grad_accum)
            nnx.update(self.model, updated)
            self.grad_accum = jtu.tree_map(jnp.zeros_like, self.grad_accum)
            print(
                f"[DenseSPMD.update] rank={self.rank} step={step} updated model params"
            )
        else:
            print(f"[DenseSPMD.update] rank={self.rank} step={step} accumulating grads")
