import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from .data_types import DPFeatures

# Select array backend for JIT-external operations based on GPU availability.
# When JAX uses GPU, CuPy provides zero-copy DLPack interop with JAX.
# When JAX falls back to CPU, NumPy is the natural choice.
_GPU_AVAILABLE = False
try:
    import cupy as cp

    # Verify CUDA is actually usable (cupy may be installed without a GPU)
    try:
        _cc = cp.cuda.Device(0).compute_capability
        _GPU_AVAILABLE = True
    except Exception:
        _GPU_AVAILABLE = False
except ImportError:
    _GPU_AVAILABLE = False


def _xp():
    """Return CuPy if GPU is available, otherwise NumPy."""
    return cp if _GPU_AVAILABLE else np


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp",),
)
class DenseSPMD:
    """Final computation node: forward pass, loss, and dense/embedding gradient computation."""

    def __init__(
        self,
        embed_dim: int = 3,
        dense_stats_dim: int = 2,
        grad_accum_steps: int = 2,
    ):
        self.embed_dim = embed_dim
        self.dense_stats_dim = dense_stats_dim
        self.grad_accum_steps = grad_accum_steps
        self._jax_initialized = False

        # JAX distributed coordination is only meaningful with accelerators.
        # On CPU-only hosts the coordination service aborts (SIGABRT), so we
        # skip initialization entirely and rely on the default CPU device.
        if _GPU_AVAILABLE:
            print(
                f"[DenseSPMD] init jax.distributed rank={self.rank} world_size={self.world_size} coordinator={self.rank0_address}",
                flush=True,
            )
            try:
                jax.distributed.initialize(
                    coordinator_address=self.rank0_address,
                    num_processes=self.world_size,
                    process_id=self.rank,
                )
                print(
                    f"[DenseSPMD] init jax.distributed success rank={self.rank}",
                    flush=True,
                )
            except Exception as e:
                print(
                    f"[DenseSPMD] init jax.distributed failed rank={self.rank} error={e}",
                    flush=True,
                )
                raise
        else:
            print(
                f"[DenseSPMD] CPU-only mode, skipping jax.distributed init rank={self.rank}",
                flush=True,
            )

    def _ensure_jax_init(self):
        if self._jax_initialized:
            return

        if _GPU_AVAILABLE:
            n_global = len(jax.devices())
            print(
                f"[DenseSPMD._ensure_jax_init] rank={self.rank} "
                f"global_devices={n_global} local_devices={len(jax.local_devices())}"
            )
            self.mesh = Mesh(jax.devices(), ("dp",))
        else:
            # CPU-only: single default device, no distributed mesh
            print(
                f"[DenseSPMD._ensure_jax_init] CPU-only rank={self.rank} devices=[cpu]"
            )
            self.mesh = Mesh(jax.local_devices(), ("dp",))

        input_dim = self.embed_dim * 2 + self.dense_stats_dim

        class DenseLayer(nnx.Module):
            def __init__(self, in_dim):
                self.linear = nnx.Linear(in_dim, 1, rngs=nnx.Rngs(0))

            def __call__(self, x):
                return self.linear(x).squeeze(-1)

        self.model = DenseLayer(input_dim)
        self.grad_accum = jtu.tree_map(
            lambda p: jnp.zeros_like(p), nnx.state(self.model)
        )
        self._jax_initialized = True

    def save_state(self) -> dict:
        self._ensure_jax_init()
        return {"model_state": nnx.state(self.model), "grad_accum": self.grad_accum}

    def restore_state(self, state: dict) -> None:
        self._ensure_jax_init()
        nnx.update(self.model, state["model_state"])
        self.grad_accum = state["grad_accum"]

    @operator(
        name="step",
        input_sharding={
            "emb1": {"type": "static", "axis": ["dp"]},
            "inv1": {"type": "static", "axis": ["dp"]},
            "emb2": {"type": "static", "axis": ["dp"]},
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
        self._ensure_jax_init()

        # -- JIT 外: GPU(CuPy) / CPU(NumPy) 操作 --
        xp = _xp()
        emb1_2d = xp.asarray(emb1.flat_values).reshape(-1, emb1.flat_values.shape[-1])
        emb2_2d = xp.asarray(emb2.flat_values).reshape(-1, emb2.flat_values.shape[-1])
        inv1_1d = xp.asarray(inv1.flat_values).reshape(-1)
        inv2_1d = xp.asarray(inv2.flat_values).reshape(-1)

        # -- 数值正确性验证: inv 索引范围 --
        assert inv1_1d.min() >= 0 and inv1_1d.max() < emb1_2d.shape[0], (
            f"[DenseSPMD] inv1 越界: max={inv1_1d.max()}, emb1_unique_count={emb1_2d.shape[0]}"
        )
        assert inv2_1d.min() >= 0 and inv2_1d.max() < emb2_2d.shape[0], (
            f"[DenseSPMD] inv2 越界: max={inv2_1d.max()}, emb2_unique_count={emb2_2d.shape[0]}"
        )

        f1_rec = emb1_2d[inv1_1d]
        f2_rec = emb2_2d[inv2_1d]
        dense_stats = xp.asarray(dp_data.dense_stats.flat_values).reshape(
            -1, dp_data.dense_stats.flat_values.shape[-1]
        )
        labels_local = (
            xp.asarray(dp_data.labels.is_click.flat_values)
            .reshape(-1)
            .astype(xp.float32)
        )

        # -- 数值正确性验证: inv 还原 --
        # f1_rec[i] == emb1_2d[inv1_1d[i]], 即 lookup 还原后每个样本的 embedding 正确
        assert f1_rec.shape[0] == inv1_1d.shape[0], (
            f"[DenseSPMD] f1_rec batch size {f1_rec.shape[0]} != inv1 size {inv1_1d.shape[0]}"
        )
        assert f2_rec.shape[0] == inv2_1d.shape[0], (
            f"[DenseSPMD] f2_rec batch size {f2_rec.shape[0]} != inv2 size {inv2_1d.shape[0]}"
        )

        dp_sharding = NamedSharding(self.mesh, P("dp"))
        replicated = NamedSharding(self.mesh, P())
        local_batch = f1_rec.shape[0]

        # -- JIT 入口桥接: CuPy → JAX (DLPack zero-copy) --
        f1_jax = jnp.asarray(f1_rec)
        f2_jax = jnp.asarray(f2_rec)
        stats_jax = jnp.asarray(dense_stats)
        labels_jax = jnp.asarray(labels_local)

        # Build global sharded arrays from per-rank local slices.
        # All 6 dp ranks collectively form the global batch of world_size * local_batch rows.
        global_f1 = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch, f1_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[f1_jax],
        )
        global_f2 = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch, f2_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[f2_jax],
        )
        global_stats = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch, stats_jax.shape[1]),
            sharding=dp_sharding,
            arrays=[stats_jax],
        )
        global_labels = jax.make_array_from_single_device_arrays(
            shape=(self.world_size * local_batch,),
            sharding=dp_sharding,
            arrays=[labels_jax],
        )

        # -- JIT 内: JAX 训练核心 --
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

        # -- JIT 输出后处理: JAX → xp --
        grad_f1_local = xp.asarray(grad_f1_global.addressable_shards[0].data)
        grad_f2_local = xp.asarray(grad_f2_global.addressable_shards[0].data)

        # scatter-add: (local_batch, emb_dim) → (K_u, emb_dim)
        g1_unique = xp.zeros_like(emb1_2d)
        if _GPU_AVAILABLE:
            cp.scatter_add(
                g1_unique,
                cp.broadcast_to(inv1_1d[:, None], grad_f1_local.shape),
                grad_f1_local,
            )
        else:
            np.add.at(g1_unique, inv1_1d, grad_f1_local)
        g2_unique = xp.zeros_like(emb2_2d)
        if _GPU_AVAILABLE:
            cp.scatter_add(
                g2_unique,
                cp.broadcast_to(inv2_1d[:, None], grad_f2_local.shape),
                grad_f2_local,
            )
        else:
            np.add.at(g2_unique, inv2_1d, grad_f2_local)

        # -- 数值正确性验证: scatter-add 梯度 --
        # scatter-add 后 g1_unique 的非零行数不应超过 unique_keys 数量
        if _GPU_AVAILABLE:
            g1_nonzero_rows = int(cp.count_nonzero(cp.any(g1_unique != 0, axis=1)))
            g2_nonzero_rows = int(cp.count_nonzero(cp.any(g2_unique != 0, axis=1)))
        else:
            g1_nonzero_rows = int(np.count_nonzero(np.any(g1_unique != 0, axis=1)))
            g2_nonzero_rows = int(np.count_nonzero(np.any(g2_unique != 0, axis=1)))
        assert g1_nonzero_rows <= emb1_2d.shape[0], (
            f"[DenseSPMD] grad1 scatter-add 非零行数 {g1_nonzero_rows} "
            f"超过 unique keys 数量 {emb1_2d.shape[0]}"
        )
        assert g2_nonzero_rows <= emb2_2d.shape[0], (
            f"[DenseSPMD] grad2 scatter-add 非零行数 {g2_nonzero_rows} "
            f"超过 unique keys 数量 {emb2_2d.shape[0]}"
        )

        dense_grad_leaves = jtu.tree_leaves(dense_param_grads)
        dense_grad_flat = xp.concatenate(
            [xp.asarray(g).ravel() for g in dense_grad_leaves]
        )

        print(
            f"[DenseSPMD.step] rank={self.rank} "
            f"f1_rec={f1_rec.shape} f2_rec={f2_rec.shape} "
            f"loss={float(loss_val):.4f} "
            f"dense_grad_norms={[float(xp.linalg.norm(xp.asarray(g))) for g in dense_grad_leaves]}"
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
