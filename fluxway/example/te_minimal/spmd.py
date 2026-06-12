import datetime

from fluxway.common.decorators import operator, spmd
from fluxway.runtime.cuda import get_cuda_debug_info, get_worker_cuda_device
from fluxway.runtime.torch_tensor import scalar_to_transport, transport_to_torch

from .data_types import TEMinimalBatch, TEMinimalOutput


class _TEMinimalBase:
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        batch_size: int = 8,
        lr: float = 1e-4,
        precision: str = "bf16",
        dp_size: int = 1,
        tp_size: int = 1,
    ):
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.batch_size = batch_size
        self.lr = lr
        self.precision = precision
        self.dp_size = dp_size
        self.tp_size = tp_size
        self._initialized = False
        self._dist_initialized_here = False

        if self.precision != "bf16":
            raise RuntimeError("TE minimal example currently only supports precision='bf16'.")
        if self.dp_size < 1 or self.tp_size < 1:
            raise RuntimeError("dp_size and tp_size must be positive.")
        if self.world_size != self.dp_size * self.tp_size:
            raise RuntimeError(
                f"TE minimal dp/tp config mismatch: world_size={self.world_size}, "
                f"dp_size={self.dp_size}, tp_size={self.tp_size}."
            )
        if self.ffn_hidden_size % self.tp_size != 0:
            raise RuntimeError(
                f"ffn_hidden_size={self.ffn_hidden_size} must be divisible by tp_size={self.tp_size}."
            )

    def _init_torch_distributed(self, torch):
        import torch.distributed as dist

        if self.world_size <= 1:
            self.dp_rank = 0
            self.tp_rank = 0
            self.tp_group = None
            self.dp_group = None
            return

        if not self.rank0_address:
            raise RuntimeError("rank0_address is required for dp/tp distributed mode.")

        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{self.rank0_address}",
                rank=self.rank,
                world_size=self.world_size,
                timeout=datetime.timedelta(minutes=30),
                device_id=self.device,
            )
            self._dist_initialized_here = True

        self.dp_rank = self.rank // self.tp_size
        self.tp_rank = self.rank % self.tp_size

        self.tp_group = None
        for dp_idx in range(self.dp_size):
            ranks = [dp_idx * self.tp_size + tp_idx for tp_idx in range(self.tp_size)]
            group = dist.new_group(ranks=ranks, backend="nccl", device_id=self.device)
            if self.rank in ranks:
                self.tp_group = group

        self.dp_group = None
        for tp_idx in range(self.tp_size):
            ranks = [dp_idx * self.tp_size + tp_idx for dp_idx in range(self.dp_size)]
            group = dist.new_group(ranks=ranks, backend="nccl", device_id=self.device)
            if self.rank in ranks:
                self.dp_group = group

        dist.barrier()

    def _average_dp_gradients(self):
        if self.dp_size <= 1:
            return

        import torch.distributed as dist

        for param in self.model.parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=self.dp_group)
                param.grad.div_(self.dp_size)

    def _make_synthetic_batch(self, torch):
        seed = 1234 + self.ctx.step + self.dp_rank * 1000003
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        x = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        target = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        return x.contiguous(), target.contiguous()

    def _train_tensors(self, torch, x, target, *, op_name: str) -> TEMinimalOutput:
        from .model import compute_mse_loss

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.model(x)
            loss = compute_mse_loss(pred, target)

        loss.backward()
        self._average_dp_gradients()
        self.optimizer.step()

        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        loss_value = float(loss.detach().float().cpu())
        if self.rank == 0 or self.ctx.step < 2:
            print(
                f"[TEMinimalSPMD.{op_name}] "
                f"rank={self.rank} dp={getattr(self, 'dp_rank', 0)} tp={getattr(self, 'tp_rank', 0)} "
                f"step={self.ctx.step} "
                f"loss={loss_value:.6f} "
                f"elapsed_ms={elapsed_ms:.3f}",
                flush=True,
            )

        return TEMinimalOutput(
            loss=scalar_to_transport(
                loss_value, dtype=torch.float32, global_shape=(self.world_size,)
            )
        )

    def _ensure_initialized(self):
        if self._initialized:
            return

        import torch

        from .model import TEMinimalMLP

        self.device = get_worker_cuda_device()
        print(
            f"[TEMinimalSPMD.pre-init] "
            f"rank={self.rank} world_size={self.world_size} "
            f"dp_size={self.dp_size} tp_size={self.tp_size} "
            f"device={self.device} cuda={get_cuda_debug_info()}",
            flush=True,
        )
        self._init_torch_distributed(torch)
        print(
            f"[TEMinimalSPMD.init] "
            f"rank={self.rank} world_size={self.world_size} "
            f"dp_rank={getattr(self, 'dp_rank', 0)} tp_rank={getattr(self, 'tp_rank', 0)} "
            f"dp_size={self.dp_size} tp_size={self.tp_size} "
            f"device={self.device} cuda={get_cuda_debug_info()}",
            flush=True,
        )

        self.model = TEMinimalMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            params_dtype=torch.bfloat16,
            tp_group=self.tp_group,
            tp_size=self.tp_size,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self._initialized = True


@spmd(
    resource={"num_cpus": 2, "num_gpus": 1, "memory_gb": 8},
    mesh_axis_names=("dp", "tp"),
)
class TEMinimalSPMD(_TEMinimalBase):
    @operator(
        name="step",
        input_sharding={
            "x": {"type": "static", "axis": ["dp"]},
            "target": {"type": "static", "axis": ["dp"]},
        },
        output_sharding={
            "loss": {"type": "static", "axis": ["dp", "tp"]},
        },
    )
    def step(self, batch: TEMinimalBatch) -> TEMinimalOutput:
        self._ensure_initialized()

        import torch

        x = transport_to_torch(batch.x, device=self.device, dtype=torch.bfloat16).contiguous()
        target = transport_to_torch(
            batch.target,
            device=self.device,
            dtype=torch.bfloat16,
        ).contiguous()
        return self._train_tensors(torch, x, target, op_name="step")


@spmd(
    resource={"num_cpus": 2, "num_gpus": 1, "memory_gb": 8},
    mesh_axis_names=("dp", "tp"),
)
class TEMinimalParallelSPMD(_TEMinimalBase):
    @operator(
        name="step_synthetic",
        is_source=True,
        output_sharding={
            "loss": {"type": "static", "axis": ["dp", "tp"]},
        },
    )
    def step_synthetic(self) -> TEMinimalOutput:
        self._ensure_initialized()

        import torch

        x, target = self._make_synthetic_batch(torch)
        return self._train_tensors(torch, x, target, op_name="step_synthetic")
