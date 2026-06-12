from fluxway.common.decorators import operator, spmd
from fluxway.runtime.cuda import get_cuda_debug_info, get_worker_cuda_device
from fluxway.runtime.torch_tensor import scalar_to_transport, transport_to_torch

from .data_types import TEMinimalBatch, TEMinimalOutput


@spmd(
    resource={"num_cpus": 2, "num_gpus": 1, "memory_gb": 8},
    mesh_axis_names=("data",),
)
class TEMinimalSPMD:
    def __init__(
        self,
        hidden_size: int,
        ffn_hidden_size: int,
        lr: float = 1e-4,
        precision: str = "bf16",
    ):
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.lr = lr
        self.precision = precision
        self._initialized = False

        if self.world_size != 1:
            raise RuntimeError("TE minimal MVP only supports world_size=1.")
        if self.precision != "bf16":
            raise RuntimeError("TE minimal MVP only supports precision='bf16'.")

    def _ensure_initialized(self):
        if self._initialized:
            return

        import torch

        from .model import TEMinimalMLP

        self.device = get_worker_cuda_device()
        print(
            f"[TEMinimalSPMD.init] "
            f"rank={self.rank} world_size={self.world_size} "
            f"device={self.device} cuda={get_cuda_debug_info()}",
            flush=True,
        )

        self.model = TEMinimalMLP(
            hidden_size=self.hidden_size,
            ffn_hidden_size=self.ffn_hidden_size,
            params_dtype=torch.bfloat16,
        ).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        self._initialized = True

    @operator(
        name="step",
        input_sharding={
            "x": {"type": "static", "axis": ["data"]},
            "target": {"type": "static", "axis": ["data"]},
        },
        output_sharding={
            "loss": {"type": "static", "axis": []},
        },
    )
    def step(self, batch: TEMinimalBatch) -> TEMinimalOutput:
        self._ensure_initialized()

        import torch

        from .model import compute_mse_loss

        x = transport_to_torch(batch.x, device=self.device, dtype=torch.bfloat16)
        target = transport_to_torch(
            batch.target,
            device=self.device,
            dtype=torch.bfloat16,
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.model(x)
            loss = compute_mse_loss(pred, target)

        loss.backward()
        self.optimizer.step()

        end.record()
        torch.cuda.synchronize()

        elapsed_ms = start.elapsed_time(end)
        loss_value = float(loss.detach().float().cpu())
        print(
            f"[TEMinimalSPMD.step] "
            f"step={self.ctx.step} "
            f"loss={loss_value:.6f} "
            f"elapsed_ms={elapsed_ms:.3f}",
            flush=True,
        )

        return TEMinimalOutput(
            loss=scalar_to_transport(loss_value, dtype=torch.float32)
        )

