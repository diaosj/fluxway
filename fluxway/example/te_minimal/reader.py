import numpy as np

from fluxway.common.decorators import operator, spmd_source
from fluxway.common.types.tensor import TransportTensor

from .data_types import TEMinimalBatch


@spmd_source(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp", "tp"),
)
class TEMinimalReader:
    def __init__(
        self,
        batch_size: int,
        hidden_size: int,
        seed: int = 1234,
        dp_size: int = 1,
        tp_size: int = 1,
    ):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seed = seed
        self.dp_size = dp_size
        self.tp_size = tp_size

    @operator(
        name="read",
        output_sharding={
            "x": {"type": "static", "axis": ["dp"]},
            "target": {"type": "static", "axis": ["dp"]},
        },
    )
    def read(self) -> TEMinimalBatch:
        dp_rank = self.rank // self.tp_size
        rng = np.random.default_rng(self.seed + self.ctx.step + dp_rank * 1000003)
        x = rng.standard_normal(
            size=(self.batch_size, self.hidden_size),
            dtype=np.float32,
        )
        target = rng.standard_normal(
            size=(self.batch_size, self.hidden_size),
            dtype=np.float32,
        )
        global_shape = (self.batch_size * self.dp_size, self.hidden_size)
        return TEMinimalBatch(
            x=TransportTensor.from_tensor(x, global_shape=global_shape),
            target=TransportTensor.from_tensor(target, global_shape=global_shape),
        )
