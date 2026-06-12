import numpy as np

from fluxway.common.decorators import operator, spmd_source
from fluxway.common.types.tensor import TransportTensor

from .data_types import TEMinimalBatch


@spmd_source(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("data",),
)
class TEMinimalReader:
    def __init__(
        self,
        batch_size: int,
        hidden_size: int,
        seed: int = 1234,
    ):
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.seed = seed

    @operator(
        name="read",
        output_sharding={
            "x": {"type": "static", "axis": ["data"]},
            "target": {"type": "static", "axis": ["data"]},
        },
    )
    def read(self) -> TEMinimalBatch:
        rng = np.random.default_rng(self.seed + self.ctx.step)
        x = rng.standard_normal(
            size=(self.batch_size, self.hidden_size),
            dtype=np.float32,
        )
        target = rng.standard_normal(
            size=(self.batch_size, self.hidden_size),
            dtype=np.float32,
        )
        global_shape = (self.batch_size * self.world_size,)
        return TEMinimalBatch(
            x=TransportTensor.from_tensor(x, global_shape=global_shape),
            target=TransportTensor.from_tensor(target, global_shape=global_shape),
        )

