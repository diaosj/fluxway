from dataclasses import dataclass

from fluxway.common.decorators import data_type, operator, spmd, spmd_source
from fluxway.common.types import Data
from fluxway.placement import ResourceProfile


@data_type()
@dataclass
class InterleavedSourceValue:
    value: Data[int]


@spmd_source(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("worker",),
    mesh_axis_shapes=(2,),
    interleaved=True,
    interleave_unit_size=1,
)
class ImmediatelyExhaustedInterleavedSource:
    @operator(
        "read",
        output_sharding={"type": "static", "axis": ["worker"]},
    )
    def read(self) -> InterleavedSourceValue | None:
        return None


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("worker",),
    mesh_axis_shapes=(1,),
)
class InterleavedSourceSink:
    @operator(
        "consume",
        input_sharding={"batch": {"type": "static", "axis": ["worker"]}},
    )
    def consume(self, batch: InterleavedSourceValue) -> None:
        del batch
