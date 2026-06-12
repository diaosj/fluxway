from dataclasses import dataclass

from fluxway.common.decorators import data_type
from fluxway.common.types.tensor import TransportTensor


@data_type()
@dataclass
class TEMinimalBatch:
    x: TransportTensor
    target: TransportTensor


@data_type()
@dataclass
class TEMinimalOutput:
    loss: TransportTensor

