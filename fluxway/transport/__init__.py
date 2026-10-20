"""Transport layer: Data transmission between SPMDs."""

# Ensure INFO logs are visible even when importing this subpackage directly.
from fluxway.common.utils.logging import ensure_logging_configured


from fluxway.common.types import TransportTensor
from fluxway.common.types.sharding import (
    ShardingSpec,
    PhysicalLayout,
    PartitionStrategy,
    ChunkPartition,
    CyclicPartition,
    FunctionPartition,
    GreedyPartition,
    # RoutingDomain abstraction (L2 contract)
    RoutingDomain,
    RoutingAxis,
    ReplicaSpec,
    # Strict-typed indices specification (L1 ↔ L2 contract)
    IndicesSpec,
    LiteralIndices,
    RangeIndices,
    IdentityIndices,
)
from .planner import TransportPlanner, TransportPlan
from .communication import AsyncDataDistributor
from .routing import TransportRoutingMode

__all__ = [
    "TransportTensor",
    "ShardingSpec",
    "PhysicalLayout",
    "PartitionStrategy",
    "ChunkPartition",
    "CyclicPartition",
    "FunctionPartition",
    "GreedyPartition",
    # RoutingDomain abstraction
    "RoutingDomain",
    "RoutingAxis",
    "ReplicaSpec",
    # IndicesSpec hierarchy
    "IndicesSpec",
    "LiteralIndices",
    "RangeIndices",
    "IdentityIndices",
    "TransportPlanner",
    "TransportPlan",
    "AsyncDataDistributor",
    "TransportRoutingMode",
]

ensure_logging_configured()
