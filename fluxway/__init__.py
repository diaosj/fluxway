"""
Fluxway: A distributed deep learning framework for large-scale training.

Features:
- Dual-Plane Propagation Protocol
- Dual-Stream Convergence
- Asynchronous Execution
- Graph-based Orchestration

Usage:
    import fluxway

    @fluxway.data_type(control_argument=["id"], data_argument=["tensor"])
    class MyData:
        id: int
        tensor: torch.Tensor

    # Use tuple-based mesh with user-defined axis names
    @fluxway.spmd(
        resource={"nums_cpu": 4, "nums_gpu": 2},
        mesh_axis_names=("data", "model"),
        mesh_axis_shapes=(4, 2),
        max_async_steps=10
    )
    class MySPMD:
        def forward(self, data):
            return process(data)
"""

__version__ = "0.1.0"

# Force logging to be visible across driver + Ray worker processes.
# Users of this repo want INFO logs by default.
from fluxway.common.utils.logging import ensure_logging_configured

# Import decorators for top-level access
from fluxway.common.decorators import data_type, spmd, operator, requires_worker_process
from fluxway.common.bind import bind_spmd

# Import common types
from fluxway.common.types import (
    SPMDConfig,
    AsyncPolicy,
    StatePolicy,
    SchedulePolicy,
    Graph,
    Node,
    Edge,
    ResourceSpec,
    # Sharding types
    ShardingSpec,
    PhysicalLayout,
    PartitionSpec,
    ChunkPartition,
    CyclicPartition,
    IndexedPartition,
    GreedyPartition,
)

# Import control plane components
from fluxway.control_plane import GraphParser, Builder, create_controller

# Import runner
from fluxway.runner import launch
from fluxway import profiler
from fluxway.visualization import graph_manifest, view_graph, visualize_graph
from fluxway.placement import (
    AFFINITY_FIRST_V1,
    PlacementRecipe,
    PlacementAttestationError,
    PlacementContractError,
    PlacementError,
    PlacementGroupTimeoutError,
    PlacementInfeasibleError,
    PlacementMaterializationError,
    PlacementRuntimeConfig,
    PlacementSolveTimeoutError,
    ResourceProfile,
    ResourceScale,
    allocated_quanta,
    affinity_constraints,
    affinity_coverage,
    all_of,
    all_workers,
    any_of,
    cardinality,
    co_location_mass,
    difference,
    disjoint,
    equal,
    instance,
    incident_spmds,
    intersection,
    isolated_spmds,
    less_equal,
    lexicographic,
    max_min,
    maximize,
    maximize_majorization,
    minimize,
    negate,
    occupancy,
    rank,
    ranks,
    resource_capacity,
    resource_fragmentation,
    resource_usage,
    subseteq,
    support,
    target_occupancy,
    union,
    weighted_dominant_share,
    workers,
)

__all__ = [
    # Version
    "__version__",
    # Runner
    "launch",
    # Decorators
    "data_type",
    "spmd",
    "operator",
    "profiler",
    "requires_worker_process",
    "bind_spmd",
    # Types
    "PartitionSpec",
    "SPMDConfig",
    "AsyncPolicy",
    "StatePolicy",
    "SchedulePolicy",
    "Graph",
    "Node",
    "Edge",
    "ResourceSpec",
    # Sharding
    "ShardingSpec",
    "PhysicalLayout",
    "PartitionSpec",
    "ChunkPartition",
    "CyclicPartition",
    "IndexedPartition",
    "GreedyPartition",
    # Control Plane
    "GraphParser",
    "Builder",
    "create_controller",
    "graph_manifest",
    "visualize_graph",
    "view_graph",
    "AFFINITY_FIRST_V1",
    "PlacementRecipe",
    "PlacementAttestationError",
    "PlacementContractError",
    "PlacementError",
    "PlacementGroupTimeoutError",
    "PlacementInfeasibleError",
    "PlacementMaterializationError",
    "PlacementRuntimeConfig",
    "PlacementSolveTimeoutError",
    "ResourceProfile",
    "ResourceScale",
    "allocated_quanta",
    "affinity_constraints",
    "affinity_coverage",
    "all_of",
    "all_workers",
    "any_of",
    "cardinality",
    "co_location_mass",
    "difference",
    "disjoint",
    "equal",
    "instance",
    "incident_spmds",
    "intersection",
    "isolated_spmds",
    "less_equal",
    "lexicographic",
    "max_min",
    "maximize",
    "maximize_majorization",
    "minimize",
    "negate",
    "occupancy",
    "rank",
    "ranks",
    "resource_capacity",
    "resource_fragmentation",
    "resource_usage",
    "subseteq",
    "support",
    "target_occupancy",
    "union",
    "weighted_dominant_share",
    "workers",
]

ensure_logging_configured()
