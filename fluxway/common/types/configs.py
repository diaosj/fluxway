"""
Core data type definitions for Fluxway framework.

Based on ARCHITECTURE.md sections 2.1.1, 3.2.1, 4.2, 4.3
"""

from typing import Optional, Any, Annotated, TypeVar
from collections.abc import Callable

from pydantic import BaseModel, Field, ConfigDict


from enum import StrEnum

T = TypeVar("T")


# Transport stream markers for fields within a @data_type
class DataFlow:
    """Marker for Worker-to-Worker (Data Plane) transport. Supports sharding."""

    pass


class ControlFlow:
    """Marker for Driver-to-Driver (Control Plane) transport. Broadcast to all workers."""

    pass


# User-facing type aliases
# Usage: field_name: Data[jnp.ndarray] or field_name: Control[float]
Data = Annotated[T, DataFlow]
Control = Annotated[T, ControlFlow]


class FluxwayBaseSchema(BaseModel):
    """Base model for all Fluxway configuration and metadata types."""

    def as_dict(self, **kwargs) -> dict[str, Any]:
        """
        Convert model to dictionary.
        Renamed from Pydantic's model_dump for better domain alignment.
        """
        return self.model_dump(**kwargs)


class TensorProxy(FluxwayBaseSchema):
    """
    A lightweight proxy representing a tensor in the computational graph.

    It uses a globally unique identifier (int TUUID) to track dependencies and
    sharding specifications independently of the variable names used in
    operator functions.
    """

    tuuid: int
    name: str | None = None  # Original name for debugging
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    def __repr__(self):
        return f"TensorProxy(tuuid={self.tuuid}, name={self.name})"


class NodeType(StrEnum):
    """Node type identifier."""

    SPMD = "spmd"  # Normal SPMD node


class ReaderState(StrEnum):
    """Reader state for data reading nodes."""

    INITIALIZING = "initializing"
    READING = "reading"
    PAUSED = "paused"  # Paused due to backpressure
    COMPLETED = "completed"
    ERROR = "error"


class MailboxStatus(FluxwayBaseSchema):
    """Mailbox status for backpressure monitoring."""

    is_full: bool = False
    capacity: int = 100
    current_size: int = 0
    full_ratio: float = 0.8  # Ratio threshold to consider mailbox as full


class SPMDTopologyInfo(FluxwayBaseSchema):
    """Information about a predecessor/successor SPMD in the graph."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    spmd_id: str
    # Physical scale of the remote node
    world_size: int = 1
    mesh: Any = None
    # TUUID-based sharding mapping: {tuuid: ShardingSpec}
    tuuid_sharding: dict[int, Any] = Field(default_factory=dict)
    # Map of local parameter/argument names to their assigned UUIDs.
    # {param_name: tuuid}
    tuuid_map: dict[str, int] = Field(default_factory=dict)
    # Optional condition function for conditional edges
    condition_fn: Callable[..., bool] | None = None
    # Link to the symmetric edge (src_node, src_op, dst_node, dst_op)
    symmetric_to: tuple[str, str, str, str] | None = None
    # Worker-local transport edge; valid only for same SPMD rank self edges.
    local_only: bool = False
    # Whether this remote node is an interleaved source
    is_interleaved: bool = False
    # Size of the interleaved group (if applicable)
    interleave_unit_size: int = 1


class SPMDTopology(FluxwayBaseSchema):
    """Topology view for a single operator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Predecessors of this operator: {spmd_id: info}
    predecessors: dict[str, SPMDTopologyInfo] = Field(default_factory=dict)
    # Successors of this operator: {spmd_id: info}
    successors: dict[str, SPMDTopologyInfo] = Field(default_factory=dict)


class SPMDConfig(FluxwayBaseSchema):
    """
    Configuration for a single SPMD (Single Program, Multiple Data) execution unit.

    This defines how the SPMD operates, including its mesh, operators, and execution
    parameters.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Mesh axis configuration (e.g., ("data", 4), ("model", 2))
    spmd_id: str = ""
    mesh: Any = None  # DeviceMesh (use Any to avoid circular import)

    # Maps operator name to its method name on the owner class.
    # OperatorManager resolves callables from operator_metas once at worker
    # initialization and caches them — zero per-call overhead.
    operators: dict[str, str] = Field(default_factory=dict)
    # Explicit OperatorMeta storage - ensures metadata survives Ray serialization
    operator_metas: dict[str, "OperatorMeta"] = Field(default_factory=dict)

    # Topology of the graph for each operator
    # {op_name: SPMDTopology}
    topology: dict[str, "SPMDTopology"] = Field(default_factory=dict)
    active_operator_names: tuple[str, ...] = Field(default_factory=tuple)

    # Static arguments for operators: {arg_name: arg_value}
    static_args: dict[str, Any] = Field(default_factory=dict)

    max_worker_step_skew: int = (
        0  # Max allowed step diff between workers in a mesh (intra-SPMD sync)
    )
    # Global step limit for the SPMD. If set, source operators will self-terminate
    max_steps: int | None = None

    # If true, worker will block identically at step=0 until a drain target is provided.
    requires_drain_at_startup: bool = False

    # If true, workers enqueue source operators immediately after topology startup.
    auto_start_source_operators: bool = True

    def require_active_operator_names(self) -> tuple[str, ...]:
        if not self.active_operator_names:
            raise RuntimeError(
                "SPMDConfig.active_operator_names is empty. Build the SPMD "
                "through the Fluxway graph builder so the active graph contract "
                "is available to the runtime."
            )
        unknown = set(self.active_operator_names) - set(self.operator_metas)
        if unknown:
            raise RuntimeError(
                "SPMDConfig.active_operator_names contains operators without "
                f"metadata: {sorted(unknown)}"
            )
        return self.active_operator_names

    # Per-node Ray runtime_env for Driver/Worker actors.
    # Passed through to .options(runtime_env=...) when creating actors.
    # Example: {"env_vars": {"JAX_PLATFORMS": "gpu"}}
    runtime_env: dict[str, Any] | None = None

    # Dedicated runtime_env for the SPMDDriver actor.
    # The driver is a control-plane actor and MUST NOT inherit GPU-related
    # env_vars (CUDA_VISIBLE_DEVICES / JAX_PLATFORMS=cuda,...) from the
    # worker side, otherwise pickle.loads of jax.Array on the driver will
    # eagerly initialize a CUDA backend in a process that holds no GPU.
    # When None, SPMDManager applies a safe default that explicitly hides
    # GPUs from the driver actor.
    driver_runtime_env: dict[str, Any] | None = None

    # If true, worker will fail immediately upon error instead of attempting recovery.
    fail_fast: bool = True

    # Transport backend for data plane RPC (e.g. "ray", "axon")
    transport_type: str = "ray"

    data_plane_batch_max_frames: int = 8
    data_plane_batch_flush_us: int = 100
    data_plane_max_in_flight: int = 50

    # Optional profiler runtime configuration. ``None`` is the zero-overhead path:
    # workers construct the original executors and data-plane handlers directly.
    profiler_config: Any | None = None

    # If True, when all received shards for a step are empty packets, the worker
    # will short-circuit: skip _assemble_input / user operator dispatch and
    # directly propagate empty packets downstream (an "empty fastpath").
    # If False (default), all steps — including all-empty ones — are dispatched
    # to the user-defined operator, letting user code decide how to handle
    # empty inputs (e.g. emit a 0-size output, or return None to propagate).
    enable_empty_fastpath: bool = False

    @property
    def num_workers(self) -> int:
        """Derive world size (total workers) from mesh size."""
        if self.mesh is None:
            return 1
        return int(self.mesh.size)


# =============================================================================
# Resource and Scheduling Types (Section 2.1.2)
# =============================================================================


class ResourceSpec(FluxwayBaseSchema):
    """Resource specification for SPMD scheduling."""

    num_gpus: int = 0
    num_cpus: float = 1.0
    memory_gb: float = 0.0
    count: int = 1
    # TODO: Add network bandwidth requirements
    # TODO: Add storage requirements


class PlacementStrategy(StrEnum):
    """Placement strategy for resource allocation."""

    PACK = "PACK"  # Pack bundles on as few nodes as possible
    SPREAD = "SPREAD"  # Spread bundles across nodes
    STRICT_PACK = "STRICT_PACK"  # Strict packing requirement
    STRICT_SPREAD = "STRICT_SPREAD"  # Strict spreading requirement


class ResourcePriority(StrEnum):
    """Resource priority for allocation and preemption decisions."""

    HIGH = "HIGH"  # High priority, less likely to be preempted
    MID = "MID"  # Medium priority (default)
    LOW = "LOW"  # Low priority, may be preempted first


class PlacementPlan(FluxwayBaseSchema):
    """Resource placement plan generated by scheduler."""

    # Mapping of SPMD ID to resource requirements
    spmd_placements: dict[str, "ResourceSpec"] = Field(default_factory=dict)

    # Mapping of SPMD ID to resource priority
    spmd_priorities: dict[str, ResourcePriority] = Field(default_factory=dict)

    # Mapping of SPMD ID to per-node placement strategy
    # Overrides the global ``strategy`` when set for a specific SPMD.
    spmd_strategies: dict[str, PlacementStrategy] = Field(default_factory=dict)

    # Shared placement groups for conditional branches
    # Key: shared_pg_id, Value: list of SPMD IDs sharing this PG
    shared_pgs: dict[str, list[str]] = Field(default_factory=dict)

    # Placement strategy (global fallback when spmd_strategies is not set)
    strategy: PlacementStrategy = PlacementStrategy.PACK


# =============================================================================
# Graph Builder Types (Based on graph_example.py)
# =============================================================================


class Edge(FluxwayBaseSchema):
    """
    Represents an edge in the task graph.

    Edges connect nodes and can specify source/target op types.
    This allows flexible connections like forward->backward for sink nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    source_node: str
    target_node: str
    source_op: str | None = None  # Source node's op name
    target_op: str | None = None  # Target node's op name
    condition_fn: Callable[..., bool] | None = None

    # Link to the symmetric edge (for automatic mirror-sharding based routing)
    # Format: (source_node, source_op, target_node, target_op)
    symmetric_to: tuple[str, str, str, str] | None = None
    # Skip global data-plane topology and deliver directly inside the worker.
    local_only: bool = False


class Node(FluxwayBaseSchema):
    """
    Represents a node in the task graph.

    Each node corresponds to an SPMD execution unit with resource requirements
    and connections to other nodes.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str = ""
    resource: ResourceSpec | None = None
    priority: ResourcePriority | None = None  # Resource priority for allocation
    placement_strategy: PlacementStrategy | None = None  # Per-node placement strategy
    spmd_config: SPMDConfig = Field(default_factory=SPMDConfig)
    node_type: NodeType = NodeType.SPMD  # Node type: SPMD


class ArgumentType(StrEnum):
    """Argument type for fluxway transfer data."""

    CONTROL = "control"  # Control argument
    DATA = "data"  # Data argument
    STATIC = "static"  # Static argument


class EdgeShardingInfo(FluxwayBaseSchema):
    """Sharding information for a single edge between SPMDs."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    world_size: int = 1
    mesh: Any = None
    # New TUUID-based sharding mapping: {tuuid: ShardingSpec}
    tuuid_sharding: dict[int, Any] = Field(default_factory=dict)
    # Map of local parameter/argument names to their assigned UUIDs.
    tuuid_map: dict[str, int] = Field(default_factory=dict)
    # Condition function for conditional edges
    condition_fn: Callable[..., bool] | None = None
    # output_param_map: source op's output mapping (sender side)
    # input_param_map: target op's input mapping (receiver side)
    output_param_map: dict[str, list[str]] = Field(default_factory=dict)
    input_param_map: dict[str, list[str]] = Field(default_factory=dict)
    is_interleaved: bool = False
    interleave_unit_size: int = 1
    local_only: bool = False


class BarrierType(StrEnum):
    """Execution barrier types for operators."""

    STANDARD = "standard"  # Wait for all expected shards/states (default)
    PERIODIC = "periodic"  # Fire every 'n' arrivals; used for streaming


class BarrierConfig(FluxwayBaseSchema):
    """Base configuration for operator execution barriers."""

    type: BarrierType = BarrierType.STANDARD


class StandardBarrierConfig(BarrierConfig):
    """Standard barrier: fires when all expected data for a step arrives."""

    type: BarrierType = BarrierType.STANDARD


class PeriodicBarrierConfig(BarrierConfig):
    """
    Periodic barrier: fires every 'n' arrivals regardless of global step boundaries.
    Commonly used for streaming or 'eager-like' processing where n=1.
    """

    type: BarrierType = BarrierType.PERIODIC
    n: int = 1


class StatePolicy(FluxwayBaseSchema):
    """State concurrency policy consumed by the worker resource-lock table."""

    access: str = "exclusive"
    resource: str | None = None

    @classmethod
    def exclusive(cls, resource: str | None = None) -> "StatePolicy":
        return cls(access="exclusive", resource=resource)

    @classmethod
    def thread_safe(cls) -> "StatePolicy":
        return cls(access="thread_safe", resource=None)

    def model_post_init(self, __context: Any) -> None:
        if self.access not in ("exclusive", "thread_safe"):
            raise ValueError(
                f"StatePolicy.access must be 'exclusive' or 'thread_safe', got {self.access!r}."
            )
        if self.resource is not None and not isinstance(self.resource, str):
            raise ValueError(
                f"StatePolicy.resource must be a string or None, got {type(self.resource).__name__}."
            )
        if self.access == "thread_safe" and self.resource is not None:
            raise ValueError(
                "StatePolicy.thread_safe() cannot be combined with a serialized resource."
            )


class SchedulePolicy(FluxwayBaseSchema):
    """Cross-operator scheduling policy lowered into the worker priority queue."""

    model_config = ConfigDict(populate_by_name=True)

    kind: str = "fifo"
    base_priority: int = 0
    boost_priority: int = 100
    trigger: str = "staleness_limit"
    phase: str | None = None
    phase_ordering: tuple[str, ...] = Field(default=(), alias="phase_order")
    phase_dependencies: tuple[tuple[str, str, int], ...] = ()
    lane: str | None = None
    lane_ordering: tuple[str, ...] = Field(default=(), alias="lane_order")
    lane_priorities: dict[str, int] = Field(default_factory=dict)
    phase_priorities: dict[str, int] = Field(default_factory=dict)
    micro_step_key: str = "step"

    @classmethod
    def fifo(cls, base_priority: int = 0) -> "SchedulePolicy":
        return cls(kind="fifo", base_priority=base_priority)

    @classmethod
    def deadline_boost(
        cls,
        base_priority: int = 0,
        boost_priority: int = 100,
        trigger: str = "staleness_limit",
    ) -> "SchedulePolicy":
        return cls(
            kind="deadline_boost",
            base_priority=base_priority,
            boost_priority=boost_priority,
            trigger=trigger,
        )

    @classmethod
    def phase_order(
        cls,
        *,
        phase: str,
        phase_order: tuple[str, ...],
        micro_step_key: str = "step",
        base_priority: int = 0,
    ) -> "SchedulePolicy":
        return cls(
            kind="phase_order",
            phase=phase,
            phase_ordering=phase_order,
            micro_step_key=micro_step_key,
            base_priority=base_priority,
        )

    @classmethod
    def phase_graph(
        cls,
        *,
        phase: str,
        phase_order: tuple[str, ...],
        dependencies: tuple[tuple[str, str, int], ...],
        lane: str | None = None,
        lane_order: tuple[str, ...] = (),
        lane_priorities: dict[str, int] | None = None,
        phase_priorities: dict[str, int] | None = None,
        micro_step_key: str = "step",
        base_priority: int = 0,
    ) -> "SchedulePolicy":
        return cls(
            kind="phase_graph",
            phase=phase,
            phase_ordering=phase_order,
            phase_dependencies=dependencies,
            lane=lane,
            lane_ordering=lane_order,
            lane_priorities={} if lane_priorities is None else lane_priorities,
            phase_priorities={} if phase_priorities is None else phase_priorities,
            micro_step_key=micro_step_key,
            base_priority=base_priority,
        )

    def model_post_init(self, __context: Any) -> None:
        if self.kind not in ("fifo", "deadline_boost", "phase_order", "phase_graph"):
            raise ValueError(f"Unsupported SchedulePolicy kind {self.kind!r}.")
        if self.trigger != "staleness_limit":
            raise ValueError(
                f"SchedulePolicy.trigger must be 'staleness_limit', got {self.trigger!r}."
            )
        if self.micro_step_key != "step":
            raise ValueError(
                f"SchedulePolicy.micro_step_key must be 'step', got {self.micro_step_key!r}."
            )
        if self.kind in ("phase_order", "phase_graph"):
            if not self.phase:
                raise ValueError(f"SchedulePolicy.{self.kind} requires a phase.")
            if not self.phase_ordering:
                raise ValueError(f"SchedulePolicy.{self.kind} requires phase_order.")
            if self.phase not in self.phase_ordering:
                raise ValueError(
                    f"phase {self.phase!r} must be present in phase_order={self.phase_ordering!r}."
                )
        if self.kind == "phase_graph":
            self._validate_phase_graph()
        self._validate_lane_schedule()

    def _validate_phase_graph(self) -> None:
        phase_order = tuple(self.phase_ordering)
        if not self.phase_dependencies:
            raise ValueError("SchedulePolicy.phase_graph requires dependencies.")
        seen: set[tuple[str, str, int]] = set()
        same_step_edges: dict[str, set[str]] = {phase: set() for phase in phase_order}
        for dependency in self.phase_dependencies:
            if len(dependency) != 3:
                raise ValueError(
                    "SchedulePolicy.phase_graph dependencies must be "
                    "(before_phase, after_phase, step_delta) triples."
                )
            before, after, step_delta = dependency
            if not before or not after:
                raise ValueError(
                    "SchedulePolicy.phase_graph dependency phases must be non-empty."
                )
            if before not in phase_order:
                raise ValueError(
                    f"phase_graph dependency before phase {before!r} is not in "
                    f"phase_order={phase_order!r}."
                )
            if after not in phase_order:
                raise ValueError(
                    f"phase_graph dependency after phase {after!r} is not in "
                    f"phase_order={phase_order!r}."
                )
            if not isinstance(step_delta, int):
                raise ValueError(
                    "SchedulePolicy.phase_graph dependency step_delta must be an int."
                )
            if step_delta < 0:
                raise ValueError(
                    "SchedulePolicy.phase_graph dependency step_delta must be non-negative."
                )
            if before == after and step_delta == 0:
                raise ValueError(
                    "SchedulePolicy.phase_graph same-step self-dependencies are invalid."
                )
            key = (before, after, step_delta)
            if key in seen:
                raise ValueError(
                    f"SchedulePolicy.phase_graph duplicate dependency {key!r}."
                )
            seen.add(key)
            if step_delta == 0:
                same_step_edges[before].add(after)
        self._validate_phase_graph_same_step_dag(same_step_edges)

    @staticmethod
    def _validate_phase_graph_same_step_dag(edges: dict[str, set[str]]) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(phase: str) -> None:
            if phase in visited:
                return
            if phase in visiting:
                raise ValueError(
                    "SchedulePolicy.phase_graph same-step dependencies must be acyclic."
                )
            visiting.add(phase)
            for successor in edges[phase]:
                visit(successor)
            visiting.remove(phase)
            visited.add(phase)

        for phase in edges:
            visit(phase)

    def _validate_lane_schedule(self) -> None:
        if self.lane is None:
            if self.lane_ordering:
                raise ValueError("SchedulePolicy.lane_order requires lane.")
            if self.lane_priorities:
                raise ValueError("SchedulePolicy.lane_priorities requires lane.")
            return
        if not self.lane:
            raise ValueError("SchedulePolicy.lane must be non-empty when provided.")
        if self.lane_ordering and self.lane not in self.lane_ordering:
            raise ValueError(
                f"lane {self.lane!r} must be present in lane_order={self.lane_ordering!r}."
            )
        unknown_lanes = set(self.lane_priorities).difference(self.lane_ordering)
        if unknown_lanes and self.lane_ordering:
            raise ValueError(
                f"lane_priorities references lanes outside lane_order: {sorted(unknown_lanes)}."
            )
        if self.phase_priorities:
            phase_order = set(self.phase_ordering)
            unknown_phases = set(self.phase_priorities).difference(phase_order)
            if unknown_phases:
                raise ValueError(
                    f"phase_priorities references phases outside phase_order: {sorted(unknown_phases)}."
                )


class AsyncPolicy(FluxwayBaseSchema):
    """
    User-facing operator execution policy.

    Factories keep beginner-facing choices compact while lowering into existing
    worker scheduler, LocalStepPlan, and state-lock fields.
    ``bounded_stale`` limits how far an operator may run ahead of dependent
    progress; it never substitutes tensors or state from another step.
    """

    mode: str = "sync"
    depth: int = 1
    commit: str = "ordered"
    state: StatePolicy = Field(default_factory=StatePolicy.exclusive)
    schedule: SchedulePolicy = Field(default_factory=SchedulePolicy.fifo)
    max_staleness: int | None = None
    phases: tuple[str, ...] = ()

    @classmethod
    def sync(cls) -> "AsyncPolicy":
        return cls(mode="sync")

    @classmethod
    def pipelined(
        cls,
        depth: int,
        commit: str = "ordered",
        schedule: SchedulePolicy | None = None,
        state: StatePolicy | None = None,
    ) -> "AsyncPolicy":
        return cls(
            mode="pipelined",
            depth=depth,
            commit=commit,
            schedule=schedule or SchedulePolicy.fifo(),
            state=state or StatePolicy.exclusive(),
        )

    @classmethod
    def bounded_stale(
        cls,
        depth: int,
        max_staleness: int,
        commit: str = "relaxed",
        schedule: SchedulePolicy | None = None,
        state: StatePolicy | None = None,
    ) -> "AsyncPolicy":
        return cls(
            mode="bounded_stale",
            depth=depth,
            commit=commit,
            max_staleness=max_staleness,
            schedule=schedule or SchedulePolicy.deadline_boost(),
            state=state or StatePolicy.thread_safe(),
        )

    @classmethod
    def pipeline_parallel(
        cls,
        depth: int,
        phases: tuple[str, ...],
        schedule: SchedulePolicy,
        commit: str = "ordered",
    ) -> "AsyncPolicy":
        return cls(
            mode="pipeline_parallel",
            depth=depth,
            commit=commit,
            phases=phases,
            schedule=schedule,
            state=StatePolicy.exclusive(),
        )

    def model_post_init(self, __context: Any) -> None:
        if self.mode not in ("sync", "pipelined", "bounded_stale", "pipeline_parallel"):
            raise ValueError(f"Unsupported AsyncPolicy mode {self.mode!r}.")
        if self.depth <= 0:
            raise ValueError(f"AsyncPolicy.depth must be positive, got {self.depth}.")
        if self.commit not in ("ordered", "relaxed"):
            raise ValueError(
                f"AsyncPolicy.commit must be 'ordered' or 'relaxed', got {self.commit!r}."
            )
        if self.mode == "sync" and self.depth != 1:
            raise ValueError("AsyncPolicy.sync() must use depth=1.")
        if self.mode == "bounded_stale":
            limit = self.max_staleness_limit()
            if limit is None:
                raise ValueError("AsyncPolicy.bounded_stale requires max_staleness.")
            if limit < 0:
                raise ValueError(
                    f"AsyncPolicy.max_staleness must be non-negative, got {limit}."
                )
            if self.depth > limit + 1:
                raise ValueError(
                    "AsyncPolicy.bounded_stale requires depth <= max_staleness + 1, "
                    f"got depth={self.depth} max_staleness={limit}."
                )
        elif self.max_staleness is not None:
            raise ValueError(
                f"AsyncPolicy mode {self.mode!r} cannot define max_staleness."
            )
        if self.mode == "pipeline_parallel":
            if not self.phases:
                raise ValueError("AsyncPolicy.pipeline_parallel requires phases.")
            if self.schedule.kind not in ("phase_order", "phase_graph"):
                raise ValueError(
                    "AsyncPolicy.pipeline_parallel requires a phase schedule."
                )

    def max_staleness_limit(self) -> int | None:
        return self.max_staleness


class OperatorContext(FluxwayBaseSchema):
    """Context provided to an operator during execution."""

    model_config = ConfigDict(frozen=True)

    physical_rank: int
    physical_world_size: int
    logical_rank: int
    logical_world_size: int
    step: int
    is_interleaved: bool = False


class OperatorMeta(FluxwayBaseSchema):
    """Metadata attached to an operator function."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    owner_class_qualname: str = ""
    owner_class_module: str = ""
    requires_worker_process: bool = False

    is_source: bool | None = None  # Explicitly mark as source (producer)
    # Whether this is an interleaved source.
    is_interleaved: bool = False
    # Size of the interleaved group (e.g., 1 for single-worker).
    interleave_unit_size: int = 1

    priority: int = 0  # Priority for execution order (lower value -> higher priority)
    max_inflight_steps: int = (
        0  # Max steps predecessors can lead (backpressure_threshold)
    )
    async_policy: AsyncPolicy = Field(default_factory=AsyncPolicy.sync)
    input_policy: str = "exact"
    local_dag_policy: str = "inline"
    state_policy: StatePolicy = Field(default_factory=StatePolicy.exclusive)
    schedule_policy: SchedulePolicy = Field(default_factory=SchedulePolicy.fifo)
    max_async_lag: int | None = None
    local_pipeline_depth: int = 1
    commit_policy: str = "ordered"
    state_access: str = "exclusive"
    state_resource: str | None = None
    # Keys use ArgumentType.value (i.e. "state" | "tensor" | "static") to avoid enum
    # serialization / circular import surprises across control-plane and worker.
    input_arguments: dict[str, list[str]] | None = None
    output_arguments: dict[str, list[str]] | None = None
    input_sharding: dict[Any, Any] = Field(default_factory=dict)
    output_sharding: dict[Any, Any] = Field(default_factory=dict)
    output_uuids: list[int] = Field(default_factory=list)

    # Implicit parameter-to-field mappings
    input_param_map: dict[str, list[str]] = Field(default_factory=dict)
    output_param_map: dict[str, list[str]] = Field(default_factory=dict)
    input_treedefs: dict[str, Any] = Field(default_factory=dict)
    output_treedef: Any | None = None

    # Linking metadata (populated by Builder)
    input_pre_sharding: dict[str, list[EdgeShardingInfo]] = Field(default_factory=dict)
    output_post_sharding: dict[str, list[EdgeShardingInfo]] = Field(
        default_factory=dict
    )

    # Execution barrier configuration
    barrier_config: StandardBarrierConfig | PeriodicBarrierConfig = Field(
        default_factory=StandardBarrierConfig
    )

    def resolve_sharding_topology(
        self, resolver_fn: Callable[[str], int | None]
    ) -> None:
        for sharding in [self.input_sharding, self.output_sharding]:
            for spec in sharding.values():
                # ShardingSpec handles both referential and algebraic logic
                if hasattr(spec, "routing") and hasattr(spec.routing, "resolve"):
                    spec.routing.resolve(resolver_fn)

    def get_arguments(self, is_output: bool, arg_type: "ArgumentType") -> list[str]:
        args_map = self.output_arguments if is_output else self.input_arguments

        if args_map and arg_type.value in args_map:
            return list(args_map[arg_type.value])

        if arg_type == ArgumentType.DATA:
            sharding = self.output_sharding if is_output else self.input_sharding
            return list(sharding.keys())

        return []

    # ---------------------------------------------------------------------
    # Compare / validate helpers (used by control-plane graph builder)
    # ---------------------------------------------------------------------

    def state_names(self, is_output: bool) -> list[str]:
        """Get state argument names from input/output arguments."""
        return self.get_arguments(is_output=is_output, arg_type=ArgumentType.CONTROL)

    def validate_and_update_state_output_match_dest_input(
        self,
        dst_meta: "OperatorMeta",
        src_spmd_class: type,
        src_method_name: str,
        ctx: str,
    ) -> None:
        """
            Validate state arguments between src output and dst input.

            Logic:
        - Compare src.output_arguments.state with dst.input_arguments.state (must match exactly)
        - If src.output_arguments.state is missing, infer from src return type fields,
          require that output fields contain all dst.input_arguments.state,
          then update src.output_arguments.state = dst.input_arguments.state
        """
        from fluxway.common.utils import infer_fluxway_output_fields

        dst_input_state = set(dst_meta.state_names(is_output=False))
        src_output_state_list = self.state_names(is_output=True)

        if src_output_state_list:
            src_output_state = set(src_output_state_list)
            if src_output_state != dst_input_state:
                raise ValueError(
                    f"State names mismatch in {ctx}: "
                    f"src_output_state={sorted(src_output_state)}, "
                    f"dst_input_state={sorted(dst_input_state)}"
                )
            return

        # src.output_state not specified: infer from return type fields
        output_fields = set(
            infer_fluxway_output_fields(src_spmd_class, src_method_name)
        )
        missing = dst_input_state - output_fields
        if missing:
            raise ValueError(
                f"State coverage mismatch in {ctx}: operator output type of "
                f"{src_spmd_class.__name__}.{src_method_name} is missing state fields "
                f"{sorted(missing)} required by destination input"
            )

        # Update src.output_arguments.state = dst.input_state
        if self.output_arguments is None:
            self.output_arguments = {}
        self.output_arguments[ArgumentType.CONTROL.value] = list(dst_input_state)


class Graph(FluxwayBaseSchema):
    """
    Complete task graph with nodes and edges.

    Built using the Builder class for programmatic construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    source_nodes: list[str] = Field(default_factory=list)  # Entry nodes (no upstream)
    sink_nodes: list[str] = Field(default_factory=list)  # Exit nodes (no downstream)


class GraphValidationResult(FluxwayBaseSchema):
    """
    Result of graph validation.

    Contains validation status and any errors/warnings found.
    """

    is_valid: bool = True
    is_acyclic: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    unreachable_nodes: list[str] = Field(default_factory=list)


# =============================================================================
# State Management Types (Section 2.1.4)
# =============================================================================


class GlobalState(FluxwayBaseSchema):
    """Global training state aggregated from all SPMDs."""

    current_step: int = 0
    # TODO: Add global metrics (loss, throughput, etc.)
    # TODO: Add checkpoint metadata
    # TODO: Add resource usage statistics


class SPMDState(FluxwayBaseSchema):
    """State of a single SPMD instance."""

    spmd_id: str = ""
    current_step: int = 0
    # TODO: Add SPMD-specific metrics
    # TODO: Add model parameters reference
    # TODO: Add optimizer state reference


class WorkerTelemetry(FluxwayBaseSchema):
    """Telemetry data from a single worker."""

    worker_id: str = ""
    # TODO: Add GPU utilization
    # TODO: Add memory usage
    # TODO: Add network I/O statistics
    # TODO: Add heartbeat timestamp


# =============================================================================
# Handle Types (Section 3.2.1, 6.1)
# =============================================================================


class LogicalHandle(FluxwayBaseSchema):
    """
    Logical handle providing a stable interface.

    Logical handles don't change when nodes are evicted and restored.
    """

    logical_id: str
    spmd_id: str
    worker_id: str


class ActualHandle(FluxwayBaseSchema):
    """
    Actual handle corresponding to a Ray Actor.

    May change when nodes are evicted and rebuilt.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    actor_handle: Any  # ray.actor.ActorHandle
    worker_id: str
    created_at: float = 0.0  # Creation timestamp


class NodeView(FluxwayBaseSchema):
    """
    Worker's view of the SPMD topology.

    Maintains logical handle mappings for predecessors and successors.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    current_spmd_id: str = ""
    current_worker_id: str = ""
    predecessor_logical_handles: dict[str, list[LogicalHandle]] = Field(
        default_factory=dict
    )
    successor_logical_handles: dict[str, list[LogicalHandle]] = Field(
        default_factory=dict
    )
    handle_mapping: dict[str, ActualHandle] = Field(default_factory=dict)

    def get_predecessor_handles(self, spmd_id: str) -> list[LogicalHandle]:
        """Get all logical handles for a predecessor SPMD."""
        # TODO: Implement handle lookup
        return self.predecessor_logical_handles.get(spmd_id, [])

    def get_successor_handles(self, spmd_id: str) -> list[LogicalHandle]:
        """Get all logical handles for a successor SPMD."""
        # TODO: Implement handle lookup
        return self.successor_logical_handles.get(spmd_id, [])

    def call_logical_handle(self, logical_id: str, method: str, *args, **kwargs):
        """
        Call a method through logical handle, auto-mapping to actual handle.

        TODO: Implement auto-mapping logic
        TODO: Handle node eviction and recovery
        TODO: Implement retry mechanism
        """
        pass

    def update_handle_mapping(self, logical_id: str, actual_handle: ActualHandle):
        """Update handle mapping when node changes."""
        # TODO: Implement mapping update
        pass

    def handle_node_eviction(self, spmd_id: str):
        """Handle node eviction and trigger recovery."""
        # TODO: Implement eviction handling
        pass


# =============================================================================
# Driver and Worker Status Types (for heartbeat and state management)
# =============================================================================


class DriverStatus(StrEnum):
    """Driver health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    ERROR = "error"
    TIMEOUT = "timeout"


class WorkerStatus(StrEnum):
    """Worker operational status."""

    INITIALIZING = "initializing"
    RUNNING = "running"
    BLOCKED = "blocked"
    ERROR = "error"


class ControllerState(StrEnum):
    """Controller state machine states.

    Lifecycle:
      INITIALIZING → PENDING → TOPOLOGY_REGISTERING → RUNNING
                                                       ↓
                                                    BLOCKED ↔ RECOVERING
                                                       ↓
                                                  COMPLETING / FAILING
                                                        ↓
                                                 COMPLETED / FAILED
    """

    INITIALIZING = "initializing"  # Allocating resources, starting SPMDs
    PENDING = "pending"  # Waiting for all drivers to init (workers_ready)
    TOPOLOGY_REGISTERING = "topology_registering"  # topology registration in progress
    RUNNING = "running"  # Normal execution
    BLOCKED = "blocked"  # Task blocked (handling exceptions)
    RECOVERING = "recovering"  # Recovering from errors
    COMPLETING = "completing"  # Final cleanup after successful execution
    FAILING = "failing"  # Final cleanup after unrecoverable failure
    COMPLETED = "completed"  # Task completed
    FAILED = "failed"  # Task failed


class ControllerEvent(StrEnum):
    """Events that trigger controller state transitions."""

    INIT_COMPLETE = "init_complete"  # Resources allocated, SPMDs started
    ALL_DRIVERS_READY = "all_drivers_ready"  # All drivers report workers_ready
    ALL_SPMDS_TOPOLOGY_READY = (
        "all_spmds_topology_ready"  # All workers running after topology
    )
    DRIVER_ERROR = "driver_error"  # Driver heartbeat timeout or error
    DRIVER_RECOVERED = "driver_recovered"  # Abnormal driver recovered
    ALL_DRIVERS_COMPLETED = "all_drivers_completed"  # All drivers report completion
    ALL_DRIVERS_DRAINED = "all_drivers_drained"  # All drivers reached DRAINED state
    RECOVERY_FAILED = "recovery_failed"  # Recovery failed
    CLEANUP_FINISHED = "cleanup_finished"  # Terminal cleanup completed


class DriverState(StrEnum):
    """SPMD Driver state machine states."""

    INITIALIZING = "initializing"  # Initializing workers
    READY = "ready"  # Ready, waiting for Controller command
    RUNNING = "running"  # Running
    BLOCKED = "blocked"  # Blocked
    DRAINING = "draining"  # Draining to target step
    DRAINED = "drained"  # Reached the drain target step
    SNAPSHOTTING = "snapshotting"  # Extracting operator states via save_state
    ERROR = "error"  # Error state
    RECOVERING = "recovering"  # Recovering
    COMPLETED = "completed"  # Completed


class DriverEvent(StrEnum):
    """Events that trigger driver state transitions."""

    INIT_COMPLETE = "init_complete"
    INIT_FAILED = "init_failed"
    START = "start"
    WORKER_ERROR = "worker_error"
    WORKER_RECOVERED = "worker_recovered"
    BLOCK_REQUESTED = "block_requested"
    UNBLOCK_REQUESTED = "unblock_requested"
    ALL_WORKERS_COMPLETED = "all_workers_completed"
    COMPLETE = "complete"
    DRAIN_REQUESTED = "drain_requested"
    DRAIN_COMPLETE = "drain_complete"
    SNAPSHOT_REQUESTED = "snapshot_requested"
    SNAPSHOT_COMPLETE = "snapshot_complete"
    SNAPSHOT_FAILED = "snapshot_failed"


class WorkerHeartbeat(FluxwayBaseSchema):
    """Worker heartbeat data."""

    rank: int
    timestamp: float
    completed_step_count: dict[str, int] = Field(
        default_factory=dict
    )  # op_type.name -> completed count
    status: WorkerStatus = WorkerStatus.RUNNING
    error_message: str | None = None
    telemetry: Optional["WorkerTelemetry"] = None
    mailbox_status: MailboxStatus | None = None  # Mailbox status for backpressure


class DriverHeartbeat(FluxwayBaseSchema):
    """Driver heartbeat data reported to Controller."""

    spmd_id: str
    timestamp: float
    status: DriverStatus = DriverStatus.HEALTHY
    state: DriverState = DriverState.INITIALIZING
    error_message: str | None = None
    completed_step_count: dict[str, int] = Field(
        default_factory=dict
    )  # op_type.name -> completed step count
    num_workers: int = 0
    healthy_num_workers: int = 0
    blocked_ops: list[str] = Field(default_factory=list)
    workers_ready: bool = False  # All workers have reported heartbeat
    workers_running: bool = False  # All workers are running main loop
    is_downstream_busy: bool = (
        False  # Whether downstream SPMDs are busy (for backpressure)
    )
    source_state: ReaderState | None = None  # Generic source state
    busy_worker_ratio: float = 0.0  # Ratio of workers with full mailbox


class ErrorPolicy(FluxwayBaseSchema):
    """Error handling policy for Controller-level error classification.

    Decision priority:
      1. recoverable_patterns match → recoverable (attempt restart)
      2. No match → fall back to SPMDConfig.fail_fast
    """

    recoverable_patterns: list[str] = Field(default_factory=list)


class HeartbeatConfig(FluxwayBaseSchema):
    """Heartbeat configuration."""

    heartbeat_interval: float = 5.0  # Heartbeat interval (seconds)
    heartbeat_timeout: float = 15.0  # Heartbeat timeout (seconds)
    max_missed_heartbeats: int = 3  # Maximum missed heartbeats before error
    blocked_timeout: float = 300.0


class CheckpointConfig(FluxwayBaseSchema):
    """Configuration for checkpoint manager."""

    backend: str  # Validated at runtime by WorkerCheckpointBackend.create()
    ckpt_dir: str
    interval_steps: int
    force_write: bool = False  # Force write even if directory already exists


class AsyncStepConfig(FluxwayBaseSchema):
    """Configuration for async step difference control."""

    max_steps: int | None = None  # Maximum steps to run (global limit)
    node_max_steps: dict[str, int] = Field(
        default_factory=dict
    )  # Per-node step overrides
    default_max_inflight_steps: int = (
        1  # Maximum generic step difference for flow control
    )
    min_lock_duration: float = (
        1.0  # Minimum lock duration (prevent frequent lock/unlock)
    )


class DriverResourceSpec(FluxwayBaseSchema):
    """Driver management resource configuration (system default)."""

    num_cpus: float = 1.0
    memory_gb: float = 1.0


class SPMDOpStep(FluxwayBaseSchema):
    """SPMD operation step tracking."""

    spmd_id: str
    completed_step_count: dict[str, int] = Field(
        default_factory=dict
    )  # op_name -> completed count


class OpLockState(FluxwayBaseSchema):
    """Operation lock state."""

    spmd_id: str
    op_name: str  # Operator name
    locked_at: float
    reason: str
