"""Compile canonical source steps before lowering to physical reader quotas."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from enum import StrEnum
from typing import Any

from fluxway.common.ingestion.protocol import (
    EncodedWorkerLayoutDescriptor,
    IngestionLayoutEpoch,
    SourceLayoutActivation,
    WorkerLayoutDescriptor,
)
from fluxway.common.ingestion.segment import (
    CanonicalWorkInterval,
    CompiledSegment,
    LayoutUnitDescriptor,
    SYNTHETIC_PAD_IDENTITY,
    SourceWorkKind,
    SourceWorkAssignment,
    _count,
    _id,
    validate_canonical_work_intervals,
)
from fluxway.common.metadata import metadata_digest
from fluxway.control_plane.logical_work_ledger import LogicalWorkEpoch


class TailLayoutMode(StrEnum):
    """Generic source-layout treatment for an incomplete cardinality."""

    DISCARD_REMAINDER = "discard_remainder"
    RETAIN_PENDING = "retain_pending"
    REJECT_REMAINDER = "reject_remainder"
    SYNTHESIZE_PADDING = "synthesize_padding"

    @classmethod
    def from_value(cls, value: str | TailLayoutMode) -> TailLayoutMode:
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            for member in cls:
                if value == member.value or value.upper() == member.name:
                    return member
        raise ValueError(f"unsupported tail layout mode: {value!r}")


@dataclass(frozen=True, slots=True)
class _LayoutCost:
    """Deterministic cost summary for one immutable physical layout."""

    distinct_entity_count: int
    transition_count: int
    fragmented_count: int
    contiguous_cardinality: int

    def __post_init__(self) -> None:
        for value, name in (
            (self.distinct_entity_count, "distinct entity count"),
            (self.transition_count, "transition count"),
            (self.fragmented_count, "fragmented count"),
            (self.contiguous_cardinality, "contiguous cardinality"),
        ):
            _count(value, name)

    @property
    def lexicographic_key(self) -> tuple[int, int, int, int]:
        return (
            self.distinct_entity_count,
            self.transition_count,
            self.fragmented_count,
            -self.contiguous_cardinality,
        )


@dataclass(frozen=True, slots=True)
class AssignmentPolicy:
    """Generic solver preferences for one immutable source layout.

    This acts as the formal input layer for the Fluxway Ingestion Solver and is
    modelled as an immutable value object.

    Attributes:
        require_sequential_order: Ensure that canonical intervals are assigned in their
            deterministic, lexicographical sequence.
        affinity_contiguous_chunks: Prioritize assigning a new interval to a reader
            that is currently stationed at the identical `storage_locator` (Owner-Affinity).
        affinity_same_file: Constrain the dispersion of a locator by preferring
            readers that have previously observed this `storage_locator` within the window.
        node_affinity_hint: Prefer readers on a deterministic or previously used
            physical node when attested reader node identities are supplied.
        allow_unit_splitting: Allow splitting an indivisible descriptor chunk
            when a reader's remaining quota is insufficient. If False, physical splitting
            will trigger an exception.

    Note:
        This class no longer accepts legacy string aliases (e.g. "ordered_prefix").
        It strictly enforces canonical mappings or value object initialization.
    """

    require_sequential_order: bool = True
    # Throughput benchmark evidence favors balanced readers; file affinity remains opt-in.
    affinity_contiguous_chunks: bool = False
    affinity_same_file: bool = False
    allow_unit_splitting: bool = True
    node_affinity_hint: bool = False

    def __post_init__(self) -> None:
        for value, name in (
            (self.require_sequential_order, "require_sequential_order"),
            (self.affinity_contiguous_chunks, "affinity_contiguous_chunks"),
            (self.affinity_same_file, "affinity_same_file"),
            (self.allow_unit_splitting, "allow_unit_splitting"),
            (self.node_affinity_hint, "node_affinity_hint"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"placement policy {name} must be boolean")
        if not self.require_sequential_order:
            raise ValueError("source layout requires ordered unit preservation")

    @classmethod
    def default(cls) -> AssignmentPolicy:
        return cls()

    @classmethod
    def from_value(
        cls, value: AssignmentPolicy | Mapping[str, Any]
    ) -> AssignmentPolicy:
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            return cls(**dict(value))
        raise TypeError("assignment policy requires a value object or mapping")


@dataclass(frozen=True, slots=True)
class RankDemand:
    rank_id: str
    cardinality: int

    def __post_init__(self) -> None:
        _id(self.rank_id, "rank id")
        _count(self.cardinality, "rank cardinality", positive=True)


@dataclass(frozen=True, slots=True)
class ReaderGroupDemand:
    worker_id: str
    ranks: tuple[RankDemand, ...]
    node_id: str | None = None

    def __post_init__(self) -> None:
        _id(self.worker_id, "reader worker id")
        if self.node_id is not None:
            _id(self.node_id, "reader worker node id")
        ranks = self.ranks
        if not isinstance(ranks, tuple):
            raise TypeError("reader group ranks must be a tuple")
        if not ranks or any(not isinstance(rank, RankDemand) for rank in ranks):
            raise TypeError("reader group requires RankDemand values")
        if len({rank.rank_id for rank in ranks}) != len(ranks):
            raise ValueError("reader group rank ids must be unique")

    @property
    def cardinality(self) -> int:
        return sum(rank.cardinality for rank in self.ranks)


@dataclass(frozen=True, slots=True)
class ReaderTopologyDescriptor:
    global_cardinality: int
    groups: tuple[ReaderGroupDemand, ...]
    execution_cycle: tuple[tuple[int, ...], ...] | None = None

    def __post_init__(self) -> None:
        _count(self.global_cardinality, "global cardinality", positive=True)
        groups = self.groups
        if not isinstance(groups, tuple):
            raise TypeError("reader topology groups must be a tuple")
        if not groups or any(
            not isinstance(group, ReaderGroupDemand) for group in groups
        ):
            raise TypeError("reader topology requires ReaderGroupDemand values")
        if len({group.worker_id for group in groups}) != len(groups):
            raise ValueError("reader topology worker ids must be unique")
        node_ids = tuple(group.node_id for group in groups)
        if any(node_id is not None for node_id in node_ids) and any(
            node_id is None for node_id in node_ids
        ):
            raise ValueError("reader topology node ids must be complete")
        ranks = [rank.rank_id for group in groups for rank in group.ranks]
        if len(set(ranks)) != len(ranks):
            raise ValueError("reader topology rank ids must be globally unique")
        cycle = (
            (tuple(range(len(groups))),)
            if self.execution_cycle is None
            else self.execution_cycle
        )
        if not isinstance(cycle, tuple) or any(
            not isinstance(ranks, tuple) for ranks in cycle
        ):
            raise TypeError("reader execution cycle must be a tuple of tuples")
        if not cycle or any(
            not ranks or len(set(ranks)) != len(ranks) for ranks in cycle
        ):
            raise ValueError("reader execution cycle requires nonempty unique ranks")
        if {rank for ranks in cycle for rank in ranks} != set(range(len(groups))):
            raise ValueError(
                "reader execution cycle must cover exactly the physical groups"
            )
        if any(
            sum(groups[rank].cardinality for rank in ranks) != self.global_cardinality
            for ranks in cycle
        ):
            raise ValueError("reader topology demands do not equal global cardinality")

    def groups_for_step(self, step: int) -> tuple[ReaderGroupDemand, ...]:
        cycle = (
            (tuple(range(len(self.groups))),)
            if self.execution_cycle is None
            else self.execution_cycle
        )
        return tuple(self.groups[rank] for rank in cycle[step % len(cycle)])

    def bind_node_ids(self, node_ids: tuple[str, ...]) -> ReaderTopologyDescriptor:
        """Return this logical topology with an attested node per reader group."""
        if not isinstance(node_ids, tuple):
            raise TypeError("reader topology node ids must be a tuple")
        if len(node_ids) != len(self.groups):
            raise ValueError(
                "reader topology node id count must match reader group count"
            )
        return replace(
            self,
            groups=tuple(
                replace(group, node_id=node_id)
                for group, node_id in zip(self.groups, node_ids, strict=True)
            ),
        )


@dataclass(frozen=True, slots=True)
class LayoutCompileRequest:
    source_identity: str
    view_identity: str
    schema_digest: str
    units: tuple[LayoutUnitDescriptor, ...]
    topology: ReaderTopologyDescriptor
    tail_layout_mode: TailLayoutMode | str
    lineage_digest: str
    assignment_policy: AssignmentPolicy = field(
        default_factory=AssignmentPolicy.default
    )

    def __post_init__(self) -> None:
        for value in (
            self.source_identity,
            self.view_identity,
            self.schema_digest,
            self.lineage_digest,
        ):
            _id(value, "layout identity")
        if not isinstance(self.topology, ReaderTopologyDescriptor):
            raise TypeError("layout requires ReaderTopologyDescriptor")
        units = tuple(self.units)
        if any(not isinstance(unit, LayoutUnitDescriptor) for unit in units):
            raise TypeError("layout requires LayoutUnitDescriptor values")
        if len({unit.unit_id for unit in units}) != len(units):
            raise ValueError("layout unit ids must be unique")
        if len({unit.cardinality_metric for unit in units}) > 1:
            raise ValueError("layout units must share one cardinality metric")
        mode = TailLayoutMode.from_value(self.tail_layout_mode)
        policy = AssignmentPolicy.from_value(self.assignment_policy)
        object.__setattr__(self, "assignment_policy", policy)
        if mode is TailLayoutMode.DISCARD_REMAINDER and any(
            unit.is_synthetic_padding for unit in units
        ):
            raise ValueError(
                "layout cannot discard remainder when units contain virtual padding"
            )

    @property
    def cardinality(self) -> int:
        return sum(unit.cardinality for unit in self.units)


@dataclass(frozen=True, slots=True)
class WorkerCompiledLayout:
    worker_id: str
    ranks: tuple[RankDemand, ...]
    segments: tuple[CompiledSegment, ...]

    @property
    def cardinality(self) -> int:
        return sum(segment.cardinality for segment in self.segments)


@dataclass(frozen=True, slots=True)
class _NodeAffinityTopology:
    node_by_worker: Mapping[str, str]
    active_nodes: tuple[str, ...]
    node_order: Mapping[str, int]


def _resolve_node_affinity_topology(
    active_groups: tuple[ReaderGroupDemand, ...],
) -> _NodeAffinityTopology:
    node_by_worker: dict[str, str] = {}
    for group in active_groups:
        node_id = group.node_id
        if node_id is None:
            raise ValueError(
                "node affinity hint requires complete active node identities"
            )
        node_by_worker[group.worker_id] = node_id
    active_nodes = tuple(dict.fromkeys(node_by_worker.values()))
    return _NodeAffinityTopology(
        node_by_worker=node_by_worker,
        active_nodes=active_nodes,
        node_order={node_id: position for position, node_id in enumerate(active_nodes)},
    )


class _WorkerAllocationStrategy:
    """Worker/file-affinity lowering with no physical-node state."""

    def __init__(
        self,
        policy: AssignmentPolicy,
        previous_by_worker: Mapping[str, str | None],
        group_order: Mapping[str, int],
    ) -> None:
        self._policy = policy
        self._previous_by_worker = previous_by_worker
        self._group_order = group_order

    def select_worker(
        self,
        available: list[str],
        storage_locator: str,
        owners: list[str],
    ) -> str:
        return min(
            available,
            key=lambda worker_id: _allocation_key(
                self._policy,
                worker_id,
                storage_locator,
                self._previous_by_worker,
                owners,
                self._group_order,
            ),
        )

    def record_worker(
        self, worker_id: str, storage_locator: str, owners: list[str]
    ) -> None:
        if worker_id not in owners:
            owners.append(worker_id)


class _NodeAffinityAllocationStrategy(_WorkerAllocationStrategy):
    """Worker/file affinity extended with an explicit soft node preference."""

    def __init__(
        self,
        policy: AssignmentPolicy,
        previous_by_worker: Mapping[str, str | None],
        group_order: Mapping[str, int],
        topology: _NodeAffinityTopology,
        owner_nodes_by_locator: dict[str, set[str]],
        preferred_node_by_locator: dict[tuple[str, tuple[str, ...]], str],
    ) -> None:
        super().__init__(policy, previous_by_worker, group_order)
        self._node_by_worker = topology.node_by_worker
        self._active_nodes = topology.active_nodes
        self._node_order = topology.node_order
        self._owner_nodes_by_locator = owner_nodes_by_locator
        self._preferred_node_by_locator = preferred_node_by_locator

    def select_worker(
        self,
        available: list[str],
        storage_locator: str,
        owners: list[str],
    ) -> str:
        owner_nodes = self._owner_nodes_by_locator.setdefault(storage_locator, set())
        preferred_key = (storage_locator, self._active_nodes)
        preferred_node = self._preferred_node_by_locator.get(preferred_key)
        if not owner_nodes and preferred_node is None:
            digest = hashlib.sha256(storage_locator.encode("utf-8")).digest()
            preferred_node = self._active_nodes[
                int.from_bytes(digest[:8], "big") % len(self._active_nodes)
            ]
            self._preferred_node_by_locator[preferred_key] = preferred_node
        return min(
            available,
            key=lambda worker_id: _node_affinity_allocation_key(
                self._policy,
                worker_id,
                storage_locator,
                self._previous_by_worker,
                owners,
                self._group_order,
                node_by_worker=self._node_by_worker,
                owner_nodes=owner_nodes,
                node_order=self._node_order,
                preferred_node=preferred_node,
            ),
        )

    def record_worker(
        self, worker_id: str, storage_locator: str, owners: list[str]
    ) -> None:
        super().record_worker(worker_id, storage_locator, owners)
        node_id = self._node_by_worker[worker_id]
        self._owner_nodes_by_locator.setdefault(storage_locator, set()).add(node_id)


@dataclass(frozen=True, slots=True)
class CompiledIngestionLayout:
    request: LayoutCompileRequest
    work_epoch: LogicalWorkEpoch
    discarded_cardinality: int
    worker_layouts: tuple[WorkerCompiledLayout, ...]
    pending_units: tuple[LayoutUnitDescriptor, ...]
    layout_digest: str

    @property
    def step_count(self) -> int:
        return self.work_epoch.end_step - self.work_epoch.start_step

    @property
    def cardinality(self) -> int:
        return sum(
            assignment.descriptor.cardinality
            for assignment in self.work_epoch.assignments
        )

    @property
    def canonical_work_intervals(self) -> tuple[CanonicalWorkInterval, ...]:
        return tuple(
            unit.canonical_work_interval(self.request.source_identity)
            for unit in self.request.units
            if not unit.is_synthetic_padding
        )

    def encoded_worker_layouts(self) -> tuple[EncodedWorkerLayoutDescriptor, ...]:
        return tuple(
            WorkerLayoutDescriptor(
                source_identity=self.request.source_identity,
                view_identity=self.request.view_identity,
                schema_digest=self.request.schema_digest,
                lineage_digest=self.request.lineage_digest,
                layout_digest=self.layout_digest,
                worker_id=layout.worker_id,
                segments=layout.segments,
            ).encode()
            for layout in self.worker_layouts
        )

    def to_source_layout_activation(
        self,
        *,
        epoch: int,
        start_position: int,
        end_position: int,
        terminal: bool,
    ) -> SourceLayoutActivation:
        if (epoch, start_position, end_position, terminal) != (
            self.work_epoch.epoch,
            self.work_epoch.start_step,
            self.work_epoch.end_step,
            self.work_epoch.terminal,
        ):
            raise ValueError(
                "activation boundaries must match compiled epoch and step_count"
            )
        return SourceLayoutActivation(
            IngestionLayoutEpoch(
                epoch=epoch,
                start_position=start_position,
                end_position=end_position,
                assigned_cardinality=self.cardinality,
                layout_id=self.layout_digest,
                worker_ids=tuple(layout.worker_id for layout in self.worker_layouts),
                terminal=terminal,
            ),
            self.encoded_worker_layouts(),
        )


class AlgebraicLayoutCompiler:
    """Mathematical solver for mapping canonical work intervals to physical layouts.

    Execution Flow (Streaming Source Contract):
    1. Resolve purely against the current immutable observation.
    2. Ignore unobserved future identities (they do not participate in topological sort).
    3. Lowering spans Quota, Canonical Coverage, Active-Cycle, and Owner-Affinity uniformly.
    4. Nonterminal incomplete observations retain pending tails without advancing layout boundaries.

    Physical Owner Semantics:
    The resulting worker distribution is a transient binding. It does not infect
    the durable `SourceWorkKey` or canonical interval. Upon `restore`, the physical
    ownership can be symmetrically reconstructed or safely relocated.
    """

    def compile(
        self,
        request: LayoutCompileRequest,
        *,
        epoch: int = 0,
        start_step: int = 0,
    ) -> CompiledIngestionLayout:
        if not isinstance(request, LayoutCompileRequest):
            raise TypeError("algebraic compiler requires LayoutCompileRequest")
        _count(epoch, "layout epoch")
        _count(start_step, "layout start step")
        mode = TailLayoutMode.from_value(request.tail_layout_mode)
        batch = request.topology.global_cardinality
        steps, residual = divmod(request.cardinality, batch)
        units = request.units
        if mode is TailLayoutMode.REJECT_REMAINDER and residual:
            raise ValueError("incomplete terminal global cardinality")
        if mode is TailLayoutMode.SYNTHESIZE_PADDING and residual:
            padding = batch - residual
            units += (
                LayoutUnitDescriptor(
                    f"{request.view_identity}:padding",
                    request.view_identity,
                    padding,
                    units[0].cardinality_metric,
                    "SplittableSpan",
                    "local",
                    SYNTHETIC_PAD_IDENTITY,
                    0,
                    "pad:virtual",
                    work_kind=SourceWorkKind.SYNTHETIC_PADDING,
                ),
            )
            steps += 1
            residual = 0
        validate_canonical_work_intervals(
            tuple(
                unit.canonical_work_interval(request.source_identity)
                for unit in units
                if not unit.is_synthetic_padding
            )
        )
        cursor = _UnitCursor(units)
        assignments = []
        segments: dict[str, list[CompiledSegment]] = {
            group.worker_id: [] for group in request.topology.groups
        }
        policy = request.assignment_policy
        owners_by_locator: dict[str, list[str]] = {}
        owner_nodes_by_locator: dict[str, set[str]] | None = (
            {} if policy.node_affinity_hint else None
        )
        preferred_node_by_locator: dict[tuple[str, tuple[str, ...]], str] | None = (
            {} if policy.node_affinity_hint else None
        )
        assignment_segment_counts: dict[str, int] = {}
        for step in range(start_step, start_step + steps):
            step_assignments = []
            global_start = step * batch
            step_descriptors = cursor.take(batch)
            valid_cardinality = sum(
                d.cardinality for d in step_descriptors if not d.is_synthetic_padding
            )
            for descriptor in step_descriptors:
                step_assignments.append(
                    SourceWorkAssignment.create(
                        epoch=epoch,
                        step=step,
                        source_identity=request.source_identity,
                        view_identity=request.view_identity,
                        schema_digest=request.schema_digest,
                        lineage_digest=request.lineage_digest,
                        descriptor=descriptor,
                        global_start=global_start,
                        global_batch_size=batch,
                        global_valid_count=valid_cardinality,
                    )
                )
                global_start += descriptor.cardinality
            assignments.extend(step_assignments)
            active_groups = request.topology.groups_for_step(step)
            remaining_by_worker = {
                group.worker_id: group.cardinality for group in active_groups
            }
            group_order = {
                group.worker_id: position
                for position, group in enumerate(active_groups)
            }
            previous_by_worker = {
                worker_id: (
                    segments[worker_id][-1].storage_locator
                    if segments[worker_id]
                    else None
                )
                for worker_id in remaining_by_worker
            }
            strategy: _WorkerAllocationStrategy
            if policy.node_affinity_hint:
                assert (
                    owner_nodes_by_locator is not None
                    and preferred_node_by_locator is not None
                )
                topology = _resolve_node_affinity_topology(active_groups)
                strategy = _NodeAffinityAllocationStrategy(
                    policy,
                    previous_by_worker,
                    group_order,
                    topology,
                    owner_nodes_by_locator,
                    preferred_node_by_locator,
                )
            else:
                strategy = _WorkerAllocationStrategy(
                    policy,
                    previous_by_worker,
                    group_order,
                )
            select_worker = strategy.select_worker
            record_worker = strategy.record_worker
            index = 0
            offset = 0
            while index < len(step_assignments):
                assignment = step_assignments[index]
                descriptor = assignment.descriptor
                owners = owners_by_locator.setdefault(descriptor.storage_locator, [])
                available = [
                    worker_id
                    for worker_id, remaining in remaining_by_worker.items()
                    if remaining
                ]
                if not available:
                    raise RuntimeError("layout lowering exhausted active reader quota")
                worker_id = select_worker(
                    available,
                    descriptor.storage_locator,
                    owners,
                )
                amount = min(
                    remaining_by_worker[worker_id],
                    descriptor.cardinality - offset,
                )
                segments[worker_id].append(
                    CompiledSegment(
                        step=step,
                        assignment_id=assignment.key.assignment_id,
                        worker_id=worker_id,
                        unit_id=descriptor.unit_id,
                        revision_identity=descriptor.revision_identity,
                        locality_key=descriptor.locality_key,
                        storage_locator=descriptor.storage_locator,
                        cardinality_metric=descriptor.cardinality_metric,
                        logical_start=descriptor.logical_start + offset,
                        logical_end=descriptor.logical_start + offset + amount,
                        opaque_token=descriptor.opaque_token,
                    )
                )
                assignment_segment_counts[assignment.key.assignment_id] = (
                    assignment_segment_counts.get(assignment.key.assignment_id, 0) + 1
                )
                if (
                    not request.assignment_policy.allow_unit_splitting
                    and assignment_segment_counts[assignment.key.assignment_id] > 1
                ):
                    raise ValueError("placement policy forbids splitting a source unit")
                record_worker(worker_id, descriptor.storage_locator, owners)
                previous_by_worker[worker_id] = descriptor.storage_locator
                remaining_by_worker[worker_id] -= amount
                offset += amount
                if offset == descriptor.cardinality:
                    index += 1
                    offset = 0
        pending = cursor.take(residual)
        work_epoch = LogicalWorkEpoch.create(
            epoch=epoch,
            start_step=start_step,
            end_step=start_step + steps,
            assignments=tuple(assignments),
            terminal=mode is not TailLayoutMode.RETAIN_PENDING,
        )
        layouts = tuple(
            WorkerCompiledLayout(
                group.worker_id, group.ranks, tuple(segments[group.worker_id])
            )
            for group in request.topology.groups
        )
        layout_cost = _layout_cost(layouts)
        digest = metadata_digest(
            {
                "plan": work_epoch.digest,
                "request": _request_to_dict(request),
                "workers": [asdict(layout) for layout in layouts],
                "layout_cost": asdict(layout_cost),
            }
        )
        return CompiledIngestionLayout(
            request,
            work_epoch,
            residual if mode is TailLayoutMode.DISCARD_REMAINDER else 0,
            layouts,
            pending if mode is TailLayoutMode.RETAIN_PENDING else (),
            digest,
        )


def _layout_cost(
    layouts: tuple[WorkerCompiledLayout, ...],
) -> _LayoutCost:
    """Evaluate the physical quality of the resultant assignment.

    Computes a deterministic cost vector $C = (c_1, c_2, c_3, c_4)$ where:
    - $c_1$: Distinct Entities. Sum of cardinality of unique owner sets for each storage locator.
    - $c_2$: Transitions. Number of times a worker switches storage locators.
    - $c_3$: Fragmentations. Number of discontinuous storage locator chunks.
    - $c_4$: Contiguous Cardinality. Aggregate volume of sequential logical observations.
    """
    entities: dict[str, set[str]] = {}
    transitions = 0
    fragmented = 0
    contiguous_cardinality = 0
    for layout in layouts:
        previous: CompiledSegment | None = None
        for segment in layout.segments:
            entities.setdefault(segment.storage_locator, set()).add(layout.worker_id)
            if previous is not None:
                if segment.storage_locator != previous.storage_locator:
                    transitions += 1
                    fragmented += 1
                elif segment.logical_start == previous.logical_end:
                    contiguous_cardinality += segment.cardinality
            previous = segment
    return _LayoutCost(
        distinct_entity_count=sum(len(owners) for owners in entities.values()),
        transition_count=transitions,
        fragmented_count=fragmented,
        contiguous_cardinality=contiguous_cardinality,
    )


def _allocation_key(
    policy: AssignmentPolicy,
    worker_id: str,
    storage_locator: str,
    previous_by_worker: Mapping[str, str | None],
    owners: list[str],
    group_order: Mapping[str, int],
) -> tuple[int, ...]:
    """Derive the deterministic allocation lexicographic key for a target worker.

    For a given storage locator chunk $P$, a candidate worker $w$ is scored
    using a lexicographical projection $K(w) = (k_1, k_2, k_3)$ where the components
    formalize Owner-Affinity and deterministic load distribution:

    1. Contiguous Prefix ($k_1 \\in \\{0, 1\\}$):
       $k_1 = 0 \\iff w.\\text{prev} = P$
       Prioritize workers whose tail storage locator exactly matches $P$.
    2. Owner Set Reunion ($k_2 \\in \\{0, 1\\} \\times \\mathbb{N}$):
       $k_{2,0} = 0 \\iff w \\in \\text{owners}(P)$
       $k_{2,1} = \\text{index}(w) \\text{ if } w \\in \\text{owners}(P) \\text{ else } |\\text{owners}(P)| + \\text{group\\_order}(w)$
       Re-use workers that have already processed $P$ within the active window.
    3. Cycle Order ($k_3 \\in \\mathbb{N}$):
       $k_3 = \\text{group\\_order}(w)$
       Fallback deterministic progression defined by the topological cycle.

    The solver lowers constraints by choosing $\\arg\\min_w K(w)$.
    """
    key: list[int] = []
    if policy.affinity_contiguous_chunks:
        key.append(0 if previous_by_worker[worker_id] == storage_locator else 1)
    if policy.affinity_same_file:
        key.append(0 if worker_id in owners else 1)
        key.append(
            owners.index(worker_id)
            if worker_id in owners
            else len(owners) + group_order[worker_id]
        )
    key.append(group_order[worker_id])
    return tuple(key)


def _node_affinity_allocation_key(
    policy: AssignmentPolicy,
    worker_id: str,
    storage_locator: str,
    previous_by_worker: Mapping[str, str | None],
    owners: list[str],
    group_order: Mapping[str, int],
    *,
    node_by_worker: Mapping[str, str],
    owner_nodes: set[str],
    node_order: Mapping[str, int],
    preferred_node: str | None,
) -> tuple[int, ...]:
    node_id = node_by_worker[worker_id]
    key = [
        0 if owner_nodes and node_id in owner_nodes else 1,
        node_order[node_id]
        if owner_nodes and node_id in owner_nodes
        else (0 if preferred_node == node_id else 1),
    ]
    key.extend(
        _allocation_key(
            policy,
            worker_id,
            storage_locator,
            previous_by_worker,
            owners,
            group_order,
        )
    )
    return tuple(key)


def compile_layout_request(
    request: LayoutCompileRequest, epoch: int, start_step: int
) -> CompiledIngestionLayout:
    """Pure process entry point for source layout compilation.

    The worker receives only immutable request data and scalar boundaries. It has
    no access to planner, ledger, cursor or asyncio state.
    """
    return AlgebraicLayoutCompiler().compile(
        request, epoch=epoch, start_step=start_step
    )


class _UnitCursor:
    def __init__(self, units: tuple[LayoutUnitDescriptor, ...]) -> None:
        self._units = units
        self._index = 0
        self._offset = 0

    def take(self, demand: int) -> tuple[LayoutUnitDescriptor, ...]:
        _count(demand, "source interval demand")
        slices = []
        while demand:
            if self._index >= len(self._units):
                raise ValueError("source demand exceeds canonical units")
            unit = self._units[self._index]
            amount = min(demand, unit.cardinality - self._offset)
            slices.append(unit.slice(self._offset, amount))
            demand -= amount
            self._offset += amount
            if self._offset == unit.cardinality:
                self._index += 1
                self._offset = 0
        return tuple(slices)


def _request_to_dict(request: LayoutCompileRequest) -> dict[str, Any]:
    value = asdict(request)
    value["tail_layout_mode"] = TailLayoutMode.from_value(request.tail_layout_mode)
    return value


def _request_from_dict(value: Any) -> LayoutCompileRequest:
    if not isinstance(value, Mapping):
        raise TypeError("layout planning request must be a mapping")
    request = dict(value)
    request["tail_layout_mode"] = TailLayoutMode.from_value(request["tail_layout_mode"])
    request["units"] = tuple(
        LayoutUnitDescriptor(
            unit_id=item["unit_id"],
            revision_identity=item["revision_identity"],
            cardinality=item["cardinality"],
            cardinality_metric=item["cardinality_metric"],
            split_policy=item["split_policy"],
            locality_key=item["locality_key"],
            storage_locator=item["storage_locator"],
            logical_start=item["logical_start"],
            opaque_token=item["opaque_token"],
            work_kind=SourceWorkKind(item["work_kind"]),
        )
        for item in value["units"]
    )
    execution_cycle = value["topology"]["execution_cycle"]
    request["topology"] = ReaderTopologyDescriptor(
        value["topology"]["global_cardinality"],
        tuple(
            ReaderGroupDemand(
                group["worker_id"],
                tuple(RankDemand(**rank) for rank in group["ranks"]),
                group.get("node_id"),
            )
            for group in value["topology"]["groups"]
        ),
        None
        if execution_cycle is None
        else tuple(tuple(ranks) for ranks in execution_cycle),
    )
    return LayoutCompileRequest(**request)
