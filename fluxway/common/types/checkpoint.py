"""Checkpoint domain identities shared across control and worker processes."""

from __future__ import annotations

import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from math import prod
from typing import Any


class CheckpointError(RuntimeError):
    pass


class CheckpointBusyError(CheckpointError):
    pass


class CheckpointObjectCleanedUpError(CheckpointError):
    pass


class EagerCheckpointUnsupportedError(CheckpointError):
    pass


class LeaseConflictError(CheckpointError):
    pass


class CheckpointCleanupBlocked(CheckpointError):
    pass


class TopologyMismatchError(CheckpointError):
    pass


class CheckpointId(str):
    def __new__(cls, value: str) -> CheckpointId:
        if not value:
            raise ValueError("checkpoint ID must be non-empty")
        return str.__new__(cls, value)

    @classmethod
    def generate(cls) -> CheckpointId:
        return cls(uuid.uuid4().hex)


class CheckpointRuntimeRole(StrEnum):
    CONTROL = "control"
    WORKER = "worker"


class CheckpointCapability(StrEnum):
    PERSISTENCE = "persistence"
    LIFECYCLE = "lifecycle"
    RETENTION = "keep"
    RECLAIMER = "cleaner"


class CheckpointCaptureMode(StrEnum):
    FENCED = "fenced"
    EAGER = "eager"


class CheckpointTransactionState(StrEnum):
    CUT_CREATED = "cut_created"
    CAPTURING = "capturing"
    CAPTURED = "captured"
    FINALIZED = "finalized"
    COMMITTED = "committed"
    PUBLISHED = "published"
    RECLAIM_PENDING = "cleanup_pending"


class PrepareScope(StrEnum):
    LEADER = "leader"
    ALL_RANKS = "all_ranks"


class CommitScope(StrEnum):
    LEADER = "leader"
    ALL_RANKS = "all_ranks"


class RestoreTopologyMode(StrEnum):
    EXACT_TOPOLOGY = "exact_topology"
    LOGICAL_PARTITION_REMAP = "logical_partition_remap"
    GLOBAL_RESHARD = "global_reshard"


@dataclass(frozen=True, slots=True)
class CheckpointCut:
    cut_id: str
    logical_step: int | None
    stream_cuts: Mapping[str, Any] = field(default_factory=dict)
    sealed_ends: Mapping[str, Any] = field(default_factory=dict)
    control_epoch: int = 0

    def __post_init__(self) -> None:
        if not self.cut_id:
            raise ValueError("checkpoint cut requires cut_id")
        if self.logical_step is not None and (
            isinstance(self.logical_step, bool) or self.logical_step < 0
        ):
            raise ValueError("logical_step must be a non-negative integer or None")
        if self.control_epoch < 0:
            raise ValueError("control_epoch must be non-negative")

    @classmethod
    def for_step(cls, cut_id: str, step: int) -> CheckpointCut:
        return cls(cut_id=cut_id, logical_step=step)


@dataclass(frozen=True, slots=True)
class CheckpointCommitPermit:
    attempt_id: str
    coordinator_generation: int
    token: bytes

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("checkpoint commit permit requires attempt_id")
        if self.coordinator_generation < 0:
            raise ValueError("checkpoint commit permit generation must be non-negative")
        if not isinstance(self.token, bytes) or not self.token:
            raise ValueError("checkpoint commit permit requires an opaque token")


@dataclass(frozen=True, slots=True)
class CommitAuthority:
    scope: str
    generation: int
    token: bytes

    def __post_init__(self) -> None:
        if not self.scope:
            raise ValueError("commit authority requires scope")
        if self.generation < 0:
            raise ValueError("commit authority generation must be non-negative")
        if not self.token:
            raise ValueError("commit authority requires token")


@dataclass(frozen=True, slots=True)
class TopologySnapshot:
    world_size: int
    mesh_axis_names: tuple[str, ...]
    mesh_axis_shapes: tuple[int, ...]
    partition_scheme_version: str

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("topology world_size must be positive")
        if len(self.mesh_axis_names) != len(self.mesh_axis_shapes):
            raise ValueError("mesh axis names and shapes must have equal length")
        if len(set(self.mesh_axis_names)) != len(self.mesh_axis_names):
            raise ValueError("mesh axis names must be unique")
        if any(size <= 0 for size in self.mesh_axis_shapes):
            raise ValueError("mesh axis shapes must be positive")
        if self.mesh_axis_shapes and prod(self.mesh_axis_shapes) != self.world_size:
            raise ValueError("mesh shape product must equal topology world_size")
        if not self.partition_scheme_version:
            raise ValueError("topology requires partition_scheme_version")


@dataclass(frozen=True, slots=True)
class CheckpointObjectKey:
    component_id: str
    checkpoint_object_id: str
    checkpoint_id: str
    logical_step: int | None
    spmd_id: str
    partition_key: str | None

    def __post_init__(self) -> None:
        for name, value in (
            ("component_id", self.component_id),
            ("checkpoint_object_id", self.checkpoint_object_id),
            ("checkpoint_id", self.checkpoint_id),
            ("spmd_id", self.spmd_id),
        ):
            if not value:
                raise ValueError(f"checkpoint_object key requires {name}")
        if self.logical_step is not None and (
            isinstance(self.logical_step, bool) or self.logical_step < 0
        ):
            raise ValueError(
                "checkpoint_object logical_step must be non-negative or None"
            )


@dataclass(frozen=True, slots=True)
class CheckpointObjectRef:
    component_id: str
    checkpoint_object_id: str
    version_id: str
    storage_locator: bytes
    format_version: int
    partition_key: str | None = None
    cleanup_group: str | None = None

    def __post_init__(self) -> None:
        if (
            not self.component_id
            or not self.checkpoint_object_id
            or not self.version_id
        ):
            raise ValueError(
                "checkpoint_object reference requires component, checkpoint_object, and version"
            )
        if not isinstance(self.storage_locator, bytes) or not self.storage_locator:
            raise ValueError(
                "checkpoint_object reference requires an opaque storage_locator"
            )
        if self.format_version <= 0:
            raise ValueError("checkpoint_object format_version must be positive")


@dataclass(frozen=True, slots=True)
class KeepPolicy:
    keep_last: int | None = None
    keep_forever: bool = False

    def __post_init__(self) -> None:
        if self.keep_last is not None and self.keep_last <= 0:
            raise ValueError("keep_last must be positive or None")
        if self.keep_forever and self.keep_last is not None:
            raise ValueError("keep_forever keep cannot define keep_last")


@dataclass(frozen=True, slots=True)
class CheckpointObjectSpec:
    checkpoint_object_id: str = "state"
    keep: KeepPolicy = field(default_factory=KeepPolicy)
    required_for_restore: bool = True
    cleanup_group: str | None = None

    def __post_init__(self) -> None:
        if not self.checkpoint_object_id:
            raise ValueError(
                "checkpoint_object checkpoint_object requires checkpoint_object_id"
            )


@dataclass(frozen=True, slots=True)
class ComponentSpec:
    component_id: str
    format_version: int = 1
    prepare_scope: PrepareScope = PrepareScope.LEADER
    commit_scope: CommitScope = CommitScope.LEADER
    restore_topology: RestoreTopologyMode = RestoreTopologyMode.EXACT_TOPOLOGY
    capabilities: frozenset[CheckpointCapability] = frozenset(
        {CheckpointCapability.PERSISTENCE, CheckpointCapability.LIFECYCLE}
    )
    checkpoint_objects: tuple[CheckpointObjectSpec, ...] = field(
        default_factory=lambda: (CheckpointObjectSpec(),)
    )
    supports_eager_capture: bool = False

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("checkpoint component requires component_id")
        if self.format_version <= 0:
            raise ValueError("backend format_version must be positive")
        checkpoint_object_ids = [
            checkpoint_object.checkpoint_object_id
            for checkpoint_object in self.checkpoint_objects
        ]
        if len(checkpoint_object_ids) != len(set(checkpoint_object_ids)):
            raise ValueError("checkpoint backend checkpoint_object IDs must be unique")
        if CheckpointCapability.PERSISTENCE not in self.capabilities:
            raise ValueError("checkpoint backend descriptor requires persistence")


@dataclass(frozen=True, slots=True)
class SaveSession:
    instance: Any
    attempt_id: str
    checkpoint_id: str
    commit_permit: CheckpointCommitPermit
    cut: CheckpointCut
    spmd_id: str
    rank: int
    source_topology: TopologySnapshot
    component_id: str
    format_version: int = 1

    def __post_init__(self) -> None:
        if not self.attempt_id or not self.checkpoint_id:
            raise ValueError("save session requires attempt and checkpoint IDs")
        if self.commit_permit.attempt_id != self.attempt_id:
            raise ValueError("save session permit belongs to another attempt")
        if not self.spmd_id or not self.component_id:
            raise ValueError("save session requires SPMD and component IDs")
        if self.rank < 0 or self.rank >= self.source_topology.world_size:
            raise ValueError("save session rank is outside source topology")
        if self.format_version <= 0:
            raise ValueError("save session format_version must be positive")

    def checkpoint_object_key(
        self, checkpoint_object_id: str, partition_key: str | None = None
    ) -> CheckpointObjectKey:
        return CheckpointObjectKey(
            component_id=self.component_id,
            checkpoint_object_id=checkpoint_object_id,
            checkpoint_id=self.checkpoint_id,
            logical_step=self.cut.logical_step,
            spmd_id=self.spmd_id,
            partition_key=partition_key,
        )

    @property
    def current_topology(self) -> None:
        return None

    def worker_save_result(self, payload: bytes = b"") -> WorkerSaveResult:
        return WorkerSaveResult(commit_metadata=payload)

    def checkpoint_object_ref(
        self,
        checkpoint_object_id: str,
        version_id: str,
        storage_locator: bytes,
        *,
        partition_key: str | None = None,
        cleanup_group: str | None = None,
    ) -> CheckpointObjectRef:
        return CheckpointObjectRef(
            component_id=self.component_id,
            checkpoint_object_id=checkpoint_object_id,
            version_id=version_id,
            storage_locator=storage_locator,
            format_version=self.format_version,
            partition_key=partition_key,
            cleanup_group=cleanup_group,
        )

    def component_checkpoint(
        self, objects: tuple[CheckpointObjectRef, ...]
    ) -> ComponentCheckpoint:
        return ComponentCheckpoint(
            component_id=self.component_id,
            format_version=self.format_version,
            shared_records=(CheckpointObjectRecord(checkpoint_objects=objects),),
        )

    def commit_authority(self, scope: str) -> CommitAuthority:
        return CommitAuthority(
            scope=scope,
            generation=self.commit_permit.coordinator_generation,
            token=self.commit_permit.token,
        )


@dataclass(frozen=True, slots=True)
class CommitSaveSession(SaveSession):
    worker_results: Mapping[int, WorkerSaveResult] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RestoreSession:
    instance: Any
    checkpoint_id: str
    cut: CheckpointCut
    spmd_id: str
    rank: int
    saved_topology: TopologySnapshot
    restore_topology: TopologySnapshot
    component_id: str
    component_checkpoint: ComponentCheckpoint
    format_version: int = 1

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.spmd_id or not self.component_id:
            raise ValueError(
                "restore session requires checkpoint, SPMD, and component IDs"
            )
        if self.rank < 0 or self.rank >= self.restore_topology.world_size:
            raise ValueError("restore session rank is outside restore topology")

    @property
    def source_topology(self) -> TopologySnapshot:
        return self.saved_topology

    @property
    def current_topology(self) -> TopologySnapshot:
        return self.restore_topology

    @property
    def commit_permit(self) -> None:
        return None

    @property
    def attempt_id(self) -> str:
        return f"restore:{self.checkpoint_id}"


@dataclass(frozen=True, slots=True)
class CheckpointRuntime:
    role: CheckpointRuntimeRole
    repository_root: str
    spmd_id: str
    rank: int | None = None

    def __post_init__(self) -> None:
        if not self.repository_root or not self.spmd_id:
            raise ValueError("backend runtime requires repository root and SPMD ID")
        if self.role is CheckpointRuntimeRole.WORKER and self.rank is None:
            raise ValueError("worker backend runtime requires rank")


@dataclass(frozen=True, slots=True)
class CheckpointObjectRecord:
    payload: bytes = b""
    checkpoint_objects: tuple[CheckpointObjectRef, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.payload, bytes):
            raise TypeError("backend record payload must be opaque bytes")


@dataclass(frozen=True, slots=True)
class WorkerSaveResult:
    record: CheckpointObjectRecord | None = None
    commit_metadata: bytes = b""

    def __post_init__(self) -> None:
        if not isinstance(self.commit_metadata, bytes):
            raise TypeError("prepared checkpoint metadata must be opaque bytes")


@dataclass(frozen=True, slots=True)
class PartitionRecord:
    partition_key: str
    source_rank: int | None
    record: CheckpointObjectRecord

    def __post_init__(self) -> None:
        if not self.partition_key:
            raise ValueError("partition record requires partition_key")
        if self.source_rank is not None and self.source_rank < 0:
            raise ValueError("partition source_rank must be non-negative")


@dataclass(frozen=True, slots=True)
class ComponentCheckpoint:
    component_id: str
    format_version: int
    shared_records: tuple[CheckpointObjectRecord, ...] = ()
    partitions: tuple[PartitionRecord, ...] = ()

    def __post_init__(self) -> None:
        if not self.component_id:
            raise ValueError("component publication requires component_id")
        if self.format_version <= 0:
            raise ValueError("backend publication format_version must be positive")
        keys = [partition.partition_key for partition in self.partitions]
        if len(keys) != len(set(keys)):
            raise ValueError("backend publication partition keys must be unique")
        records = (
            *self.shared_records,
            *(partition.record for partition in self.partitions),
        )
        if any(
            checkpoint_object.component_id != self.component_id
            for record in records
            for checkpoint_object in record.checkpoint_objects
        ):
            raise ValueError(
                "checkpoint_object component does not match its publication"
            )
        versions_by_checkpoint_object: dict[str, set[str]] = {}
        for record in records:
            for checkpoint_object in record.checkpoint_objects:
                versions_by_checkpoint_object.setdefault(
                    checkpoint_object.checkpoint_object_id, set()
                ).add(checkpoint_object.version_id)
        if any(
            len(versions) != 1 for versions in versions_by_checkpoint_object.values()
        ):
            raise ValueError(
                "one checkpoint_object publication must use one logical checkpoint_object version"
            )


@dataclass(frozen=True, slots=True)
class SpmdCheckpoint:
    source_topology: TopologySnapshot
    runtime_state: Mapping[str, Any]
    components: Mapping[str, ComponentCheckpoint]

    def __post_init__(self) -> None:
        if any(
            component_id != publication.component_id
            for component_id, publication in self.components.items()
        ):
            raise ValueError("component publication does not match its mapping key")


@dataclass(frozen=True, slots=True)
class StepReference:
    step: int

    def __post_init__(self) -> None:
        if isinstance(self.step, bool) or self.step < 0:
            raise ValueError("step reference must be non-negative")


@dataclass(frozen=True, slots=True)
class NameReference:
    name: str
    keep_forever: bool = True

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name reference requires a name")


@dataclass(frozen=True, slots=True)
class CheckpointIdReference:
    checkpoint_id: str

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint reference requires checkpoint_id")


CheckpointReference = StepReference | NameReference | CheckpointIdReference


@dataclass(frozen=True, slots=True)
class CheckpointManifest:
    schema_version: int
    checkpoint_id: str
    attempt_id: str
    coordinator_generation: int
    cut: CheckpointCut
    checkpoint_sequence: int
    references: tuple[CheckpointReference, ...]
    framework_state: Mapping[str, Any]
    spmds: Mapping[str, SpmdCheckpoint]

    def __post_init__(self) -> None:
        if self.schema_version != 4:
            raise ValueError("only checkpoint manifest schema version 4 is supported")
        if not self.checkpoint_id:
            raise ValueError("checkpoint manifest requires checkpoint_id")
        if self.checkpoint_sequence < 0:
            raise ValueError("checkpoint_sequence must be non-negative")
        if not self.attempt_id:
            raise ValueError("checkpoint manifest requires attempt_id")
        if self.coordinator_generation < 0:
            raise ValueError("checkpoint manifest coordinator generation is invalid")


@dataclass(frozen=True, slots=True)
class CheckpointAssembly:
    cut: CheckpointCut
    framework_state: Mapping[str, Any]
    spmds: Mapping[str, SpmdCheckpoint]


@dataclass(frozen=True, slots=True)
class RestoreRequest:
    checkpoint_id: str
    source_topology: TopologySnapshot
    current_topology: TopologySnapshot


@dataclass(frozen=True, slots=True)
class SaveCheckpointRequest:
    attempt_id: str
    cut: CheckpointCut
    references: tuple[CheckpointReference, ...]
    capture_mode: CheckpointCaptureMode = CheckpointCaptureMode.FENCED

    def __post_init__(self) -> None:
        if not self.attempt_id:
            raise ValueError("save checkpoint request requires attempt_id")
        if not self.references:
            raise ValueError("save checkpoint request requires at least one reference")
        step_references = tuple(
            reference.step
            for reference in self.references
            if isinstance(reference, StepReference)
        )
        if step_references and self.cut.logical_step is None:
            raise ValueError("step references require a logical checkpoint step")
        if any(step != self.cut.logical_step for step in step_references):
            raise ValueError("step reference must match the checkpoint cut")
        names = tuple(
            reference.name
            for reference in self.references
            if isinstance(reference, NameReference)
        )
        if len(names) != len(set(names)):
            raise ValueError(
                "save checkpoint request contains duplicate named references"
            )
        if any(
            isinstance(reference, CheckpointIdReference)
            for reference in self.references
        ):
            raise ValueError(
                "checkpoint references select existing manifests and cannot publish new ones"
            )

    @classmethod
    def for_step(
        cls,
        step: int,
        *,
        names: tuple[str, ...] = (),
        capture_mode: CheckpointCaptureMode = CheckpointCaptureMode.FENCED,
    ) -> SaveCheckpointRequest:
        attempt_id = uuid.uuid4().hex
        references: tuple[CheckpointReference, ...] = (
            StepReference(step),
            *(NameReference(name) for name in names),
        )
        return cls(
            attempt_id=attempt_id,
            cut=CheckpointCut.for_step(uuid.uuid4().hex, step),
            references=references,
            capture_mode=capture_mode,
        )


@dataclass(frozen=True, slots=True)
class CheckpointHold:
    lease_id: str
    checkpoint_object: CheckpointObjectRef
    owner_id: str
    expires_at: float
    generation: int
    request_id: str

    @property
    def expired(self) -> bool:
        return self.expires_at <= time.time()


@dataclass(frozen=True, slots=True)
class PendingCleanup:
    checkpoint_id: str
    spmd_id: str
    component_id: str
    reservation: CleanupReservation

    def __post_init__(self) -> None:
        if not self.checkpoint_id or not self.spmd_id or not self.component_id:
            raise ValueError(
                "pending cleanup requires snapshot, SPMD, and component IDs"
            )
        if any(
            checkpoint_object.component_id != self.component_id
            for checkpoint_object in self.reservation.claimed
        ):
            raise ValueError(
                "pending cleanup checkpoint_object belongs to another component"
            )


@dataclass(frozen=True, slots=True)
class CleanupReservationRequest:
    permit: CheckpointCommitPermit
    checkpoint_object_id: str
    ordered_candidates: tuple[CheckpointObjectRef, ...]
    required_count: int
    protected_versions: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if not self.checkpoint_object_id:
            raise ValueError("cleanup reservation requires checkpoint_object")
        if self.required_count < 0:
            raise ValueError("keep required_count must be non-negative")


@dataclass(frozen=True, slots=True)
class CleanupReservation:
    attempt_id: str
    checkpoint_object_id: str
    claimed: tuple[CheckpointObjectRef, ...]
    generation: int
