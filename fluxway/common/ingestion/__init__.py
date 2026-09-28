"""Common protocols, descriptors, and segment models for ingestion pipelines."""

from fluxway.common.ingestion.segment import (
    CanonicalWorkInterval,
    LayoutUnitDescriptor,
    SourceWorkAssignment,
    SourceWorkAcknowledgement,
    SourceWorkItem,
    SourceWorkKey,
    SourceWorkLease,
    SplitPolicy,
    CompiledSegment,
    validate_canonical_work_intervals,
)
from fluxway.common.ingestion.protocol import (
    EncodedWorkerLayoutDescriptor,
    SourceLayoutCompletion,
    IngestionLayoutActivation,
    SourceLayoutActivation,
    IngestionLayoutEpoch,
    WorkerLayoutDescriptor,
)
from fluxway.common.ingestion.receiver import (
    IngestionLayoutReceiver,
    ingestion_worker_id,
)

__all__ = [
    "CanonicalWorkInterval",
    "LayoutUnitDescriptor",
    "SourceWorkAssignment",
    "SourceWorkAcknowledgement",
    "SourceWorkItem",
    "SourceWorkKey",
    "SourceWorkLease",
    "SplitPolicy",
    "CompiledSegment",
    "validate_canonical_work_intervals",
    "EncodedWorkerLayoutDescriptor",
    "SourceLayoutCompletion",
    "IngestionLayoutActivation",
    "SourceLayoutActivation",
    "IngestionLayoutEpoch",
    "WorkerLayoutDescriptor",
    "IngestionLayoutReceiver",
    "ingestion_worker_id",
]
