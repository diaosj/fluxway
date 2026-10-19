"""Common protocols, descriptors, and segment models for ingestion pipelines."""

from fluxway.common.ingestion.segment import (
    CanonicalWorkInterval,
    LayoutUnitDescriptor,
    SourceWorkAssignment,
    SourceWorkAcknowledgement,
    SourceWorkItem,
    SourceWorkKey,
    SourceWorkKind,
    source_work_global_valid_count,
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
from fluxway.common.ingestion.source_scan import (
    OpaqueSourceCursor,
    SourceBindingIdentity,
    SourceScanResult,
    SourceScanBudget,
    SourceScanIdentity,
    SourceScanStatus,
)

__all__ = [
    "CanonicalWorkInterval",
    "LayoutUnitDescriptor",
    "SourceWorkAssignment",
    "SourceWorkAcknowledgement",
    "SourceWorkItem",
    "SourceWorkKey",
    "SourceWorkKind",
    "source_work_global_valid_count",
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
    "OpaqueSourceCursor",
    "SourceBindingIdentity",
    "SourceScanResult",
    "SourceScanBudget",
    "SourceScanIdentity",
    "SourceScanStatus",
]
