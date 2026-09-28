"""Control Plane: Core coordination layer for the training framework."""

# Ensure INFO logs are visible even when importing this subpackage directly.
from fluxway.common.utils.logging import ensure_logging_configured


from .graph import Builder
from .scheduler import Scheduler
from .spmd_manager import SPMDManager, SPMDHandle
from .state_manager import StateManager
from .monitor import Monitor
from .heartbeat import HeartbeatManager
from .state_machine import ControllerStateMachine
from .async_manager import AsyncStepManager
from .errors import ExecutionFailedError
from .components import (
    ControlPlaneComponent,
    ControlPlaneComponentRegistry,
    TransactionalControlPlaneComponent,
    DriverComponentContext,
    DriverComponentRegistry,
)
from .ledger_client import SourceWorkClient
from .source_control import SourceControlRuntime
from .logical_work_ledger import LogicalWorkEpoch, LogicalWorkLedger
from .ingestion import (
    AlgebraicLayoutCompiler,
    CanonicalWorkInterval,
    CatalogCheckpointImage,
    CatalogCursor,
    CatalogViewObservation,
    CompiledIngestionLayout,
    CompiledSegment,
    DirectIngestionSourceDescriptor,
    EncodedWorkerLayoutDescriptor,
    InMemoryIngestionSourceAdapter,
    IngestionCatalogCheckpointComponent,
    SourceLayoutCompletion,
    IngestionLayoutActivation,
    IngestionLayoutActivationComponent,
    SourceLayoutActivation,
    IngestionLayoutEpoch,
    IngestionLayoutPlanningComponent,
    IngestionPlacementContext,
    IngestionLayoutReceiver,
    IngestionSourceAdapter,
    LayoutCompileRequest,
    AssignmentPolicy,
    LayoutUnitDescriptor,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    WorkerCompiledLayout,
    WorkerLayoutDescriptor,
    ingestion_worker_id,
    validate_canonical_work_intervals,
)
from .execution import start, GraphExecution
from .checkpointer import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
    SaveSession,
    CheckpointCut,
    FluxwayCheckpointer,
    StateCheckpointComponent,
    SafeTensorsCheckpointComponent,
    WorkerLocalCheckpointComponent,
    OrbaxCheckpointComponent,
)

__all__ = [
    "Builder",
    "Scheduler",
    "SPMDManager",
    "SPMDHandle",
    "StateManager",
    "Monitor",
    "HeartbeatManager",
    "ControllerStateMachine",
    "AsyncStepManager",
    "ExecutionFailedError",
    "ControlPlaneComponent",
    "ControlPlaneComponentRegistry",
    "TransactionalControlPlaneComponent",
    "DriverComponentContext",
    "DriverComponentRegistry",
    "SourceWorkClient",
    "SourceControlRuntime",
    "LogicalWorkEpoch",
    "LogicalWorkLedger",
    "EncodedWorkerLayoutDescriptor",
    "DirectIngestionSourceDescriptor",
    "IngestionLayoutActivation",
    "IngestionLayoutActivationComponent",
    "SourceLayoutActivation",
    "SourceLayoutCompletion",
    "IngestionLayoutEpoch",
    "WorkerLayoutDescriptor",
    "AlgebraicLayoutCompiler",
    "CanonicalWorkInterval",
    "CompiledIngestionLayout",
    "CompiledSegment",
    "IngestionLayoutPlanningComponent",
    "IngestionPlacementContext",
    "IngestionLayoutReceiver",
    "LayoutCompileRequest",
    "AssignmentPolicy",
    "LayoutUnitDescriptor",
    "RankDemand",
    "ReaderGroupDemand",
    "ReaderTopologyDescriptor",
    "WorkerCompiledLayout",
    "validate_canonical_work_intervals",
    "CatalogCheckpointImage",
    "CatalogCursor",
    "IngestionCatalogCheckpointComponent",
    "CatalogViewObservation",
    "InMemoryIngestionSourceAdapter",
    "IngestionSourceAdapter",
    "ingestion_worker_id",
    "start",
    "GraphExecution",
    "FluxwayCheckpointer",
    "SaveSession",
    "CheckpointComponentRegistration",
    "CheckpointComponentRegistry",
    "CheckpointCut",
    "StateCheckpointComponent",
    "SafeTensorsCheckpointComponent",
    "WorkerLocalCheckpointComponent",
    "OrbaxCheckpointComponent",
]

ensure_logging_configured()
