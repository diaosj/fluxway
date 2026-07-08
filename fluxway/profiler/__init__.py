"""Optional low-overhead profiler runtime for Fluxway."""

from fluxway.profiler.config import ProfilerConfig, ProfilerDiagnostic
from fluxway.profiler.errors import (
    CausalClockViolation,
    ClockSyncUnstable,
    MissingAxonNativeHlc,
    MissingTraceMetadata,
    ProfilerBufferExhausted,
)
from fluxway.profiler.record import TraceCarrier, TraceEventKind, TraceMetadata
from fluxway.profiler.runtime import ProfilerRuntimeFactory
from fluxway.profiler.user_api import enabled, event, scope

__all__ = [
    "CausalClockViolation",
    "ClockSyncUnstable",
    "MissingAxonNativeHlc",
    "MissingTraceMetadata",
    "ProfilerBufferExhausted",
    "ProfilerConfig",
    "ProfilerDiagnostic",
    "ProfilerRuntimeFactory",
    "TraceCarrier",
    "TraceEventKind",
    "TraceMetadata",
    "enabled",
    "event",
    "scope",
]
