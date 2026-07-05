from __future__ import annotations

from typing import ClassVar


class ProfilerError(RuntimeError):
    code: ClassVar[str] = "profiler_error"

    def __init__(self, message: str, **context: object) -> None:
        if not message:
            raise ValueError("profiler error message must be non-empty.")
        self.message = message
        self.context = dict(context)
        super().__init__(self._format_message())

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": self.message,
            "context": dict(self.context),
        }

    def _format_message(self) -> str:
        if not self.context:
            return f"{self.code}: {self.message}"
        pairs = " ".join(f"{key}={self.context[key]!r}" for key in sorted(self.context))
        return f"{self.code}: {self.message} {pairs}"


class ProfilerBufferExhausted(ProfilerError):
    code = "profiler_buffer_exhausted"

    def __init__(self, *, capacity: int, attempted_index: int) -> None:
        super().__init__(
            "Profiler trace arena capacity was exhausted.",
            capacity=capacity,
            attempted_index=attempted_index,
        )


class MissingAxonNativeHlc(ProfilerError):
    code = "missing_axon_native_hlc"

    def __init__(self, *, result_type: str, has_metadata: bool) -> None:
        super().__init__(
            "Axon invoke result must expose native HLC metadata.",
            result_type=result_type,
            has_metadata=has_metadata,
        )


class ClockSyncUnstable(ProfilerError):
    code = "clock_sync_unstable"

    def __init__(
        self,
        *,
        sender_ts_ns: int,
        receiver_ts_ns: int,
        max_uncertainty_ns: int,
    ) -> None:
        super().__init__(
            "Physical clock synchronization exceeded uncertainty bound.",
            sender_ts_ns=sender_ts_ns,
            receiver_ts_ns=receiver_ts_ns,
            max_uncertainty_ns=max_uncertainty_ns,
        )


class CausalClockViolation(ProfilerError):
    code = "causal_clock_violation"

    def __init__(
        self,
        *,
        sender_hlc_physical_ns: int,
        sender_hlc_logical: int,
        receiver_hlc_physical_ns: int,
        receiver_hlc_logical: int,
    ) -> None:
        super().__init__(
            "Receiver HLC must be greater than sender HLC.",
            sender_hlc_physical_ns=sender_hlc_physical_ns,
            sender_hlc_logical=sender_hlc_logical,
            receiver_hlc_physical_ns=receiver_hlc_physical_ns,
            receiver_hlc_logical=receiver_hlc_logical,
        )


class MissingTraceMetadata(ProfilerError):
    code = "missing_trace_metadata"

    def __init__(self, *, metadata_kind: str, ident: int) -> None:
        super().__init__(
            "Trace metadata is required to write a readable profiler trace.",
            metadata_kind=metadata_kind,
            ident=ident,
        )
