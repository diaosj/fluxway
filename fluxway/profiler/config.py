from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class ProfilerDiagnostic(StrEnum):
    HEARTBEAT = "heartbeat"


@dataclass(frozen=True, slots=True)
class ProfilerConfig:
    trace_dir: str
    run_id: str
    buffer_bytes_per_process: int = 64 * 1024 * 1024
    clock_sync_rounds: int = 8
    max_clock_uncertainty_ns: int = 1_000_000
    fail_on_buffer_exhaustion: bool = True
    output_file: str | None = None
    diagnostics: frozenset[ProfilerDiagnostic] = frozenset()
    skip_initial_steps: int = 0
    steady_state_only: bool = False

    def __post_init__(self) -> None:
        if not self.trace_dir:
            raise ValueError("ProfilerConfig.trace_dir must be non-empty.")
        if not self.run_id:
            raise ValueError("ProfilerConfig.run_id must be non-empty.")
        if self.buffer_bytes_per_process <= 0:
            raise ValueError(
                "ProfilerConfig.buffer_bytes_per_process must be positive."
            )
        if self.clock_sync_rounds <= 0:
            raise ValueError("ProfilerConfig.clock_sync_rounds must be positive.")
        if self.max_clock_uncertainty_ns < 0:
            raise ValueError(
                "ProfilerConfig.max_clock_uncertainty_ns must be non-negative."
            )
        if self.skip_initial_steps < 0:
            raise ValueError("ProfilerConfig.skip_initial_steps must be non-negative.")
        if not isinstance(self.diagnostics, frozenset):
            raise TypeError("ProfilerConfig.diagnostics must be a frozenset")
        if any(not isinstance(item, ProfilerDiagnostic) for item in self.diagnostics):
            raise TypeError(
                "ProfilerConfig.diagnostics must contain ProfilerDiagnostic values"
            )

    @property
    def run_dir(self) -> Path:
        return Path(self.trace_dir) / self.run_id

    @property
    def output_path(self) -> Path:
        if self.output_file is not None:
            return Path(self.output_file)
        return self.run_dir / "trace.pftrace"
