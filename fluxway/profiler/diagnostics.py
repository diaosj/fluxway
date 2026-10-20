from __future__ import annotations

from contextlib import AbstractAsyncContextManager

from fluxway.common.types import DriverHeartbeat
from fluxway.profiler.recorder import TraceRecorder


class HeartbeatProfilerObserver:
    def __init__(self, recorder: TraceRecorder) -> None:
        self._recorder = recorder

    def receive_driver_heartbeat(
        self, heartbeat: DriverHeartbeat
    ) -> AbstractAsyncContextManager[None]:
        return self._recorder.heartbeat_receive(
            spmd_id=heartbeat.spmd_id,
            state=heartbeat.state.name,
            healthy_workers=heartbeat.healthy_num_workers,
            num_workers=heartbeat.num_workers,
        )

    def monitor_loop(self, *, spmd_count: int) -> AbstractAsyncContextManager[None]:
        return self._recorder.heartbeat_monitor_loop(spmd_count=spmd_count)

    def timeout_check(self, *, spmd_id: str, elapsed_ms: int) -> None:
        self._recorder.heartbeat_timeout_check(
            spmd_id=spmd_id,
            elapsed_ms=elapsed_ms,
        )

    def blocked_watchdog(self, *, timeout_ms: int) -> AbstractAsyncContextManager[None]:
        return self._recorder.heartbeat_blocked_watchdog(timeout_ms=timeout_ms)
