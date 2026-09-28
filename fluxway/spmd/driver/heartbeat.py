import threading
import time
import asyncio
from typing import Any
from collections.abc import Callable

from fluxway.common.types import (
    DriverEvent,
    DriverState,
    DriverStatus,
    DriverHeartbeat,
    ExecutionFailure,
    HeartbeatConfig,
    DEFAULT_INACTIVITY_TIMEOUT_SECS,
)
from fluxway.common.utils.logging import get_logger

logger = get_logger(__name__)


class HeartbeatManager:
    """Manages heartbeat loop for the SPMD Driver."""

    def __init__(
        self,
        spmd_id: str,
        num_workers: int,
        heartbeat_config: HeartbeatConfig,
        profile_heartbeat: bool,
        get_driver_state: Callable[[], DriverState],
        get_driver_failure: Callable[[], ExecutionFailure | None],
        get_workers_ready: Callable[[], bool],
        get_topology_ready: Callable[[], bool],
        get_worker_heartbeat_times: Callable[[], dict[int, float]],
        controller_handle: Any | None,
        trigger_state_event: Callable[[DriverEvent, Any], Any],
        receive_worker_failure: Callable[[int, ExecutionFailure], Any],
        profiler_runtime: Any = None,
        get_worker_heartbeats: Callable[[], dict[int, Any]] | None = None,
        inactivity_timeout_s: float = DEFAULT_INACTIVITY_TIMEOUT_SECS,
    ):
        self.spmd_id = spmd_id
        self.num_workers = num_workers
        self._heartbeat_config = heartbeat_config
        self._profile_heartbeat = profile_heartbeat
        self.get_driver_state = get_driver_state
        self.get_driver_failure = get_driver_failure
        self.get_workers_ready = get_workers_ready
        self.get_topology_ready = get_topology_ready
        self.get_worker_heartbeat_times = get_worker_heartbeat_times
        self.controller_handle = controller_handle
        self.trigger_state_event = trigger_state_event
        self.receive_worker_failure = receive_worker_failure
        self._profiler_runtime = profiler_runtime
        self.get_worker_heartbeats = get_worker_heartbeats
        self._inactivity_timeout_s = inactivity_timeout_s

        self._heartbeat_running = False
        self._heartbeat_thread: threading.Thread | None = None
        self._init_timeout = 120.0
        self._init_start_time = 0.0

    def start(self, init_start_time: float) -> None:
        if self._heartbeat_running:
            return

        self._init_start_time = init_start_time
        self._heartbeat_running = True
        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop, daemon=True
        )
        self._heartbeat_thread.start()

    async def stop(self) -> None:
        self._heartbeat_running = False
        heartbeat_thread = self._heartbeat_thread
        if (
            heartbeat_thread is not None
            and heartbeat_thread.is_alive()
            and heartbeat_thread is not threading.current_thread()
        ):
            timeout = self._heartbeat_config.heartbeat_interval + 1.0
            if self._profile_heartbeat and self._profiler_runtime:
                async with self._profiler_runtime.recorder.driver_heartbeat_thread_join(
                    timeout_ms=int(timeout * 1000),
                ):
                    await asyncio.to_thread(heartbeat_thread.join, timeout)
            else:
                await asyncio.to_thread(heartbeat_thread.join, timeout)
            if heartbeat_thread.is_alive():
                logger.warning(
                    f"Heartbeat thread did not terminate within timeout spmd_id={self.spmd_id}"
                )
        self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        while self._heartbeat_running:
            try:
                self._send_heartbeat()

                driver_state = self.get_driver_state()
                if (
                    driver_state == DriverState.INITIALIZING
                    and self._init_start_time > 0
                ):
                    elapsed = time.time() - self._init_start_time
                    if elapsed > self._init_timeout:
                        reported = len(self.get_worker_heartbeat_times())
                        logger.error(
                            f"Worker init timeout spmd_id={self.spmd_id} "
                            f"elapsed={elapsed:.1f}s "
                            f"reported_workers={reported}/{self.num_workers}"
                        )
                        self.trigger_state_event(
                            DriverEvent.INIT_FAILED,
                            ExecutionFailure(
                                spmd_id=self.spmd_id,
                                rank=None,
                                op_name=None,
                                step=None,
                                phase="DRIVER_INITIALIZATION",
                                exception_type="WorkerInitializationTimeout",
                                message=(
                                    f"Worker initialization timed out after "
                                    f"{elapsed:.3f}s; "
                                    f"{reported}/"
                                    f"{self.num_workers} workers reported."
                                ),
                                traceback=None,
                            ),
                        )
            except Exception as e:
                logger.error(
                    f"Error sending heartbeat error={e} spmd_id={self.spmd_id}"
                )

            time.sleep(self._heartbeat_config.heartbeat_interval)

    def _send_heartbeat(self) -> None:
        if self.controller_handle is None:
            return

        healthy_count = 0
        current_time = time.time()
        timed_out_rank: int | None = None
        timed_out_elapsed = 0.0

        driver_state = self.get_driver_state()
        timeout_limit = (
            self._inactivity_timeout_s
            if driver_state is DriverState.SNAPSHOTTING
            else self._heartbeat_config.heartbeat_timeout
        )

        worker_times = self.get_worker_heartbeat_times()

        for rank in range(self.num_workers):
            last_time = worker_times.get(rank, 0)
            elapsed = current_time - last_time
            if elapsed < timeout_limit:
                healthy_count += 1
            elif timed_out_rank is None:
                timed_out_rank = rank
                timed_out_elapsed = elapsed

        if timed_out_rank is not None and driver_state is not DriverState.INITIALIZING:
            phase = (
                "CHECKPOINT"
                if driver_state is DriverState.SNAPSHOTTING
                else "WORKER_HEARTBEAT"
            )
            self.receive_worker_failure(
                timed_out_rank,
                ExecutionFailure(
                    spmd_id=self.spmd_id,
                    rank=timed_out_rank,
                    op_name=None,
                    step=None,
                    phase=phase,
                    exception_type="WorkerHeartbeatTimeout",
                    message=(
                        f"Worker heartbeat timed out after {timed_out_elapsed:.3f}s; "
                        f"limit is {timeout_limit:.3f}s."
                    ),
                    traceback=None,
                ),
            )
            driver_state = self.get_driver_state()

        if (
            driver_state == DriverState.SNAPSHOTTING
            and self.get_worker_heartbeats is not None
        ):
            worker_hbs = self.get_worker_heartbeats()
            for rank in range(self.num_workers):
                hb = worker_hbs.get(rank)
                if (
                    hb is not None
                    and hb.checkpoint_id is not None
                    and hb.checkpoint_progress_timestamp is not None
                    and hb.checkpoint_progress_timestamp > 0.0
                ):
                    inactivity = current_time - hb.checkpoint_progress_timestamp
                    if inactivity > self._inactivity_timeout_s:
                        self.receive_worker_failure(
                            rank,
                            ExecutionFailure(
                                spmd_id=self.spmd_id,
                                rank=rank,
                                op_name=None,
                                step=None,
                                phase="CHECKPOINT",
                                exception_type="WorkerCheckpointInactivityTimeout",
                                message=(
                                    f"Worker checkpoint activity stalled after {inactivity:.3f}s "
                                    f"during snapshot {hb.checkpoint_id}; "
                                    f"limit is {self._inactivity_timeout_s:.3f}s."
                                ),
                                traceback=None,
                            ),
                        )
                        driver_state = self.get_driver_state()
                        break

        if driver_state is DriverState.ERROR:
            status = DriverStatus.ERROR
        elif healthy_count < self.num_workers:
            status = DriverStatus.WARNING
        else:
            status = DriverStatus.HEALTHY

        heartbeat = DriverHeartbeat(
            spmd_id=self.spmd_id,
            timestamp=time.time(),
            status=status,
            state=driver_state,
            failure=(
                self.get_driver_failure() if driver_state is DriverState.ERROR else None
            ),
            num_workers=self.num_workers,
            healthy_num_workers=healthy_count,
            workers_ready=self.get_workers_ready(),
            topology_ready=self.get_topology_ready(),
        )

        try:
            self.controller_handle.receive_driver_heartbeat.remote(heartbeat)
            logger.heartbeat(
                f"Driver heartbeat sent status={status.name} "
                f"state={heartbeat.state.name} "
                f"healthy_workers={healthy_count}/{self.num_workers} "
                f"spmd_id={self.spmd_id}"
            )
        except Exception as e:
            logger.error(f"Failed to send heartbeat error={e} spmd_id={self.spmd_id}")
