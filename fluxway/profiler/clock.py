from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

from fluxway.profiler.errors import ClockSyncUnstable, MissingAxonNativeHlc
from fluxway.profiler.record import TraceCarrier, TraceClockSource, merge_hlc


@dataclass(frozen=True, slots=True)
class ClockCalibration:
    monotonic_to_epoch_offset_ns: int
    uncertainty_ns: int

    @classmethod
    def measure(
        cls,
        *,
        rounds: int,
        max_uncertainty_ns: int,
    ) -> ClockCalibration:
        if rounds <= 0:
            raise ValueError("clock calibration rounds must be positive.")
        best_offset = 0
        best_uncertainty = 1 << 62
        best_start = 0
        best_end = 0
        for _ in range(rounds):
            start = time.perf_counter_ns()
            epoch = time.time_ns()
            end = time.perf_counter_ns()
            midpoint = (start + end) // 2
            uncertainty = (end - start) // 2
            if uncertainty < best_uncertainty:
                best_offset = epoch - midpoint
                best_uncertainty = uncertainty
                best_start = start
                best_end = end
        if best_uncertainty > max_uncertainty_ns:
            raise ClockSyncUnstable(
                sender_ts_ns=best_start,
                receiver_ts_ns=best_end,
                max_uncertainty_ns=max_uncertainty_ns,
            )
        return cls(
            monotonic_to_epoch_offset_ns=best_offset,
            uncertainty_ns=best_uncertainty,
        )


class RayFluxwayHlcBridge:
    def make_send_carrier(self, carrier: TraceCarrier, *parts: object) -> TraceCarrier:
        return carrier.child(*parts)

    def receive(self, carrier: TraceCarrier) -> TraceCarrier:
        physical, logical = merge_hlc(time.perf_counter_ns(), 0, carrier)
        return TraceCarrier(
            run_id_hash=carrier.run_id_hash,
            flow_id=carrier.flow_id,
            task_id=carrier.task_id,
            parent_task_id=carrier.parent_task_id,
            sender_local_ts_ns=carrier.sender_local_ts_ns,
            sender_hlc_physical_ns=carrier.sender_hlc_physical_ns,
            sender_hlc_logical=carrier.sender_hlc_logical,
            receiver_hlc_physical_ns=physical,
            receiver_hlc_logical=logical,
            clock_source=int(TraceClockSource.FLUXWAY_RAY),
        )


class AxonNativeHlcBridge:
    def receive_result(self, carrier: TraceCarrier, result: Any) -> TraceCarrier:
        physical = getattr(result, "native_hlc_physical_ns", None)
        logical = getattr(result, "native_hlc_logical", None)
        if physical is None or logical is None:
            metadata = getattr(result, "metadata", None)
            if metadata is not None:
                physical = getattr(metadata, "native_hlc_physical_ns", physical)
                logical = getattr(metadata, "native_hlc_logical", logical)
        if physical is None or logical is None or int(physical) <= 0:
            raise MissingAxonNativeHlc(
                result_type=type(result).__qualname__,
                has_metadata=metadata is not None,
            )
        return TraceCarrier(
            run_id_hash=carrier.run_id_hash,
            flow_id=carrier.flow_id,
            task_id=carrier.task_id,
            parent_task_id=carrier.parent_task_id,
            sender_local_ts_ns=carrier.sender_local_ts_ns,
            sender_hlc_physical_ns=carrier.sender_hlc_physical_ns,
            sender_hlc_logical=carrier.sender_hlc_logical,
            receiver_hlc_physical_ns=int(physical),
            receiver_hlc_logical=int(logical),
            clock_source=int(TraceClockSource.AXON_NATIVE),
        )
