from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any

import numpy as np
import ray


def make_payload(
    dp_rank: int,
    step: int,
    table: int,
    ps_rank: int,
    tokens_per_dp: int,
    ps_count: int,
    dim: int,
) -> np.ndarray:
    token_ids = np.arange(tokens_per_dp, dtype=np.int64)
    routed = token_ids[token_ids % ps_count == ps_rank]
    base = (
        step * 100_000 + dp_rank * 10_000 + table * 1_000 + ps_rank * 100 + routed
    ).astype(np.float32)
    cols = (np.arange(dim, dtype=np.float32) + 1.0) / 4096.0
    return np.ascontiguousarray(base.reshape(-1, 1) + cols.reshape(1, -1))


def native_sender_compute(tokens_per_dp: int, dim: int) -> float:
    values = np.arange(tokens_per_dp * dim, dtype=np.float32).reshape(
        tokens_per_dp, dim
    )
    return float(np.square(values, dtype=np.float32).sum(dtype=np.float64))


def frame_payloads(
    dp_rank: int,
    step: int,
    ps_rank: int,
    tokens_per_dp: int,
    ps_count: int,
    table_count: int,
    dim: int,
) -> tuple[float, int, tuple[np.ndarray, ...]]:
    payloads = tuple(
        make_payload(dp_rank, step, table, ps_rank, tokens_per_dp, ps_count, dim)
        for table in range(table_count)
    )
    checksum = float(sum(p.sum(dtype=np.float64) for p in payloads))
    token_rows = int(sum(p.shape[0] for p in payloads))
    return checksum, token_rows, payloads


@ray.remote
def receiver_assemble_compute(
    step: int,
    ps_rank: int,
    edge_count: int,
    token_rows: int,
    expected_checksum: float,
    receiver_compute_rounds: int,
    *payload_segments: np.ndarray,
) -> dict[str, Any]:
    start_ns = time.perf_counter_ns()
    if edge_count != len(payload_segments):
        raise ValueError(
            f"edge_count={edge_count} does not match segment count={len(payload_segments)}."
        )
    checksum = float(sum(segment.sum(dtype=np.float64) for segment in payload_segments))
    if not np.isclose(checksum, expected_checksum, rtol=1e-6, atol=1e-3):
        raise ValueError(
            f"checksum mismatch step={step} ps_rank={ps_rank}: "
            f"got {checksum}, expected {expected_checksum}."
        )
    accumulator = checksum
    for _ in range(receiver_compute_rounds):
        accumulator += checksum * 1e-12
    end_ns = time.perf_counter_ns()
    return {
        "step": step,
        "ps_rank": ps_rank,
        "edge_count": edge_count,
        "token_rows": token_rows,
        "checksum": checksum,
        "accumulator": accumulator,
        "duration_s": (end_ns - start_ns) / 1_000_000_000.0,
    }


@ray.remote(max_concurrency=128)  # type: ignore
class SparseParameterServer:
    def __init__(
        self,
        ps_rank: int,
        dp_count: int,
        table_count: int,
        receiver_compute_rounds: int,
    ) -> None:
        self.ps_rank = ps_rank
        self.dp_count = dp_count
        self.table_count = table_count
        self.receiver_compute_rounds = receiver_compute_rounds
        self.rpc_count = 0
        self.edge_count = 0
        self.step_slots: dict[int, dict[str, Any]] = {}
        self.done_refs: list[Any] = []

    async def ping(self) -> int:
        return self.rpc_count

    async def deposit_frame(
        self,
        step: int,
        sender_rank: int,
        sender_world_size: int,
        receiver_rank: int,
        frame_edge_count: int,
        token_rows: int,
        payload_checksum: float,
        *payload_segments: np.ndarray,
    ) -> None:
        if sender_world_size != self.dp_count:
            raise ValueError(
                f"sender_world_size={sender_world_size} does not match dp_count={self.dp_count}."
            )
        if receiver_rank != self.ps_rank:
            raise ValueError(
                f"receiver_rank={receiver_rank} does not match ps_rank={self.ps_rank}."
            )
        if frame_edge_count != self.table_count:
            raise ValueError(
                f"frame_edge_count={frame_edge_count} does not match table_count={self.table_count}."
            )
        if frame_edge_count != len(payload_segments):
            raise ValueError(
                f"frame_edge_count={frame_edge_count} does not match payload segments={len(payload_segments)}."
            )
        slot = self.step_slots.setdefault(
            step,
            {
                "senders": set(),
                "segments": [],
                "edge_count": 0,
                "token_rows": 0,
                "checksum": 0.0,
            },
        )
        senders = slot["senders"]
        if sender_rank in senders:
            raise ValueError(
                f"duplicate sender_rank={sender_rank} for step={step} ps_rank={self.ps_rank}."
            )
        senders.add(sender_rank)
        slot["segments"].extend(payload_segments)
        slot["edge_count"] += frame_edge_count
        slot["token_rows"] += token_rows
        slot["checksum"] += payload_checksum
        self.rpc_count += 1
        self.edge_count += frame_edge_count

        if len(senders) == self.dp_count:
            segments = tuple(slot["segments"])
            ref = receiver_assemble_compute.remote(
                step,
                self.ps_rank,
                int(slot["edge_count"]),
                int(slot["token_rows"]),
                float(slot["checksum"]),
                self.receiver_compute_rounds,
                *segments,
            )
            self.done_refs.append(ref)

    def summary(self) -> dict[str, Any]:
        if self.done_refs:
            done = ray.get(self.done_refs)
        else:
            done = []
        checksum = float(sum(item["checksum"] for item in done))
        latency_values = [float(item["duration_s"]) for item in done]
        return {
            "ps_rank": self.ps_rank,
            "rpc_count": self.rpc_count,
            "edge_count": self.edge_count,
            "checksum": checksum,
            "assembled_steps": len(done),
            "token_rows": int(sum(item["token_rows"] for item in done)),
            "receiver_step_latency_p50_s": statistics.median(latency_values)
            if latency_values
            else 0.0,
        }


@ray.remote
def send_target_frame(
    dp_rank: int,
    step: int,
    ps_rank: int,
    ps_actor: Any,
    dp_count: int,
    ps_count: int,
    table_count: int,
    tokens_per_dp: int,
    dim: int,
    sender_compute_rounds: int,
) -> dict[str, Any]:
    compute_checksum = 0.0
    for _ in range(sender_compute_rounds):
        compute_checksum += native_sender_compute(tokens_per_dp, dim)
    checksum, token_rows, payloads = frame_payloads(
        dp_rank, step, ps_rank, tokens_per_dp, ps_count, table_count, dim
    )
    ray.get(
        ps_actor.deposit_frame.remote(
            step,
            dp_rank,
            dp_count,
            ps_rank,
            table_count,
            token_rows,
            checksum,
            *payloads,
        )
    )
    return {
        "rpc_count": 1,
        "edge_count": table_count,
        "token_rows": token_rows,
        "checksum": checksum,
        "compute_checksum": compute_checksum,
    }


@ray.remote
def send_successor_frames(
    dp_rank: int,
    step: int,
    ps_actors: list[Any],
    dp_count: int,
    ps_count: int,
    table_count: int,
    tokens_per_dp: int,
    dim: int,
    sender_compute_rounds: int,
) -> dict[str, Any]:
    compute_checksum = 0.0
    for _ in range(sender_compute_rounds):
        compute_checksum += native_sender_compute(tokens_per_dp, dim)
    refs = []
    rpc_count = 0
    edge_count = 0
    token_rows_total = 0
    checksum_total = 0.0
    for ps_rank, ps_actor in enumerate(ps_actors):
        checksum, token_rows, payloads = frame_payloads(
            dp_rank, step, ps_rank, tokens_per_dp, ps_count, table_count, dim
        )
        refs.append(
            ps_actor.deposit_frame.remote(
                step,
                dp_rank,
                dp_count,
                ps_rank,
                table_count,
                token_rows,
                checksum,
                *payloads,
            )
        )
        rpc_count += 1
        edge_count += table_count
        token_rows_total += token_rows
        checksum_total += checksum
    ray.get(refs)
    return {
        "rpc_count": rpc_count,
        "edge_count": edge_count,
        "token_rows": token_rows_total,
        "checksum": checksum_total,
        "compute_checksum": compute_checksum,
    }


@ray.remote
def send_step_frames(
    step: int,
    ps_actors: list[Any],
    dp_count: int,
    ps_count: int,
    table_count: int,
    tokens_per_dp: int,
    dim: int,
    sender_compute_rounds: int,
) -> dict[str, Any]:
    refs = []
    rpc_count = 0
    edge_count = 0
    token_rows_total = 0
    checksum_total = 0.0
    compute_checksum = 0.0
    for dp_rank in range(dp_count):
        for _ in range(sender_compute_rounds):
            compute_checksum += native_sender_compute(tokens_per_dp, dim)
        for ps_rank, ps_actor in enumerate(ps_actors):
            checksum, token_rows, payloads = frame_payloads(
                dp_rank, step, ps_rank, tokens_per_dp, ps_count, table_count, dim
            )
            refs.append(
                ps_actor.deposit_frame.remote(
                    step,
                    dp_rank,
                    dp_count,
                    ps_rank,
                    table_count,
                    token_rows,
                    checksum,
                    *payloads,
                )
            )
            rpc_count += 1
            edge_count += table_count
            token_rows_total += token_rows
            checksum_total += checksum
    ray.get(refs)
    return {
        "rpc_count": rpc_count,
        "edge_count": edge_count,
        "token_rows": token_rows_total,
        "checksum": checksum_total,
        "compute_checksum": compute_checksum,
    }


def launch_workload(
    args: argparse.Namespace, mode: str, ps_actors: list[Any]
) -> list[Any]:
    refs: list[Any] = []
    if mode == "per-target":
        for step in range(args.steps):
            for dp_rank in range(args.dp):
                for ps_rank, ps_actor in enumerate(ps_actors):
                    refs.append(
                        send_target_frame.remote(
                            dp_rank,
                            step,
                            ps_rank,
                            ps_actor,
                            args.dp,
                            args.ps,
                            args.tables,
                            args.tokens_per_dp,
                            args.dim,
                            args.sender_compute_rounds,
                        )
                    )
    elif mode == "per-successor":
        for step in range(args.steps):
            for dp_rank in range(args.dp):
                refs.append(
                    send_successor_frames.remote(
                        dp_rank,
                        step,
                        ps_actors,
                        args.dp,
                        args.ps,
                        args.tables,
                        args.tokens_per_dp,
                        args.dim,
                        args.sender_compute_rounds,
                    )
                )
    elif mode == "per-step":
        for step in range(args.steps):
            refs.append(
                send_step_frames.remote(
                    step,
                    ps_actors,
                    args.dp,
                    args.ps,
                    args.tables,
                    args.tokens_per_dp,
                    args.dim,
                    args.sender_compute_rounds,
                )
            )
    else:
        raise ValueError(f"unsupported mode {mode!r}.")
    return refs


def wait_with_responsiveness(
    refs: list[Any], ps_actors: list[Any]
) -> tuple[list[Any], dict[str, float]]:
    pending = list(refs)
    completed: list[Any] = []
    latencies = []
    while pending:
        probe_start = time.perf_counter()
        ray.get([actor.ping.remote() for actor in ps_actors])
        latencies.append(time.perf_counter() - probe_start)
        done, pending = ray.wait(pending, num_returns=1, timeout=0.01)
        if done:
            completed.extend(ray.get(done))
    if not latencies:
        raise RuntimeError("event-loop responsiveness probe did not run.")
    return completed, {
        "probe_count": float(len(latencies)),
        "event_loop_ping_p50_ms": statistics.median(latencies) * 1000.0,
        "event_loop_ping_max_ms": max(latencies) * 1000.0,
    }


def run_mode(args: argparse.Namespace, mode: str) -> dict[str, Any]:
    ps_actors = [
        SparseParameterServer.remote(  # type: ignore[attr-defined]
            rank, args.dp, args.tables, args.receiver_compute_rounds
        )
        for rank in range(args.ps)
    ]
    start = time.perf_counter()
    refs = launch_workload(args, mode, ps_actors)
    send_results, responsiveness = wait_with_responsiveness(refs, ps_actors)
    ps_summaries = ray.get([actor.summary.remote() for actor in ps_actors])
    elapsed = time.perf_counter() - start

    send_rpc_count = sum(int(item["rpc_count"]) for item in send_results)
    send_edge_count = sum(int(item["edge_count"]) for item in send_results)
    recv_rpc_count = sum(int(item["rpc_count"]) for item in ps_summaries)
    recv_edge_count = sum(int(item["edge_count"]) for item in ps_summaries)
    checksum_sent = float(sum(float(item["checksum"]) for item in send_results))
    checksum_received = float(sum(float(item["checksum"]) for item in ps_summaries))
    token_rows = sum(int(item["token_rows"]) for item in send_results)
    expected_rpc_count = args.steps * args.dp * args.ps
    expected_edge_count = expected_rpc_count * args.tables
    expected_token_rows = args.steps * args.dp * args.tables * args.tokens_per_dp

    if send_rpc_count != expected_rpc_count or recv_rpc_count != expected_rpc_count:
        raise AssertionError(
            f"{mode} rpc_count mismatch: send={send_rpc_count} recv={recv_rpc_count} "
            f"expected={expected_rpc_count}."
        )
    if send_edge_count != expected_edge_count or recv_edge_count != expected_edge_count:
        raise AssertionError(
            f"{mode} edge_count mismatch: send={send_edge_count} recv={recv_edge_count} "
            f"expected={expected_edge_count}."
        )
    if args.tables > 1 and send_rpc_count == expected_edge_count:
        raise AssertionError(f"{mode} regressed to per-tensor RPC.")
    if token_rows != expected_token_rows:
        raise AssertionError(
            f"{mode} token rows mismatch: got={token_rows} expected={expected_token_rows}."
        )
    if not np.isclose(checksum_sent, checksum_received, rtol=1e-6, atol=1e-3):
        raise AssertionError(
            f"{mode} checksum mismatch: sent={checksum_sent} received={checksum_received}."
        )

    ps_latency = [
        float(item["receiver_step_latency_p50_s"])
        for item in ps_summaries
        if float(item["receiver_step_latency_p50_s"]) > 0.0
    ]
    return {
        "mode": mode,
        "elapsed_s": elapsed,
        "tokens_per_second": expected_token_rows / elapsed,
        "step_latency_s": elapsed / args.steps,
        "receiver_step_latency_p50_s": statistics.median(ps_latency)
        if ps_latency
        else 0.0,
        "rpc_count": send_rpc_count,
        "edge_count": send_edge_count,
        "checksum": checksum_received,
        "event_loop_responsiveness": responsiveness,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", type=int, default=4)
    parser.add_argument("--ps", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--tables", type=int, default=4)
    parser.add_argument("--tokens-per-dp", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=8)
    parser.add_argument("--sender-compute-rounds", type=int, default=1)
    parser.add_argument("--receiver-compute-rounds", type=int, default=1000)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["per-target", "per-successor", "per-step"],
        choices=["per-target", "per-successor", "per-step"],
    )
    parser.add_argument("--json-output", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dp <= 0 or args.ps <= 0 or args.steps <= 0:
        raise ValueError("dp, ps, and steps must be positive.")
    if args.tokens_per_dp % args.ps != 0:
        raise ValueError("tokens_per_dp must be divisible by ps.")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)
    try:
        results = [run_mode(args, mode) for mode in args.modes]
    finally:
        ray.shutdown()

    winner = max(results, key=lambda item: float(item["tokens_per_second"]))
    payload = {"results": results, "winner": winner["mode"]}
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.json_output:
        with open(args.json_output, "w", encoding="utf-8") as handle:
            handle.write(text + "\n")


if __name__ == "__main__":
    main()
