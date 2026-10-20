from __future__ import annotations

import threading
from collections import defaultdict

from fluxway.common.types import AsyncStepConfig, FlowControlState, Graph
from fluxway.spmd.driver.credit import (
    CreditEdge,
    CreditWindow,
    build_credit_edges,
    credit_topology_for_driver,
)


class FlowControlManager:
    def __init__(self, graph: Graph, config: AsyncStepConfig | None = None):
        self._epoch = 0
        self._credit_edges = build_credit_edges(
            graph.require_compiled_graph(),
            config or AsyncStepConfig(),
        )
        self._edges_by_target_op: dict[tuple[str, str], list[CreditEdge]] = defaultdict(
            list
        )
        self._allowed_by_edge_id: dict[str, int] = {}
        self._states_by_spmd_id: dict[str, FlowControlState] = {}
        self._lock = threading.Lock()

        for edge in self._credit_edges:
            self._edges_by_target_op[(edge.target_spmd_id, edge.target_op)].append(edge)
            self._allowed_by_edge_id[edge.edge_id] = edge.max_inflight_steps

    @property
    def epoch(self) -> int:
        return self._epoch

    def credit_edges(self) -> tuple[CreditEdge, ...]:
        return self._credit_edges

    def topology_for_driver(
        self, spmd_id: str
    ) -> tuple[tuple[CreditEdge, ...], tuple[CreditEdge, ...]]:
        return credit_topology_for_driver(spmd_id, self._credit_edges)

    def initial_windows_for_driver(self, spmd_id: str) -> tuple[CreditWindow, ...]:
        with self._lock:
            return self._windows_for_driver_locked(spmd_id)

    def observe_op_completed(
        self, spmd_id: str, op_name: str, completed_count: int
    ) -> dict[str, tuple[CreditWindow, ...]]:
        if completed_count < 0:
            raise ValueError(
                f"completed_count must be non-negative for {spmd_id}.{op_name}, "
                f"got {completed_count}."
            )

        changed_source_spmds: set[str] = set()
        with self._lock:
            for edge in self._edges_by_target_op.get((spmd_id, op_name), ()):
                allowed_step = completed_count + edge.max_inflight_steps
                current = self._allowed_by_edge_id[edge.edge_id]
                if allowed_step <= current:
                    continue
                self._allowed_by_edge_id[edge.edge_id] = allowed_step
                changed_source_spmds.add(edge.source_spmd_id)

            windows = {
                source_spmd_id: self._windows_for_driver_locked(source_spmd_id)
                for source_spmd_id in sorted(changed_source_spmds)
            }
            return windows

    def report_state(self, state: FlowControlState) -> None:
        if state.epoch != self._epoch:
            raise ValueError(
                f"Flow-control state epoch mismatch for {state.spmd_id!r}: "
                f"expected={self._epoch}, got={state.epoch}."
            )
        with self._lock:
            self._states_by_spmd_id[state.spmd_id] = state

    def state_for_driver(self, spmd_id: str) -> FlowControlState | None:
        with self._lock:
            return self._states_by_spmd_id.get(spmd_id)

    def _windows_for_driver_locked(self, spmd_id: str) -> tuple[CreditWindow, ...]:
        allowed_by_op: dict[str, int] = {}
        for edge in self._credit_edges:
            if edge.source_spmd_id != spmd_id:
                continue
            allowed = self._allowed_by_edge_id[edge.edge_id]
            current = allowed_by_op.get(edge.source_op)
            if current is None or allowed < current:
                allowed_by_op[edge.source_op] = allowed

        return tuple(
            CreditWindow(
                epoch=self._epoch,
                op_name=op_name,
                allowed_step=allowed_step,
            )
            for op_name, allowed_step in sorted(allowed_by_op.items())
        )
