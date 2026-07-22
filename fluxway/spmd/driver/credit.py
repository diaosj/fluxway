from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any

from fluxway.common.compiled_graph import CompiledDependency, CompiledGraph


@dataclass(frozen=True, slots=True)
class CreditUpdate:
    epoch: int
    sequence: int
    allowed_step: int


@dataclass(frozen=True, slots=True)
class CreditWindow:
    epoch: int
    op_name: str
    allowed_step: int


@dataclass(frozen=True, slots=True)
class CreditEdge:
    edge_id: str
    source_spmd_id: str
    source_op: str
    target_spmd_id: str
    target_op: str
    max_inflight_steps: int


class UpstreamCreditState:
    def __init__(self, edges: tuple[CreditEdge, ...], epoch: int):
        self._epoch = epoch
        self._edges_by_id = {edge.edge_id: edge for edge in edges}
        self._edge_ids_by_source_op: dict[str, list[str]] = defaultdict(list)
        self._updates_by_edge_id: dict[str, CreditUpdate] = {}
        for edge in edges:
            if edge.max_inflight_steps < 0:
                raise ValueError(
                    f"Credit edge {edge.edge_id!r} has negative max_inflight_steps="
                    f"{edge.max_inflight_steps}."
                )
            self._edge_ids_by_source_op[edge.source_op].append(edge.edge_id)
            self._updates_by_edge_id[edge.edge_id] = CreditUpdate(
                epoch=self._epoch,
                sequence=0,
                allowed_step=edge.max_inflight_steps,
            )

    def source_ops(self) -> tuple[str, ...]:
        return tuple(sorted(self._edge_ids_by_source_op))

    def receive(self, edge_id: str, update: CreditUpdate) -> bool:
        if edge_id not in self._edges_by_id:
            raise ValueError(f"Unknown upstream credit edge {edge_id!r}.")
        if update.epoch != self._epoch:
            raise ValueError(
                f"Credit update epoch mismatch for {edge_id!r}: "
                f"expected={self._epoch}, got={update.epoch}."
            )
        current = self._updates_by_edge_id[edge_id]
        if update.sequence <= current.sequence:
            return False
        if update.allowed_step < current.allowed_step:
            raise ValueError(
                f"Credit update cannot decrease allowed_step for {edge_id!r}: "
                f"current={current.allowed_step}, got={update.allowed_step}."
            )
        self._updates_by_edge_id[edge_id] = CreditUpdate(
            epoch=update.epoch,
            sequence=update.sequence,
            allowed_step=update.allowed_step,
        )
        return True

    def allowed_step(self, source_op: str) -> int:
        edge_ids = self._edge_ids_by_source_op.get(source_op)
        if not edge_ids:
            raise ValueError(
                f"No upstream credit edges registered for op {source_op!r}."
            )
        return min(
            self._updates_by_edge_id[edge_id].allowed_step for edge_id in edge_ids
        )

    def allowed_steps(self) -> dict[str, int]:
        return {
            source_op: self.allowed_step(source_op) for source_op in self.source_ops()
        }


def edge_credit_id(edge: CompiledDependency) -> str:
    source = edge.spec.source
    target = edge.spec.target
    source_op = source.op_id
    target_op = target.op_id
    if not source_op or not target_op:
        raise ValueError(
            "Flow-control credit edges require source_op and target_op: "
            f"{source.spmd_id}->{target.spmd_id} source_op={source_op!r} "
            f"target_op={target_op!r}."
        )
    return f"{source.spmd_id}.{source_op}->{target.spmd_id}.{target_op}"


def build_credit_edges(
    graph: CompiledGraph,
    async_step_config: Any,
) -> tuple[CreditEdge, ...]:
    credit_edges: list[CreditEdge] = []
    default_limit = int(getattr(async_step_config, "default_max_inflight_steps", 1))
    if default_limit < 0:
        raise ValueError(
            f"default_max_inflight_steps must be non-negative, got {default_limit}."
        )
    for dependency in graph.dependencies:
        edge = dependency.spec
        if edge.local_only:
            continue
        if not edge.source.op_id or not edge.target.op_id:
            raise ValueError(
                "Non-local flow-control edges require explicit source_op and target_op: "
                f"{edge.source.spmd_id}->{edge.target.spmd_id}."
            )
        limit = dependency.target_operator.spec.runtime_policy.max_inflight_steps
        if limit < 0:
            raise ValueError(
                f"Credit edge {edge_credit_id(dependency)!r} has negative max_inflight_steps={limit}."
            )
        credit_edges.append(
            CreditEdge(
                edge_id=edge_credit_id(dependency),
                source_spmd_id=edge.source.spmd_id,
                source_op=edge.source.op_id,
                target_spmd_id=edge.target.spmd_id,
                target_op=edge.target.op_id,
                max_inflight_steps=limit,
            )
        )
    return tuple(credit_edges)


def credit_topology_for_driver(
    spmd_id: str,
    all_credit_edges: tuple[CreditEdge, ...],
) -> tuple[tuple[CreditEdge, ...], tuple[CreditEdge, ...]]:
    upstream_edges = tuple(
        edge for edge in all_credit_edges if edge.source_spmd_id == spmd_id
    )
    downstream_edges = tuple(
        edge for edge in all_credit_edges if edge.target_spmd_id == spmd_id
    )
    return upstream_edges, downstream_edges


class CreditManager:
    def __init__(self, spmd_id: str, synchronizer: Any):
        self.spmd_id = spmd_id
        self.synchronizer = synchronizer
        self._credit_epoch = 0
        self._upstream_credit_state = UpstreamCreditState((), self._credit_epoch)
        self._upstream_credit_edges_by_id: dict[str, CreditEdge] = {}
        self._downstream_credit_edges_by_target_op: dict[str, list[CreditEdge]] = defaultdict(list)
        self._upstream_driver_handles: dict[str, Any] = {}
        self._controller_credit_windows_by_op: dict[str, int] = {}
        self._peer_credit_hints_by_op: dict[str, int] = {}
        self._credit_sequence_by_edge_id: dict[str, int] = {}
        self._last_sent_allowed_by_edge_id: dict[str, int] = {}

    def get_credit_epoch(self) -> int:
        return self._credit_epoch

    def register_credit_topology(
        self,
        upstream_driver_handles: dict[str, Any],
        upstream_edges: tuple[CreditEdge, ...],
        downstream_edges: tuple[CreditEdge, ...],
        epoch: int = 0,
    ) -> None:
        self._credit_epoch = epoch
        self._upstream_driver_handles = dict(upstream_driver_handles)
        self._upstream_credit_state = UpstreamCreditState(
            tuple(upstream_edges), self._credit_epoch
        )
        self._upstream_credit_edges_by_id = {
            edge.edge_id: edge for edge in upstream_edges
        }
        self._downstream_credit_edges_by_target_op = defaultdict(list)
        self._controller_credit_windows_by_op = {}
        self._peer_credit_hints_by_op = {}
        self._credit_sequence_by_edge_id = {}
        self._last_sent_allowed_by_edge_id = {}

        for edge in downstream_edges:
            handle = self._upstream_driver_handles.get(edge.source_spmd_id)
            if handle is None:
                raise ValueError(
                    f"Missing upstream driver handle for credit edge {edge.edge_id!r} "
                    f"on driver {self.spmd_id!r}."
                )
            self._downstream_credit_edges_by_target_op[edge.target_op].append(edge)
            self._credit_sequence_by_edge_id[edge.edge_id] = 0
            self._last_sent_allowed_by_edge_id[edge.edge_id] = edge.max_inflight_steps

        initial_allowed_steps = self._upstream_credit_state.allowed_steps()
        self._peer_credit_hints_by_op = dict(initial_allowed_steps)
        self._controller_credit_windows_by_op = dict(initial_allowed_steps)
        self._apply_effective_credit_windows(tuple(initial_allowed_steps))

    def apply_credit_window(self, windows: tuple[CreditWindow, ...]) -> None:
        changed_ops: set[str] = set()
        for window in windows:
            if window.epoch != self._credit_epoch:
                raise ValueError(
                    f"Credit window epoch mismatch for {self.spmd_id}.{window.op_name}: "
                    f"expected={self._credit_epoch}, got={window.epoch}."
                )
            allowed_step = window.allowed_step
            if allowed_step < 0:
                raise ValueError(
                    f"Credit window cannot be negative for {self.spmd_id}.{window.op_name}: "
                    f"{allowed_step}."
                )
            if window.op_name not in self._peer_credit_hints_by_op:
                raise ValueError(
                    f"Driver {self.spmd_id!r} received controller credit window "
                    f"for unknown upstream op {window.op_name!r}."
                )
            current = self._controller_credit_windows_by_op.get(window.op_name)
            if current is not None and allowed_step <= current:
                continue
            if current != allowed_step:
                self._controller_credit_windows_by_op[window.op_name] = allowed_step
                changed_ops.add(window.op_name)
        self._apply_effective_credit_windows(changed_ops)

    def receive_credit_update(
        self, downstream_id: str, edge_id: str, update: CreditUpdate
    ) -> None:
        edge = self._upstream_credit_edges_by_id.get(edge_id)
        if edge is None:
            raise ValueError(
                f"Driver {self.spmd_id!r} received unknown credit edge {edge_id!r}."
            )
        if edge.target_spmd_id != downstream_id:
            raise ValueError(
                f"Credit edge {edge_id!r} downstream mismatch: "
                f"expected={edge.target_spmd_id!r}, got={downstream_id!r}."
            )
        if self._upstream_credit_state.receive(edge_id, update):
            self._peer_credit_hints_by_op[edge.source_op] = (
                self._upstream_credit_state.allowed_step(edge.source_op)
            )
            self._apply_effective_credit_windows((edge.source_op,))

    def _apply_effective_credit_windows(
        self, op_names: tuple[str, ...] | set[str]
    ) -> None:
        for op_name in op_names:
            controller_allowed = self._controller_credit_windows_by_op.get(op_name)
            peer_allowed = self._peer_credit_hints_by_op.get(op_name)
            if controller_allowed is None:
                raise RuntimeError(
                    f"Driver {self.spmd_id!r} has no controller credit window "
                    f"for op {op_name!r}."
                )
            if peer_allowed is None:
                raise RuntimeError(
                    f"Driver {self.spmd_id!r} has no peer credit hint "
                    f"for op {op_name!r}."
                )
            self.synchronizer.update_allowed_step(
                op_name, min(controller_allowed, peer_allowed)
            )

    def send_credit_updates(self, op_name: str, completed_count: int) -> None:
        edges = self._downstream_credit_edges_by_target_op.get(op_name, ())
        for edge in edges:
            allowed_step = completed_count + edge.max_inflight_steps
            last_allowed = self._last_sent_allowed_by_edge_id[edge.edge_id]
            if allowed_step <= last_allowed:
                continue
            handle = self._upstream_driver_handles.get(edge.source_spmd_id)
            if handle is None:
                raise ValueError(
                    f"Missing upstream driver handle for credit edge {edge.edge_id!r} "
                    f"on driver {self.spmd_id!r}."
                )
            sequence = self._credit_sequence_by_edge_id[edge.edge_id] + 1
            self._credit_sequence_by_edge_id[edge.edge_id] = sequence
            self._last_sent_allowed_by_edge_id[edge.edge_id] = allowed_step
            handle.receive_credit_update.remote(
                self.spmd_id,
                edge.edge_id,
                CreditUpdate(
                    epoch=self._credit_epoch,
                    sequence=sequence,
                    allowed_step=allowed_step,
                ),
            )
