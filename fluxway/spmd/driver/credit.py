from __future__ import annotations

from dataclasses import dataclass
from collections import defaultdict
from typing import Any


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


def edge_credit_id(edge: Any) -> str:
    source_op = edge.source_op
    target_op = edge.target_op
    if not source_op or not target_op:
        raise ValueError(
            "Flow-control credit edges require source_op and target_op: "
            f"{edge.source_node}->{edge.target_node} source_op={source_op!r} "
            f"target_op={target_op!r}."
        )
    return f"{edge.source_node}.{source_op}->{edge.target_node}.{target_op}"


def build_credit_edges(graph: Any, async_step_config: Any) -> tuple[CreditEdge, ...]:
    credit_edges: list[CreditEdge] = []
    default_limit = int(getattr(async_step_config, "default_max_inflight_steps", 1))
    if default_limit < 0:
        raise ValueError(
            f"default_max_inflight_steps must be non-negative, got {default_limit}."
        )
    for edge in graph.edges:
        if getattr(edge, "local_only", False):
            continue
        if not edge.source_op or not edge.target_op:
            raise ValueError(
                "Non-local flow-control edges require explicit source_op and target_op: "
                f"{edge.source_node}->{edge.target_node}."
            )
        limit = default_limit
        target_node = graph.nodes.get(edge.target_node)
        if target_node is not None and target_node.spmd_config is not None:
            meta = target_node.spmd_config.operator_metas.get(edge.target_op)
            if meta is not None:
                limit = int(getattr(meta, "max_inflight_steps", limit))
        if limit < 0:
            raise ValueError(
                f"Credit edge {edge_credit_id(edge)!r} has negative max_inflight_steps={limit}."
            )
        credit_edges.append(
            CreditEdge(
                edge_id=edge_credit_id(edge),
                source_spmd_id=edge.source_node,
                source_op=edge.source_op,
                target_spmd_id=edge.target_node,
                target_op=edge.target_op,
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
