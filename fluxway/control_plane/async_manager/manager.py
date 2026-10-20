from __future__ import annotations

import threading

from fluxway.common.types import AsyncStepConfig, Graph, SPMDOpStep


class AsyncStepManager:
    def __init__(self, graph: Graph, config: AsyncStepConfig | None = None):
        self._graph = graph
        self._config = config or AsyncStepConfig()
        self._spmd_steps: dict[str, SPMDOpStep] = {
            node_name: SPMDOpStep(spmd_id=node_name) for node_name in graph.nodes
        }
        self._operator_ids_by_spmd = {
            spmd_id: graph.require_compiled_graph().spmd(spmd_id).operator_ids
            for spmd_id in graph.require_compiled_graph().spmd_ids
        }
        self._lock = threading.Lock()

    def report_op_completed(
        self, spmd_id: str, op_name: str, completed_count: int
    ) -> None:
        with self._lock:
            if spmd_id not in self._spmd_steps:
                self._spmd_steps[spmd_id] = SPMDOpStep(spmd_id=spmd_id)
            spmd_progress = self._spmd_steps[spmd_id]
            spmd_progress.completed_step_count[op_name] = max(
                spmd_progress.completed_step_count.get(op_name, 0),
                completed_count,
            )

    def get_spmd_steps(self, spmd_id: str) -> SPMDOpStep | None:
        with self._lock:
            return self._spmd_steps.get(spmd_id)

    def get_all_spmd_steps(self) -> dict[str, SPMDOpStep]:
        with self._lock:
            return {
                key: SPMDOpStep(**value.as_dict())
                for key, value in self._spmd_steps.items()
            }

    def is_all_completed(self) -> bool:
        with self._lock:
            global_max_steps = self._config.max_steps
            for node_id, node in self._graph.nodes.items():
                node_max_steps = node.spmd_config.max_steps
                if node_max_steps is None:
                    node_max_steps = global_max_steps
                if node_max_steps is None:
                    return False
                spmd_progress = self._spmd_steps.get(node_id)
                if spmd_progress is None:
                    return False
                for op_name in self._operator_ids_by_spmd[node_id]:
                    if (
                        spmd_progress.completed_step_count.get(op_name, 0)
                        < node_max_steps
                    ):
                        return False
            return True

    def reset(self) -> None:
        with self._lock:
            for spmd_step in self._spmd_steps.values():
                spmd_step.completed_step_count.clear()
