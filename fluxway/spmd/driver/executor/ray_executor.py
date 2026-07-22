import asyncio
from typing import Any
from collections.abc import Mapping

from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fluxway.common.types import SPMDConfig
from fluxway.spmd.worker import SPMDWorker
from fluxway.placement import PlacementAttestationError
from fluxway.placement.worker_group import (
    attest_node_ids,
    ray_actor_options,
    worker_tensor_device,
)
from fluxway.common.utils.logging import get_logger
from fluxway.spmd.driver.executor.base import AbstractWorkerExecutor

logger = get_logger(__name__)


class RayWorkerExecutor(AbstractWorkerExecutor):
    """Ray-based implementation of the Worker Executor."""

    def __init__(
        self,
        config: SPMDConfig,
        placement_group: Any | None,
        worker_bundle_indices: list[int],
        worker_resource: Mapping[str, Any],
        expected_node_ids: list[str],
        checkpoint_backend: Any | None = None,
    ):
        self.config = config
        self.placement_group = placement_group
        self.worker_bundle_indices = worker_bundle_indices
        self.worker_resource = worker_resource
        self.expected_node_ids = expected_node_ids
        self._checkpoint_backend = checkpoint_backend
        self._runtime_env = config.runtime_env
        self._worker_handles: dict[int, Any] = {}
        self._attested_node_ids: tuple[str, ...] | None = None
        self.spmd_id = None

    def _build_worker_opts(self, bundle_idx: int, name: str) -> dict[str, Any]:
        """Build .options() kwargs for a worker actor."""
        opts: dict[str, Any] = {"name": name}
        if self.placement_group is not None:
            opts["scheduling_strategy"] = PlacementGroupSchedulingStrategy(
                placement_group=self.placement_group,
                placement_group_bundle_index=bundle_idx,
                placement_group_capture_child_tasks=False,
            )
        if self._runtime_env:
            opts["runtime_env"] = self._runtime_env

        opts.update(ray_actor_options(self.worker_resource))
        logger.debug(f"Worker opts for {name}: {opts}")
        return opts

    def _compute_rank_info(self, node_ids: dict[int, str]) -> dict[int, dict[str, int]]:
        from collections import defaultdict

        node_groups: dict[str, list[int]] = defaultdict(list)
        for rank, node_id in node_ids.items():
            node_groups[node_id].append(rank)

        for ranks in node_groups.values():
            ranks.sort()

        sorted_nodes = sorted(node_groups, key=lambda node_id: node_groups[node_id][0])
        node_to_rank = {
            node_id: node_rank for node_rank, node_id in enumerate(sorted_nodes)
        }

        return {
            rank: {
                "local_rank": local_rank,
                "local_world_size": len(ranks),
                "node_rank": node_to_rank[node_id],
                "num_nodes": len(sorted_nodes),
            }
            for node_id, ranks in node_groups.items()
            for local_rank, rank in enumerate(ranks)
        }

    def get_attested_node_ids(self) -> tuple[str, ...]:
        if self._attested_node_ids is None:
            raise PlacementAttestationError(
                f"SPMD {self.spmd_id!r} has not attested worker node IDs"
            )
        return self._attested_node_ids

    def get_worker_handles(self) -> dict[int, Any]:
        return dict(self._worker_handles)

    async def start_workers(
        self, spmd_id: str, driver_handle: Any, controller_handle: Any
    ) -> None:
        self.spmd_id = spmd_id
        for rank, bundle_idx in enumerate(self.worker_bundle_indices):
            opts = self._build_worker_opts(
                bundle_idx, f"spmd_worker_{self.spmd_id}_{rank}"
            )
            worker = SPMDWorker.options(**opts).remote(  # type: ignore[attr-defined]
                config=self.config,
                rank=rank,
                spmd_id=self.spmd_id,
                driver_handle=driver_handle,
                checkpoint_backend=self._checkpoint_backend,
                tensor_device=worker_tensor_device(self.worker_resource),
                control_plane_handle=controller_handle,
            )
            self._worker_handles[rank] = worker
            logger.debug(f"Started worker rank={rank} spmd_id={self.spmd_id}")

        node_id_refs = {
            rank: worker.get_node_id.remote()
            for rank, worker in self._worker_handles.items()
        }
        ranks = list(node_id_refs.keys())
        node_id_values = await asyncio.gather(*node_id_refs.values())
        node_ids = dict(zip(ranks, node_id_values, strict=True))
        actual = tuple(node_ids[rank] for rank in range(len(node_ids)))
        self._attested_node_ids = actual
        expected = tuple(self.expected_node_ids)
        attest_node_ids(self.spmd_id, expected, actual)
        logger.info(
            "placement.worker.attested spmd_id=%s node_ids=%s",
            self.spmd_id,
            actual,
        )

        rank_info = self._compute_rank_info(node_ids)

        if 0 in self._worker_handles:
            rank0_ip = await self._worker_handles[0].get_node_ip.remote()
            if rank_info[0]["num_nodes"] == 1:
                rank0_ip = "127.0.0.1"

            max_retries = 5
            for attempt in range(max_retries):
                rank0_port = await self._worker_handles[0].find_free_port.remote()
                rank0_address = f"{rank0_ip}:{rank0_port}"
                try:
                    await asyncio.gather(
                        *[
                            worker.setup_and_start.remote(
                                **rank_info[rank],
                                rank0_address=rank0_address,
                            )
                            for rank, worker in self._worker_handles.items()
                        ]
                    )
                    break
                except Exception as e:
                    if (
                        "already in use" in str(e).lower()
                        or "address already in use" in str(e).lower()
                        or "eaddrinuse" in str(e).lower()
                    ):
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.1 * (2**attempt))
                            continue
                    raise
        else:
            rank0_address = None
            if self._worker_handles:
                await asyncio.gather(
                    *[
                        worker.setup_and_start.remote(
                            **rank_info[rank],
                            rank0_address=rank0_address,
                        )
                        for rank, worker in self._worker_handles.items()
                    ]
                )

    async def stop_workers(self) -> list[Exception]:
        shutdown_errors: list[Exception] = []
        stop_refs: list[tuple[int, Any]] = []
        for rank, worker in self._worker_handles.items():
            try:
                stop_refs.append((rank, worker.stop.remote()))
            except Exception as e:
                shutdown_errors.append(e)

        worker_handles = list(self._worker_handles.items())
        if stop_refs:
            results = await asyncio.gather(
                *(ref for _, ref in stop_refs), return_exceptions=True
            )
            for (rank, _), result in zip(stop_refs, results, strict=True):
                if isinstance(result, Exception):
                    shutdown_errors.append(result)

        detach_refs: list[tuple[int, Any]] = []
        for rank, worker in worker_handles:
            try:
                detach_refs.append((rank, worker.set_driver_handle.remote(None)))
            except Exception as e:
                shutdown_errors.append(e)

        if detach_refs:
            detach_results = await asyncio.gather(
                *(ref for _, ref in detach_refs), return_exceptions=True
            )
            for (rank, _), result in zip(detach_refs, detach_results, strict=True):
                if isinstance(result, Exception):
                    shutdown_errors.append(result)

        self._worker_handles.clear()
        return shutdown_errors

    async def inspect_operator_state(self) -> dict[int, Any]:
        tasks = {
            rank: worker.inspect_operator_state.remote()
            for rank, worker in self._worker_handles.items()
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values())
        return dict(zip(tasks.keys(), results, strict=True))

    async def export_profiler_snapshots(self) -> dict[Any, Any]:
        tasks = {
            rank: worker.export_profiler_snapshot.remote()
            for rank, worker in self._worker_handles.items()
        }
        if not tasks:
            return {}
        results = await asyncio.gather(*tasks.values())
        return {
            rank: snapshot
            for rank, snapshot in zip(tasks.keys(), results, strict=True)
            if snapshot is not None
        }

    def register_topology(
        self,
        predecessor_handlers: dict,
        successor_handlers: dict,
        predecessor_count: int,
        successor_count: int,
        local_spmd_code: int,
        spmd_id_by_code: dict,
    ) -> None:
        for worker in self._worker_handles.values():
            worker.set_expected_topology.remote(predecessor_count, successor_count)
            worker.register_spmd_code_map.remote(local_spmd_code, spmd_id_by_code)
        for spmd_id, handlers in predecessor_handlers.items():
            for worker in self._worker_handles.values():
                worker.register_predecessor_handlers.remote(spmd_id, handlers)
        for spmd_id, handlers in successor_handlers.items():
            for worker in self._worker_handles.values():
                worker.register_successor_handlers.remote(spmd_id, handlers)

    async def set_drain_target_count(self, target_count: int) -> None:
        await asyncio.gather(
            *[
                worker.set_drain_target_count.remote(target_count)
                for worker in self._worker_handles.values()
            ]
        )

    def clear_drain_target_count(self) -> None:
        for worker in self._worker_handles.values():
            worker.clear_drain_target_count.remote()

    async def prepare_checkpoint_artifacts(
        self, ckpt_dir: str, target_step_count: int
    ) -> None:
        if 0 in self._worker_handles:
            await self._worker_handles[0].prepare_checkpoint_artifacts.remote(
                ckpt_dir, target_step_count
            )

    async def save_state_to_disk(
        self, ckpt_dir: str, target_step_count: int
    ) -> list[tuple[int, Any, Any, Any]]:
        ranks = list(self._worker_handles.keys())
        results = await asyncio.gather(
            *(
                self._worker_handles[r].save_state_to_disk.remote(
                    ckpt_dir, target_step_count
                )
                for r in ranks
            )
        )
        return [
            (rank, res[0], res[1], res[2])
            for rank, res in zip(ranks, results, strict=True)
        ]

    async def commit_checkpoint_artifacts(
        self, ckpt_dir: str, target_step_count: int, gathered_metadata: dict
    ) -> None:
        if 0 in self._worker_handles:
            await self._worker_handles[0].commit_checkpoint_artifacts.remote(
                ckpt_dir, target_step_count, gathered_metadata
            )

    async def abort_checkpoint_artifacts(
        self, ckpt_dir: str, target_step_count: int
    ) -> None:
        if 0 in self._worker_handles:
            await self._worker_handles[0].abort_checkpoint_artifacts.remote(
                ckpt_dir, target_step_count
            )

    async def load_state_from_disk(
        self, spmd_state: dict, rank_storage: dict, step: int
    ) -> None:
        tasks = [
            worker.load_state_from_disk.remote(spmd_state, rank_storage.get(rank), step)
            for rank, worker in self._worker_handles.items()
        ]
        await asyncio.gather(*tasks)

    def block_worker_op(self, rank: int, op_name: str, op_runnable_epoch: int) -> None:
        worker = self._worker_handles.get(rank)
        if worker is not None:
            worker.set_op_runnable.remote(op_name, False, op_runnable_epoch)

    def unblock_worker_op(
        self, rank: int, op_name: str, op_runnable_epoch: int
    ) -> None:
        worker = self._worker_handles.get(rank)
        if worker is not None:
            worker.set_op_runnable.remote(op_name, True, op_runnable_epoch)

    def broadcast_control_argument(
        self, target_op: str, names: list[str], step: int, values: list[Any]
    ) -> None:
        for rank, worker in self._worker_handles.items():
            worker.receive_control.remote(target_op, step, names, values)

    def pause_sources(self) -> None:
        for worker in self._worker_handles.values():
            worker.pause_sources.remote()

    def resume_sources(self) -> None:
        for worker in self._worker_handles.values():
            worker.resume_sources.remote()

    async def start_source_operators(self) -> None:
        tasks = [
            worker.start_source_operators.remote()
            for worker in self._worker_handles.values()
        ]
        if tasks:
            await asyncio.gather(*tasks)
