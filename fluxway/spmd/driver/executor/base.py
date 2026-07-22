from typing import Any


class AbstractWorkerExecutor:
    """Abstract base class for executing commands across a fleet of SPMD workers."""

    async def start_workers(
        self, spmd_id: str, driver_handle: Any, controller_handle: Any
    ) -> None:
        raise NotImplementedError

    async def stop_workers(self) -> list[Exception]:
        raise NotImplementedError

    def get_attested_node_ids(self) -> tuple[str, ...]:
        raise NotImplementedError

    def get_worker_handles(self) -> dict[int, Any]:
        raise NotImplementedError

    async def inspect_operator_state(self) -> dict[int, Any]:
        raise NotImplementedError

    async def export_profiler_snapshots(self) -> dict[Any, Any]:
        raise NotImplementedError

    def register_topology(
        self,
        predecessor_handlers: dict,
        successor_handlers: dict,
        predecessor_count: int,
        successor_count: int,
        local_spmd_code: int,
        spmd_id_by_code: dict,
    ) -> None:
        raise NotImplementedError

    async def set_drain_target_count(self, target_count: int) -> None:
        raise NotImplementedError

    def clear_drain_target_count(self) -> None:
        raise NotImplementedError

    async def prepare_checkpoint_artifacts(
        self, ckpt_dir: str, target_step_count: int
    ) -> None:
        raise NotImplementedError

    async def save_state_to_disk(
        self, ckpt_dir: str, target_step_count: int
    ) -> list[tuple[int, Any, Any, Any]]:
        raise NotImplementedError

    async def commit_checkpoint_artifacts(
        self, ckpt_dir: str, target_step_count: int, gathered_metadata: dict
    ) -> None:
        raise NotImplementedError

    async def abort_checkpoint_artifacts(
        self, ckpt_dir: str, target_step_count: int
    ) -> None:
        raise NotImplementedError

    async def load_state_from_disk(
        self, spmd_state: dict, rank_storage: dict, step: int
    ) -> None:
        raise NotImplementedError

    def block_worker_op(self, rank: int, op_name: str, op_runnable_epoch: int) -> None:
        raise NotImplementedError

    def unblock_worker_op(
        self, rank: int, op_name: str, op_runnable_epoch: int
    ) -> None:
        raise NotImplementedError

    def broadcast_control_argument(
        self, target_op: str, names: list[str], step: int, values: list[Any]
    ) -> None:
        raise NotImplementedError

    def pause_sources(self) -> None:
        raise NotImplementedError

    def resume_sources(self) -> None:
        raise NotImplementedError

    async def start_source_operators(self) -> None:
        raise NotImplementedError
