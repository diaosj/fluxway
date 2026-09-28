"""Orbax checkpoint component with collective shared nested PyTree state."""

from __future__ import annotations

import asyncio
import hashlib
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from etils import epath
import msgpack

from fluxway.common.types.checkpoint import (
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointObjectRef,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CommitSaveSession,
    RestoreSession,
    RestoreTopologyMode,
    SaveSession,
    WorkerSaveResult,
)
from fluxway.control_plane.checkpointer._filesystem_component import (
    FilesystemCheckpointComponent,
    _as_async,
)


def _barrier_sync_key(context: SaveSession | RestoreSession, operation: str) -> str:
    if not operation:
        raise ValueError("Orbax barrier operation must be non-empty")
    identity = "\x00".join(
        (
            context.spmd_id,
            context.checkpoint_id,
            context.attempt_id,
            context.component_id,
            operation,
        )
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return f"fluxway_{operation}_{digest}"


def _jax_process_count() -> int:
    try:
        import jax

        distributed = jax.distributed
        return int(jax.process_count()) if distributed.is_initialized() else 1
    except (ImportError, AttributeError, RuntimeError):
        return 1


def _require_collective_contract(world_size: int) -> bool:
    if world_size <= 0:
        raise ValueError(
            f"Orbax checkpoint topology world_size must be positive, got {world_size}"
        )
    process_count = _jax_process_count()
    if world_size == 1:
        if process_count != 1:
            raise RuntimeError(
                f"single-rank Orbax checkpoint requires one JAX process, got {process_count}"
            )
        return False
    if process_count != world_size:
        raise RuntimeError(
            "Orbax checkpoint requires one initialized JAX process per SPMD rank: "
            f"spmd_world_size={world_size} jax_process_count={process_count}"
        )
    return True


class OrbaxCheckpointComponent(FilesystemCheckpointComponent):
    def __init__(
        self,
        component_id: str,
        save_state: Callable[[Any, SaveSession], Any],
        restore_state: Callable[[Any, Any, RestoreSession], Any],
        restore_layout: Callable[[Any, RestoreSession], Any],
        *,
        finalize_state: Callable[[Any, SaveSession], Any] | None = None,
        retention_domain: str | None = None,
    ) -> None:
        super().__init__(
            component_id,
            save_state,
            restore_state,
            finalize_state=finalize_state,
            restore_topology=RestoreTopologyMode.GLOBAL_RESHARD,
            retention_domain=retention_domain,
        )
        self._restore_layout = _as_async(restore_layout, "restore_layout")
        self._handler = None

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_handler"] = None
        return state

    def _get_handler(self):
        if self._handler is None:
            import orbax.checkpoint as ocp

            self._handler = ocp.StandardCheckpointHandler()
        return self._handler

    async def _open_resources(self, runtime: CheckpointRuntime) -> None:
        if runtime.role is CheckpointRuntimeRole.WORKER:
            self._get_handler()

    async def save(self, session: SaveSession) -> WorkerSaveResult:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        import orbax.checkpoint as ocp

        state = await self._save_state(session.instance, session)
        shared_staging = self._staging_root(session) / "shared"
        completion_marker = shared_staging.parent / ".shared-save-complete"
        shared_final = self._final_component_root(session) / "state" / "shared"
        shared_staging.parent.mkdir(parents=True, exist_ok=True)

        collective = _require_collective_contract(session.source_topology.world_size)

        def write_global() -> None:
            options = ocp.options.MultiprocessingOptions(
                **(
                    {"barrier_sync_key_prefix": _barrier_sync_key(session, "save")}
                    if collective
                    else {}
                )
            )
            checkpointer = ocp.Checkpointer(
                self._get_handler(),
                multiprocessing_options=options,
            )
            try:
                checkpointer.save(
                    epath.Path(shared_staging),
                    force=False,
                    args=ocp.args.StandardSave(state),
                )
            finally:
                checkpointer.close()

        if collective or session.rank == 0:
            await asyncio.to_thread(write_global)
            if session.rank == 0:
                completion_marker.write_text(session.checkpoint_id, encoding="utf-8")
        else:
            raise RuntimeError("single-rank Orbax save cannot run on a nonzero rank")

        checkpoint_object = CheckpointObjectRef(
            component_id=self.descriptor.component_id,
            checkpoint_object_id="state",
            version_id=session.checkpoint_id,
            storage_locator=shared_final.as_uri().encode("utf-8"),
            format_version=self.descriptor.format_version,
        )
        return WorkerSaveResult(
            record=CheckpointObjectRecord(
                payload=b"",
                checkpoint_objects=(checkpoint_object,),
            ),
            commit_metadata=msgpack.packb(
                {"shared_staging_uri": shared_staging.as_uri()},
                use_bin_type=True,
            ),
        )

    async def commit_save(
        self, session: CommitSaveSession
    ) -> CheckpointComponentPublication:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        if not session.worker_results:
            raise ValueError("Orbax commit_save requires rank records")
        prepared_values = tuple(session.worker_results.values())
        records = [prepared.record for prepared in prepared_values]
        if any(record is None for record in records):
            raise ValueError("Orbax commit_save received an empty rank record")
        typed_records = [record for record in records if record is not None]

        first_record = typed_records[0]
        if len(first_record.checkpoint_objects) != 1:
            raise ValueError(
                "Orbax rank record must contain exactly one shared checkpoint_object"
            )
        shared_checkpoint_object = first_record.checkpoint_objects[0]
        first_metadata = msgpack.unpackb(prepared_values[0].commit_metadata, raw=False)
        if not isinstance(first_metadata, Mapping):
            raise TypeError("Orbax rank record has no commit_save metadata")
        shared_staging_value = first_metadata.get("shared_staging_uri")
        if not isinstance(shared_staging_value, str):
            raise TypeError("Orbax shared staging URI is missing")

        shared_staging = self._file_path(shared_staging_value.encode("utf-8"))
        shared_final = self._file_path(shared_checkpoint_object.storage_locator)
        if not shared_staging.is_dir():
            raise FileNotFoundError(
                f"Orbax shared staging checkpoint_object is missing: {shared_staging}"
            )
        if shared_final.exists():
            raise FileExistsError(
                f"Orbax shared final checkpoint_object exists: {shared_final}"
            )

        shared_final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(shared_staging, shared_final)

        shared_record = CheckpointObjectRecord(
            payload=b"",
            checkpoint_objects=(shared_checkpoint_object,),
        )
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
            format_version=self.descriptor.format_version,
            shared_records=(shared_record,),
            partitions=(),
        )

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker(session)
        publication = session.component_checkpoint
        if not publication.shared_records:
            raise ValueError("Orbax publication contains no shared records")
        if len(publication.shared_records) != 1:
            raise ValueError("Orbax publication has multiple shared records")
        shared = publication.shared_records[0]
        layout = await self._restore_layout(session.instance, session)
        state = await self._read_global(shared, layout, session)
        await self._restore_state(session.instance, state, session)

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        if session.rank == 0:
            paths = [
                self._staging_root(session) / "shared",
                self._staging_root(session) / ".shared-save-complete",
                self._final_component_root(session) / "state" / "shared",
            ]
            for path in paths:
                try:
                    await asyncio.to_thread(shutil.rmtree, path)
                except FileNotFoundError:
                    pass

    async def _write_state(
        self, state: Any, directory: Path, context: SaveSession
    ) -> Mapping[str, Any]:
        raise RuntimeError("Orbax uses its collective save implementation")

    async def _read_state(
        self, record: CheckpointObjectRecord, context: RestoreSession
    ) -> Any:
        raise RuntimeError("Orbax uses shared restore records")

    async def _finish_save_io(self, context: SaveSession) -> None:
        return None

    async def _close_resources(self) -> None:
        if self._handler is not None:
            self._handler.close()
            self._handler = None

    async def _read_global(
        self,
        record: CheckpointObjectRecord,
        restore_layout: Any,
        context: RestoreSession,
    ) -> Any:
        import orbax.checkpoint as ocp

        if len(record.checkpoint_objects) != 1:
            raise ValueError("Orbax shared record must contain one checkpoint_object")
        checkpoint_path = epath.Path(
            self._file_path(record.checkpoint_objects[0].storage_locator)
        )

        def read() -> Any:
            options = ocp.options.MultiprocessingOptions(
                barrier_sync_key_prefix=_barrier_sync_key(context, "restore")
            )
            checkpointer = ocp.Checkpointer(
                self._get_handler(), multiprocessing_options=options
            )
            try:
                return checkpointer.restore(
                    checkpoint_path,
                    args=ocp.args.StandardRestore(item=restore_layout),
                )
            finally:
                checkpointer.close()

        return await asyncio.to_thread(read)
