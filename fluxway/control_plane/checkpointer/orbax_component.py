"""Orbax checkpoint component with shared global and partition-local records."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import os
import shutil
import msgpack
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from etils import epath
from ray import cloudpickle

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointComponentPublication,
    CheckpointObjectRecord,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    PartitionRecord,
    WorkerSaveResult,
    RestoreTopologyMode,
    CheckpointRetentionPolicy,
    TopologyMismatchError,
)
from fluxway.control_plane.checkpointer._filesystem_component import (
    FilesystemCheckpointComponent,
    _as_async,
)


def _escape(value: str) -> str:
    return value.replace("%", "%25").replace("/", "%2F")


def _get_path_str(path: tuple) -> str:
    import jax.tree_util as jtu

    parts = []
    for item in path:
        if isinstance(item, jtu.DictKey):
            parts.append(_escape(str(item.key)))
        elif isinstance(item, jtu.SequenceKey):
            parts.append(str(item.idx))
        elif hasattr(item, "name"):
            parts.append(_escape(item.name))
        else:
            parts.append(_escape(str(item)))
    return "/".join(parts)


def _is_global_leaf(leaf: Any) -> bool:
    import jax

    if not isinstance(leaf, jax.Array | jax.ShapeDtypeStruct):
        return False
    if hasattr(leaf, "sharding") and isinstance(
        leaf.sharding, jax.sharding.SingleDeviceSharding
    ):
        return False
    return True


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


class OrbaxCheckpointComponent(FilesystemCheckpointComponent):
    def __init__(
        self,
        component_id: str,
        save_state,
        restore_state,
        restore_layout,
        *,
        retention: CheckpointRetentionPolicy = CheckpointRetentionPolicy(),
    ) -> None:
        super().__init__(
            component_id,
            save_state,
            restore_state,
            retention=retention,
            restore_topology=RestoreTopologyMode.GLOBAL_RESHARD,
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
        import jax
        import jax.tree_util as jtu
        import orbax.checkpoint as ocp

        state = await self._save_state(session.instance, session)
        leaves_with_path, treedef = jtu.tree_flatten_with_path(state)
        global_dict: dict[str, Any] = {}
        local_dict: dict[str, Any] = {}
        for path, leaf in leaves_with_path:
            path_string = _get_path_str(path)
            checkpoint_object = global_dict if _is_global_leaf(leaf) else local_dict
            checkpoint_object[path_string] = leaf

        pytree_meta = base64.b64encode(cloudpickle.dumps(treedef)).decode("ascii")
        shared_staging = self._staging_root(session) / "shared"
        shared_final = self._final_component_root(session) / "state" / "shared"
        partition_key = f"rank/{session.rank}"
        local_staging = self._staging_path(session, partition_key)
        local_final = self._final_path(session, partition_key)

        if global_dict:
            shared_staging.parent.mkdir(parents=True, exist_ok=True)

            def write_global() -> None:
                options = ocp.options.MultiprocessingOptions(
                    barrier_sync_key_prefix=_barrier_sync_key(session, "save")
                )
                checkpointer = ocp.Checkpointer(
                    self._get_handler(), multiprocessing_options=options
                )
                try:
                    checkpointer.save(
                        epath.Path(shared_staging),
                        force=False,
                        args=ocp.args.StandardSave(global_dict),
                    )
                finally:
                    checkpointer.close()

            await asyncio.to_thread(write_global)

        if local_dict:
            if local_staging.exists():
                raise FileExistsError(
                    f"Orbax local staging directory exists: {local_staging}"
                )
            local_staging.mkdir(parents=True, exist_ok=False)
            local_state = jtu.tree_map(
                lambda value: np.asarray(value)
                if isinstance(value, jax.Array)
                else value,
                local_dict,
            )
            await asyncio.to_thread(self._write_local, local_staging, local_state)

        checkpoint_objects: list[CheckpointObjectRef] = []
        if global_dict:
            checkpoint_objects.append(
                CheckpointObjectRef(
                    component_id=self.descriptor.component_id,
                    checkpoint_object_id="state",
                    version_id=session.checkpoint_id,
                    storage_locator=shared_final.as_uri().encode("utf-8"),
                    format_version=self.descriptor.format_version,
                )
            )
        if local_dict:
            checkpoint_objects.append(
                CheckpointObjectRef(
                    component_id=self.descriptor.component_id,
                    checkpoint_object_id="state",
                    version_id=session.checkpoint_id,
                    storage_locator=local_final.as_uri().encode("utf-8"),
                    format_version=self.descriptor.format_version,
                    partition_key=partition_key,
                )
            )
        return WorkerSaveResult(
            record=CheckpointObjectRecord(
                payload=msgpack.packb(
                    {
                        "pytree_meta": pytree_meta,
                        "has_global_state": bool(global_dict),
                        "has_local_state": bool(local_dict),
                    },
                    use_bin_type=True,
                ),
                checkpoint_objects=tuple(checkpoint_objects),
            ),
            commit_metadata=msgpack.packb(
                {
                    "shared_staging_uri": shared_staging.as_uri()
                    if global_dict
                    else None,
                    "local_staging_uri": local_staging.as_uri() if local_dict else None,
                },
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
        payloads = [
            msgpack.unpackb(record.payload, raw=False) for record in typed_records
        ]
        pytree_values = {payload["pytree_meta"] for payload in payloads}
        if len(pytree_values) != 1:
            raise ValueError("Orbax ranks produced different pytree structures")

        moved: list[tuple[Path, Path]] = []
        shared_record: CheckpointObjectRecord | None = None
        partitions: list[PartitionRecord] = []
        try:
            first_record = typed_records[0]
            global_checkpoint_objects = [
                checkpoint_object
                for checkpoint_object in first_record.checkpoint_objects
                if checkpoint_object.partition_key is None
            ]
            if global_checkpoint_objects:
                if len(global_checkpoint_objects) != 1:
                    raise ValueError(
                        "Orbax rank record has multiple shared checkpoint_objects"
                    )
                first_metadata = msgpack.unpackb(
                    prepared_values[0].commit_metadata, raw=False
                )
                if not isinstance(first_metadata, Mapping):
                    raise TypeError("Orbax rank record has no commit_save metadata")
                shared_staging_value = first_metadata["shared_staging_uri"]
                if not isinstance(shared_staging_value, str):
                    raise TypeError("Orbax shared staging URI is missing")
                shared_staging = self._file_path(shared_staging_value.encode("utf-8"))
                shared_final = self._file_path(
                    global_checkpoint_objects[0].storage_locator
                )
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
                moved.append((shared_staging, shared_final))
                shared_record = CheckpointObjectRecord(
                    payload=msgpack.packb(
                        {
                            "pytree_meta": msgpack.unpackb(
                                first_record.payload, raw=False
                            )["pytree_meta"],
                            "has_global_state": True,
                        },
                        use_bin_type=True,
                    ),
                    checkpoint_objects=(global_checkpoint_objects[0],),
                )

            for rank, prepared in sorted(session.worker_results.items()):
                record = prepared.record
                if record is None:
                    raise ValueError(f"Orbax rank {rank} has no record")
                local_checkpoint_objects = [
                    checkpoint_object
                    for checkpoint_object in record.checkpoint_objects
                    if checkpoint_object.partition_key is not None
                ]
                if not local_checkpoint_objects:
                    continue
                if len(local_checkpoint_objects) != 1:
                    raise ValueError(
                        f"Orbax rank {rank} has multiple local checkpoint_objects"
                    )
                metadata = msgpack.unpackb(prepared.commit_metadata, raw=False)
                if not isinstance(metadata, Mapping):
                    raise TypeError(f"Orbax rank {rank} has no commit_save metadata")
                local_staging_value = metadata["local_staging_uri"]
                if not isinstance(local_staging_value, str):
                    raise TypeError(f"Orbax rank {rank} local staging URI is missing")
                local_staging = self._file_path(local_staging_value.encode("utf-8"))
                local_final = self._file_path(
                    local_checkpoint_objects[0].storage_locator
                )
                if not local_staging.is_dir():
                    raise FileNotFoundError(
                        f"Orbax rank {rank} local staging checkpoint_object is missing"
                    )
                if local_final.exists():
                    raise FileExistsError(
                        f"Orbax rank {rank} local final checkpoint_object exists"
                    )
                local_final.parent.mkdir(parents=True, exist_ok=True)
                os.replace(local_staging, local_final)
                moved.append((local_staging, local_final))
                partitions.append(
                    PartitionRecord(
                        partition_key=local_checkpoint_objects[0].partition_key or "",
                        source_rank=rank,
                        record=CheckpointObjectRecord(
                            payload=msgpack.packb(
                                {
                                    "pytree_meta": msgpack.unpackb(
                                        record.payload, raw=False
                                    )["pytree_meta"],
                                    "has_local_state": True,
                                },
                                use_bin_type=True,
                            ),
                            checkpoint_objects=(local_checkpoint_objects[0],),
                        ),
                    )
                )
        except BaseException:
            for staging, final in reversed(moved):
                if final.exists():
                    os.replace(final, staging)
            raise
        return CheckpointComponentPublication(
            component_id=self.descriptor.component_id,
            format_version=self.descriptor.format_version,
            shared_records=(shared_record,) if shared_record is not None else (),
            partitions=tuple(partitions),
        )

    async def restore(self, session: RestoreSession) -> None:
        self._require_worker(session)
        publication = session.component_checkpoint
        current = session.current_topology or session.source_topology
        if publication.partitions and current != session.source_topology:
            raise TopologyMismatchError(
                "Orbax publication contains rank-local state and requires exact topology"
            )
        layout = await self._restore_layout(session.instance, session)
        global_values: dict[str, Any] = {}
        local_values: dict[str, Any] = {}
        pytree_meta: str | None = None

        if publication.shared_records:
            if len(publication.shared_records) != 1:
                raise ValueError("Orbax publication has multiple shared records")
            shared = publication.shared_records[0]
            pytree_meta = msgpack.unpackb(shared.payload, raw=False)["pytree_meta"]
            global_values = await self._read_global(shared, layout, session)

        partition_key = f"rank/{session.rank}"
        local_matches = [
            partition.record
            for partition in publication.partitions
            if partition.partition_key == partition_key
        ]
        if local_matches:
            if len(local_matches) != 1:
                raise ValueError("Orbax publication has duplicate local partitions")
            local = local_matches[0]
            pytree_meta = msgpack.unpackb(local.payload, raw=False)["pytree_meta"]
            local_values = await self._read_local(local)
        elif publication.partitions:
            raise KeyError(f"Orbax local partition not found: {partition_key!r}")

        if pytree_meta is None:
            raise ValueError("Orbax publication contains no state records")
        state = self._reconstruct_state(pytree_meta, {**global_values, **local_values})
        await self._restore_state(session.instance, state, session)

    async def rollback_save(self, session: SaveSession) -> None:
        self._require_permit(session, session.commit_permit)
        self._require_worker(session)
        partition_key = f"rank/{session.rank}"
        paths = [
            self._staging_path(session, partition_key),
            self._final_path(session, partition_key),
        ]
        if session.rank == 0:
            paths.extend(
                (
                    self._staging_root(session) / "shared",
                    self._final_component_root(session) / "state" / "shared",
                )
            )
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
        raise RuntimeError("Orbax uses shared and partition restore records")

    async def _finish_save_io(self, context: SaveSession) -> None:
        return None

    async def _close_resources(self) -> None:
        if self._handler is not None:
            self._handler.close()
            self._handler = None

    @staticmethod
    def _write_local(directory: Path, local_state: Any) -> None:
        import jax.tree_util as jtu

        leaves, treedef = jtu.tree_flatten(local_state)
        for index, leaf in enumerate(leaves):
            numpy_path = directory / f"leaf_{index}.npy"
            if isinstance(leaf, np.ndarray):
                np.save(str(numpy_path), leaf)
            else:
                with (directory / f"leaf_{index}.pkl").open("xb") as stream:
                    cloudpickle.dump(leaf, stream)
        with (directory / "treedef.pkl").open("xb") as stream:
            cloudpickle.dump(treedef, stream)

    async def _read_global(
        self,
        record: CheckpointObjectRecord,
        restore_layout: Any,
        context: RestoreSession,
    ) -> dict[str, Any]:
        import jax.tree_util as jtu
        import orbax.checkpoint as ocp

        if len(record.checkpoint_objects) != 1:
            raise ValueError("Orbax shared record must contain one checkpoint_object")
        checkpoint_object: dict[str, Any] = {}
        for path, leaf in jtu.tree_flatten_with_path(restore_layout)[0]:
            if _is_global_leaf(leaf):
                checkpoint_object[_get_path_str(path)] = leaf
        if not checkpoint_object:
            raise ValueError("Orbax restore layout contains no globally sharded arrays")
        checkpoint_path = epath.Path(
            self._file_path(record.checkpoint_objects[0].storage_locator)
        )

        def read() -> dict[str, Any]:
            options = ocp.options.MultiprocessingOptions(
                barrier_sync_key_prefix=_barrier_sync_key(context, "restore")
            )
            checkpointer = ocp.Checkpointer(
                self._get_handler(), multiprocessing_options=options
            )
            try:
                return checkpointer.restore(
                    checkpoint_path,
                    args=ocp.args.StandardRestore(item=checkpoint_object),
                )
            finally:
                checkpointer.close()

        return await asyncio.to_thread(read)

    async def _read_local(self, record: CheckpointObjectRecord) -> dict[str, Any]:
        if len(record.checkpoint_objects) != 1:
            raise ValueError("Orbax local record must contain one checkpoint_object")
        directory = self._file_path(record.checkpoint_objects[0].storage_locator)

        def read() -> dict[str, Any]:
            with (directory / "treedef.pkl").open("rb") as stream:
                treedef = cloudpickle.load(stream)
            leaves = []
            for index in range(treedef.num_leaves):
                numpy_path = directory / f"leaf_{index}.npy"
                pickle_path = directory / f"leaf_{index}.pkl"
                if numpy_path.exists():
                    leaves.append(np.load(str(numpy_path)))
                elif pickle_path.exists():
                    with pickle_path.open("rb") as stream:
                        leaves.append(cloudpickle.load(stream))
                else:
                    raise FileNotFoundError(
                        f"Orbax local leaf {index} is missing in {directory}"
                    )
            return treedef.unflatten(leaves)

        return await asyncio.to_thread(read)

    @staticmethod
    def _reconstruct_state(pytree_meta: str, values: dict[str, Any]) -> Any:
        import jax.tree_util as jtu

        treedef = cloudpickle.loads(base64.b64decode(pytree_meta))
        dummy_tree = jtu.tree_unflatten(treedef, list(range(treedef.num_leaves)))
        paths, _ = jtu.tree_flatten_with_path(dummy_tree)
        leaves = []
        for path, _ in paths:
            path_string = _get_path_str(path)
            if path_string not in values:
                raise KeyError(f"Orbax restored state is missing path {path_string!r}")
            leaves.append(values[path_string])
        return jtu.tree_unflatten(treedef, leaves)
