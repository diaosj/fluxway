"""Controller-side global checkpoint orchestration."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointObjectCleanedUpError,
    CheckpointRuntime,
    CheckpointRuntimeRole,
    CheckpointCaptureMode,
    CheckpointCut,
    CheckpointTransactionState,
    CheckpointManifest,
    SaveCheckpointRequest,
    CheckpointIdReference,
    CheckpointId,
    NameReference,
    StepReference,
    CheckpointHold,
)
from fluxway.control_plane.checkpointer.backend import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.control_plane.checkpointer.keep import (
    CheckpointCleanupCoordinator,
)

logger = logging.getLogger(__name__)


class FluxwayCheckpointer:
    def __init__(
        self,
        controller: Any,
        repository_dir: str,
        *,
        interval_steps: int,
        capture_mode: CheckpointCaptureMode = CheckpointCaptureMode.FENCED,
        component_registries: Mapping[str, CheckpointComponentRegistry] | None = None,
    ) -> None:
        if interval_steps < 1:
            raise ValueError("interval_steps must be positive")
        self._controller = controller
        self._repository = CheckpointRepository(repository_dir)
        self._coordinator_generation = self._repository.claim_coordinator_generation()
        self._repository_dir = str(self._repository.root)
        self._interval_steps = interval_steps
        self._capture_mode = capture_mode
        self._registries = dict(component_registries or {})
        self._keep = CheckpointCleanupCoordinator(self._registries, self._repository)
        self._control_open = False
        self._closed = False

    @property
    def interval_steps(self) -> int:
        return self._interval_steps

    @property
    def capture_mode(self) -> CheckpointCaptureMode:
        return self._capture_mode

    @property
    def repository(self) -> CheckpointRepository:
        return self._repository

    async def save(self, request: SaveCheckpointRequest) -> CheckpointManifest:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is closed")
        if not isinstance(request, SaveCheckpointRequest):
            raise TypeError("FluxwayCheckpointer.save requires SaveCheckpointRequest")
        await self._ensure_control_backends()
        await self._repair_pending_cleanups()
        checkpoint_id = CheckpointId.generate()
        permit = self._repository.issue_commit_permit(
            request.attempt_id, self._coordinator_generation
        )
        self._repository.write_transaction_state(
            checkpoint_id, CheckpointTransactionState.CUT_CREATED
        )
        published = False
        reservations = ()
        try:
            assembly = await self._controller.request_snapshot(
                str(checkpoint_id),
                request.attempt_id,
                permit,
                request.cut,
                self._repository_dir,
                request.capture_mode,
            )
            self._validate_assembly_cut(request, assembly.cut)
            sequence = self._repository.next_checkpoint_sequence()
            manifest = CheckpointManifest(
                schema_version=4,
                checkpoint_id=str(checkpoint_id),
                attempt_id=request.attempt_id,
                coordinator_generation=permit.coordinator_generation,
                cut=assembly.cut,
                checkpoint_sequence=sequence,
                references=request.references,
                framework_state=assembly.framework_state,
                spmds=assembly.spmds,
            )
            self._validate_manifest_progress(manifest)
            self._repository.write_transaction_state(
                str(checkpoint_id), CheckpointTransactionState.COMMITTED
            )
            reservations = await self._keep.reserve(
                permit, manifest, self._repository.list()
            )
            self._repository.require_commit_permit(permit)
            await self._keep.publish(reservations, permit)
            pending_cleanups = self._keep.pending_cleanups(
                str(checkpoint_id), reservations
            )
            self._repository.publish(manifest, permit, pending_cleanups)
            published = True
            await self._controller.mark_snapshot_published(str(checkpoint_id), permit)
            await self._finalize_framework_state(manifest)
            await self._keep.cleanup(pending_cleanups)
            logger.info(
                "checkpoint published snapshot=%s logical_step=%s",
                checkpoint_id,
                manifest.cut.logical_step,
            )
            return manifest
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            if not published:
                try:
                    await self._keep.rollback_save(reservations, permit)
                except BaseException as error:
                    cleanup_errors.append(error)
                try:
                    await self._controller.abort_snapshot_checkpoint_objects(
                        str(checkpoint_id), permit
                    )
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint save and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def restore(
        self,
        reference: StepReference | CheckpointIdReference | str = "latest",
    ) -> int | None:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is closed")
        await self._ensure_control_backends()
        if reference == "latest":
            manifest = self._repository.latest()
            if manifest is None:
                return None
        else:
            if isinstance(reference, str):
                raise TypeError(
                    "restore reference must be latest, StepReference, or CheckpointIdReference"
                )
            manifest = self._repository.resolve(reference)
        self._validate_manifest_progress(manifest)
        self._assert_restorable(manifest)
        await self._controller.restore_snapshot(
            manifest.checkpoint_id,
            manifest,
            self._repository_dir,
        )
        return manifest.cut.logical_step

    async def restore_latest(self) -> int | None:
        return await self.restore()

    async def restore_step(self, step: int) -> int | None:
        return await self.restore(StepReference(step))

    async def restore_named(self, name: str) -> int | None:
        return await self.restore(NameReference(name))

    async def restore_checkpoint(self, checkpoint_id: str) -> int | None:
        return await self.restore(CheckpointIdReference(checkpoint_id))

    def list_steps(self) -> tuple[int, ...]:
        return self._repository.list_steps()

    def latest(self) -> CheckpointManifest | None:
        return self._repository.latest()

    async def hold_checkpoint_object(
        self,
        spmd_id: str,
        checkpoint_object: CheckpointObjectRef,
        owner_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold:
        self._require_open()
        if not self._repository.contains_checkpoint_object(spmd_id, checkpoint_object):
            raise KeyError(
                "cannot lease an checkpoint_object that is not a live published reference"
            )
        await self._ensure_control_backends()
        registration = self._require_backend(spmd_id, checkpoint_object.component_id)
        if registration.object_lifecycle is None:
            raise RuntimeError(
                "backend does not expose checkpoint object lifecycle capability"
            )
        return await registration.object_lifecycle.hold_checkpoint_object(
            checkpoint_object, owner_id, ttl_seconds, request_id
        )

    async def renew_checkpoint_hold(
        self,
        spmd_id: str,
        lease: CheckpointHold,
        ttl_seconds: float,
    ) -> CheckpointHold:
        self._require_open()
        await self._ensure_control_backends()
        registration = self._require_backend(
            spmd_id, lease.checkpoint_object.component_id
        )
        if registration.object_lifecycle is None:
            raise RuntimeError(
                "backend does not expose checkpoint object lifecycle capability"
            )
        return await registration.object_lifecycle.renew_checkpoint_hold(
            lease, ttl_seconds
        )

    async def release_checkpoint_hold(
        self, spmd_id: str, lease: CheckpointHold
    ) -> None:
        self._require_open()
        await self._ensure_control_backends()
        registration = self._require_backend(
            spmd_id, lease.checkpoint_object.component_id
        )
        if registration.object_lifecycle is None:
            raise RuntimeError(
                "backend does not expose checkpoint object lifecycle capability"
            )
        await registration.object_lifecycle.release_checkpoint_hold(lease)

    async def repair_pending_cleanups(self) -> int:
        self._require_open()
        await self._ensure_control_backends()
        return await self._repair_pending_cleanups()

    async def close(self) -> None:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is already closed")
        if self._control_open:
            errors: list[BaseException] = []
            for registry in reversed(tuple(self._registries.values())):
                try:
                    await registry.close_all()
                except BaseException as error:
                    errors.append(error)
            if errors:
                if len(errors) == 1:
                    raise errors[0]
                raise BaseExceptionGroup(
                    "control checkpoint backend close failed", errors
                )
        self._closed = True

    async def _ensure_control_backends(self) -> None:
        if self._control_open:
            return
        opened: list[CheckpointComponentRegistry] = []
        try:
            for spmd_id, registry in self._registries.items():
                await registry.open_all(
                    CheckpointRuntime(
                        role=CheckpointRuntimeRole.CONTROL,
                        repository_root=self._repository_dir,
                        spmd_id=spmd_id,
                    )
                )
                opened.append(registry)
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            for registry in reversed(opened):
                try:
                    await registry.close_all()
                except BaseException as error:
                    cleanup_errors.append(error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "control backend open and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        self._control_open = True

    async def _repair_pending_cleanups(self) -> int:
        pending = self._repository.list_pending_cleanups()
        if not pending:
            return 0
        await self._keep.cleanup(pending)
        return len(pending)

    def _require_backend(
        self, spmd_id: str, component_id: str
    ) -> CheckpointComponentRegistration:
        registry = self._registries.get(spmd_id)
        if registry is None:
            raise KeyError(
                f"checkpoint backend registry not found for SPMD {spmd_id!r}"
            )
        return registry.require(component_id)

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("FluxwayCheckpointer is closed")

    def _assert_restorable(self, manifest: CheckpointManifest) -> None:
        for spmd_id, snapshot in manifest.spmds.items():
            registry = self._registries.get(spmd_id)
            if registry is None:
                raise KeyError(
                    f"checkpoint backend registry not found for SPMD {spmd_id!r}"
                )
            for component_id, publication in snapshot.components.items():
                backend = registry.require(component_id)
                declared_objects = {
                    checkpoint_object.checkpoint_object_id: checkpoint_object
                    for checkpoint_object in backend.descriptor.checkpoint_objects
                }
                records = (
                    *publication.shared_records,
                    *(partition.record for partition in publication.partitions),
                )
                for record in records:
                    for object_ref in record.checkpoint_objects:
                        declared_object = declared_objects.get(
                            object_ref.checkpoint_object_id
                        )
                        if declared_object is None:
                            raise ValueError(
                                f"manifest checkpoint object {object_ref.checkpoint_object_id!r} "
                                f"is not declared by backend {component_id!r}"
                            )
                        if declared_object.required_for_restore and (
                            self._repository.is_checkpoint_object_cleaned_up(
                                spmd_id, object_ref
                            )
                        ):
                            raise CheckpointObjectCleanedUpError(
                                "checkpoint object was cleaned up: "
                                f"spmd={spmd_id!r} backend={component_id!r} "
                                f"checkpoint_object={object_ref.checkpoint_object_id!r} "
                                f"version={object_ref.version_id!r}"
                            )

    @staticmethod
    def _validate_assembly_cut(
        request: SaveCheckpointRequest, resolved: CheckpointCut
    ) -> None:
        if not isinstance(resolved, CheckpointCut):
            raise TypeError("checkpoint assembly returned an invalid checkpoint cut")
        if resolved.cut_id != request.cut.cut_id:
            raise ValueError("checkpoint assembly changed the checkpoint cut ID")
        if resolved.logical_step != request.cut.logical_step:
            raise ValueError("checkpoint assembly changed the logical checkpoint step")
        if resolved.control_epoch != request.cut.control_epoch:
            raise ValueError("checkpoint assembly changed the control epoch")

    @staticmethod
    def _validate_manifest_progress(manifest: CheckpointManifest) -> None:
        ledger = manifest.framework_state.get("control_ledger")
        if ledger is None:
            return
        if not isinstance(ledger, Mapping):
            raise TypeError("manifest control ledger state must be a mapping")
        durable_cuts = ledger.get("durable_cuts")
        sealed_ends = ledger.get("sealed_ends")
        if not isinstance(durable_cuts, Mapping):
            raise TypeError("manifest control ledger durable cuts must be a mapping")
        if not isinstance(sealed_ends, Mapping):
            raise TypeError("manifest control ledger sealed ends must be a mapping")
        if dict(durable_cuts) != dict(manifest.cut.stream_cuts):
            raise ValueError(
                "CheckpointCut stream cuts disagree with framework ledger state"
            )
        if dict(sealed_ends) != dict(manifest.cut.sealed_ends):
            raise ValueError(
                "CheckpointCut sealed ends disagree with framework ledger state"
            )

    async def _finalize_framework_state(self, manifest: CheckpointManifest) -> None:
        if manifest.cut.logical_step is None:
            raise RuntimeError(
                "controller framework ledger finalization requires logical_step"
            )
        await self._controller.finalize_snapshot(
            manifest.cut.logical_step, manifest.framework_state
        )
