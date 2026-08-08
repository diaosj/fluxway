"""Keep policy calculation over opaque backend-owned checkpoint_objects."""

from __future__ import annotations

from dataclasses import dataclass

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointCommitPermit,
    NameReference,
    PendingCleanup,
    CleanupReservation,
    CleanupReservationRequest,
    CheckpointManifest,
)
from fluxway.control_plane.checkpointer.backend import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)
from fluxway.control_plane.checkpointer.repository import CheckpointRepository


@dataclass(frozen=True, slots=True)
class BackendCleanupReservation:
    spmd_id: str
    component_id: str
    reservation: CleanupReservation


@dataclass(frozen=True, slots=True)
class _PublishedTargetVersion:
    checkpoint_sequence: int
    version_id: str
    checkpoint_objects: tuple[CheckpointObjectRef, ...]
    protected: bool


class CheckpointCleanupCoordinator:
    """Calculates pressure and delegates all atomic state to checkpoint_object backends."""

    def __init__(
        self,
        registries: dict[str, CheckpointComponentRegistry],
        repository: CheckpointRepository,
    ) -> None:
        self._registries = dict(registries)
        self._repository = repository

    async def reserve(
        self,
        permit: CheckpointCommitPermit,
        new_manifest: CheckpointManifest,
        existing_manifests: tuple[CheckpointManifest, ...],
    ) -> tuple[BackendCleanupReservation, ...]:
        if not permit.attempt_id:
            raise ValueError("cleanup reservation requires commit_save permit")
        existing = self._group_checkpoint_objects(existing_manifests)
        added = self._group_checkpoint_objects((new_manifest,))
        reservations: list[BackendCleanupReservation] = []
        try:
            for key, new_versions in added.items():
                spmd_id, component_id, checkpoint_object_id = key
                registration = self._require_backend(spmd_id, component_id)
                checkpoint_object = self._require_checkpoint_object(
                    registration, checkpoint_object_id
                )
                maximum = checkpoint_object.keep.keep_last
                if maximum is None or checkpoint_object.keep.keep_forever:
                    continue
                previous = existing.get(key, ())
                previous_by_version = self._unique_versions(previous)
                new_by_version = self._unique_versions(new_versions)
                required = max(
                    0, len(previous_by_version) + len(new_by_version) - maximum
                )
                if required == 0:
                    continue
                ordered = tuple(
                    checkpoint_object
                    for item in sorted(
                        previous_by_version.values(),
                        key=lambda item: item.checkpoint_sequence,
                    )
                    for checkpoint_object in item.checkpoint_objects
                )
                protected = frozenset(
                    item.version_id
                    for item in previous_by_version.values()
                    if item.protected
                )
                keep = registration.object_lifecycle
                if keep is None:
                    raise RuntimeError(
                        "checkpoint object lifecycle capability is not registered"
                    )
                reservation = await keep.reserve_cleanup(
                    CleanupReservationRequest(
                        permit=permit,
                        checkpoint_object_id=checkpoint_object_id,
                        ordered_candidates=ordered,
                        required_count=required,
                        protected_versions=protected,
                    )
                )
                claimed_by_version = self._checkpoint_objects_by_version(
                    reservation.claimed
                )
                if len(claimed_by_version) != required:
                    raise RuntimeError(
                        "checkpoint backend returned an incomplete cleanup reservation "
                        f"backend={component_id!r} checkpoint_object={checkpoint_object_id!r} "
                        f"required_versions={required} "
                        f"claimed_versions={len(claimed_by_version)}"
                    )
                for version_id, claimed in claimed_by_version.items():
                    candidate = previous_by_version.get(version_id)
                    if candidate is None or frozenset(claimed) != frozenset(
                        candidate.checkpoint_objects
                    ):
                        raise RuntimeError(
                            "checkpoint backend returned a partial checkpoint_object-version "
                            f"reservation backend={component_id!r} "
                            f"checkpoint_object={checkpoint_object_id!r} version={version_id!r}"
                        )
                reservations.append(
                    BackendCleanupReservation(
                        spmd_id=spmd_id,
                        component_id=component_id,
                        reservation=reservation,
                    )
                )
        except BaseException as primary_error:
            cleanup_errors = await self._abort_all(tuple(reservations), permit)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "cleanup reservation and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        return tuple(reservations)

    async def publish(
        self,
        reservations: tuple[BackendCleanupReservation, ...],
        permit: CheckpointCommitPermit,
    ) -> None:
        try:
            for binding in reservations:
                registration = self._require_backend(
                    binding.spmd_id, binding.component_id
                )
                keep = registration.object_lifecycle
                if keep is None:
                    raise RuntimeError(
                        "checkpoint object lifecycle capability is not registered"
                    )
                await keep.commit_cleanup(binding.reservation, permit)
        except BaseException as primary_error:
            cleanup_errors = await self._abort_all(reservations, permit)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "keep publication and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def rollback_save(
        self,
        reservations: tuple[BackendCleanupReservation, ...],
        permit: CheckpointCommitPermit,
    ) -> None:
        errors = await self._abort_all(reservations, permit)
        self._raise_errors("keep rollback_save failed", errors)

    async def cleanup(
        self,
        pending_cleanups: tuple[PendingCleanup, ...],
    ) -> None:
        errors: list[BaseException] = []
        for pending in pending_cleanups:
            registration = self._require_backend(pending.spmd_id, pending.component_id)
            cleaner = registration.cleaner
            if cleaner is None:
                raise RuntimeError("cleaner capability is not registered")
            try:
                await cleaner.cleanup(pending.reservation)
                for checkpoint_object in pending.reservation.claimed:
                    self._repository.mark_checkpoint_object_cleaned_up(
                        pending.spmd_id, checkpoint_object
                    )
                self._repository.complete_pending_cleanup(pending)
            except BaseException as error:
                errors.append(error)
        self._raise_errors("keep cleanup failed", errors)

    @staticmethod
    def pending_cleanups(
        checkpoint_id: str,
        reservations: tuple[BackendCleanupReservation, ...],
    ) -> tuple[PendingCleanup, ...]:
        return tuple(
            PendingCleanup(
                checkpoint_id=checkpoint_id,
                spmd_id=binding.spmd_id,
                component_id=binding.component_id,
                reservation=binding.reservation,
            )
            for binding in reservations
        )

    async def _abort_all(
        self,
        reservations: tuple[BackendCleanupReservation, ...],
        permit: CheckpointCommitPermit,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        for binding in reversed(reservations):
            registration = self._require_backend(binding.spmd_id, binding.component_id)
            keep = registration.object_lifecycle
            if keep is None:
                raise RuntimeError(
                    "checkpoint object lifecycle capability is not registered"
                )
            try:
                await keep.rollback_cleanup(binding.reservation, permit)
            except BaseException as error:
                errors.append(error)
        return errors

    def _require_backend(
        self, spmd_id: str, component_id: str
    ) -> CheckpointComponentRegistration:
        registry = self._registries.get(spmd_id)
        if registry is None:
            raise KeyError(f"checkpoint backend registry not found for {spmd_id!r}")
        return registry.require(component_id)

    @staticmethod
    def _require_checkpoint_object(
        registration: CheckpointComponentRegistration, checkpoint_object_id: str
    ):
        matches = [
            checkpoint_object
            for checkpoint_object in registration.descriptor.checkpoint_objects
            if checkpoint_object.checkpoint_object_id == checkpoint_object_id
        ]
        if len(matches) != 1:
            raise KeyError(
                f"checkpoint_object checkpoint_object {checkpoint_object_id!r} is not declared by "
                f"backend {registration.descriptor.component_id!r}"
            )
        return matches[0]

    def _group_checkpoint_objects(
        self, manifests: tuple[CheckpointManifest, ...]
    ) -> dict[tuple[str, str, str], tuple[_PublishedTargetVersion, ...]]:
        active_pins = self._active_keep_forever_checkpoint_ids(manifests)
        grouped: dict[tuple[str, str, str], list[_PublishedTargetVersion]] = {}
        for manifest in manifests:
            protected = manifest.checkpoint_id in active_pins
            for spmd_id, snapshot in manifest.spmds.items():
                for component_id, publication in snapshot.components.items():
                    records = [
                        *publication.shared_records,
                        *(partition.record for partition in publication.partitions),
                    ]
                    by_checkpoint_object: dict[str, list[CheckpointObjectRef]] = {}
                    for record in records:
                        for checkpoint_object in record.checkpoint_objects:
                            if checkpoint_object.component_id != component_id:
                                raise ValueError(
                                    "checkpoint_object component does not match publication: "
                                    f"checkpoint_object={checkpoint_object.component_id!r} "
                                    f"publication={component_id!r}"
                                )
                            if self._repository.is_checkpoint_object_cleaned_up(
                                spmd_id, checkpoint_object
                            ):
                                continue
                            by_checkpoint_object.setdefault(
                                checkpoint_object.checkpoint_object_id, []
                            ).append(checkpoint_object)
                    for (
                        checkpoint_object_id,
                        checkpoint_objects,
                    ) in by_checkpoint_object.items():
                        version_ids = {
                            checkpoint_object.version_id
                            for checkpoint_object in checkpoint_objects
                        }
                        if len(version_ids) != 1:
                            raise ValueError(
                                "one checkpoint_object publication must use one logical "
                                "checkpoint_object version"
                            )
                        grouped.setdefault(
                            (spmd_id, component_id, checkpoint_object_id), []
                        ).append(
                            _PublishedTargetVersion(
                                manifest.checkpoint_sequence,
                                next(iter(version_ids)),
                                tuple(checkpoint_objects),
                                protected,
                            )
                        )
        return {key: tuple(values) for key, values in grouped.items()}

    @staticmethod
    def _active_keep_forever_checkpoint_ids(
        manifests: tuple[CheckpointManifest, ...],
    ) -> frozenset[str]:
        latest_by_name: dict[str, CheckpointManifest] = {}
        for manifest in manifests:
            for reference in manifest.references:
                if (
                    not isinstance(reference, NameReference)
                    or not reference.keep_forever
                ):
                    continue
                previous = latest_by_name.get(reference.name)
                if previous is None or (
                    manifest.checkpoint_sequence > previous.checkpoint_sequence
                ):
                    latest_by_name[reference.name] = manifest
        return frozenset(manifest.checkpoint_id for manifest in latest_by_name.values())

    @staticmethod
    def _unique_versions(
        versions: tuple[_PublishedTargetVersion, ...],
    ) -> dict[str, _PublishedTargetVersion]:
        unique: dict[str, _PublishedTargetVersion] = {}
        for item in versions:
            previous = unique.get(item.version_id)
            if previous is None:
                unique[item.version_id] = item
                continue
            if frozenset(previous.checkpoint_objects) != frozenset(
                item.checkpoint_objects
            ):
                raise ValueError(
                    "one checkpoint_object version ID resolves to multiple references: "
                    f"{item.version_id!r}"
                )
            if item.protected and not previous.protected:
                unique[item.version_id] = item
        return unique

    @staticmethod
    def _checkpoint_objects_by_version(
        checkpoint_objects: tuple[CheckpointObjectRef, ...],
    ) -> dict[str, tuple[CheckpointObjectRef, ...]]:
        grouped: dict[str, list[CheckpointObjectRef]] = {}
        for checkpoint_object in checkpoint_objects:
            grouped.setdefault(checkpoint_object.version_id, []).append(
                checkpoint_object
            )
        return {version_id: tuple(items) for version_id, items in grouped.items()}

    @staticmethod
    def _raise_errors(message: str, errors: list[BaseException]) -> None:
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup(message, errors)
