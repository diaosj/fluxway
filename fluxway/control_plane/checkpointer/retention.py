"""Retention policy calculation over opaque component-owned checkpoint_objects."""

from __future__ import annotations

from dataclasses import dataclass

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointCommitPermit,
    NameReference,
    CleanupReservationRequest,
    CheckpointManifest,
    CleanupReservationBinding,
    CheckpointCleanupIntent,
    CheckpointRetentionPolicy,
)
from fluxway.control_plane.checkpointer.component import (
    CheckpointComponentRegistration,
    CheckpointComponentRegistry,
)
from fluxway.control_plane.checkpointer.repository import CheckpointRepository
from fluxway.common.io_lane import IOLane


@dataclass(frozen=True, slots=True)
class _PublishedTargetVersion:
    artifact_id: str
    checkpoint_sequence: int
    version_id: str
    checkpoint_objects: tuple[CheckpointObjectRef, ...]
    protected: bool


class CheckpointRetentionCoordinator:
    """Calculates pressure and delegates all atomic state to checkpoint_object components."""

    def __init__(
        self,
        registries: dict[str, CheckpointComponentRegistry],
        repository: CheckpointRepository,
        retention_policy: CheckpointRetentionPolicy,
        retention_domains: dict[str, CheckpointRetentionPolicy] | None = None,
        *,
        io_lane: IOLane,
    ) -> None:
        self._registries = dict(registries)
        self._repository = repository
        self._io_lane = io_lane
        self._retention_policy = retention_policy
        self._retention_domains = dict(retention_domains or {})

    async def reserve(
        self,
        permit: CheckpointCommitPermit,
        new_manifest: CheckpointManifest,
        existing_manifests: tuple[CheckpointManifest, ...],
    ) -> tuple[CleanupReservationBinding, ...]:
        if not permit.attempt_id:
            raise ValueError("cleanup reservation requires commit_save permit")
        cleaned = await self._io_lane.call(
            self._repository.cleaned_checkpoint_objects,
            (*existing_manifests, new_manifest),
        )
        existing = self._group_checkpoint_objects(existing_manifests, cleaned)
        added = self._group_checkpoint_objects((new_manifest,), cleaned)
        artifact_limit = self._artifact_retention_limit(new_manifest)
        reservations: list[CleanupReservationBinding] = []
        try:
            for key, new_versions in added.items():
                spmd_id, component_id, checkpoint_object_id = key
                registration = self._require_component(spmd_id, component_id)
                self._require_checkpoint_object(registration, checkpoint_object_id)
                if artifact_limit is None:
                    continue
                previous = existing.get(key, ())
                previous_by_version = self._unique_versions(previous)
                new_by_version = self._unique_versions(new_versions)
                required = max(
                    0, len(previous_by_version) + len(new_by_version) - artifact_limit
                )
                if required == 0:
                    continue
                candidates = tuple(
                    sorted(
                        previous_by_version.values(),
                        key=lambda item: item.checkpoint_sequence,
                    )[:required]
                )
                protected = frozenset(
                    item.version_id
                    for item in previous_by_version.values()
                    if item.protected
                )
                lifecycle = registration.object_lifecycle
                if lifecycle is None:
                    raise RuntimeError(
                        "checkpoint object lifecycle capability is not registered"
                    )
                by_artifact: dict[str, list[_PublishedTargetVersion]] = {}
                for candidate in candidates:
                    by_artifact.setdefault(candidate.artifact_id, []).append(candidate)
                for artifact_id, artifact_candidates in by_artifact.items():
                    reservation = await lifecycle.reserve_cleanup(
                        CleanupReservationRequest(
                            permit=permit,
                            checkpoint_object_id=checkpoint_object_id,
                            ordered_candidates=tuple(
                                checkpoint_object
                                for item in artifact_candidates
                                for checkpoint_object in item.checkpoint_objects
                            ),
                            required_count=len(artifact_candidates),
                            protected_versions=protected,
                        )
                    )
                    claimed_by_version = self._checkpoint_objects_by_version(
                        reservation.claimed
                    )
                    if len(claimed_by_version) != len(artifact_candidates):
                        raise RuntimeError(
                            "checkpoint component returned an incomplete cleanup reservation"
                        )
                    expected = {
                        item.version_id: frozenset(item.checkpoint_objects)
                        for item in artifact_candidates
                    }
                    if any(
                        version_id not in expected
                        or frozenset(claimed) != expected[version_id]
                        for version_id, claimed in claimed_by_version.items()
                    ):
                        raise RuntimeError(
                            "checkpoint component returned a partial checkpoint_object-version reservation"
                        )
                    reservations.append(
                        CleanupReservationBinding(
                            artifact_id=artifact_id,
                            spmd_id=spmd_id,
                            component_id=component_id,
                            reservation=reservation,
                        )
                    )
            self._validate_artifact_intents(tuple(reservations), existing_manifests)
        except BaseException as primary_error:
            cleanup_errors = await self._rollback_reservations(
                tuple(reservations), permit
            )
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "cleanup reservation and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        return tuple(reservations)

    async def publish(
        self,
        reservations: tuple[CleanupReservationBinding, ...],
        permit: CheckpointCommitPermit,
    ) -> None:
        try:
            for binding in reservations:
                registration = self._require_component(
                    binding.spmd_id, binding.component_id
                )
                lifecycle = registration.object_lifecycle
                if lifecycle is None:
                    raise RuntimeError(
                        "checkpoint object lifecycle capability is not registered"
                    )
                await lifecycle.commit_cleanup(binding.reservation, permit)
        except BaseException as primary_error:
            cleanup_errors = await self._rollback_reservations(reservations, permit)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "retention publication and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def rollback_save(
        self,
        reservations: tuple[CleanupReservationBinding, ...],
        permit: CheckpointCommitPermit,
    ) -> None:
        errors = await self._rollback_reservations(reservations, permit)
        self._raise_errors("retention rollback failed", errors)

    async def cleanup(
        self,
        cleanup_intents: tuple[CheckpointCleanupIntent, ...],
    ) -> None:
        errors: list[BaseException] = []
        for intent in cleanup_intents:
            try:
                for binding in intent.reservations:
                    registration = self._require_component(
                        binding.spmd_id, binding.component_id
                    )
                    cleanup = registration.object_cleanup
                    if cleanup is None:
                        raise RuntimeError(
                            "object cleanup capability is not registered"
                        )
                    await cleanup.cleanup(binding.reservation)
                    for checkpoint_object in binding.reservation.claimed:
                        await self._io_lane.call(
                            self._repository.mark_checkpoint_object_cleaned_up,
                            binding.spmd_id,
                            checkpoint_object,
                        )
                await self._io_lane.call(
                    self._repository.complete_cleanup_intent, intent
                )
            except BaseException as error:
                errors.append(error)
        self._raise_errors("retention cleanup failed", errors)

    @staticmethod
    def cleanup_intents(
        reservations: tuple[CleanupReservationBinding, ...],
    ) -> tuple[CheckpointCleanupIntent, ...]:
        if not reservations:
            return ()
        return tuple(
            CheckpointCleanupIntent(artifact_id=artifact_id, reservations=bindings)
            for artifact_id, bindings in _group_by_artifact(reservations).items()
        )

    async def _rollback_reservations(
        self,
        reservations: tuple[CleanupReservationBinding, ...],
        permit: CheckpointCommitPermit,
    ) -> list[BaseException]:
        errors: list[BaseException] = []
        for binding in reversed(reservations):
            registration = self._require_component(
                binding.spmd_id, binding.component_id
            )
            lifecycle = registration.object_lifecycle
            if lifecycle is None:
                raise RuntimeError(
                    "checkpoint object lifecycle capability is not registered"
                )
            try:
                await lifecycle.rollback_cleanup(binding.reservation, permit)
            except BaseException as error:
                errors.append(error)
        return errors

    def _require_component(
        self, spmd_id: str, component_id: str
    ) -> CheckpointComponentRegistration:
        registry = self._registries.get(spmd_id)
        if registry is None:
            raise KeyError(f"checkpoint component registry not found for {spmd_id!r}")
        return registry.require(component_id)

    def _policy_for(
        self, registration: CheckpointComponentRegistration
    ) -> CheckpointRetentionPolicy:
        domain = registration.descriptor.retention_domain
        if domain is None:
            return self._retention_policy
        try:
            return self._retention_domains[domain]
        except KeyError as error:
            raise ValueError(
                f"retention domain is not configured: {domain!r}"
            ) from error

    def _artifact_retention_limit(self, manifest: CheckpointManifest) -> int | None:
        limits: list[int] = []
        for spmd_id, snapshot in manifest.spmds.items():
            for component_id in snapshot.components:
                registration = self._require_component(spmd_id, component_id)
                if not registration.descriptor.checkpoint_objects:
                    continue
                policy = self._policy_for(registration)
                if policy.retain_forever:
                    return None
                if policy.retain_last is None:
                    continue
                limits.append(policy.retain_last)
        if not limits:
            return None
        return max(limits)

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
                f"component {registration.descriptor.component_id!r}"
            )
        return matches[0]

    def _group_checkpoint_objects(
        self,
        manifests: tuple[CheckpointManifest, ...],
        cleaned: frozenset[tuple[str, CheckpointObjectRef]],
    ) -> dict[tuple[str, str, str], tuple[_PublishedTargetVersion, ...]]:
        active_pins = self._active_retain_forever_checkpoint_ids(manifests)
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
                            if (spmd_id, checkpoint_object) in cleaned:
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
                                manifest.checkpoint_id,
                                manifest.checkpoint_sequence,
                                next(iter(version_ids)),
                                tuple(checkpoint_objects),
                                protected,
                            )
                        )
        return {key: tuple(values) for key, values in grouped.items()}

    def _validate_artifact_intents(
        self,
        reservations: tuple[CleanupReservationBinding, ...],
        manifests: tuple[CheckpointManifest, ...],
    ) -> None:
        manifest_by_id = {manifest.checkpoint_id: manifest for manifest in manifests}
        covered = {
            (
                binding.artifact_id,
                binding.spmd_id,
                binding.component_id,
                binding.reservation.checkpoint_object_id,
            )
            for binding in reservations
        }
        for artifact_id in {binding.artifact_id for binding in reservations}:
            manifest = manifest_by_id.get(artifact_id)
            if manifest is None:
                raise RuntimeError(
                    f"cleanup reservation references unknown artifact {artifact_id!r}"
                )
            for spmd_id, snapshot in manifest.spmds.items():
                registry = self._registries.get(spmd_id)
                if registry is None:
                    raise KeyError(
                        f"checkpoint component registry not found for {spmd_id!r}"
                    )
                for component_id in snapshot.components:
                    registration = registry.require(component_id)
                    required = {
                        item.checkpoint_object_id
                        for item in registration.descriptor.checkpoint_objects
                        if item.required_for_restore
                    }
                    if not required:
                        continue
                    covered_ids = {
                        object_id
                        for current_artifact, current_spmd, current_component, object_id in covered
                        if current_artifact == artifact_id
                        and current_spmd == spmd_id
                        and current_component == component_id
                    }
                    if not required.issubset(covered_ids):
                        raise RuntimeError(
                            "cleanup reservation does not cover complete restore artifact "
                            f"artifact={artifact_id!r} spmd={spmd_id!r} component={component_id!r}"
                        )

    @staticmethod
    def _active_retain_forever_checkpoint_ids(
        manifests: tuple[CheckpointManifest, ...],
    ) -> frozenset[str]:
        latest_by_name: dict[str, CheckpointManifest] = {}
        for manifest in manifests:
            for reference in manifest.references:
                if (
                    not isinstance(reference, NameReference)
                    or not reference.retain_forever
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


def _group_by_artifact(
    reservations: tuple[CleanupReservationBinding, ...],
) -> dict[str, tuple[CleanupReservationBinding, ...]]:
    grouped: dict[str, list[CleanupReservationBinding]] = {}
    for reservation in reservations:
        grouped.setdefault(reservation.artifact_id, []).append(reservation)
    return {artifact_id: tuple(items) for artifact_id, items in grouped.items()}
