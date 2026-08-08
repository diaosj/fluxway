"""Capability-composed checkpoint backend registrations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from fluxway.common.types.checkpoint import (
    CheckpointObjectRef,
    CheckpointCapability,
    ComponentCheckpoint,
    CheckpointRuntime,
    ComponentSpec,
    CheckpointCommitPermit,
    SaveSession,
    CommitSaveSession,
    RestoreSession,
    WorkerSaveResult,
    CheckpointHold,
    CleanupReservation,
    CleanupReservationRequest,
)


class CheckpointComponent(ABC):
    @property
    @abstractmethod
    def descriptor(self) -> ComponentSpec: ...

    @abstractmethod
    async def open(self, runtime: CheckpointRuntime) -> None: ...

    @abstractmethod
    async def start_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def save(self, session: SaveSession) -> WorkerSaveResult: ...

    @abstractmethod
    async def finish_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def commit_save(self, session: CommitSaveSession) -> ComponentCheckpoint: ...

    @abstractmethod
    async def rollback_save(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def on_checkpoint_available(self, session: SaveSession) -> None: ...

    @abstractmethod
    async def restore(self, session: RestoreSession) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...


class CheckpointObjectLifecycle(ABC):
    @abstractmethod
    async def hold_checkpoint_object(
        self,
        checkpoint_object: CheckpointObjectRef,
        owner_id: str,
        ttl_seconds: float,
        request_id: str,
    ) -> CheckpointHold: ...

    @abstractmethod
    async def renew_checkpoint_hold(
        self, lease: CheckpointHold, ttl_seconds: float
    ) -> CheckpointHold: ...

    @abstractmethod
    async def release_checkpoint_hold(self, lease: CheckpointHold) -> None: ...

    @abstractmethod
    async def reserve_cleanup(
        self, request: CleanupReservationRequest
    ) -> CleanupReservation: ...

    @abstractmethod
    async def commit_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None: ...

    @abstractmethod
    async def rollback_cleanup(
        self,
        reservation: CleanupReservation,
        permit: CheckpointCommitPermit,
    ) -> None: ...


class CheckpointObjectCleaner(ABC):
    @abstractmethod
    async def cleanup(
        self,
        reservation: CleanupReservation,
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class CheckpointComponentRegistration:
    descriptor: ComponentSpec
    component: CheckpointComponent
    object_lifecycle: CheckpointObjectLifecycle | None = None
    cleaner: CheckpointObjectCleaner | None = None

    def __post_init__(self) -> None:
        declared = self.descriptor.capabilities
        actual = {CheckpointCapability.PERSISTENCE}
        actual.add(CheckpointCapability.LIFECYCLE)
        if self.object_lifecycle is not None:
            actual.add(CheckpointCapability.RETENTION)
        if self.cleaner is not None:
            actual.add(CheckpointCapability.RECLAIMER)
        if declared != actual:
            raise ValueError(
                "checkpoint backend capability declaration does not match registration"
            )
        if any(
            checkpoint_object.keep.keep_last is not None
            for checkpoint_object in self.descriptor.checkpoint_objects
        ) and (self.object_lifecycle is None or self.cleaner is None):
            raise ValueError(
                "finite checkpoint_object cleanup requires keep and cleaner capabilities"
            )


class CheckpointComponentRegistry:
    """Immutable registrations with startup-validated, direct capability dispatch."""

    def __init__(
        self, registrations: tuple[CheckpointComponentRegistration, ...]
    ) -> None:
        component_ids = [
            registration.descriptor.component_id for registration in registrations
        ]
        if len(component_ids) != len(set(component_ids)):
            raise ValueError("checkpoint component IDs must be unique; duplicate found")
        self._registrations = registrations
        self._by_id = {
            registration.descriptor.component_id: registration
            for registration in registrations
        }

    def __iter__(self):
        return iter(self._registrations)

    def __len__(self) -> int:
        return len(self._registrations)

    def require(self, component_id: str) -> CheckpointComponentRegistration:
        registration = self._by_id.get(component_id)
        if registration is None:
            raise KeyError(f"checkpoint component is not registered: {component_id!r}")
        return registration

    async def open_all(self, runtime: CheckpointRuntime) -> None:
        opened: list[CheckpointComponent] = []
        try:
            for registration in self._registrations:
                await registration.component.open(runtime)
                opened.append(registration.component)
        except BaseException as primary_error:
            cleanup_errors: list[BaseException] = []
            for lifecycle in reversed(opened):
                try:
                    await lifecycle.close()
                except BaseException as close_error:
                    cleanup_errors.append(close_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint backend open and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise

    async def close_all(self) -> None:
        errors: list[BaseException] = []
        for registration in reversed(self._registrations):
            try:
                await registration.component.close()
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise BaseExceptionGroup("checkpoint backend close failed", errors)
