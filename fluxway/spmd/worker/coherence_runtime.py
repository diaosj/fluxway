from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from fluxway.common.allocation import AllocationRegistry
from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    GrantToken,
    RetirementInvariantError,
    CoherenceInstance,
    GrantPending,
    GrantReady,
    GrantRejected,
)
from fluxway.common.types.tensor import TransportTensor
from fluxway.transport.communication import TransportFrameBatch
from fluxway.spmd.worker.data_plane import CompiledSendPlan
from fluxway.spmd.worker.coherence_sequencer import CoherenceSequencer


@dataclass(frozen=True, slots=True)
class ParentReturnRoute:
    subject: CoherenceSubject
    route_code: int
    epoch: int
    parent_spmd_id: str
    parent_rank: int


ParentReturnSender = Callable[[tuple[ParentReturnRoute, ...]], Awaitable[None]]


class InputGrant(Protocol):
    @property
    def parent_spmd_id(self) -> str | None: ...

    @property
    def parent_rank(self) -> int: ...

    @property
    def subject(self) -> CoherenceSubject: ...

    @property
    def route_code(self) -> int: ...

    @property
    def epoch(self) -> int: ...

    @property
    def local_access_token(self) -> GrantToken: ...

    @property
    def parent_return_required(self) -> bool: ...


CoherenceDependency = CoherenceInstance


class CoherenceBatchRetireRuntime:
    """Adapts ordered coherence retirement to a pure data-plane enqueue."""

    def __init__(
        self,
        directory: CoherenceDirectory,
        sequencer: CoherenceSequencer[Callable[[], Awaitable[Any]]],
    ) -> None:
        self._directory = directory
        self._admission_runtime = CoherenceAdmissionRuntime(directory)
        self._sequencer = sequencer

    def retire(
        self,
        compiled_plan: CompiledSendPlan,
        batch: TransportFrameBatch,
        enqueue: Callable[[], Awaitable[Any]],
    ) -> Coroutine[Any, Any, Any]:
        instances = self._batch_instances(compiled_plan, batch)

        async def admitted_delivery() -> Any:
            tokens = await self._delegate_batch_reads(compiled_plan, batch)
            completed = False

            def complete(*, delivered: bool) -> None:
                nonlocal completed
                if completed:
                    return
                completed = True
                for token in reversed(tokens):
                    self._directory.complete_child_delivery(
                        token,
                        delivered=delivered,
                    )

            try:
                result = await enqueue()
            except BaseException:
                complete(delivered=False)
                raise
            complete(delivered=True)
            return result

        if not instances:
            return admitted_delivery()
        return self._sequencer.commit_payload(instances, admitted_delivery)

    @staticmethod
    def _batch_instances(
        compiled_plan: CompiledSendPlan,
        batch: TransportFrameBatch,
    ) -> tuple[CoherenceInstance, ...]:
        instances: list[CoherenceInstance] = []
        seen: set[CoherenceInstance] = set()
        for item in batch.delegations:
            instance = compiled_plan.coherence_instance(
                item.transfer_code,
                item.epoch,
            )
            if instance not in seen:
                seen.add(instance)
                instances.append(instance)
        return tuple(instances)

    async def _delegate_batch_reads(
        self,
        compiled_plan: CompiledSendPlan,
        batch: TransportFrameBatch,
    ) -> tuple[GrantToken, ...]:
        tokens: list[GrantToken] = []
        try:
            for subject, epoch, child_slot in _delegated_reads(compiled_plan, batch):
                result = self._directory.delegate_local_payload(
                    subject, epoch, child_slot
                )
                if isinstance(result, GrantReady):
                    tokens.append(self._directory.register_child_delivery(result.token))
                elif isinstance(result, GrantPending):
                    token = await self._admission_runtime.await_grant(result)
                    tokens.append(self._directory.register_child_delivery(token))
                elif isinstance(result, GrantRejected):
                    raise result.error
                else:
                    raise TypeError(
                        f"unsupported grant result={type(result).__name__}."
                    )
        except BaseException:
            for token in reversed(tokens):
                self._directory.complete_child_delivery(token, delivered=False)
            raise
        return tuple(tokens)

    async def _complete_delivery(
        self,
        tokens: tuple[GrantToken, ...],
        delivery: Awaitable[Any],
    ) -> Any:
        completed = False

        def complete(*, delivered: bool) -> None:
            nonlocal completed
            if completed:
                return
            completed = True
            for token in reversed(tokens):
                self._directory.complete_child_delivery(token, delivered=delivered)

        try:
            result = await delivery
        except BaseException:
            complete(delivered=False)
            raise
        complete(delivered=True)
        return result


class CoherenceAdmissionRuntime:
    """Event-loop adapter for synchronous directory progress."""

    def __init__(self, directory: CoherenceDirectory) -> None:
        self._directory = directory

    async def await_grant(self, pending: GrantPending) -> GrantToken:
        try:
            while True:
                result = self._directory.pending_result(pending.request_id)
                if result is not None:
                    result = self._directory.take_pending_result(pending.request_id)
                    if isinstance(result, GrantRejected):
                        raise result.error
                    if not isinstance(result, GrantReady):
                        raise TypeError(
                            f"unsupported pending grant result={type(result).__name__}."
                        )
                    return result.token
                event = asyncio.Event()
                remove = self._directory.add_progress_listener(event.set)
                try:
                    result = self._directory.pending_result(pending.request_id)
                    if result is None:
                        await event.wait()
                finally:
                    remove()
        except asyncio.CancelledError:
            self._directory.cancel_pending_request(pending.request_id)
            raise

    async def admit(self, instance: CoherenceInstance) -> None:
        await self._wait_until(
            lambda: self._directory.can_admit(instance),
        )
        self._directory.admit(instance)

    async def await_quiescent(self, instance: CoherenceInstance) -> None:
        await self.admit(instance)
        await self._wait_until(lambda: self._directory.is_quiescent(instance))

    async def await_quiescent_many(
        self, instances: Iterable[CoherenceInstance]
    ) -> None:
        unique: list[CoherenceInstance] = []
        seen: set[CoherenceInstance] = set()
        for instance in instances:
            if instance not in seen:
                seen.add(instance)
                unique.append(instance)
        for instance in unique:
            await self.admit(instance)
        await self._wait_until(
            lambda: all(self._directory.is_quiescent(instance) for instance in unique)
        )

    async def await_deliveries_complete(
        self,
        instance: CoherenceInstance,
        child_bitmap: int,
    ) -> None:
        await self.admit(instance)
        await self._wait_until(
            lambda: self._directory.deliveries_complete(instance, child_bitmap)
        )

    async def _wait_until(self, condition: Callable[[], bool]) -> None:
        if condition():
            return
        event = asyncio.Event()
        remove = self._directory.add_progress_listener(event.set)
        try:
            while not condition():
                event.clear()
                if condition():
                    break
                await event.wait()
        finally:
            remove()


def _delegated_reads(
    compiled_plan: CompiledSendPlan,
    batch: TransportFrameBatch,
) -> tuple[tuple[CoherenceSubject, int, int], ...]:
    reads: list[tuple[CoherenceSubject, int, int]] = []
    seen: set[tuple[int, int, int]] = set()
    for item in batch.delegations:
        key = (item.transfer_code, item.epoch, batch.target_rank)
        if key in seen:
            continue
        seen.add(key)
        instance = compiled_plan.coherence_instance(item.transfer_code, item.epoch)
        reads.append(
            (
                instance.subject,
                instance.epoch,
                compiled_plan.coherence_child_slot(
                    item.transfer_code, batch.target_rank
                ),
            )
        )
    return tuple(reads)


def _ordered_dependencies(
    dependencies: Iterable[CoherenceDependency],
) -> tuple[CoherenceDependency, ...]:
    values: list[CoherenceDependency] = []
    seen: set[CoherenceDependency] = set()
    for dependency in dependencies:
        if dependency not in seen:
            seen.add(dependency)
            values.append(dependency)
    return tuple(values)


class ParentGrantReturnRuntime:
    """Owns the parent-return transaction and transport ordering."""

    def __init__(
        self,
        directory: CoherenceDirectory,
        send: ParentReturnSender,
    ) -> None:
        self._directory = directory
        self._send = send

    async def return_parent_grants(
        self,
        routes: tuple[ParentReturnRoute, ...],
    ) -> frozenset[CoherenceInstance]:
        if not routes:
            raise ValueError("parent return requires at least one route.")
        for route in routes:
            if not self._directory.parent_return_ready(route.subject, route.epoch):
                raise RetirementInvariantError(
                    "parent return route is not quiescent: "
                    f"subject={route.subject} epoch={route.epoch}."
                )
        tokens = tuple(
            self._directory.begin_parent_return(route.subject, route.epoch)
            for route in routes
        )
        groups: dict[tuple[str, int, int], list[ParentReturnRoute]] = {}
        for route in routes:
            groups.setdefault(
                (route.parent_spmd_id, route.parent_rank, route.epoch),
                [],
            ).append(route)
        await asyncio.gather(*(self._send(tuple(group)) for group in groups.values()))
        for token in tokens:
            self._directory.complete_parent_return(token)
        return frozenset(CoherenceInstance(token.key, token.epoch) for token in tokens)


def _parent_return_routes_for(
    grants: Iterable[InputGrant],
    revoked_instances: frozenset[CoherenceInstance],
    parent_handlers: Mapping[str, Mapping[int, object]],
) -> tuple[ParentReturnRoute, ...]:
    routes: list[ParentReturnRoute] = []
    for grant in grants:
        if not grant.parent_return_required:
            continue
        instance = CoherenceInstance(grant.subject, grant.epoch)
        if instance in revoked_instances:
            continue
        parent_spmd_id = grant.parent_spmd_id
        if parent_spmd_id is None:
            raise ValueError(
                "parent_spmd_id cannot be None if parent_return_required is True."
            )
        handlers = parent_handlers.get(parent_spmd_id)
        if handlers is None or grant.parent_rank not in handlers:
            raise ValueError(
                "missing direct parent handler "
                f"sender={parent_spmd_id}:{grant.parent_rank}."
            )
        routes.append(
            ParentReturnRoute(
                subject=grant.subject,
                route_code=grant.route_code,
                epoch=grant.epoch,
                parent_spmd_id=parent_spmd_id,
                parent_rank=grant.parent_rank,
            )
        )
    return tuple(routes)


@dataclass(frozen=True, slots=True)
class InputCoherenceReleasePlan:
    grants: tuple[InputGrant, ...]
    alias_dependencies: tuple[CoherenceDependency, ...]
    parent_return_routes: tuple[ParentReturnRoute, ...]
    quiescence_dependencies: tuple[CoherenceDependency, ...]
    tensors: tuple[TransportTensor, ...]
    instances: tuple[CoherenceInstance, ...]

    @classmethod
    def create(
        cls,
        *,
        grants: Iterable[InputGrant],
        alias_dependencies: Iterable[CoherenceDependency],
        revoked_instances: frozenset[CoherenceInstance],
        parent_handlers: Mapping[str, Mapping[int, object]],
        tensors: Iterable[TransportTensor],
    ) -> InputCoherenceReleasePlan:
        grant_values = tuple(grants)
        alias_values = _ordered_dependencies(alias_dependencies)
        tensor_values = tuple(tensors)
        grant_instances = tuple(
            CoherenceInstance(grant.subject, grant.epoch) for grant in grant_values
        )
        parent_routes = _parent_return_routes_for(
            grant_values,
            revoked_instances,
            parent_handlers,
        )
        tensor_instances = tuple(
            instance
            for tensor in tensor_values
            for instance in tensor._coherence_instances
        )
        quiescence_dependencies = _ordered_dependencies(grant_instances + alias_values)
        instances = _ordered_dependencies(tensor_instances + grant_instances)
        return cls(
            grants=grant_values,
            alias_dependencies=alias_values,
            parent_return_routes=parent_routes,
            quiescence_dependencies=quiescence_dependencies,
            tensors=tensor_values,
            instances=instances,
        )


@dataclass(frozen=True, slots=True)
class InputCoherenceRelease:
    """Completes one operator input's coherence lifecycle synchronously."""

    directory: CoherenceDirectory
    admission_runtime: CoherenceAdmissionRuntime
    parent_return_runtime: ParentGrantReturnRuntime
    allocation_registry: AllocationRegistry
    plan: InputCoherenceReleasePlan

    async def complete(self) -> None:
        self._release_local_access()
        await self._await_quiescence()
        self._invalidate_tensors()
        returned_instances = await self._return_parent_grants()
        self._release_payloads()
        self._retire_instances(returned_instances)

    def _release_local_access(self) -> None:
        for grant in self.plan.grants:
            self.directory.release(grant.local_access_token)

    async def _await_quiescence(self) -> None:
        await self.admission_runtime.await_quiescent_many(
            self.plan.quiescence_dependencies
        )

    def _invalidate_tensors(self) -> None:
        for tensor in self.plan.tensors:
            tensor._invalidate_runtime_access()

    async def _return_parent_grants(self) -> frozenset[CoherenceInstance]:
        if not self.plan.parent_return_routes:
            return frozenset()
        return await self.parent_return_runtime.return_parent_grants(
            self.plan.parent_return_routes
        )

    def _release_payloads(self) -> None:
        for tensor in self.plan.tensors:
            tensor._release_runtime_payload()

    def _retire_instances(
        self,
        returned_instances: frozenset[CoherenceInstance],
    ) -> None:
        for instance in self.plan.instances:
            if instance in returned_instances:
                self.directory.require_returned_instance(instance, returned_instances)
            else:
                self.directory.require_current_entry_retirement(instance)
            if not self.allocation_registry.is_bound(instance):
                continue
            self.allocation_registry.retire(instance)
            self.allocation_registry.advance_watermark(
                instance.subject,
                instance.epoch + 1,
            )
