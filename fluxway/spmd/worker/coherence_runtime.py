from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Coroutine, Iterable, Mapping
from dataclasses import dataclass
import logging
from typing import Any, Protocol

from fluxway.common.allocation import AllocationRegistry
from fluxway.common.payload_lifetime import PayloadLifetime, payload_handle
from fluxway.rpc.protocol import ReceiverIngressCompletion
from fluxway.common.coherence import (
    CoherenceDirectory,
    CoherenceSubject,
    CoherenceStateTransitionError,
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
BatchEnqueue = Callable[[TransportFrameBatch], Awaitable[Any]]


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

logger = logging.getLogger(__name__)


class _ChildDeliveryReceipts:
    __slots__ = ("_completed", "_directory", "_tokens")

    def __init__(self, directory: CoherenceDirectory) -> None:
        self._directory = directory
        self._tokens: list[GrantToken] = []
        self._completed = False

    def add(self, token: GrantToken) -> None:
        if self._completed:
            raise RuntimeError("cannot register a child delivery after completion.")
        self._tokens.append(self._directory.register_child_delivery(token))

    def succeed(self) -> None:
        self._complete(delivered=True)

    def fail(self) -> None:
        self._complete(delivered=False)

    def _complete(self, *, delivered: bool) -> None:
        if self._completed:
            raise RuntimeError("child delivery receipts completed twice.")
        self._completed = True
        for token in reversed(self._tokens):
            self._directory.complete_child_delivery(token, delivered=delivered)


class CoherenceBatchRetireRuntime:
    """Owns one admitted publication's complete target fanout."""

    def __init__(
        self,
        directory: CoherenceDirectory,
        sequencer: CoherenceSequencer[Callable[[], Awaitable[Any]]],
    ) -> None:
        self._directory = directory
        self._admission_runtime = CoherenceAdmissionRuntime(directory)
        self._sequencer = sequencer
        self._next_publication_id = 1

    def retire_publication(
        self,
        compiled_plan: CompiledSendPlan,
        batches: Iterable[TransportFrameBatch],
        enqueue: BatchEnqueue,
    ) -> Coroutine[Any, Any, Any]:
        publication_batches = tuple(batches)
        if not publication_batches:
            raise ValueError("coherence publication requires at least one batch.")
        publication = compiled_plan.describe_publication(publication_batches)
        instances = tuple(item.instance for item in publication.publications)
        publication_id = self._next_publication_id
        self._next_publication_id += 1
        expected_children = tuple(
            (
                instance,
                _instance_child_slot(compiled_plan, batch, instance),
                batch.context_index,
                batch.target_rank,
                batch.batch_seq,
            )
            for batch in publication_batches
            for instance in _batch_instances(compiled_plan, batch)
        )
        if len(set(expected_children)) != len(expected_children):
            raise CoherenceStateTransitionError(
                "coherence publication contains duplicate target batch identities."
            )
        handles = _publication_payload_handles(publication_batches)
        lifetime = (
            PayloadLifetime(publication_id, handles, expected_children)
            if expected_children
            else None
        )

        async def admitted_delivery() -> Any:
            receipts = _ChildDeliveryReceipts(self._directory)
            try:
                await self._delegate_publication_reads(
                    compiled_plan,
                    publication_batches,
                    receipts,
                )
                result = await _await_all_or_cancel(
                    tuple(enqueue(batch) for batch in publication_batches)
                )
                for completion in result:
                    if not isinstance(completion, ReceiverIngressCompletion):
                        raise TypeError(
                            "coherence publication enqueue must return "
                            "ReceiverIngressCompletion."
                        )
                    if not completion.receiver_accepted:
                        raise CoherenceStateTransitionError(
                            "receiver ingress completion was not accepted."
                        )
                    if not completion.producer_retention_released:
                        raise CoherenceStateTransitionError(
                            "carrier retained producer payload without a release proof."
                        )
                for batch in publication_batches:
                    for instance in _batch_instances(compiled_plan, batch):
                        child_slot = _instance_child_slot(
                            compiled_plan, batch, instance
                        )
                        if lifetime is None:
                            raise CoherenceStateTransitionError(
                                "coherence batch lifetime is missing for an instance."
                            )
                        lifetime.accept_child(
                            batch.context_index,
                            batch.target_rank,
                            batch.batch_seq,
                            child_slot,
                            instance,
                        )
                if lifetime is not None:
                    lifetime.release_transport_retention()
            except BaseException as error:
                if lifetime is not None:
                    lifetime.fence(error)
                receipts.fail()
                raise
            receipts.succeed()
            return tuple(completion.result for completion in result)

        if not instances:
            return admitted_delivery()

        return self._sequencer.commit_payload(instances, admitted_delivery)

    async def _delegate_publication_reads(
        self,
        compiled_plan: CompiledSendPlan,
        batches: tuple[TransportFrameBatch, ...],
        receipts: _ChildDeliveryReceipts,
    ) -> None:
        seen: set[tuple[CoherenceSubject, int, int]] = set()
        delegation_tasks: list[Coroutine[Any, Any, None]] = []
        for batch in batches:
            for subject, epoch, child_slot in _delegated_reads(compiled_plan, batch):
                key = (subject, epoch, child_slot)
                if key in seen:
                    continue
                seen.add(key)
                delegation_tasks.append(
                    self._delegate_read(subject, epoch, child_slot, receipts)
                )
        if delegation_tasks:
            await _await_all_or_cancel(tuple(delegation_tasks))

    async def _delegate_read(
        self,
        subject: CoherenceSubject,
        epoch: int,
        child_slot: int,
        receipts: _ChildDeliveryReceipts,
    ) -> None:
        result = self._directory.delegate_local_payload(subject, epoch, child_slot)
        if isinstance(result, GrantReady):
            token = result.token
        elif isinstance(result, GrantPending):
            token = await self._admission_runtime.await_grant(result)
        elif isinstance(result, GrantRejected):
            raise result.error
        else:
            raise TypeError(f"unsupported grant result={type(result).__name__}.")
        receipts.add(token)


async def _await_all_or_cancel(
    awaitables: tuple[Awaitable[Any], ...],
) -> tuple[Any, ...]:
    if not awaitables:
        raise ValueError("coherence publication requires at least one enqueue.")
    tasks = tuple(asyncio.create_task(awaitable) for awaitable in awaitables)
    try:
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        failures: list[BaseException] = []
        for task in done:
            if task.cancelled():
                raise asyncio.CancelledError()
            error = task.exception()
            if error is not None:
                failures.append(error)
        if failures:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if len(failures) == 1:
                raise failures[0]
            raise BaseExceptionGroup("coherence publication enqueue failed", failures)
        if pending:
            await asyncio.gather(*pending)
        return tuple(task.result() for task in tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class CoherenceAdmissionRuntime:
    """Event-loop adapter for synchronous directory progress."""

    def __init__(self, directory: CoherenceDirectory) -> None:
        self._directory = directory

    async def await_grant(self, pending: GrantPending) -> GrantToken:
        try:

            def poll_result():
                return self._directory.pending_result(pending.request_id)

            result = poll_result()
            if result is None:
                event = asyncio.Event()
                remove = self._directory.add_progress_listener(event.set)
                try:
                    while (result := poll_result()) is None:
                        event.clear()
                        if (result := poll_result()) is not None:
                            break
                        await event.wait()
                finally:
                    remove()

            taken = self._directory.take_pending_result(pending.request_id)
            if isinstance(taken, GrantRejected):
                raise taken.error
            if not isinstance(taken, GrantReady):
                raise TypeError(
                    f"unsupported pending grant result={type(taken).__name__}."
                )
            return taken.token
        except asyncio.CancelledError:
            self._directory.cancel_pending_request(pending.request_id)
            raise

    async def admit(self, instance: CoherenceInstance) -> None:
        await self._wait_until(
            lambda: self._directory.can_admit(instance),
        )
        self._directory.admit(instance)

    async def await_quiescent(self, instance: CoherenceInstance) -> None:
        if not self._directory.is_quiescent(instance):
            await self.admit(instance)
        await self._wait_until(lambda: self._directory.is_quiescent(instance))

    async def await_quiescent_many(
        self, instances: Iterable[CoherenceInstance]
    ) -> None:
        unique = tuple(dict.fromkeys(instances))
        for instance in unique:
            if not self._directory.is_quiescent(instance):
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


def _instance_child_slot(
    compiled_plan: CompiledSendPlan,
    batch: TransportFrameBatch,
    instance: CoherenceInstance,
) -> int:
    slots = {
        compiled_plan.coherence_child_slot(item.transfer_code, batch.target_rank)
        for item in batch.delegations
        if compiled_plan.coherence_instance(item.transfer_code, item.epoch) == instance
    }
    if len(slots) != 1:
        raise CoherenceStateTransitionError(
            "one instance in a transport batch must map to exactly one child slot."
        )
    return next(iter(slots))


def _batch_instances(
    compiled_plan: CompiledSendPlan,
    batch: TransportFrameBatch,
) -> tuple[CoherenceInstance, ...]:
    return tuple(
        dict.fromkeys(
            compiled_plan.coherence_instance(item.transfer_code, item.epoch)
            for item in batch.delegations
        )
    )


def _publication_payload_handles(
    batches: tuple[TransportFrameBatch, ...],
) -> tuple[Any, ...]:
    seen: set[int] = set()
    handles = []
    for batch in batches:
        for _header, segments in batch.frames:
            for segment in segments:
                marker = id(segment)
                if marker in seen:
                    continue
                seen.add(marker)
                handles.append(payload_handle(segment))
    return tuple(handles)


def _ordered_dependencies(
    dependencies: Iterable[CoherenceDependency],
) -> tuple[CoherenceDependency, ...]:
    return tuple(dict.fromkeys(dependencies))


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
    seen_instances: set[CoherenceInstance] = set()
    for grant in grants:
        if not grant.parent_return_required:
            continue
        instance = CoherenceInstance(grant.subject, grant.epoch)
        if instance in revoked_instances:
            continue
        if instance in seen_instances:
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
        seen_instances.add(instance)
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
                self.allocation_registry.forget_retired(instance)
                continue
            self.allocation_registry.detach(instance)
            self.allocation_registry.forget_retired(instance)
