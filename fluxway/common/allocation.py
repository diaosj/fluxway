from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

from fluxway.common.coherence import (
    CoherenceSubject,
    RetirementInvariantError,
    CoherenceInstance,
    UnsupportedAllocationHandoffError,
)


class AllocationRelation(IntEnum):
    DISJOINT = 0
    OVERLAP = 1
    EXACT = 2


@dataclass(frozen=True, slots=True)
class AllocationIdentity:
    backend: str
    device: str
    root_address: int
    root_nbytes: int
    view_address: int
    view_nbytes: int

    def relation(self, other: AllocationIdentity) -> AllocationRelation:
        if self.backend != other.backend or self.device != other.device:
            return AllocationRelation.DISJOINT
        if (
            self.root_address == other.root_address
            and self.root_nbytes == other.root_nbytes
            and self.view_address == other.view_address
            and self.view_nbytes == other.view_nbytes
        ):
            return AllocationRelation.EXACT
        self_end = self.view_address + self.view_nbytes
        other_end = other.view_address + other.view_nbytes
        if self_end <= other.view_address or other_end <= self.view_address:
            return AllocationRelation.DISJOINT
        return AllocationRelation.OVERLAP


@dataclass(frozen=True, slots=True)
class AllocationDescriptor:
    identity: AllocationIdentity
    owner: Any


@dataclass(frozen=True, slots=True)
class AllocationBinding:
    instance: CoherenceInstance
    descriptor: AllocationDescriptor


class AllocationRegistry:
    def __init__(self) -> None:
        self._bindings: dict[CoherenceInstance, AllocationBinding] = {}
        self._retired: dict[CoherenceSubject, deque[CoherenceInstance]] = defaultdict(
            deque
        )

    def bind(
        self,
        instance: CoherenceInstance,
        descriptor: AllocationDescriptor,
    ) -> AllocationBinding:
        if instance in self._bindings:
            raise UnsupportedAllocationHandoffError(
                f"allocation instance is already bound: {instance}."
            )
        binding = AllocationBinding(instance, descriptor)
        self._bindings[instance] = binding
        return binding

    def handoff(
        self,
        old: CoherenceInstance,
        new: CoherenceInstance,
        descriptor: AllocationDescriptor,
    ) -> AllocationBinding:
        self.require_handoff(old, new, descriptor)
        del self._bindings[old]
        binding = AllocationBinding(new, descriptor)
        self._bindings[new] = binding
        self._retired[old.subject].append(old)
        return binding

    def require_handoff(
        self,
        old: CoherenceInstance,
        new: CoherenceInstance,
        descriptor: AllocationDescriptor,
    ) -> None:
        old_binding = self._bindings.get(old)
        if old_binding is None:
            raise UnsupportedAllocationHandoffError(
                f"allocation handoff source is not bound: {old}."
            )
        if new in self._bindings:
            raise UnsupportedAllocationHandoffError(
                f"allocation handoff destination is already bound: {new}."
            )
        if (
            old_binding.descriptor.identity.relation(descriptor.identity)
            is not AllocationRelation.EXACT
        ):
            raise UnsupportedAllocationHandoffError(
                "allocation handoff requires an exact allocation and byte span."
            )

    def detach(self, instance: CoherenceInstance) -> AllocationBinding:
        """Drop an identity binding after payload ownership has transferred."""
        try:
            return self._bindings.pop(instance)
        except KeyError as exc:
            raise RetirementInvariantError(
                f"cannot detach an unbound tensor instance: {instance}."
            ) from exc

    def forget_retired(self, instance: CoherenceInstance) -> None:
        """Remove a logically retired handoff tombstone after proof consumption."""
        retired = self._retired.get(instance.subject)
        if retired is None:
            return
        try:
            retired.remove(instance)
        except ValueError as exc:
            raise RetirementInvariantError(
                f"retired allocation instance is not tracked: {instance}."
            ) from exc
        if not retired:
            del self._retired[instance.subject]

    def binding(self, instance: CoherenceInstance) -> AllocationBinding:
        try:
            return self._bindings[instance]
        except KeyError as exc:
            raise UnsupportedAllocationHandoffError(
                f"tensor instance has no allocation binding: {instance}."
            ) from exc

    def is_bound(self, instance: CoherenceInstance) -> bool:
        return instance in self._bindings

    def fence(self) -> None:
        self._bindings.clear()
        self._retired.clear()
