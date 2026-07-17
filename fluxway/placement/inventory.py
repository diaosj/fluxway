from __future__ import annotations

from dataclasses import dataclass

from fluxway.common.compiled_graph import CompiledGraph, CompiledSPMDDeclaration

from .mathematics import SPMDClassRef


@dataclass(frozen=True, slots=True)
class PlacementInventory:
    declarations: tuple[CompiledSPMDDeclaration, ...]

    @classmethod
    def from_compiled_graph(cls, graph: CompiledGraph) -> PlacementInventory:
        return cls(declarations=graph.declarations)

    @property
    def spmd_ids(self) -> tuple[str, ...]:
        return tuple(declaration.spmd_id for declaration in self.declarations)

    @property
    def resource_names(self) -> frozenset[str]:
        return frozenset(
            resource
            for declaration in self.declarations
            for vector in (
                declaration.resource_profile.base,
                declaration.resource_profile.scale.step
                if declaration.resource_profile.scale is not None
                else {},
            )
            for resource, quantity in vector.items()
            if quantity > 0
        )

    def spmd_ids_for(self, class_ref: SPMDClassRef) -> tuple[str, ...]:
        return tuple(
            declaration.spmd_id
            for declaration in self.declarations
            if declaration.spmd_class is class_ref.spmd_class
        )
