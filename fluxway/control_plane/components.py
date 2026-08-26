"""Shared registries for Controller-owned and Driver-owned modules."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from fluxway.common.types import CheckpointCut, ControllerState, DriverState

type ComponentState = ControllerState | DriverState


def driver_component_id(spmd_id: str, operator_name: str) -> str:
    if not isinstance(spmd_id, str) or not spmd_id:
        raise ValueError("driver component spmd_id must be non-empty")
    if not isinstance(operator_name, str) or not operator_name:
        raise ValueError("driver component operator_name must be non-empty")
    return f"{spmd_id}:{operator_name}"


@dataclass(frozen=True, slots=True)
class DriverComponentContext:
    component_id: str
    spmd_id: str
    operator_name: str
    operator_spec: Any
    static_args: Mapping[str, Any]
    controller_handle: Any
    world_size: int
    interleave_unit_size: int
    report_failure: Any


@runtime_checkable
class ControlPlaneComponent(Protocol):
    """Protocol for pluggable control-plane modules hosted by Controller or Driver.

    Mechanism & Architecture:
    -------------------------
    1. Hosting & Registration:
       Components are registered in a `_ComponentRegistry` (specifically `ControlPlaneComponentRegistry`
       for Controller-owned modules or `DriverComponentRegistry` for SPMD Driver-owned modules).
       Each component must define a unique `component_id`.

    2. Lifecycle State Broadcast:
       Whenever the host Controller or Driver undergoes a state transition in its Finite State
       Machine (FSM), the host calls `registry.notify_state(state)`. The registry sequentially
       invokes `await component.on_state(state)` for each registered component.
       `ComponentState` represents the host FSM's current state:
       - `ControllerState` (e.g., INITIALIZING, PENDING, TOPOLOGY_REGISTERING, READY, RUNNING, COMPLETING, FAILING, COMPLETED, FAILED)
       - `DriverState` (e.g., INITIALIZING, READY, RESTORING, RUNNING, BLOCKED, DRAINING, DRAINED, SNAPSHOTTING, ERROR, COMPLETED)

    3. State Checkpointing & Recovery:
       During checkpointing or fault recovery operations, the host calls `registry.checkpoint_state(cut)`
       to aggregate state dicts from all components via `checkpoint_state(cut)`. Upon restart or restore,
       `registry.restore_state(checkpoint_dict, cut)` distributes state back to each component via `restore_state(state, cut)`.

    Example:
    --------
    >>> class StepCounterComponent:
    ...     def __init__(self, component_id: str = "step_counter"):
    ...         self.component_id = component_id
    ...         self.step_count = 0
    ...
    ...     async def on_state(self, state: ComponentState) -> None:
    ...         if state == DriverState.RUNNING:
    ...             print(f"Component {self.component_id} activated on state: {state}")
    ...
    ...     def checkpoint_state(self, cut: CheckpointCut) -> dict[str, Any]:
    ...         return {"step_count": self.step_count}
    ...
    ...     def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
    ...         self.step_count = int(state.get("step_count", 0))
    """

    component_id: str

    async def on_state(self, state: ComponentState) -> None:
        """Handle host FSM lifecycle state transitions.

        Args:
            state: The new state of the host. For Controller-hosted components, this is a
                `ControllerState` (e.g. READY, RUNNING, COMPLETING). For Driver-hosted
                components, this is a `DriverState` (e.g. READY, RUNNING, DRAINING).
        """
        ...

    def checkpoint_state(self, cut: CheckpointCut) -> Mapping[str, Any]:
        """Serialize internal component state into a JSON-serializable mapping for checkpointing."""
        ...

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        """Restore internal component state from a previously saved checkpoint mapping."""
        ...


class _ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, ControlPlaneComponent] = {}

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self._components)

    def register(self, component: ControlPlaneComponent) -> None:
        component_id = getattr(component, "component_id", None)
        if not isinstance(component_id, str) or not component_id:
            raise TypeError("control-plane component_id must be a non-empty string")
        if component_id in self._components:
            raise ValueError(f"duplicate control-plane component: {component_id!r}")
        for name in ("on_state", "checkpoint_state", "restore_state"):
            if not callable(getattr(component, name, None)):
                raise TypeError(
                    f"control-plane component {component_id!r} must implement {name}"
                )
        self._components[component_id] = component

    async def notify_state(self, state: ComponentState) -> None:
        for component in tuple(self._components.values()):
            result = component.on_state(state)
            if not inspect.isawaitable(result):
                raise TypeError(
                    f"component {component.component_id!r} on_state must be awaitable"
                )
            await result

    def checkpoint_state(self, cut: CheckpointCut) -> dict[str, dict[str, Any]]:
        if not isinstance(cut, CheckpointCut):
            raise TypeError(
                f"component checkpoint requires CheckpointCut, got {type(cut).__name__}"
            )
        return {
            component_id: dict(component.checkpoint_state(cut))
            for component_id, component in self._components.items()
        }

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        if not isinstance(cut, CheckpointCut):
            raise TypeError(
                f"component restore requires CheckpointCut, got {type(cut).__name__}"
            )
        if not isinstance(state, Mapping):
            raise TypeError("component checkpoint must be a mapping")
        expected = set(self._components)
        actual = set(state)
        if expected != actual:
            raise ValueError(
                "component checkpoint ids do not match graph: "
                f"expected={sorted(expected)} actual={sorted(actual)}"
            )
        for component_id, component in self._components.items():
            component.restore_state(state[component_id], cut)


class ControlPlaneComponentRegistry(_ComponentRegistry):
    """Registry for global Controller-owned modules."""


class DriverComponentRegistry(_ComponentRegistry):
    """Registry for modules owned by one SPMD Driver."""


def build_driver_component_registry(
    config: Any,
    *,
    spmd_id: str,
    controller_handle: Any,
    report_failure: Any,
) -> DriverComponentRegistry:
    registry = DriverComponentRegistry()
    graph_view = config.require_graph_view()
    static_args = dict(config.static_args)
    for operator_name in graph_view.operator_ids:
        spec = graph_view.operator(operator_name).spec
        descriptor = spec.source
        if descriptor is None:
            continue
        build = getattr(descriptor, "build_component", None)
        if not callable(build):
            raise TypeError(
                f"source descriptor for {spmd_id}:{operator_name} must implement "
                "build_component(context)"
            )
        context = DriverComponentContext(
            component_id=driver_component_id(spmd_id, operator_name),
            spmd_id=spmd_id,
            operator_name=operator_name,
            operator_spec=spec,
            static_args=static_args,
            controller_handle=controller_handle,
            world_size=graph_view.world_size,
            interleave_unit_size=spec.interleave_unit_size,
            report_failure=report_failure,
        )
        registry.register(cast(ControlPlaneComponent, build(context)))
    return registry
