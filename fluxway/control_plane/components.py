"""Shared registries for Controller-owned and Driver-owned modules."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from fluxway.common.types import CheckpointCut, ControllerState, DriverState
from fluxway.common.driver_control import DriverComponentCapabilities
from fluxway.common.metadata import metadata_digest

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
    capabilities: DriverComponentCapabilities


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


@runtime_checkable
class TransactionalControlPlaneComponent(Protocol):
    """Explicit Controller transaction capability."""

    component_id: str

    def prepare_checkpoint(
        self, checkpoint_id: str, permit: Any, cut: CheckpointCut, state_digest: str
    ) -> Any: ...

    def apply_prepared(self, prepared_state: Any) -> None: ...

    def abort_prepared(self, checkpoint_id: str, permit: Any) -> None: ...


class _ComponentRegistry:
    def __init__(self) -> None:
        self._components: dict[str, ControlPlaneComponent] = {}

    @property
    def component_ids(self) -> tuple[str, ...]:
        return tuple(self._components)

    def components_implementing(self, *names: str) -> tuple[Any, ...]:
        if not names or any(not isinstance(name, str) or not name for name in names):
            raise ValueError("component capability names must be non-empty strings")
        return tuple(
            component
            for component in self._components.values()
            if all(callable(getattr(component, name, None)) for name in names)
        )

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
        transaction_names = (
            "prepare_checkpoint",
            "apply_prepared",
            "abort_prepared",
        )
        transaction_methods = tuple(
            callable(getattr(component, name, None)) for name in transaction_names
        )
        if any(transaction_methods) and not all(transaction_methods):
            raise TypeError(
                f"control-plane component {component_id!r} must implement the complete "
                "transaction capability"
            )
        self._components[component_id] = component

    def register_transactional(
        self, component: TransactionalControlPlaneComponent
    ) -> None:
        if not isinstance(component, TransactionalControlPlaneComponent):
            raise TypeError(
                f"control-plane component {component.component_id!r} must implement "
                "the explicit transaction capability"
            )
        self.register(component)

    async def dispatch_request(self, component_id: str, request: Any) -> object:
        if not isinstance(component_id, str) or not component_id:
            raise ValueError("component dispatch requires a component id")
        try:
            component = self._components[component_id]
        except KeyError as error:
            raise ValueError(
                f"component dispatch references unknown component {component_id!r}"
            ) from error
        dispatch = getattr(component, "dispatch", None)
        if not callable(dispatch):
            raise TypeError(f"component {component_id!r} lacks dispatch capability")
        result = dispatch(request)
        if not inspect.isawaitable(result):
            raise TypeError(f"component {component_id!r} dispatch must be awaitable")
        return await result

    async def dispatch_unaddressed_request(self, request: Any) -> object:
        dispatchable = tuple(
            component
            for component in self._components.values()
            if callable(getattr(component, "dispatch", None))
        )
        if len(dispatchable) != 1:
            raise RuntimeError(
                "unaddressed component dispatch requires exactly one dispatchable component"
            )
        return await self.dispatch_request(dispatchable[0].component_id, request)

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

    def prepare_checkpoint_transaction(
        self,
        checkpoint_id: str,
        permit: Any,
        cut: CheckpointCut,
        component_states: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(component_states, Mapping):
            raise TypeError("component transaction states must be a mapping")
        prepared: dict[str, Any] = {}
        for component_id, component in self._components.items():
            if not isinstance(component, TransactionalControlPlaneComponent):
                continue
            if component_id not in component_states:
                raise ValueError(
                    f"component transaction state is missing {component_id!r}"
                )
            state = component_states[component_id]
            state_digest = metadata_digest(state)
            result = component.prepare_checkpoint(
                checkpoint_id, permit, cut, state_digest
            )
            if inspect.isawaitable(result):
                raise TypeError(
                    f"component {component_id!r} prepare_checkpoint must be synchronous"
                )
            prepared[component_id] = result
        return prepared

    def apply_prepared_checkpoint_transaction(
        self,
        checkpoint_id: str,
        permit: Any,
        cut: CheckpointCut,
        component_states: Mapping[str, Any],
        prepared: Mapping[str, Any],
    ) -> None:
        if not isinstance(component_states, Mapping) or not isinstance(
            prepared, Mapping
        ):
            raise TypeError(
                "component transaction state and prepared values must be mappings"
            )
        for component_id, component in self._components.items():
            if not isinstance(component, TransactionalControlPlaneComponent):
                continue
            component_prepared = prepared.get("components", prepared)
            if (
                component_id not in component_states
                or component_id not in component_prepared
            ):
                raise ValueError(
                    f"component transaction state is missing {component_id!r}"
                )
            result = component.apply_prepared(component_prepared[component_id])
            if result is not None:
                raise TypeError(
                    f"component {component_id!r} apply_prepared must return None"
                )

    def abort_checkpoint_transaction(self, checkpoint_id: str, permit: Any) -> None:
        for component_id, component in self._components.items():
            if not isinstance(component, TransactionalControlPlaneComponent):
                continue
            result = component.abort_prepared(checkpoint_id, permit)
            if inspect.isawaitable(result):
                raise TypeError(
                    f"component {component_id!r} abort_prepared must be synchronous"
                )


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
    capabilities: DriverComponentCapabilities,
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
            capabilities=capabilities,
        )
        registry.register(cast(ControlPlaneComponent, build(context)))
    return registry
