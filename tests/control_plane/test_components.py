import asyncio
from types import SimpleNamespace

import pytest

from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types import CheckpointCut, ControllerState
from fluxway.control_plane.components import (
    ControlPlaneComponentRegistry,
    DriverComponentContext,
    build_driver_component_registry,
    driver_component_id,
)


class _Component:
    component_id = "component"

    def __init__(self):
        self.states = []
        self.restored = None
        self.last_cut = None

    async def on_state(self, state):
        self.states.append(state)

    def checkpoint_state(self, cut: CheckpointCut):
        self.last_cut = cut
        return {
            "states": [state.value for state in self.states],
            "cut_id": cut.cut_id,
        }

    def restore_state(self, state, cut: CheckpointCut):
        self.restored = state
        self.last_cut = cut


def test_source_descriptor_is_operator_metadata_only():
    class Descriptor:
        def build_component(self, context):
            return _Component()

    descriptor = Descriptor()
    spec = OperatorSpec(name="read", is_source=True, source=descriptor)
    assert spec.source is descriptor
    with pytest.raises(ValueError, match="is_source=True"):
        OperatorSpec(name="compute", source=descriptor)


def test_registry_forwards_lifecycle_and_round_trips_domain_state():
    component = _Component()
    registry = ControlPlaneComponentRegistry()
    registry.register(component)
    cut = CheckpointCut.for_step("cut-1", 1)

    async def scenario():
        await registry.notify_state(ControllerState.RUNNING)
        await registry.notify_state(ControllerState.COMPLETING)

    asyncio.run(scenario())
    state = registry.checkpoint_state(cut)
    assert state == {
        "component": {
            "states": ["running", "completing"],
            "cut_id": "cut-1",
        }
    }
    assert component.last_cut == cut

    restore_cut = CheckpointCut.for_step("cut-restore", 1)
    registry.restore_state(state, restore_cut)
    assert component.restored == state["component"]
    assert component.last_cut == restore_cut


def test_registry_rejects_non_checkpoint_cut():
    component = _Component()
    registry = ControlPlaneComponentRegistry()
    registry.register(component)

    with pytest.raises(TypeError, match="CheckpointCut"):
        registry.checkpoint_state(None)  # type: ignore

    with pytest.raises(TypeError, match="CheckpointCut"):
        registry.restore_state({"component": {}}, 1)  # type: ignore


def test_driver_registry_uses_compiled_source_descriptor_and_canonical_id():
    component = _Component()

    class Descriptor:
        def build_component(self, context):
            assert isinstance(context, DriverComponentContext)
            component.component_id = context.component_id
            return component

    source_spec = OperatorSpec(
        name="read",
        is_source=True,
        source=Descriptor(),
    )
    compute_spec = OperatorSpec(name="compute")
    compiled = {
        "read": SimpleNamespace(spec=source_spec),
        "compute": SimpleNamespace(spec=compute_spec),
    }
    view = SimpleNamespace(
        operator_ids=("read", "compute"),
        world_size=8,
        operator=lambda name: compiled[name],
    )
    config = SimpleNamespace(
        static_args={},
        operator_specs={"stale": compute_spec},
        require_graph_view=lambda: view,
    )

    registry = build_driver_component_registry(
        config,
        spmd_id="reader",
        controller_handle=object(),
        report_failure=lambda error: None,
    )

    assert registry.component_ids == ("reader:read",)


def test_driver_component_id_is_the_canonical_identity_constructor():
    assert driver_component_id("reader", "read") == "reader:read"
    with pytest.raises(ValueError, match="spmd_id"):
        driver_component_id("", "read")
    with pytest.raises(ValueError, match="operator_name"):
        driver_component_id("reader", "")
