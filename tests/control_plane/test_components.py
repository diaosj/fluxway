import asyncio
from types import SimpleNamespace

import pytest

from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types import CheckpointCut, ControllerState
from fluxway.control_plane.components import (
    ControlPlaneComponentRegistry,
    DriverComponentRegistry,
    DriverComponentContext,
    build_driver_component_registry,
    driver_component_id,
)
from fluxway.common.driver_control import DriverComponentCapabilities


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
        requires_source_control_runtime = False

        def build_component(self, context):
            return _Component()

    descriptor = Descriptor()
    spec = OperatorSpec(name="read", is_source=True, source=descriptor)
    assert spec.source is descriptor
    with pytest.raises(ValueError, match="is_source=True"):
        OperatorSpec(name="compute", source=descriptor)


def test_controlled_source_descriptor_requires_all_scan_capabilities():
    class Descriptor:
        requires_source_control_runtime = True

        def build_component(self, context):
            return _Component()

        def build_reader_topology(self, **kwargs):
            return None

    with pytest.raises(TypeError, match="bounded scan capabilities"):
        OperatorSpec(name="read", is_source=True, source=Descriptor())


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
        requires_source_control_runtime = False

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
        capabilities=DriverComponentCapabilities(
            worker_control=SimpleNamespace(dispatch=lambda request: None)
        ),
    )

    assert registry.component_ids == ("reader:read",)


def test_driver_component_id_is_the_canonical_identity_constructor():
    assert driver_component_id("reader", "read") == "reader:read"
    with pytest.raises(ValueError, match="spmd_id"):
        driver_component_id("", "read")
    with pytest.raises(ValueError, match="operator_name"):
        driver_component_id("reader", "")


@pytest.mark.asyncio
async def test_driver_registry_dispatches_only_explicit_component_capability():
    class Dispatchable(_Component):
        component_id = "reader:read"

        async def dispatch(self, request):
            return (self.component_id, request)

    registry = DriverComponentRegistry()
    component = Dispatchable()
    registry.register(component)

    assert await registry.dispatch_request("reader:read", {"epoch": 1}) == (
        "reader:read",
        {"epoch": 1},
    )
    with pytest.raises(ValueError, match="unknown component"):
        await registry.dispatch_request("reader:missing", object())

    plain = _Component()
    plain.component_id = "plain"
    registry.register(plain)
    with pytest.raises(TypeError, match="dispatch capability"):
        await registry.dispatch_request("plain", object())


@pytest.mark.asyncio
async def test_controller_registry_routes_one_unaddressed_typed_notice():
    class Dispatchable(_Component):
        component_id = "ledger"

        async def dispatch(self, request):
            return request

    registry = ControlPlaneComponentRegistry()
    marker = object()
    registry.register(Dispatchable())
    assert await registry.dispatch_unaddressed_request(marker) is marker

    class AnotherDispatchable(Dispatchable):
        component_id = "another"

    registry.register(AnotherDispatchable())
    with pytest.raises(RuntimeError, match="exactly one"):
        await registry.dispatch_unaddressed_request(marker)
