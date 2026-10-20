from __future__ import annotations

import asyncio

import pytest

from fluxway.common.operator_spec import OperatorSpec
from fluxway.common.types import CheckpointCut, DriverState
from fluxway.control_plane.components import DriverComponentContext
from fluxway.common.driver_control import (
    DriverComponentCapabilities,
    IngestionLayoutActivationRequest,
)
from fluxway.control_plane.ingestion import (
    AlgebraicLayoutCompiler,
    TailLayoutMode,
    IngestionLayoutActivationComponent,
    LayoutCompileRequest,
    AssignmentPolicy,
    LayoutUnitDescriptor,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
    ingestion_worker_id,
)


def _envelope(unit_id: str = "unit"):
    worker_ids = tuple(ingestion_worker_id("reader", "read", rank) for rank in range(2))
    compiled = AlgebraicLayoutCompiler().compile(
        LayoutCompileRequest(
            source_identity="source",
            view_identity="view",
            schema_digest="schema",
            lineage_digest="lineage",
            units=(
                LayoutUnitDescriptor(
                    unit_id,
                    "revision",
                    4,
                    "rows",
                    "SplittableSpan",
                    "node",
                    "object",
                    0,
                    "range",
                ),
            ),
            topology=ReaderTopologyDescriptor(
                4,
                tuple(
                    ReaderGroupDemand(
                        worker_id,
                        (RankDemand(f"{worker_id}:logical-rank-0", 2),),
                    )
                    for worker_id in worker_ids
                ),
            ),
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
        ),
        epoch=3,
        start_step=8,
    )
    return compiled.to_source_layout_activation(
        epoch=3,
        start_position=8,
        end_position=9,
        terminal=True,
    )


def _capabilities(callback):
    class Transport:
        async def dispatch(self, request):
            assert isinstance(request, IngestionLayoutActivationRequest)
            return await callback(
                request.rank, request.operator_name, request.activation
            )

    return DriverComponentCapabilities(worker_control=Transport())


@pytest.mark.asyncio
async def test_driver_component_delivers_targeted_descriptors_concurrently():
    started: list[int] = []
    release = asyncio.Event()
    delivered = {}

    async def activate(rank, operator_name, activation):
        started.append(rank)
        if len(started) == 2:
            release.set()
        await release.wait()
        delivered[rank] = (operator_name, activation)

    context = DriverComponentContext(
        component_id="reader:read",
        spmd_id="reader",
        operator_name="read",
        operator_spec=OperatorSpec(name="read", is_source=True),
        static_args={},
        controller_handle=object(),
        world_size=2,
        interleave_unit_size=1,
        report_failure=lambda error: None,
        capabilities=_capabilities(activate),
    )
    component = IngestionLayoutActivationComponent(context)

    await component.activate(_envelope())

    assert started == [0, 1]
    assert set(delivered) == {0, 1}
    for rank, (operator_name, activation) in delivered.items():
        assert operator_name == "read"
        assert activation.descriptor.worker_id == ingestion_worker_id(
            "reader", "read", rank
        )
        assert not hasattr(activation, "payload")
        assert not hasattr(activation, "tensor")


@pytest.mark.asyncio
async def test_driver_component_rejects_another_worker_set_before_delivery():
    delivered = []

    async def activate(*args):
        delivered.append(args)

    context = DriverComponentContext(
        component_id="other:read",
        spmd_id="other",
        operator_name="read",
        operator_spec=OperatorSpec(name="read", is_source=True),
        static_args={},
        controller_handle=object(),
        world_size=2,
        interleave_unit_size=1,
        report_failure=lambda error: None,
        capabilities=_capabilities(activate),
    )
    component = IngestionLayoutActivationComponent(context)

    with pytest.raises(ValueError, match="worker identities"):
        await component.activate(_envelope())
    assert delivered == []


@pytest.mark.asyncio
async def test_activation_component_is_checkpoint_neutral():
    async def activate(*args):
        return None

    context = DriverComponentContext(
        component_id="reader:read",
        spmd_id="reader",
        operator_name="read",
        operator_spec=OperatorSpec(name="read", is_source=True),
        static_args={},
        controller_handle=object(),
        world_size=2,
        interleave_unit_size=1,
        report_failure=lambda error: None,
        capabilities=_capabilities(activate),
    )
    component = IngestionLayoutActivationComponent(context)

    await component.on_state(DriverState.RUNNING)
    cut = CheckpointCut.for_position("cut", 10)
    assert component.checkpoint_state(cut) == {}
    component.restore_state({}, cut)


@pytest.mark.asyncio
async def test_activation_replay_is_idempotent_and_rebound_after_restore():
    calls: list[int] = []

    async def activate(rank, operator_name, activation):
        calls.append(rank)

    context = DriverComponentContext(
        component_id="reader:read",
        spmd_id="reader",
        operator_name="read",
        operator_spec=OperatorSpec(name="read", is_source=True),
        static_args={},
        controller_handle=object(),
        world_size=2,
        interleave_unit_size=1,
        report_failure=lambda error: None,
        capabilities=_capabilities(activate),
    )
    component = IngestionLayoutActivationComponent(context)
    envelope = _envelope()

    await component.activate(envelope)
    await component.activate(envelope)

    assert calls == [0, 1]
    state = component.checkpoint_state(CheckpointCut.for_position("cut", 10))
    restored = IngestionLayoutActivationComponent(context)
    restored.restore_state(state, CheckpointCut.for_position("cut", 10))
    await restored.activate(envelope)
    assert calls == [0, 1, 0, 1]


@pytest.mark.asyncio
async def test_activation_replay_conflict_is_rejected_before_dispatch():
    calls: list[int] = []

    async def activate(rank, operator_name, activation):
        calls.append(rank)

    context = DriverComponentContext(
        component_id="reader:read",
        spmd_id="reader",
        operator_name="read",
        operator_spec=OperatorSpec(name="read", is_source=True),
        static_args={},
        controller_handle=object(),
        world_size=2,
        interleave_unit_size=1,
        report_failure=lambda error: None,
        capabilities=_capabilities(activate),
    )
    component = IngestionLayoutActivationComponent(context)

    await component.activate(_envelope())
    with pytest.raises(ValueError, match="conflicting"):
        await component.activate(_envelope("different-unit"))
    assert calls == [0, 1]
