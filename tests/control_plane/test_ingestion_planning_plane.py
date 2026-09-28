from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace

import pytest

from fluxway.control_plane.ingestion import (
    CatalogViewObservation,
    TailLayoutMode,
    InMemoryIngestionSourceAdapter,
    IngestionLayoutPlanningComponent,
    IngestionPlacementContext,
    AssignmentPolicy,
)
from fluxway.control_plane.logical_work_ledger import LogicalWorkLedger
from fluxway.common.types import CheckpointCut


@pytest.mark.asyncio
async def test_planner_returns_fully_bound_node_affinity_request(source_layout) -> None:
    logical_request = source_layout(
        demands=(2, 2), cardinality=4, terminal=True
    ).request
    request = replace(
        logical_request,
        assignment_policy=AssignmentPolicy(node_affinity_hint=True),
    )

    class Descriptor:
        def build_ingestion_plan(self, *, spmd_id, operator_name, config):
            return request

    planner = IngestionLayoutPlanningComponent("bound")
    try:
        bound = await planner.build_ingestion_plan(
            Descriptor(),
            spmd_id="source",
            operator_name="read",
            config=None,
            placement_context=IngestionPlacementContext(lambda: ("node-a", "node-b")),
        )
        assert tuple(group.node_id for group in bound.topology.groups) == (
            "node-a",
            "node-b",
        )
    finally:
        await planner.close()


@pytest.mark.asyncio
async def test_planning_does_not_use_default_executor(source_layout) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "isolated", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    loop = asyncio.get_running_loop()
    blocker = threading.Event()

    def hold_default_executor() -> None:
        blocker.wait()

    default_executor = ThreadPoolExecutor(max_workers=1)
    loop.set_default_executor(default_executor)
    entered = asyncio.Event()

    def occupy_default_executor() -> None:
        loop.call_soon_threadsafe(entered.set)
        hold_default_executor()

    class Descriptor:
        def build_ingestion_plan(self, *, spmd_id, operator_name, config):
            assert threading.current_thread() is not threading.main_thread()
            return compiled.request

    workers = [loop.run_in_executor(None, occupy_default_executor)]
    try:
        await entered.wait()
        request = await asyncio.wait_for(
            planner.build_ingestion_plan(
                Descriptor(), spmd_id="source", operator_name="read", config=None
            ),
            timeout=5,
        )
        activation = await asyncio.wait_for(
            planner.plan_and_publish(
                ledger,
                request,
                epoch=0,
                start_position=0,
                cursor={"position": 2},
                observation_identity=("source", "schema"),
            ),
            timeout=15,
        )
        assert activation.epoch.epoch == 0
        assert len(planner.windows) == 1
        assert ledger.checkpoint_state()["epochs"]
    finally:
        blocker.set()
        await asyncio.gather(*workers)
        await planner.close()


@pytest.mark.asyncio
async def test_failed_publication_does_not_install_planner_or_cursor(
    source_layout, monkeypatch
) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)
    planner = IngestionLayoutPlanningComponent(
        "atomic", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)

    def reject(*args, **kwargs):
        raise RuntimeError("prepare rejected")

    monkeypatch.setattr(ledger, "prepare_publication", reject)
    with pytest.raises(RuntimeError, match="prepare rejected"):
        await planner.plan_and_publish(
            ledger,
            compiled.request,
            epoch=0,
            start_position=0,
            cursor={"position": 2},
            observation_identity=("source", "schema"),
        )
    assert planner.windows == ()
    image = planner.checkpoint_state(CheckpointCut.for_position("cut", 0))["image"]
    assert image["cursor"] == {}
    assert ledger.checkpoint_state()["epochs"] == []
    await planner.close()


@pytest.mark.asyncio
async def test_adapter_publication_installs_cursor_with_window(source_layout) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)
    observation = CatalogViewObservation.create(
        source_identity=compiled.request.source_identity,
        view_identity=compiled.request.view_identity,
        schema_digest=compiled.request.schema_digest,
        lineage_digest=compiled.request.lineage_digest,
        units=compiled.request.units,
        terminal=True,
        cursor={"position": 2},
    )
    adapter = InMemoryIngestionSourceAdapter((observation,))
    planner = IngestionLayoutPlanningComponent(
        "cursor", topology=compiled.request.topology
    )
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    await planner.plan_adapter_window(
        adapter,
        ledger,
        {"index": 0},
        cardinality_limit=2,
        topology=compiled.request.topology,
        tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
        assignment_policy=AssignmentPolicy.default(),
        epoch=0,
        start_position=0,
    )
    image = planner.checkpoint_state(CheckpointCut.for_position("cut", 2))["image"]
    assert image["cursor"] == {"position": 2}
    assert image["observation_identity"] == ["source", "schema"]
    assert len(image["windows"]) == 1
    await planner.close()


@pytest.mark.asyncio
async def test_close_joins_accepted_publication(source_layout) -> None:
    compiled = source_layout(demands=(1,), cardinality=2, terminal=True)

    class GatedPlanner(IngestionLayoutPlanningComponent):
        def __init__(self) -> None:
            super().__init__("close", topology=compiled.request.topology)
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def _compile_prepared(self, request, *, epoch, start_position):
            self.entered.set()
            await self.release.wait()
            return compiled

    planner = GatedPlanner()
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    publication = asyncio.create_task(
        planner.plan_and_publish(ledger, compiled.request, epoch=0, start_position=0)
    )
    await planner.entered.wait()
    closing = asyncio.create_task(planner.close())
    await asyncio.sleep(0)
    assert not closing.done()
    planner.release.set()
    await publication
    await closing
    assert len(planner.windows) == 1
    with pytest.raises(RuntimeError, match="closed"):
        await planner.plan_and_publish(
            ledger, compiled.request, epoch=0, start_position=0
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_waiter", [False, True])
async def test_close_settles_capture_and_publication(source_layout, cancel_waiter):
    compiled = source_layout(demands=(1,), cardinality=2)
    loop = asyncio.get_running_loop()
    entered = asyncio.Event()
    release = threading.Event()

    class Adapter:
        def capture_view(self, cursor, *, cardinality_limit):
            loop.call_soon_threadsafe(entered.set)
            if not release.wait(20):
                raise TimeoutError("test release barrier was abandoned")
            return CatalogViewObservation.create(
                source_identity="source",
                view_identity="r",
                schema_digest="schema",
                lineage_digest="r",
                units=compiled.request.units,
                terminal=True,
                cursor={"position": 2},
            )

        def encode_cursor(self, cursor):
            return dict(cursor)

        def decode_cursor(self, state):
            return dict(state)

    planner = IngestionLayoutPlanningComponent("capture-close")
    ledger = LogicalWorkLedger(max_in_flight_steps=2)
    caller = asyncio.create_task(
        planner.plan_adapter_window(
            Adapter(),
            ledger,
            {},
            cardinality_limit=2,
            topology=compiled.request.topology,
            tail_layout_mode=TailLayoutMode.REJECT_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
            epoch=0,
            start_position=0,
        )
    )
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 5)
        if cancel_waiter:
            caller.cancel()
            with pytest.raises(asyncio.CancelledError):
                await caller
        closing = asyncio.create_task(planner.close())
        done, _ = await asyncio.wait((closing,), timeout=0.05)
        assert not done
        closing.cancel()
        with pytest.raises(asyncio.CancelledError):
            await closing
        release.set()
        if not cancel_waiter:
            await caller
        await asyncio.wait_for(planner.close(), 20)
        image = planner.checkpoint_state(CheckpointCut.for_position("cut", 0))["image"]
        assert image["cursor"] == {"position": 2}
        assert len(image["windows"]) == len(ledger.checkpoint_state()["epochs"]) == 1
        assert not planner._operations
        assert not planner._adapter_lane._thread.is_alive()
        assert not planner._lifecycle_lane._thread.is_alive()
    finally:
        release.set()
        await asyncio.gather(caller, return_exceptions=True)
        if closing is not None:
            await asyncio.gather(closing, return_exceptions=True)
        await planner.close()


@pytest.mark.asyncio
async def test_cancelled_waiter_failure_is_reported_after_resource_cleanup():
    entered = asyncio.Event()

    class Descriptor:
        async def build_ingestion_plan(self, **kwargs):
            loop.call_soon_threadsafe(entered.set)
            await asyncio.wrap_future(released)
            raise ValueError("descriptor failed")

    loop = asyncio.get_running_loop()
    released = Future()
    planner = IngestionLayoutPlanningComponent("failure-close")
    caller = asyncio.create_task(
        planner.build_ingestion_plan(
            Descriptor(), spmd_id="source", operator_name="read", config=None
        )
    )
    await asyncio.wait_for(entered.wait(), 5)
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
    released.set_result(None)
    with pytest.raises(ExceptionGroup) as caught:
        await planner.close()
    assert len(caught.value.exceptions) == 1
    assert str(caught.value.exceptions[0]) == "descriptor failed"
    assert not planner._adapter_lane._thread.is_alive()
