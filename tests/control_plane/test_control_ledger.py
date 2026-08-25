from __future__ import annotations

import asyncio
import copy
from types import SimpleNamespace

import pytest

from fluxway.control_plane.control_ledger import ControlLedger
from fluxway.control_plane.components import driver_component_id
from fluxway.control_plane.controller import ControllerActor
from fluxway.control_plane.ledger_client import ControlRecord, ControlStreamExhausted
from fluxway.control_plane.ledger_client import ControlLedgerClient
from fluxway.common.types.checkpoint import SpmdCheckpoint, TopologySnapshot
from fluxway.common.types import DrainBoundary


CONTROLLER_CLASS = ControllerActor.__ray_metadata__.modified_class  # type: ignore[attr-defined]


def test_control_ledger_implements_the_single_terminal_interface() -> None:
    ledger = ControlLedger(1)

    assert isinstance(ledger, ControlLedgerClient)
    assert hasattr(ledger, "seal")
    assert not hasattr(ledger, "close")
    assert not hasattr(ledger, "finish")


def _record(position: int, *, stream_id: str = "stream", payload=None):
    return ControlRecord(
        stream_id=stream_id,
        position=position,
        payload={"value": position} if payload is None else payload,
    )


@pytest.mark.asyncio
async def test_append_receive_checkpoint_and_commit() -> None:
    ledger = ControlLedger(2)
    await ledger.append((_record(0), _record(1)))
    assert await ledger.receive("stream", 0) == _record(0)

    state = ledger.checkpoint_state(cuts={"stream": 1})
    assert state["durable_cuts"] == {"stream": 1}
    assert state["records"] == [_record(1).to_dict()]
    assert (await ledger.progress("stream")).durable_cut == 0

    await ledger.commit_checkpoint(state)
    assert (await ledger.progress("stream")).durable_cut == 1
    assert ledger.record_count == 1


@pytest.mark.asyncio
async def test_try_append_reports_capacity_without_waiting() -> None:
    ledger = ControlLedger(1)
    assert await ledger.try_append((_record(0),))
    assert not await ledger.try_append((_record(1),))

    assert await ledger.receive("stream", 0) == _record(0)
    state = ledger.checkpoint_state(cuts={"stream": 1})
    await ledger.commit_checkpoint(state)

    assert await ledger.try_append((_record(1),))


@pytest.mark.asyncio
async def test_receive_waits_for_append_without_polling() -> None:
    ledger = ControlLedger(1)
    waiting = asyncio.create_task(ledger.receive("stream", 0))
    await asyncio.sleep(0)
    assert not waiting.done()

    await ledger.append((_record(0),))
    assert await waiting == _record(0)


@pytest.mark.asyncio
async def test_seal_wakes_receiver_at_exclusive_boundary_with_eof() -> None:
    ledger = ControlLedger(1)
    waiting = asyncio.create_task(ledger.receive("stream", 0))
    await asyncio.sleep(0)

    await ledger.seal("stream")
    await asyncio.sleep(0)

    with pytest.raises(ControlStreamExhausted):
        await waiting
    assert ledger.checkpoint_state(cuts={"stream": 0})["sealed_ends"] == {"stream": 0}
    state = ledger.checkpoint_state(cuts={"stream": 0})
    await ledger.commit_checkpoint(state)


@pytest.mark.asyncio
async def test_append_batch_is_atomic_when_cancelled_under_backpressure() -> None:
    ledger = ControlLedger(2)
    await ledger.append((_record(0),))
    blocked = asyncio.create_task(ledger.append((_record(1), _record(2))))
    await asyncio.sleep(0)
    assert not blocked.done()

    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    assert ledger.record_count == 1
    assert (await ledger.progress("stream")).published_end == 1


@pytest.mark.asyncio
async def test_append_rejects_batch_larger_than_capacity() -> None:
    ledger = ControlLedger(2)
    with pytest.raises(ValueError, match="exceeds ledger capacity"):
        await ledger.append((_record(0), _record(1), _record(2)))
    assert ledger.record_count == 0


@pytest.mark.asyncio
async def test_append_is_idempotent_and_rejects_gaps_and_conflicts() -> None:
    ledger = ControlLedger(2)
    await ledger.append((_record(0),))
    await ledger.append((_record(0),))
    assert ledger.record_count == 1

    with pytest.raises(ValueError, match="conflicting payload"):
        await ledger.append((_record(0, payload={"other": True}),))
    with pytest.raises(ValueError, match="create a gap"):
        await ledger.append((_record(2),))


@pytest.mark.asyncio
async def test_terminal_is_stream_scoped_and_immutable() -> None:
    ledger = ControlLedger(2)
    await ledger.seal("a")
    assert (await ledger.progress("a")).sealed_end == 0
    assert (await ledger.progress("b")).sealed_end is None
    with pytest.raises(RuntimeError, match="sealed"):
        await ledger.append((_record(0, stream_id="a"),))
    await ledger.append((_record(0, stream_id="b"),))


@pytest.mark.asyncio
async def test_vector_cut_restores_every_stream_without_topology_state() -> None:
    ledger = ControlLedger(4)
    await ledger.append(
        (
            _record(0, stream_id="a"),
            _record(1, stream_id="a"),
            _record(0, stream_id="b"),
        )
    )
    await ledger.receive("a", 0)
    await ledger.receive("b", 0)
    state = ledger.checkpoint_state(cuts={"a": 1, "b": 1})
    assert "owner" not in repr(state)
    assert "worker" not in repr(state)

    restored = ControlLedger(4)
    restored.restore_state(state)
    assert await restored.receive("a", 1) == _record(1, stream_id="a")


@pytest.mark.asyncio
async def test_checkpoint_requires_received_prefix() -> None:
    ledger = ControlLedger(1)
    await ledger.append((_record(0),))
    with pytest.raises(RuntimeError, match="were not received"):
        ledger.checkpoint_state(cuts={"stream": 1})


@pytest.mark.asyncio
async def test_restore_accepts_runtime_capacity_change_when_replay_fits() -> None:
    ledger = ControlLedger(3)
    await ledger.append((_record(0), _record(1), _record(2)))
    await ledger.receive("stream", 0)
    state = ledger.checkpoint_state(cuts={"stream": 1})

    assert "capacity" not in state
    restored = ControlLedger(2)
    restored.restore_state(state)
    assert await restored.receive("stream", 1) == _record(1)
    assert await restored.receive("stream", 2) == _record(2)


@pytest.mark.asyncio
async def test_restore_rejects_replay_larger_than_runtime_capacity() -> None:
    ledger = ControlLedger(3)
    await ledger.append((_record(0), _record(1), _record(2)))
    state = ledger.checkpoint_state(cuts={"stream": 0})

    with pytest.raises(ValueError, match="exceeds per-stream ledger capacity"):
        ControlLedger(2).restore_state(state)


@pytest.mark.asyncio
async def test_capacity_is_reserved_per_stream() -> None:
    ledger = ControlLedger(1)
    await ledger.append(
        (
            _record(0, stream_id="a"),
            _record(0, stream_id="b"),
        )
    )
    assert ledger.record_count == 2
    assert await ledger.receive("a", 0) == _record(0, stream_id="a")
    assert await ledger.receive("b", 0) == _record(0, stream_id="b")


def test_control_record_defensively_copies_payload() -> None:
    payload = {"nested": [1]}
    record = _record(0, payload=payload)
    payload["nested"].append(2)
    exported = record.payload
    exported["nested"].append(3)
    assert record.payload == {"nested": [1]}


@pytest.mark.asyncio
async def test_append_updates_retained_record_index_incrementally() -> None:
    ledger = ControlLedger(2)
    retained_index = ledger._records

    await ledger.append((_record(0),))
    await ledger.append((_record(1),))

    assert ledger._records is retained_index
    assert ledger.record_count == 2


@pytest.mark.asyncio
async def test_receive_does_not_recopy_immutable_control_record(monkeypatch) -> None:
    record = _record(0, payload={"ranges": [{"offset": value} for value in range(64)]})
    ledger = ControlLedger(1)
    await ledger.append((record,))
    copies = 0
    original = copy.deepcopy

    def count_deepcopy(value):
        nonlocal copies
        copies += 1
        return original(value)

    monkeypatch.setattr(
        "fluxway.control_plane.ledger_client.copy.deepcopy", count_deepcopy
    )

    received = await ledger.receive("stream", 0)

    assert received is record
    assert copies == 0


@pytest.mark.asyncio
async def test_controller_builds_vector_cut_from_source_operator_snapshots() -> None:
    stream_a = driver_component_id("source_a", "read")
    stream_b = driver_component_id("source_b", "read")
    ledger = ControlLedger(2)
    await ledger.append(
        (
            _record(0, stream_id=stream_a),
            _record(0, stream_id=stream_b),
            _record(1, stream_id=stream_b),
        )
    )
    await ledger.receive(stream_a, 0)
    await ledger.receive(stream_b, 0)
    await ledger.receive(stream_b, 1)

    source_spec = SimpleNamespace(
        source=object(),
        has_source_descriptor=True,
        is_source=True,
        is_interleaved=False,
    )
    views = {
        spmd_id: SimpleNamespace(
            operator_ids=("read",),
            operator=lambda name: SimpleNamespace(spec=source_spec),
        )
        for spmd_id in ("source_a", "source_b")
    }
    compiled = SimpleNamespace(
        spmd_ids=("source_a", "source_b"),
        spmd=lambda spmd_id: views[spmd_id],
    )
    controller = CONTROLLER_CLASS.__new__(CONTROLLER_CLASS)
    controller._graph = SimpleNamespace(require_compiled_graph=lambda: compiled)
    controller._control_ledger = ledger
    controller._control_plane_components = SimpleNamespace(checkpoint_state=lambda: {})
    controller._drain_boundary = DrainBoundary()
    payload = {
        "source_a": SpmdCheckpoint(
            TopologySnapshot(1, (), (), "v1"),
            {"drain_boundary": DrainBoundary(committed_cut=1).to_dict()},
            {},
        ),
        "source_b": SpmdCheckpoint(
            TopologySnapshot(1, (), (), "v1"),
            {"drain_boundary": DrainBoundary(committed_cut=2).to_dict()},
            {},
        ),
    }

    cuts = controller.control_stream_cuts(payload)
    assert cuts == {stream_a: 1, stream_b: 2}
    assert (
        controller.checkpoint_state(cuts=cuts)["control_ledger"]["durable_cuts"] == cuts
    )


@pytest.mark.parametrize(
    "state",
    (
        {"control_ledger": {}},
        {"components": {}, "drain_boundary": DrainBoundary().to_dict()},
        {"control_ledger": {}, "components": {}},
    ),
)
def test_controller_rejects_incomplete_control_checkpoint(state) -> None:
    controller = CONTROLLER_CLASS.__new__(CONTROLLER_CLASS)
    controller._control_ledger = ControlLedger(1)
    controller._control_plane_components = SimpleNamespace(
        restore_state=lambda component_state: None
    )

    with pytest.raises(ValueError, match="fields do not match protocol"):
        controller.restore_checkpoint_state(state)
