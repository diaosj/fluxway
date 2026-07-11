import asyncio
import typing
from types import SimpleNamespace

import pytest

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.mesh import DeviceMesh
from fluxway.profiler.record import TraceArena
from fluxway.profiler.recorder import TraceRecorder


def _make_config(mesh, *op_names):
    from fluxway.common.types.configs import OperatorMeta, ArgumentType

    metas = {}
    for name in op_names:
        meta = OperatorMeta(
            name=name,
            input_arguments={ArgumentType.DATA.value: ["in"]},
            output_arguments={ArgumentType.DATA.value: ["out"]},
            input_sharding={"in": {"type": "replicated"}},
            output_sharding={"out": {"type": "replicated"}},
        )
        metas[name] = meta

    return SPMDConfig(
        spmd_id="test",
        mesh=mesh,
        operator_metas=metas,
        active_operator_names=tuple(op_names),
    )


@pytest.mark.asyncio
async def test_spmd_worker_delivery_token_registration_is_not_throttled():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")
    config.data_plane_max_in_flight = 1

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    worker = cls(config, rank=0)
    release_first = asyncio.Event()
    started: list[str] = []

    async def delivery(name: str) -> str:
        started.append(name)
        if name == "first":
            await release_first.wait()
        return name

    first = await worker._delivery_runtime.submit(
        delivery("first"), schedule_key=(0, 0, 0, "op")
    )
    second_submit = asyncio.create_task(
        worker._delivery_runtime.submit(
            delivery("second"), schedule_key=(0, 0, 1, "op")
        )
    )
    await asyncio.sleep(0.01)

    assert started == ["first"]
    assert second_submit.done()
    second = await second_submit
    assert not second.done()

    release_first.set()
    assert await first == "first"
    assert await second == "second"
    assert started == ["first", "second"]


@pytest.mark.asyncio
async def test_spmd_worker_delivery_runtime_profiler_spans_lifecycle():
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")
    config.data_plane_max_in_flight = 1

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    worker = cls(config, rank=0)
    recorder = TraceRecorder(
        TraceArena(1024 * 1024),
        run_id="delivery-test",
        process_name="worker",
    )
    worker._profiler_runtime = typing.cast(
        typing.Any, SimpleNamespace(enabled=True, recorder=recorder)
    )
    worker._delivery_runtime._profiler_runtime = worker._profiler_runtime

    release_first = asyncio.Event()
    started: list[str] = []

    async def delivery(name: str) -> str:
        started.append(name)
        if name == "first":
            await release_first.wait()
        return name

    first = await worker._delivery_runtime.submit(
        delivery("first"), schedule_key=(0, 0, 0, "op")
    )
    second = await worker._delivery_runtime.submit(
        delivery("second"), schedule_key=(0, 0, 1, "op")
    )
    await asyncio.sleep(0.01)
    assert started == ["first"]

    release_first.set()
    await worker._delivery_runtime.drain("unit_test")
    assert await first == "first"
    assert await second == "second"

    snapshot = recorder.arena.snapshot()
    names = [recorder.metadata.name_for(int(row[4])) for row in snapshot]

    assert names.count("delivery.register_token") == 2
    assert names.count("delivery.semaphore_wait") == 2
    assert names.count("delivery.run") == 2
    assert names.count("delivery.fence") == 1
    assert "data_plane.delivery_fence" not in names
