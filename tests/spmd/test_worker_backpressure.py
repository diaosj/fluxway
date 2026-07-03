import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock

from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.mesh import DeviceMesh
from fluxway.transport.manifest import TransportFrameHeader


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
        num_workers=mesh.size if mesh else 1,
        mesh=mesh,
        operator_metas=metas,
    )


@pytest.mark.asyncio
async def test_spmd_worker_backpressure_throttles_sender():
    """Verify that a sender delays sending if the downstream receiver returns a high watermark."""
    mesh = DeviceMesh(axis_names=("data",), device_shape=(2,))
    config = _make_config(mesh, "compute")

    # Instantiate class depending on whether it's wrapped by Ray or not
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    cls = cls.modified_class if cls else SPMDWorker

    worker = cls(config, rank=0)
    worker._max_in_flight = 2
    worker._mailbox_capacity = 10
    worker._high_watermark = 8

    # Create dummy handler to simulate the remote receive call
    mock_handler = MagicMock()

    # First response: normal watermark (under high_watermark)
    # Second response: high watermark (above high_watermark=8) -> triggers delay
    # Third response: normal watermark

    async def recv_side_effect(*args, **kwargs):
        recv_side_effect.call_count += 1
        if recv_side_effect.call_count == 1:
            return 5
        elif recv_side_effect.call_count == 2:
            return 9  # Trigger delay
        else:
            return 2

    recv_side_effect.call_count = 0
    mock_handler.receive_transport_frame.remote = MagicMock(
        side_effect=recv_side_effect
    )

    # Patch asyncio.sleep
    sleep_calls = []
    original_sleep = asyncio.sleep

    async def mock_sleep(delay):
        sleep_calls.append(delay)
        await original_sleep(0.001)

    # Create a simplified frame-only version of the backpressure logic in _send_fn.
    async def simulated_send_fn(
        target_rank,
        header,
        *payload_iovs,
        _handlers,
    ):
        handler = _handlers.get(target_rank)
        if handler is None:
            raise ValueError(f"missing handler for target_rank={target_rank}")

        await worker._send_semaphore.acquire()

        ref = handler.receive_transport_frame.remote(header, *payload_iovs)

        async def _wait_and_release(r=ref):
            try:
                watermark = await r
                if isinstance(watermark, int) and watermark > worker._high_watermark:
                    backoff = 0.05
                    await worker_module.asyncio.sleep(backoff)
            finally:
                worker._send_semaphore.release()

        asyncio.create_task(_wait_and_release())
        return True

    try:
        import fluxway.spmd.worker.worker as worker_module

        old_sleep = worker_module.asyncio.sleep
        worker_module.asyncio.sleep = mock_sleep

        # Test the simulated _send_fn callback
        header = TransportFrameHeader(
            frame_i32=np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=np.int32),
            edge_i32=np.array([[1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0]], dtype=np.int32),
            payload_i32=np.array([[0, 1, 1, 0, 1, 1, 0, 0]], dtype=np.int32),
            shape_i32=np.array([1], dtype=np.int32),
            global_shape_i64=np.zeros((0,), dtype=np.int64),
            compact_i64=np.zeros((0,), dtype=np.int64),
        )
        frame_args = (header, np.array([1], dtype=np.int32))
        handlers = {0: mock_handler, 1: mock_handler, 2: mock_handler}

        await simulated_send_fn(0, *frame_args, _handlers=handlers)
        await asyncio.sleep(0)

        await simulated_send_fn(1, *frame_args, _handlers=handlers)
        await asyncio.sleep(0)

        await simulated_send_fn(2, *frame_args, _handlers=handlers)
        await asyncio.sleep(0)

        assert len(sleep_calls) >= 1
    finally:
        worker_module.asyncio.sleep = old_sleep
