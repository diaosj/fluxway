import numpy as np
import pytest

from fluxway.common.tensor_backend import ensure_cpu_serializable
from fluxway.rpc.protocol import (
    AxonDataPlane,
    DataPlaneRegistry,
    PayloadDeviceClass,
    RayActorDataPlane,
    TransportFrameEnvelope,
    payload_device_class,
    validate_payload_iov_device_class,
)
from fluxway.transport.manifest import TransportFrameHeader


def _make_header() -> TransportFrameHeader:
    frame_i32 = np.arange(8, dtype=np.int32)
    frame_i32[0] = 1
    frame_i32[5] = 1
    frame_i32[6] = 2
    edge_i32 = np.zeros((1, 11), dtype=np.int32)
    edge_i32[0, 2] = 0
    edge_i32[0, 4] = 2
    edge_i32[0, 6] = 1
    payload_i32 = np.zeros((2, 8), dtype=np.int32)
    payload_i32[:, 0] = 0
    payload_i32[0, 1] = 4
    payload_i32[1, 1] = 6
    payload_i32[0, 2] = 1
    payload_i32[1, 2] = 1
    payload_i32[0, 3] = 0
    payload_i32[1, 3] = 0
    payload_i32[0, 4] = 1
    payload_i32[1, 4] = 1
    payload_i32[0, 6] = 0
    payload_i32[1, 6] = 1
    shape_i32 = np.array([2], dtype=np.int32)
    global_shape_i64 = np.array([2**35], dtype=np.int64)
    compact_i64 = np.array([2**33, 1, 4, 1], dtype=np.int64)
    return TransportFrameHeader(
        frame_i32=frame_i32,
        edge_i32=edge_i32,
        payload_i32=payload_i32,
        shape_i32=shape_i32,
        global_shape_i64=global_shape_i64,
        compact_i64=compact_i64,
    )


class _FakeCudaPayload:
    dtype = np.dtype(np.float32)
    __cuda_array_interface__ = {
        "shape": (1,),
        "typestr": "<f4",
        "data": (1, False),
        "version": 3,
    }


class _DLPackOnlyPayload:
    dtype = np.dtype(np.float32)

    def __dlpack__(self, stream=None):
        return object()


class _DLPackCpuPayload(_DLPackOnlyPayload):
    def __dlpack_device__(self):
        return (1, 0)


def test_registry():
    class MockHandle:
        pass

    handle = MockHandle()
    dp = DataPlaneRegistry.create("ray", handle, sender_spmd_code=0)
    assert isinstance(dp, RayActorDataPlane)


def test_registry_creates_axon_data_plane():
    class MockHandle:
        payload_device_class = "cpu"
        completion_mode = "receiver_success"

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            return True

        async def invoke_control(self, op_name, step, names, values, sender_spmd_code):
            return True

    dp = DataPlaneRegistry.create("axon", MockHandle(), sender_spmd_code=0)
    assert isinstance(dp, AxonDataPlane)


def test_axon_data_plane_requires_explicit_receiver_success_completion_mode():
    class MissingCompletionMode:
        payload_device_class = "cpu"

    with pytest.raises(TypeError, match="completion_mode"):
        AxonDataPlane(MissingCompletionMode(), sender_spmd_code=0)

    class SendEnqueuedHandle:
        payload_device_class = "cpu"
        completion_mode = "send_enqueued"

    with pytest.raises(ValueError, match="receiver_success"):
        AxonDataPlane(SendEnqueuedHandle(), sender_spmd_code=0)


def test_ensure_cpu_serializable():
    # Test 1: normal array (should pass through)
    a = np.array([1, 2, 3])
    b = ensure_cpu_serializable(a)
    assert a is b

    # Test 2: memory view without owndata
    # create a large array and take a view
    base = np.zeros(1000)
    view = base[10:20]

    assert not view.flags.owndata

    fixed = ensure_cpu_serializable(view)

    assert fixed is not view
    assert fixed.flags.owndata
    assert np.array_equal(fixed, view)


@pytest.mark.asyncio
async def test_ray_actor_data_plane_sends_header_and_payload_buffers():
    class RemoteMethod:
        def __init__(self):
            self.args = None

        async def remote(self, *args):
            self.args = args
            return True

    class MockHandle:
        def __init__(self):
            self.receive_transport_frames = RemoteMethod()

    handle = MockHandle()
    adapter = RayActorDataPlane(handle, sender_spmd_code=7)
    ids = np.array([2**40], dtype=np.uint64)
    emb = np.arange(4, dtype=np.float32)
    header = _make_header()

    await adapter.receive_transport_frames(
        (
            TransportFrameEnvelope(
                header=header,
                payload_segments=(ids, emb),
            ),
        )
    )

    args = handle.receive_transport_frames.args
    assert args is not None
    frames = args[0]
    assert len(frames) == 1
    assert frames[0].header is header
    assert isinstance(frames[0].header, TransportFrameHeader)
    assert isinstance(frames[0].payload_segments[0], np.ndarray)
    assert isinstance(frames[0].payload_segments[1], np.ndarray)
    assert frames[0].payload_segments[0].dtype == np.uint64
    assert not any(
        isinstance(arg, (str, list, dict))
        for frame in frames
        for arg in frame.payload_segments
    )


def test_payload_device_class_detects_cpu_cuda_and_dlpack_cpu():
    assert payload_device_class(np.zeros(1, dtype=np.float32)) is PayloadDeviceClass.CPU
    assert payload_device_class(_FakeCudaPayload()) is PayloadDeviceClass.CUDA
    assert payload_device_class(_DLPackCpuPayload()) is PayloadDeviceClass.CPU


def test_payload_device_class_rejects_unverifiable_dlpack_payload():
    with pytest.raises(TypeError, match="__dlpack_device__"):
        payload_device_class(_DLPackOnlyPayload())


def test_validate_payload_iov_device_class_rejects_mixed_payloads():
    with pytest.raises(ValueError, match="must share one device class"):
        validate_payload_iov_device_class(
            (np.zeros(1, dtype=np.float32), _FakeCudaPayload())
        )


def test_validate_payload_iov_device_class_rejects_destination_mismatch():
    with pytest.raises(ValueError, match="source=cpu destination=cuda"):
        validate_payload_iov_device_class(
            (np.zeros(1, dtype=np.float32),),
            expected_device_class=PayloadDeviceClass.CUDA,
        )


@pytest.mark.asyncio
async def test_axon_data_plane_validates_and_invokes_flat_payload_iovs():
    class MockHandle:
        payload_device_class = "cpu"
        completion_mode = "receiver_success"

        def __init__(self):
            self.args = None

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            self.args = (frames, payload_device_class, sender_spmd_code)
            return True

    handle = MockHandle()
    adapter = AxonDataPlane(handle, sender_spmd_code=11)
    header = _make_header()
    ids = np.array([2**40], dtype=np.uint64)
    emb = np.arange(4, dtype=np.float32)

    result = await adapter.receive_transport_frames(
        (TransportFrameEnvelope(header=header, payload_segments=(ids, emb)),)
    )

    assert result is True
    assert handle.args is not None
    assert handle.args[0][0].header is header
    assert handle.args[1] == "cpu"
    assert handle.args[2] == 11
    assert handle.args[0][0].payload_segments[0] is ids
    assert handle.args[0][0].payload_segments[1] is emb


@pytest.mark.asyncio
async def test_axon_data_plane_rejects_payload_before_invoking_handle():
    class MockHandle:
        payload_device_class = "cuda"
        completion_mode = "receiver_success"

        def __init__(self):
            self.called = False

        async def invoke_transport_frames(
            self, frames, payload_device_class, sender_spmd_code
        ):
            self.called = True
            return True

    handle = MockHandle()
    adapter = AxonDataPlane(handle, sender_spmd_code=0)

    with pytest.raises(ValueError, match="source=cpu destination=cuda"):
        await adapter.receive_transport_frames(
            (
                TransportFrameEnvelope(
                    header=_make_header(),
                    payload_segments=(
                        np.zeros(1, dtype=np.float32),
                        np.ones(1, dtype=np.float32),
                    ),
                ),
            )
        )
    assert handle.called is False
