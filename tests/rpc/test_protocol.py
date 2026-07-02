import numpy as np
import pytest

from fluxway.rpc.protocol import DataPlaneRegistry, RayActorDataPlane
from fluxway.common.tensor_backend import ensure_cpu_serializable


def test_registry():
    class MockHandle:
        pass

    handle = MockHandle()
    dp = DataPlaneRegistry.create("ray", handle, sender_spmd_code=0)
    assert isinstance(dp, RayActorDataPlane)


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
async def test_ray_actor_data_plane_sends_flat_transport_frame_arguments():
    class RemoteMethod:
        def __init__(self):
            self.args = None

        async def remote(self, *args):
            self.args = args
            return True

    class MockHandle:
        def __init__(self):
            self.receive_transport_frame = RemoteMethod()

    handle = MockHandle()
    adapter = RayActorDataPlane(handle, sender_spmd_code=7)
    frame_i32 = np.arange(8, dtype=np.int32)
    edge_i32 = np.zeros((1, 11), dtype=np.int32)
    payload_i32 = np.zeros((2, 8), dtype=np.int32)
    shape_i32 = np.array([2], dtype=np.int32)
    global_shape_i64 = np.array([2**35], dtype=np.int64)
    compact_i64 = np.array([2**33, 1, 4, 1], dtype=np.int64)
    ids = np.array([2**40], dtype=np.uint64)
    emb = np.arange(4, dtype=np.float32)

    await adapter.receive_transport_frame(
        frame_i32,
        edge_i32,
        payload_i32,
        shape_i32,
        global_shape_i64,
        compact_i64,
        ids,
        emb,
    )

    args = handle.receive_transport_frame.args
    assert args is not None
    assert args[:6] == (
        frame_i32,
        edge_i32,
        payload_i32,
        shape_i32,
        global_shape_i64,
        compact_i64,
    )
    assert isinstance(args[6], np.ndarray)
    assert isinstance(args[7], np.ndarray)
    assert args[6].dtype == np.uint64
    assert not any(isinstance(arg, (str, list, dict)) for arg in args)
