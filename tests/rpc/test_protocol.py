import numpy as np
from fluxway.rpc.protocol import DataPlaneRegistry, RayActorDataPlane
from fluxway.common.tensor_backend import ensure_cpu_serializable


def test_registry():
    class MockHandle:
        pass

    handle = MockHandle()
    dp = DataPlaneRegistry.create("ray", handle)
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
