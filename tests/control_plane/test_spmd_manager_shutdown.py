from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fluxway.control_plane.spmd_manager.manager import SPMDHandle, SPMDManager


class _AsyncRemoteMethod:
    def __init__(self):
        self.calls = []

    def remote(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self._result()

    async def _result(self):
        return None


@pytest.mark.asyncio
async def test_stop_spmd_releases_driver_without_force_kill():
    manager = SPMDManager()
    handle = SPMDHandle("dense")

    stop_method = _AsyncRemoteMethod()
    driver_handle = SimpleNamespace(stop=stop_method)
    handle.set_driver_handle(driver_handle)
    manager.register_spmd("dense", handle)

    await manager.stop_spmd("dense", save_checkpoint=False)

    assert stop_method.calls == [((False,), {})]
    assert handle.driver_handle is None


@pytest.mark.asyncio
async def test_stop_spmd_can_preserve_driver_reference_during_terminal_reclaim():
    manager = SPMDManager()
    handle = SPMDHandle("dense")

    stop_method = _AsyncRemoteMethod()
    driver_handle = SimpleNamespace(stop=stop_method)
    handle.set_driver_handle(driver_handle)
    manager.register_spmd("dense", handle)

    await manager.stop_spmd("dense", save_checkpoint=False, release_driver_handle=False)

    assert stop_method.calls == [((False,), {})]
    assert handle.driver_handle is driver_handle


@pytest.mark.asyncio
async def test_stop_spmd_propagates_driver_shutdown_failure_after_state_reclaim():
    manager = SPMDManager()
    handle = SPMDHandle("metrics")

    stop_method = _AsyncRemoteMethod()
    stop_method._result = AsyncMock(side_effect=RuntimeError("drain failed"))
    driver_handle = SimpleNamespace(stop=stop_method)
    handle.set_driver_handle(driver_handle)
    manager.register_spmd("metrics", handle)

    with pytest.raises(RuntimeError, match="drain failed"):
        await manager.stop_spmd("metrics", save_checkpoint=False)

    assert handle.driver_handle is None
