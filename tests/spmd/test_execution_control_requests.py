from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from fluxway.spmd.worker import SPMDWorker
from fluxway.spmd.worker.operator_manager.dispatcher import (
    ExecutionResult,
    ExecutionStatus,
)


def test_execution_result_requires_a_tuple_of_control_requests() -> None:
    marker = object()
    result = ExecutionResult(
        ExecutionStatus.SOURCE_EXHAUSTED, control_requests=(marker,)
    )
    assert result.control_requests == (marker,)
    with pytest.raises(TypeError, match="must be a tuple"):
        ExecutionResult(ExecutionStatus.SOURCE_EXHAUSTED, control_requests=[marker])
    with pytest.raises(ValueError, match="cannot contain None"):
        ExecutionResult(ExecutionStatus.SOURCE_EXHAUSTED, control_requests=(None,))


@pytest.mark.asyncio
async def test_worker_forwards_generic_control_requests_without_source_introspection() -> (
    None
):
    worker_class = SPMDWorker.__ray_metadata__.modified_class  # type: ignore[attr-defined]
    worker = object.__new__(worker_class)
    marker = object()
    forward = AsyncMock()
    worker._driver_handle = SimpleNamespace(
        forward_control_plane_request=SimpleNamespace(remote=forward)
    )

    await worker_class._forward_control_requests(worker, (marker,))

    forward.assert_awaited_once_with(marker)
