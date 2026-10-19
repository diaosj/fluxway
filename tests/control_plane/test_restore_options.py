import asyncio
import inspect
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

import fluxway
from fluxway.common.types import ControllerState
from fluxway.common.types.checkpoint import validate_restore_options
from fluxway.control_plane.lifecycle import ExecutionLifecycle


def _options() -> dict[str, dict[int, dict[str, bytes]]]:
    return {"Embedding_user": {0: {"user_embedding": b"restore"}}}


def test_public_start_exposes_restore_options() -> None:
    parameter = inspect.signature(fluxway.start).parameters["restore_options"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


def test_restore_options_validator_rejects_invalid_shape() -> None:
    with pytest.raises(TypeError, match="non-empty bytes"):
        validate_restore_options({"Embedding_user": {0: {"embedding": b""}}})
    with pytest.raises(TypeError, match="non-negative integers"):
        validate_restore_options({"Embedding_user": {True: {"embedding": b"x"}}})


def test_restore_options_validator_accepts_opaque_mapping() -> None:
    validate_restore_options(_options())


def test_lifecycle_forwards_restore_options_without_decoding() -> None:
    options = _options()
    checkpointer = SimpleNamespace(restore=AsyncMock(return_value=37))
    controller = SimpleNamespace(
        state_machine=SimpleNamespace(state=ControllerState.READY),
        _checkpointer=checkpointer,
    )

    async def scenario() -> None:
        restored = await ExecutionLifecycle(controller).restore(
            "latest", restore_options=options
        )
        assert restored == 37
        checkpointer.restore.assert_awaited_once_with("latest", restore_options=options)

    asyncio.run(scenario())
