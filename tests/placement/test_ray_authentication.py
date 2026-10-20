import os

import pytest
from ray._raylet import (
    AuthenticationMode,
    AuthenticationTokenLoader,
    get_authentication_mode,
)

from tests.ray_cluster_process import _ray_authentication


def test_cluster_authentication_is_runtime_only_and_restored_after_failure():
    names = ("RAY_AUTH_MODE", "RAY_AUTH_TOKEN", "RAY_AUTH_TOKEN_PATH")
    original = {name: os.environ.get(name) for name in names}
    with pytest.raises(RuntimeError, match="cluster startup failed"):
        with _ray_authentication():
            assert get_authentication_mode() == AuthenticationMode.TOKEN
            assert AuthenticationTokenLoader.instance().has_token()
            assert "RAY_AUTH_TOKEN_PATH" not in os.environ
            fresh_token = os.environ["RAY_AUTH_TOKEN"] != original["RAY_AUTH_TOKEN"]
            assert fresh_token
            raise RuntimeError("cluster startup failed")
    restored = all(os.environ.get(name) == value for name, value in original.items())
    assert restored
