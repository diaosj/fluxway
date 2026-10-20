import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("missing", ["jax", "orbax"])
def test_orbax_dependencies_are_lazy_and_missing_backend_fails_explicitly(
    tmp_path, missing
):
    program = textwrap.dedent(
        """
        import asyncio
        import sys

        class BlockBackend:
            blocked = ("jax", "orbax", "etils")
            attempts = []

            def find_spec(self, fullname, path=None, target=None):
                if any(fullname == name or fullname.startswith(name + ".")
                       for name in self.blocked):
                    self.attempts.append(fullname)
                    raise ModuleNotFoundError("backend blocked by test", name=fullname)

        blocker = BlockBackend()
        sys.meta_path.insert(0, blocker)
        import fluxway
        from fluxway.control_plane import OrbaxCheckpointComponent as public
        from fluxway.control_plane.checkpointer import OrbaxCheckpointComponent
        from fluxway.common.types.checkpoint import CheckpointRuntime, CheckpointRuntimeRole

        assert public is OrbaxCheckpointComponent
        assert not any(name.split(".")[0] in ("orbax", "etils") for name in blocker.attempts)
        assert not any(name.split(".")[0] in blocker.blocked for name in sys.modules)
        blocker.attempts.clear()
        component = public("test", lambda *_: None, lambda *_: None, lambda *_: {})
        assert not blocker.attempts, blocker.attempts

        blocker.blocked = (sys.argv[2],)
        runtime = CheckpointRuntime(CheckpointRuntimeRole.WORKER, sys.argv[1], "train", 0)
        try:
            asyncio.run(component.open(runtime))
        except ImportError as error:
            assert "OrbaxCheckpointComponent requires" in str(error)
            assert "pip install fluxway[jax]" in str(error)
            assert isinstance(error.__cause__, ImportError)
            assert blocker.attempts
        else:
            raise AssertionError("missing backend was not rejected")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(tmp_path), missing],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
