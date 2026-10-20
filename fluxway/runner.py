import os
import sys
from typing import Any
from collections.abc import Callable, Awaitable
import logging
import ray

from fluxway.common import asyncio_runtime

logger = logging.getLogger(__name__)


class _EnvRestorer:
    """Context manager to restore environment variables."""

    def __init__(self, keys):
        self.keys = keys
        self.saved = {}

    def __enter__(self):
        for k in self.keys:
            self.saved[k] = os.environ.get(k)

    def __exit__(self, exc_type, exc_val, exc_tb):
        for k, v in self.saved.items():
            if v is None and k in os.environ:
                del os.environ[k]
            elif v is not None:
                os.environ[k] = v


def run_in_sandbox(
    entry_fn: Callable[..., Awaitable[Any]],
    *args: Any,
    ray_init_kwargs: dict[str, Any] | None = None,
    runtime_env: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """Run an async entry function inside a zero-GPU Ray sandbox.

    Runs ``entry_fn`` inside an isolated Ray Task with **zero GPUs** so that
    the main / Jupyter process never triggers eager GPU memory allocation when
    JAX or PyTorch is imported.

    All stateful orchestration (scheduling, heartbeat, checkpoint, drain,
    backpressure, dynamic graph reconfiguration, …) is the exclusive
    responsibility of the **Controller** created inside ``entry_fn``.
    This function intentionally uses a Ray Task (not an Actor) to make
    that boundary unambiguous.

    Args:
        entry_fn: An async coroutine containing the user's training logic
                  (graph construction → Controller creation → execution).
        *args:    Positional arguments forwarded to ``entry_fn``.
        ray_init_kwargs:
                  Optional kwargs passed to ``ray.init()`` when Ray has not
                  been initialised yet.  Defaults to
                  ``ignore_reinit_error=True`` for Jupyter compatibility.
        runtime_env:
                  Optional Ray runtime_env dict forwarded to the sandbox task
                  (e.g. ``{"env_vars": {"MY_VAR": "1"}}``).
        **kwargs: Keyword arguments forwarded to ``entry_fn``.

    Returns:
        Whatever ``entry_fn`` returns.
    """
    if not ray.is_initialized():
        init_kw = (ray_init_kwargs or {}).copy()
        init_kw.setdefault("ignore_reinit_error", True)
        if "address" not in init_kw and os.environ.get("RAY_ADDRESS"):
            init_kw["address"] = os.environ["RAY_ADDRESS"]
        ray.init(**init_kw)

    sandbox_env = {
        "JAX_PLATFORMS": "cpu",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    if runtime_env is not None and "env_vars" in runtime_env:
        sandbox_env.update(runtime_env["env_vars"])

    sandbox_runtime_env = dict(runtime_env or {})
    sandbox_runtime_env["env_vars"] = dict(sandbox_env)

    task_options: dict[str, Any] = {
        "num_cpus": 0,
        "num_gpus": 0,
        "runtime_env": sandbox_runtime_env,
    }

    @ray.remote(**task_options)
    def _sandbox(fn, a, kw):
        # Re-apply the same sandbox environment around user orchestration code
        # so direct environment reads observe the launch contract for the whole
        # entry_fn execution.
        with _EnvRestorer(sandbox_env.keys()):
            for k, v in sandbox_env.items():
                os.environ[k] = v
            return asyncio_runtime.run(fn(*a, **kw))

    ref = _sandbox.remote(entry_fn, args, kwargs)
    try:
        return ray.get(ref)
    except KeyboardInterrupt:
        logger.warning("Interrupted — cancelling sandbox task …")
        ray.cancel(ref, force=True)
        sys.exit(1)
