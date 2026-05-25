from .checkpoint import FluxwayCheckpointer
from .backend import WorkerCheckpointBackend
from .safetensors import SafeTensorsBackend

try:
    from .orbax import OrbaxBackend  # type: ignore
except ModuleNotFoundError as exc:
    _ORBAX_IMPORT_ERROR = exc

    class OrbaxBackend(WorkerCheckpointBackend, backend_name="orbax"):  # type: ignore[misc]
        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError(
                "OrbaxBackend requires optional jax/orbax dependencies."
            ) from _ORBAX_IMPORT_ERROR

        async def save(self, *args, **kwargs):
            pass

        async def load(self, *args, **kwargs):
            pass

        def finalize(self) -> None:
            pass


__all__ = [
    "FluxwayCheckpointer",
    "WorkerCheckpointBackend",
    "SafeTensorsBackend",
    "OrbaxBackend",
]
