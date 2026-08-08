"""
Per-Worker safetensors serialization utilities.

Called inside Worker Ray Actor processes (not the Controller).
Tensors are stored in a ``.safetensors`` file; non-tensor metadata
(scalars, strings, lists) is stored in a companion ``.meta.json`` sidecar.
"""

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import asyncio
import numpy as np

from fluxway.common.types.checkpoint import (
    CheckpointObjectRecord,
    SaveSession,
    RestoreSession,
    RestoreTopologyMode,
    KeepPolicy,
)
from fluxway.control_plane.checkpointer._filesystem_backend import (
    FilesystemCheckpointComponent,
)

_SEP = "\x00"  # Separator for composite keys; cannot appear in field names


class SafeTensorsCheckpointComponent(FilesystemCheckpointComponent):
    """safetensors tensors + JSON sidecar."""

    def __init__(
        self,
        component_id: str,
        capture_state,
        restore_state,
        *,
        keep: KeepPolicy = KeepPolicy(),
    ) -> None:
        super().__init__(
            component_id,
            capture_state,
            restore_state,
            keep=keep,
            restore_topology=RestoreTopologyMode.EXACT_TOPOLOGY,
        )

    async def _write_state(
        self, state: Any, directory: Path, context: SaveSession
    ) -> Mapping[str, Any]:
        tensors_path = directory / "state.safetensors"
        metadata_path = directory / "state.meta.json"
        await asyncio.to_thread(
            save_worker_state, state, str(tensors_path), str(metadata_path)
        )
        return {"tensors": "state.safetensors", "metadata": "state.meta.json"}

    async def _read_state(
        self, record: CheckpointObjectRecord, context: RestoreSession
    ) -> Any:
        if len(record.checkpoint_objects) != 1:
            raise ValueError(
                "SafeTensors record must contain exactly one checkpoint_object"
            )
        directory = self._file_path(record.checkpoint_objects[0].storage_locator)
        return await asyncio.to_thread(
            load_worker_state,
            str(directory / "state.safetensors"),
            str(directory / "state.meta.json"),
        )

    async def _finalize_io(self, context: SaveSession) -> None:
        return None

    async def _close_resources(self) -> None:
        return None


def _try_jax_flatten(value: Any) -> dict[str, np.ndarray] | None:
    try:
        import jax
    except ImportError:
        return None

    leaves_with_paths = jax.tree_util.tree_leaves_with_path(value)
    if not leaves_with_paths:
        return None
    result = {}
    for path, leaf in leaves_with_paths:
        path_str = "/".join(str(p.key) if hasattr(p, "key") else str(p) for p in path)
        result[path_str] = np.asarray(leaf)
    return result


def _flatten_worker_state(
    state: dict[str, Any], prefix: str = ""
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Flatten state dict into (flat_tensors, meta_fields), recursing into nested dicts."""
    flat_tensors: dict[str, np.ndarray] = {}
    meta_fields: dict[str, Any] = {}

    for field_key, value in state.items():
        key = f"{prefix}{_SEP}{field_key}" if prefix else field_key

        if isinstance(value, np.ndarray):
            flat_tensors[key] = value
        elif hasattr(value, "detach"):  # PyTorch tensor
            flat_tensors[key] = value.detach().cpu().numpy()
        elif hasattr(value, "__jax_array__") or (
            not isinstance(value, int | float | bool | str | dict | list | type(None))
            and type(value).__module__.startswith("jax")
        ):
            flat_tensors[key] = np.asarray(value)
        elif isinstance(value, dict):
            if not value:
                meta_fields[key] = {}
            else:
                inner_tensors, inner_meta = _flatten_worker_state(value, prefix=key)
                flat_tensors.update(inner_tensors)
                meta_fields.update(inner_meta)
        else:
            # Primitives must be checked before _try_jax_flatten: JAX treats
            # Python scalars as traceable leaves, producing an empty-path entry.
            if isinstance(
                value, int | float | bool | str | bytes | list | tuple | type(None)
            ):
                meta_fields[key] = value
            else:
                jax_leaves = _try_jax_flatten(value)
                if jax_leaves is not None and len(jax_leaves) > 0:
                    for path_str, arr in jax_leaves.items():
                        flat_tensors[f"{key}{_SEP}{path_str}"] = arr
                else:
                    raise TypeError(
                        f"_flatten_worker_state: non-serializable value at key '{key}': "
                        f"{type(value)!r}. Only NumPy/JAX/PyTorch arrays, JAX pytrees, "
                        f"and JSON-compatible primitives (int, float, bool, str, list, "
                        f"dict, None) are supported."
                    )

    return flat_tensors, meta_fields


def _unflatten_worker_state(
    flat_tensors: dict[str, Any], meta_fields: dict[str, Any]
) -> dict[str, Any]:
    """Reconstruct state dict from flat tensors + JSON metadata."""

    def _set_nested(d: dict, parts: list, value: Any) -> None:
        for part in parts[:-1]:
            d = d.setdefault(part, {})
        leaf = parts[-1]
        existing = d.get(leaf)
        # Merge sibling keys that share a parent (pytree group dicts)
        if isinstance(existing, dict) and isinstance(value, dict):
            existing.update(value)
        else:
            d[leaf] = value

    state: dict[str, Any] = {}

    for key, arr in flat_tensors.items():
        parts = key.split(_SEP)
        _set_nested(state, parts, arr)

    for key, val in meta_fields.items():
        parts = key.split(_SEP)
        _set_nested(state, parts, val)

    return state


def save_worker_state(state: dict[str, Any], tensors_path: str, meta_path: str) -> None:
    from safetensors.numpy import save_file

    flat_tensors, meta_fields = _flatten_worker_state(state)
    Path(tensors_path).parent.mkdir(parents=True, exist_ok=True)
    if flat_tensors:
        save_file(flat_tensors, tensors_path)
    with open(meta_path, "w") as f:
        json.dump(meta_fields, f)


def load_worker_state(tensors_path: str, meta_path: str) -> dict[str, Any]:
    """
    Deserialize one Worker's state dict from disk.

    Args:
        tensors_path: Absolute path to the ``.safetensors`` file.
        meta_path:    Absolute path to the ``.meta.json`` sidecar.

    Returns:
        State dict returned to the registered restore callback.
    """
    from safetensors.numpy import load_file

    flat_tensors: dict[str, np.ndarray] = {}
    if Path(tensors_path).exists():
        flat_tensors = load_file(tensors_path)

    if not Path(meta_path).exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")

    with open(meta_path) as f:
        meta_fields = json.load(f)

    return _unflatten_worker_state(flat_tensors, meta_fields)
