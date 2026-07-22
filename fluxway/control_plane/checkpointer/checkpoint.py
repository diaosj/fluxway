import itertools
import pickle
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
from collections.abc import Callable

from fluxway.control_plane.checkpointer.directory import CheckpointDirectory
from fluxway.control_plane.checkpointer.protocol import validate_checkpoint_state

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PathReplacement:
    final_path: Path
    previous_path: Path | None


@dataclass(frozen=True, slots=True)
class _PreparedStep:
    payload_tmp_path: Path
    replacements: tuple[_PathReplacement, ...]


class FluxwayCheckpointer:
    """
    Controller-side checkpoint orchestrator.

    Manages only the per-step payload.pkl that maps each (spmd_id, rank) to
    the path metadata returned by the Worker's WorkerCheckpointBackend.  The
    actual tensor serialization is performed by each Worker in its own process.

    Lifecycle initiated by Controller.report_op_completed (event-driven):
      1. save(step)  →  request_snapshot  →  Workers write tensors  →  payload written
      2. restore(step)  →  payload read  →  Workers read tensors

    Args:
        controller:     Ray Actor handle of the owning Controller.
        ckpt_dir:       Root directory on a shared filesystem for all checkpoints.
        interval_steps: How many completed steps between automatic checkpoints.
        max_to_keep:    Maximum number of on-disk checkpoint directories to retain.
        validate_deps:  Optional callable invoked at construction to validate optional
                        dependencies (e.g. ``import safetensors``). Raises if missing.
    """

    def __init__(
        self,
        controller,
        ckpt_dir: str,
        *,
        interval_steps: int,
        force_write: bool = False,
        max_to_keep: int = 2,
        validate_deps: Callable | None = None,
    ) -> None:
        if interval_steps < 1:
            raise ValueError(f"interval_steps must be >= 1, got {interval_steps}")
        if max_to_keep <= 0:
            raise ValueError(f"max_to_keep must be positive, got {max_to_keep}")

        if validate_deps is not None:
            validate_deps()

        self._controller = controller
        self._ckpt_dir = str(CheckpointDirectory.open(ckpt_dir).path)
        self._interval_steps = interval_steps
        self._force_write = force_write
        self._max_to_keep = max_to_keep
        self._snapshot_counter = itertools.count()
        self._prepared_steps: dict[int, _PreparedStep] = {}
        self._published_steps: dict[int, _PreparedStep] = {}

    @property
    def interval_steps(self) -> int:
        return self._interval_steps

    # =========================================================================
    # Public API
    # =========================================================================

    def list_steps(self) -> list[int]:
        """Return all valid checkpoint steps on disk, ascending."""
        root = Path(self._ckpt_dir)
        if not root.exists():
            return []
        return sorted(
            int(d.name)
            for d in root.iterdir()
            if d.is_dir() and d.name.isdigit() and (d / "payload.pkl").exists()
        )

    def latest_step(self) -> int | None:
        steps = self.list_steps()
        return steps[-1] if steps else None

    async def save(self, step: int) -> None:
        """Trigger a global snapshot at ``step`` and persist its payload."""
        snapshot_id = f"ckpt_{step}_{next(self._snapshot_counter)}"
        logger.info(f"Requesting snapshot '{snapshot_id}' at step {step}")
        published = False
        try:
            payload = await self._controller.request_snapshot(
                snapshot_id, step, self._ckpt_dir
            )
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"request_snapshot returned {type(payload).__name__} instead of dict; "
                    "controller may not be fully initialized."
                )
            control_state = payload.get("__fluxway_control_plane__")
            if not isinstance(control_state, dict):
                raise ValueError(
                    "snapshot payload is missing Fluxway control-plane state"
                )
            self._prepare(step, payload)
            await self._controller.commit_snapshot_artifacts(snapshot_id)
            self._publish(step)
            published = True
        except BaseException as primary_error:
            cleanup_errors = []
            if not published:
                try:
                    await self._controller.abort_snapshot_artifacts(snapshot_id)
                except BaseException as abort_error:
                    cleanup_errors.append(abort_error)
                try:
                    self._discard_prepared(step)
                except BaseException as discard_error:
                    cleanup_errors.append(discard_error)
            if cleanup_errors:
                raise BaseExceptionGroup(
                    "checkpoint save and cleanup failed",
                    [primary_error, *cleanup_errors],
                ) from None
            raise
        try:
            await self._controller.finalize_snapshot(step, control_state)
        finally:
            self._cleanup_published(step)
            self._evict_old_checkpoints()
        logger.info(f"Checkpoint step {step} saved ({list(payload.keys())})")

    async def restore(self, step: Literal["latest"] | int | None = None) -> int | None:
        """
        Restore state from a checkpoint.

        Returns the step restored, or None if no checkpoint exists and the latest
        checkpoint was requested.
        Raises FileNotFoundError if a specific step is requested but absent.
        """
        if step is None or step == "latest":
            step = self.latest_step()
            if step is None:
                logger.info("No existing checkpoint found.")
                return None
        if isinstance(step, bool) or not isinstance(step, int):
            raise TypeError(
                "Checkpoint restore selector must be 'latest', int, or None, "
                f"got {step!r}."
            )

        # Single filesystem read — avoids TOCTOU between list_steps() and _load().
        try:
            payload = self._load(step)
        except FileNotFoundError as e:
            available = self.list_steps()
            raise FileNotFoundError(
                f"Checkpoint step {step} not found in {self._ckpt_dir}. Available: {available}"
            ) from e

        if not payload:
            raise RuntimeError(
                f"_load({step}) returned empty — checkpoint may be corrupt."
            )
        snapshot_id = f"restore_{step}_{next(self._snapshot_counter)}"
        await self._controller.restore_snapshot(snapshot_id, step, payload)
        logger.info(f"Checkpoint step {step} restored.")
        return step

    # =========================================================================
    # Internal Persistence (format-agnostic payload (pickle))
    # =========================================================================

    def _persist(self, step: int, payload: dict[str, Any]) -> None:
        """Write the payload that maps spmd_id to {spmd_state, rank_storage}."""
        self._prepare(step, payload)
        self._publish(step)
        self._cleanup_published(step)
        self._evict_old_checkpoints()

    def _prepare(self, step: int, payload: dict[str, Any]) -> None:
        validate_checkpoint_state(payload, path="payload")
        step_dir = Path(self._ckpt_dir) / str(step)
        payload_path = step_dir / "payload.pkl"

        if step in self._prepared_steps or step in self._published_steps:
            raise RuntimeError(
                f"checkpoint step {step} already has an active transaction"
            )
        planned_paths: list[tuple[Path, Path, Path | None]] = []
        renamed_paths: set[str] = set()
        for spmd_entry in payload.values():
            for meta in spmd_entry.get("rank_storage", {}).values():
                if "orbax_tmp_path" in meta and "orbax_final_path" in meta:
                    o_tmp = Path(meta["orbax_tmp_path"])
                    o_final = Path(meta["orbax_final_path"])

                    if str(o_final) not in renamed_paths:
                        if o_tmp.exists():
                            previous_path = None
                            if o_final.exists():
                                if not self._force_write:
                                    raise FileExistsError(
                                        f"Orbax checkpoint directory already exists at {o_final}. "
                                        "Set force_write=True to overwrite."
                                    )
                                previous_path = o_final.with_name(
                                    f"{o_final.name}.fluxway-previous"
                                )
                                if previous_path.exists():
                                    raise FileExistsError(
                                        "checkpoint rollback path already exists: "
                                        f"{previous_path}"
                                    )
                            planned_paths.append((o_tmp, o_final, previous_path))
                            renamed_paths.add(str(o_final))
                        elif not o_final.exists():
                            raise FileNotFoundError(
                                "Orbax checkpoint has neither prepared nor final path: "
                                f"prepared={o_tmp} final={o_final}"
                            )

        if payload_path.exists() and not self._force_write:
            raise FileExistsError(
                f"Checkpoint step {step} already exists at {payload_path}. "
                "Set force_write=True in CheckpointConfig to overwrite."
            )
        step_dir.mkdir(parents=True, exist_ok=True)
        tmp_man = step_dir / "payload.pkl.tmp"
        if tmp_man.exists():
            raise FileExistsError(f"checkpoint prepare file already exists: {tmp_man}")
        try:
            with open(tmp_man, "wb") as f:
                pickle.dump(payload, f)
        except BaseException:
            tmp_man.unlink(missing_ok=True)
            raise
        replacements: list[_PathReplacement] = []
        try:
            for prepared_path, final_path, previous_path in planned_paths:
                if previous_path is not None:
                    final_path.rename(previous_path)
                try:
                    prepared_path.rename(final_path)
                except BaseException:
                    if previous_path is not None:
                        previous_path.rename(final_path)
                    raise
                replacements.append(_PathReplacement(final_path, previous_path))
        except BaseException:
            self._rollback_replacements(replacements)
            tmp_man.unlink(missing_ok=True)
            raise
        self._prepared_steps[step] = _PreparedStep(tmp_man, tuple(replacements))

    def _publish(self, step: int) -> None:
        step_dir = Path(self._ckpt_dir) / str(step)
        payload_path = step_dir / "payload.pkl"
        transaction = self._prepared_steps.get(step)
        if transaction is None:
            raise RuntimeError(f"checkpoint step {step} has no prepared transaction")
        tmp_man = transaction.payload_tmp_path
        if not tmp_man.exists():
            raise FileNotFoundError(f"checkpoint prepare file is missing: {tmp_man}")
        tmp_man.replace(payload_path)
        self._published_steps[step] = self._prepared_steps.pop(step)

    def _discard_prepared(self, step: int) -> None:
        transaction = self._prepared_steps.pop(step, None)
        if transaction is None:
            return
        transaction.payload_tmp_path.unlink(missing_ok=True)
        self._rollback_replacements(transaction.replacements)

    def _cleanup_published(self, step: int) -> None:
        transaction = self._published_steps.pop(step, None)
        if transaction is None:
            raise RuntimeError(f"checkpoint step {step} has no published transaction")
        for replacement in transaction.replacements:
            if replacement.previous_path is not None:
                self._remove_path(replacement.previous_path)

    @classmethod
    def _rollback_replacements(
        cls, replacements: list[_PathReplacement] | tuple[_PathReplacement, ...]
    ) -> None:
        for replacement in reversed(replacements):
            cls._remove_path(replacement.final_path)
            if replacement.previous_path is not None:
                replacement.previous_path.rename(replacement.final_path)

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not path.exists():
            return
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink()

    def _load(self, step: int) -> dict[str, Any]:
        """Read and return the payload dict for ``step``."""
        payload_path = Path(self._ckpt_dir) / str(step) / "payload.pkl"
        if not payload_path.exists():
            raise FileNotFoundError(
                f"FluxwayCheckpointer: payload not found: {payload_path}"
            )
        with open(payload_path, "rb") as f:
            payload = pickle.load(f)
        if not isinstance(payload, dict):
            raise TypeError("Fluxway checkpoint payload root must be a dictionary")
        return payload

    def _evict_old_checkpoints(self) -> None:
        """Remove oldest checkpoint directories exceeding max_to_keep."""
        steps = self.list_steps()
        for old_step in steps[: -self._max_to_keep]:
            shutil.rmtree(Path(self._ckpt_dir) / str(old_step))
