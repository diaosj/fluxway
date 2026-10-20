from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import shutil


CHECKPOINT_ROOT_MARKER = ".fluxway-checkpoint-root"
_CHECKPOINT_ROOT_FORMAT = "fluxway-checkpoint-root\n"


class CheckpointDirectoryOwnershipError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointDirectory:
    path: Path

    @classmethod
    def open(cls, checkpoint_dir: str | Path) -> CheckpointDirectory:
        path = Path(os.path.abspath(checkpoint_dir))
        if path.is_symlink() or path.exists():
            cls._validate_existing(path)
            return cls(path)

        path.mkdir(parents=True, exist_ok=False)
        cls._write_marker(path)
        return cls(path)

    @classmethod
    def clear(cls, checkpoint_dir: str | Path) -> None:
        path = Path(os.path.abspath(checkpoint_dir))
        if not path.exists():
            return
        cls._validate_owned(path)
        shutil.rmtree(path)

    @staticmethod
    def _validate_existing(path: Path) -> None:
        CheckpointDirectory._validate_path(path)
        marker = path / CHECKPOINT_ROOT_MARKER
        if marker.exists() or marker.is_symlink():
            CheckpointDirectory._validate_marker(marker, path)

    @staticmethod
    def _validate_owned(path: Path) -> None:
        CheckpointDirectory._validate_path(path)
        CheckpointDirectory._validate_marker(path / CHECKPOINT_ROOT_MARKER, path)

    @staticmethod
    def _validate_path(path: Path) -> None:
        if path.is_symlink() or not path.is_dir():
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint path {path} is not an owned directory."
            )

    @staticmethod
    def _validate_marker(marker: Path, path: Path) -> None:
        if marker.is_symlink() or not marker.is_file():
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint directory {path} is not owned by Fluxway."
            )
        try:
            marker_format = marker.read_text(encoding="ascii")
        except (OSError, UnicodeError) as error:
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint directory {path} has an invalid ownership marker."
            ) from error
        if marker_format != _CHECKPOINT_ROOT_FORMAT:
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint directory {path} has an invalid ownership marker."
            )

    @staticmethod
    def _write_marker(path: Path) -> None:
        marker = path / CHECKPOINT_ROOT_MARKER
        with marker.open("x", encoding="ascii") as stream:
            stream.write(_CHECKPOINT_ROOT_FORMAT)
