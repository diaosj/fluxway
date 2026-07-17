from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
import shutil


CHECKPOINT_ROOT_MARKER = ".fluxway-checkpoint-root"
_CHECKPOINT_ROOT_FORMAT = "fluxway-checkpoint-root-v1\n"


class CheckpointDirectoryOwnershipError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class CheckpointDirectory:
    path: Path

    @classmethod
    def open(cls, checkpoint_dir: str | Path) -> CheckpointDirectory:
        path = Path(os.path.abspath(checkpoint_dir))
        if path.exists():
            cls._validate_owned(path)
            return cls(path)

        path.mkdir(parents=True, exist_ok=False)
        marker = path / CHECKPOINT_ROOT_MARKER
        with marker.open("x", encoding="ascii") as stream:
            stream.write(_CHECKPOINT_ROOT_FORMAT)
        return cls(path)

    @classmethod
    def clear(cls, checkpoint_dir: str | Path) -> None:
        path = Path(os.path.abspath(checkpoint_dir))
        if not path.exists():
            return
        cls._validate_owned(path)
        shutil.rmtree(path)

    @staticmethod
    def _validate_owned(path: Path) -> None:
        marker = path / CHECKPOINT_ROOT_MARKER
        if path.is_symlink() or not path.is_dir():
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint path {path} is not an owned directory."
            )
        if marker.is_symlink() or not marker.is_file():
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint directory {path} is not owned by Fluxway."
            )
        if marker.read_text(encoding="ascii") != _CHECKPOINT_ROOT_FORMAT:
            raise CheckpointDirectoryOwnershipError(
                f"Checkpoint directory {path} has an invalid ownership marker."
            )
