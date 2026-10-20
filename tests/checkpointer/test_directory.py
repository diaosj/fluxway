from pathlib import Path

import pytest

from fluxway.control_plane.checkpointer.directory import (
    CHECKPOINT_ROOT_MARKER,
    CheckpointDirectory,
    CheckpointDirectoryOwnershipError,
)


@pytest.mark.parametrize(
    "marker_contents", ["business-marker\n", "fluxway-checkpoint-root-v1\n"]
)
def test_open_rejects_an_invalid_existing_marker(
    tmp_path: Path, marker_contents: str
) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / CHECKPOINT_ROOT_MARKER).write_text(marker_contents, encoding="ascii")

    with pytest.raises(
        CheckpointDirectoryOwnershipError, match="invalid ownership marker"
    ):
        CheckpointDirectory.open(root)


def test_open_rejects_a_symlink_root(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "repository"
    link.symlink_to(target, target_is_directory=True)

    with pytest.raises(
        CheckpointDirectoryOwnershipError, match="not an owned directory"
    ):
        CheckpointDirectory.open(link)


def test_open_rejects_a_symlink_marker(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    marker_owner = CheckpointDirectory.open(tmp_path / "marker-owner")
    marker_target = marker_owner.path / CHECKPOINT_ROOT_MARKER
    (root / CHECKPOINT_ROOT_MARKER).symlink_to(marker_target)

    with pytest.raises(CheckpointDirectoryOwnershipError, match="not owned"):
        CheckpointDirectory.open(root)


def test_clear_rejects_an_existing_unmarked_directory(tmp_path: Path) -> None:
    root = tmp_path / "repository"
    root.mkdir()
    (root / "business-repair").write_text("ok", encoding="utf-8")

    with pytest.raises(CheckpointDirectoryOwnershipError, match="not owned"):
        CheckpointDirectory.clear(root)

    assert root.is_dir()
