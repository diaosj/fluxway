import textwrap
from pathlib import Path

import pytest

import fluxway
from fluxway.common.decorators import (
    _is_worker_process_required,
    _validate_worker_process_marker,
)


def test_requires_worker_process_marks_class_for_worker_process_lane():
    @fluxway.requires_worker_process
    class Trainer:
        pass

    assert _is_worker_process_required(Trainer) is True


def test_missing_worker_process_marker_fails_for_jax_distributed_initialize(tmp_path):
    module_path = Path(tmp_path) / "missing_marker.py"
    module_path.write_text(
        textwrap.dedent(
            """
            class Trainer:
                def __init__(self):
                    import jax
                    jax.distributed.initialize(
                        coordinator_address=self.rank0_address,
                        num_processes=self.world_size,
                        process_id=self.rank,
                    )
            """
        )
    )

    import importlib.util

    spec = importlib.util.spec_from_file_location("missing_marker", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with pytest.raises(RuntimeError, match="@fluxway.requires_worker_process"):
        _validate_worker_process_marker(module.Trainer)
