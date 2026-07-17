import textwrap
from pathlib import Path

import pytest

import fluxway
from fluxway.common.mesh import DeviceMesh
from fluxway.common.decorators import (
    _is_worker_process_required,
    _validate_worker_process_marker,
    operator,
    requires_worker_process,
    spmd,
)
from fluxway import OperatorSpec
from fluxway.spmd.worker.worker import SPMDWorker
from tests.compiled_graph_fixtures import single_spmd_config


def test_requires_worker_process_marks_class_for_worker_process_lane():
    @fluxway.requires_worker_process
    class Trainer:
        pass

    assert _is_worker_process_required(Trainer) is True


def test_requires_worker_process_lowers_to_operator_meta():
    @spmd()
    @requires_worker_process
    class Trainer:
        @operator()
        def step(self):
            return None

    config = Trainer.get_spmd_config()  # type: ignore

    assert config.operator_specs["step"].requires_worker_process is True


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
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    with pytest.raises(RuntimeError, match="@fluxway.requires_worker_process"):
        _validate_worker_process_marker(module.Trainer)


def test_operator_preinstantiation_creates_each_owner_once_in_worker():
    created = []

    @requires_worker_process
    class WorkerOperator:
        def __init__(self) -> None:
            created.append("worker")

        def update(self) -> None:
            return None

    class StandardOperator:
        def __init__(self) -> None:
            created.append("standard")

    config = single_spmd_config(
        name="operator-preinstantiation",
        spmd_id="test",
        mesh=DeviceMesh(axis_names=("data",), device_shape=(1,)),
        operators={
            "worker": OperatorSpec(
                name="worker",
                owner=WorkerOperator,
                requires_worker_process=True,
            ),
            "worker_update": OperatorSpec(
                name="worker_update",
                owner=WorkerOperator,
                requires_worker_process=True,
            ),
            "standard": OperatorSpec(
                name="standard",
                owner=StandardOperator,
                requires_worker_process=False,
            ),
        },
    )
    worker_type = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]

    worker_type._pre_instantiate_operators(
        config,
        rank=0,
        static_args={},
    )
    assert sorted(created) == ["standard", "worker"]
