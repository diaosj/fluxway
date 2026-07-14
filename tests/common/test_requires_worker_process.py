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
from fluxway.common.types import OperatorMeta, SPMDConfig
from fluxway.spmd.worker.worker import SPMDWorker


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

    assert config.operator_metas["step"].requires_worker_process is True


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


def test_operator_preinstantiation_respects_process_location(monkeypatch):
    created = []

    @requires_worker_process
    class WorkerOperator:
        def __init__(self) -> None:
            created.append("worker")

    class ExecutionActorOperator:
        def __init__(self) -> None:
            created.append("execution_actor")

    classes = {
        "WorkerOperator": WorkerOperator,
        "ExecutionActorOperator": ExecutionActorOperator,
    }
    monkeypatch.setattr(
        "fluxway.spmd.worker.worker._resolve_owner_class",
        lambda _, qualname: classes[qualname],
    )
    config = SPMDConfig(
        mesh=DeviceMesh(axis_names=("data",), device_shape=(1,)),
        operator_metas={
            "worker": OperatorMeta(
                name="worker",
                owner_class_module="test",
                owner_class_qualname="WorkerOperator",
                requires_worker_process=True,
            ),
            "execution_actor": OperatorMeta(
                name="execution_actor",
                owner_class_module="test",
                owner_class_qualname="ExecutionActorOperator",
                requires_worker_process=False,
            ),
        },
    )
    worker_type = SPMDWorker.__ray_actor_class__  # pyrefly: ignore [missing-attribute]

    worker_type._pre_instantiate_operators(
        config,
        rank=0,
        operator_location="worker",
    )
    assert created == ["worker"]

    created.clear()
    worker_type._pre_instantiate_operators(
        config,
        rank=0,
        operator_location="execution_actor",
    )
    assert created == ["execution_actor"]
