"""
Tests for OperatorSpec serialization behavior.

These tests verify that OperatorSpec modifications survive cross-process
serialization (pickle), which is critical for Ray-based worker communication.

Background:
    Python pickles module-level functions by reference (module + qualified_name).
    When unpickled in a different process, Python imports the module fresh,
    getting the original function with original attributes.

    This means modifications to function attributes (like _flux_operator_meta)
    made after class definition are lost when the function is sent to workers.

    The solution is to store OperatorSpec explicitly in SPMDConfig fields
    rather than relying on function attributes.
"""

import importlib.util
import os
import pickle
import subprocess
import sys
import tempfile

import pytest

sys.path.append(os.getcwd())

from fluxway.common.compiled_graph import (
    compile_graph,
)
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.operator_policy import ArgumentType
from fluxway import OperatorSpec
from fluxway.common.types.sharding import PartitionSpec, ShardingSpec
from fluxway.common.decorators import (
    operator,
    spmd,
    _instance_storage,
)
from fluxway.spmd.worker.worker import SPMDWorker
from tests.compiled_graph_fixtures import compiled_spmd_spec


def dummy_my_function():
    pass


class TestPickleBehavior:
    """Tests demonstrating Python pickle behavior with function attributes."""

    def test_same_process_pickle_preserves_function_attributes(self):
        """
        Within the same process, pickle preserves function attributes.

        This works because pickle returns the same object reference.
        """
        my_function = dummy_my_function

        my_function.meta = {"key": "original"}  # type: ignore[attr-defined]
        my_function.meta["key"] = "modified"  # type: ignore[attr-defined]
        my_function.meta["new_key"] = "new_value"  # type: ignore[attr-defined]  # type: ignore

        pickled = pickle.dumps(my_function)
        restored = pickle.loads(pickled)

        # Same process: restored IS my_function (same object reference)
        assert restored is my_function
        assert restored.meta == {"key": "modified", "new_key": "new_value"}  # type: ignore[attr-defined]  # type: ignore

    def test_cross_process_pickle_loses_function_attribute_modifications(self):
        """
        Cross-process pickle loses function attribute modifications.

        This is the root cause of the sharding metadata issue:
        - Control plane modifies OperatorSpec on functions
        - Ray sends SPMDConfig (with functions) to workers via pickle
        - Workers import module fresh, getting original OperatorSpec
        """
        # Create a temporary module file
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, dir=tempfile.gettempdir()
        ) as f:
            f.write("""
def test_function():
    pass
test_function.meta = {"key": "default_value"}
""")
            module_path = f.name
            pickle_path = module_path + ".pkl"

        try:
            spec = importlib.util.spec_from_file_location("test_module", module_path)
            assert spec is not None and spec.loader is not None
            test_module = importlib.util.module_from_spec(spec)
            sys.modules["test_module"] = test_module
            spec.loader.exec_module(test_module)

            fn = test_module.test_function
            assert fn.meta == {"key": "default_value"}

            # Modify the metadata (simulating Builder modification)
            fn.meta["key"] = "modified_in_main_process"
            fn.meta["new_key"] = "added_in_main"
            assert fn.meta == {
                "key": "modified_in_main_process",
                "new_key": "added_in_main",
            }

            # Pickle to bytes
            pickled = pickle.dumps(fn)

            # Save pickle for child process
            with open(pickle_path, "wb") as pf:
                pf.write(pickled)

            # Child process script
            child_script = f"""
import pickle
import sys
import importlib.util

spec = importlib.util.spec_from_file_location("test_module", "{module_path}")
test_module = importlib.util.module_from_spec(spec)
sys.modules['test_module'] = test_module
spec.loader.exec_module(test_module)

with open("{pickle_path}", 'rb') as f:
    restored_fn = pickle.loads(f.read())

# Print the meta value for assertion
print(repr(restored_fn.meta))
"""

            # Run child process
            result = subprocess.run(
                [sys.executable, "-c", child_script],
                capture_output=True,
                text=True,
                timeout=10,
            )

            # Child process should see ORIGINAL meta, not modified
            assert result.returncode == 0, f"Child process failed: {result.stderr}"
            child_output = result.stdout.strip()

            # The key assertion: modifications are LOST in cross-process pickle
            assert child_output == "{'key': 'default_value'}", (
                f"Expected original meta, but got: {child_output}"
            )

        finally:
            # Reclaim
            if "test_module" in sys.modules:
                del sys.modules["test_module"]
            os.unlink(module_path)
            if os.path.exists(pickle_path):
                os.unlink(pickle_path)


def dummy_op(x):
    return x


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(2,))
class _ResolvableModel:
    @operator(name="forward", static_argument=["x"])
    def forward(self, x):
        return x


@spmd(mesh_axis_names=("dp",), mesh_axis_shapes=(1,))
class _InstantiableModel:
    @operator(name="run", static_argument=["x"])
    def run(self, x):
        return x


def _compiled_class_config(spmd_class: type, spmd_id: str) -> SPMDConfig:
    declaration = spmd_class.get_spmd_config()
    graph = compile_graph(
        name=f"{spmd_id}-owner-transport",
        spmds=(
            compiled_spmd_spec(
                spmd_id=spmd_id,
                spmd_class=spmd_class,
                mesh=declaration.mesh,
                operators=declaration.operator_specs,
            ),
        ),
        dependencies=(),
    )
    return declaration.model_copy(
        update={"spmd_id": spmd_id, "graph_view": graph.spmd(spmd_id)}
    )


class TestOperatorSpecSerialization:
    """Tests for OperatorSpec serialization in SPMDConfig using actual types."""

    def test_real_pydantic_model_preserves_modifications_in_pickle(self):
        """
        OperatorSpec (Pydantic model) preserves modifications when pickled.
        """
        meta = OperatorSpec(name="test_op", input_arguments=None, output_arguments=None)
        updated = meta.updated(requires_worker_process=True)

        # Pickle and unpickle
        pickled = pickle.dumps(updated)
        restored = pickle.loads(pickled)

        assert restored == updated
        assert restored.requires_worker_process is True
        assert meta.requires_worker_process is False

    def test_real_spmd_config_preserves_modifications(self):
        """
        SPMDConfig with explicit meta fields preserves OperatorSpec modifications.
        """
        meta = OperatorSpec(
            name="test_op",
            input_arguments={ArgumentType.DATA.value: ["input"]},
            output_arguments={ArgumentType.DATA.value: ["output"]},
            input_sharding={"input": ShardingSpec.chunk(PartitionSpec())},
            output_sharding={"output": ShardingSpec.chunk(PartitionSpec())},
        )
        mesh = DeviceMesh(("data",), (1,))
        plan = compile_graph(
            name="metadata-serialization",
            spmds=(
                compiled_spmd_spec(
                    spmd_id="model",
                    mesh=mesh,
                    operators={"test_op": meta},
                ),
            ),
            dependencies=(),
        )
        config = SPMDConfig(
            spmd_id="model",
            mesh=mesh,
            operator_specs={"test_op": meta},
            graph_view=plan.spmd("model"),
        )

        # Pickle and unpickle (simulating Ray serialization)
        pickled = pickle.dumps(config)
        restored = pickle.loads(pickled)

        assert "test_op" in restored.operator_specs
        restored_operator = restored.require_graph_view().operator("test_op")
        assert restored_operator.spec is restored.operator_specs["test_op"]
        assert restored_operator.input_sharding == meta.input_sharding
        assert restored_operator.output_sharding == meta.output_sharding


class TestOwnerClassTransport:
    """Tests verifying that owner classes survive worker transport."""

    def test_free_function_operator_remains_unbound(self):
        wrapped = operator(name="free_step", static_argument=["x"])(dummy_op)
        assert wrapped._fluxway_operator_spec.owner is None

    def test_class_method_operator_meta_records_owner_info(self):
        config = _ResolvableModel.get_spmd_config()  # type: ignore
        meta = config.operator_specs["forward"]
        assert meta.owner is _ResolvableModel
        assert meta.owner_class_qualname == "_ResolvableModel"
        assert meta.owner_class_module == __name__

    def test_owner_class_survives_pickle(self):
        config = _ResolvableModel.get_spmd_config()  # type: ignore
        restored_config = pickle.loads(pickle.dumps(config))

        meta = restored_config.operator_specs["forward"]
        assert meta.owner is _ResolvableModel

    def test_pre_instantiate_operators_uses_owner_meta(self):
        config = _compiled_class_config(_InstantiableModel, "instantiable")

        _instance_storage.clear()
        SPMDWorker._pre_instantiate_operators(  # type: ignore
            config, rank=0, static_args={}
        )

        cls = config.operator_specs["run"].owner
        assert cls is not None
        assert id(cls) in _instance_storage

    def test_pre_instantiate_operators_logs_init_boundaries(self, caplog):
        config = _compiled_class_config(_InstantiableModel, "instantiable")

        _instance_storage.clear()
        caplog.set_level("DEBUG", logger="fluxway.spmd.worker.worker")

        SPMDWorker._pre_instantiate_operators(  # type: ignore
            config, rank=0, static_args={}
        )

        begin_records = [
            record
            for record in caplog.records
            if "operator_init_begin" in record.getMessage()
        ]
        complete_records = [
            record
            for record in caplog.records
            if "operator_init_complete" in record.getMessage()
        ]
        assert begin_records
        assert complete_records
        assert all(record.levelname == "DEBUG" for record in begin_records)
        assert all(record.levelname == "DEBUG" for record in complete_records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
