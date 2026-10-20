import unittest
from dataclasses import dataclass
from fluxway.common.decorators import data_type, operator, _get_fluxway_streams
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.types.sharding import ShardingConfig

# --- Test Data Types ---


@data_type()
@dataclass
class DeepLeaf:
    val: TransportTensor


@data_type()
@dataclass
class MidLevel:
    leaf: DeepLeaf
    direct: TransportTensor


@data_type()
@dataclass
class RootLevel:
    mid: MidLevel
    top_val: TransportTensor


# --- Test Operators ---

spec = {"type": "static", "axis": ["dp"]}


class _RecursiveOperators:
    @operator(
        name="test_op",
        input_sharding={
            "inp": {
                "mid": {
                    "leaf": spec,
                    "direct": {"type": "dynamic", "axis": ["shard"]},
                },
                "top_val": spec,
            }
        },
    )
    def my_op(self, inp: RootLevel):
        return inp

    @operator(name="uniform_op", input_sharding={"inp": spec})
    def uniform_op(self, inp: RootLevel):
        return inp

    @operator(name="output_op", output_sharding=spec)
    def output_op(self) -> RootLevel:
        raise NotImplementedError


# --- Test Class ---


class TestRecursiveDataType(unittest.TestCase):
    def test_recursive_stream_extraction(self):
        """Verify that _get_fluxway_streams finds nested paths with correct dot notation."""
        data_f, control_f = _get_fluxway_streams(RootLevel)

        expected_data = ["mid.leaf.val", "mid.direct", "top_val"]
        self.assertCountEqual(data_f, expected_data)

    def test_recursive_sharding_resolution(self):
        """Test how @operator resolves nested input_sharding specs using parameter prefixes."""
        meta = _RecursiveOperators.my_op._fluxway_operator_spec
        sharding = meta.input_sharding

        # Keys should be: param_name.field_path
        self.assertEqual(
            sharding["mid.leaf.val"], ShardingConfig.model_validate(spec).to_spec()
        )
        self.assertTrue(sharding["mid.direct"].is_dynamic)
        self.assertEqual(
            sharding["top_val"], ShardingConfig.model_validate(spec).to_spec()
        )

    def test_uniform_sharding_resolution(self):
        """Test applying one spec to an entire nested @data_type parameter branch."""
        sharding = _RecursiveOperators.uniform_op._fluxway_operator_spec.input_sharding
        for field in ["mid.leaf.val", "mid.direct", "top_val"]:
            self.assertEqual(
                sharding[field], ShardingConfig.model_validate(spec).to_spec()
            )

    def test_output_sharding_resolution(self):
        """Test applying uniform spec to @data_type return (no parameter prefix for returns)."""
        sharding = _RecursiveOperators.output_op._fluxway_operator_spec.output_sharding
        for field in ["mid.leaf.val", "mid.direct", "top_val"]:
            self.assertEqual(
                sharding[field], ShardingConfig.model_validate(spec).to_spec()
            )


if __name__ == "__main__":
    unittest.main()
