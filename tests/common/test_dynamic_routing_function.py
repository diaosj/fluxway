import numpy as np
import pytest
import ray.cloudpickle

from fluxway.common.types.sharding import FunctionPartition, P, ShardingConfig


def offset_router(offset: int):
    def route(values, *, partition_count, domain_size):
        del domain_size
        return (values + offset) % partition_count

    return route


def test_custom_dynamic_routing_accepts_captured_closure() -> None:
    route = offset_router(3)
    config = ShardingConfig(
        type="dynamic",
        axis=("shard",),
        strategy=FunctionPartition(target=route),
    )

    spec = config.to_spec()
    restored = ray.cloudpickle.loads(ray.cloudpickle.dumps(spec))
    values = np.asarray([0, 1, 7, 8], dtype=np.int32)

    assert restored.partition == P("shard")
    assert restored.is_dynamic
    assert isinstance(restored.strategy, FunctionPartition)
    np.testing.assert_array_equal(
        np.asarray(restored.strategy(values, dim_size=16, num_workers=8)),
        np.asarray([3, 4, 2, 3]),
    )


def test_custom_dynamic_routing_accepts_lambda() -> None:
    strategy = FunctionPartition(
        target=lambda values, *, partition_count, domain_size: (
            (values + domain_size) % partition_count
        )
    )

    strategy.validate_contract()


def test_custom_dynamic_routing_rejects_wrong_contract() -> None:
    strategy = FunctionPartition(target=lambda values: values)

    with pytest.raises(TypeError, match="routing contract"):
        strategy.validate_contract()


def test_custom_dynamic_routing_arguments_must_be_hashable() -> None:
    with pytest.raises(TypeError, match="must be hashable"):
        FunctionPartition(
            target=offset_router(1),
            arguments={"invalid": []},
        )
