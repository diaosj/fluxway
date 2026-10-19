import numpy as np
import pytest
import ray.cloudpickle

from fluxway.common.mesh import DeviceMesh
from fluxway.common.tensor_backend import use_backend
from fluxway.common.types.sharding import (
    FunctionPartition,
    LiteralIndices,
    P,
    RoutingDomain,
    ShardingConfig,
)
from fluxway.transport.planner import TransportPlanner


class NativeRanks:
    def __init__(self, ranks):
        self.ranks = ranks

    def __dlpack__(self, *args, **kwargs):
        return self.ranks.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.ranks.__dlpack_device__()


class NativeRankProvider:
    def __init__(self, result):
        self.result = result

    def __call__(self, values, *, partition_count, domain_size):
        return self.result


@pytest.mark.parametrize("dtype", ("int32", "uint32", "int64", "uint64"))
def test_native_output_shares_buffer_and_reaches_planner(dtype) -> None:
    ranks = np.asarray([2, 6, 0, 1, 0], dtype=dtype)
    provider = FunctionPartition(target=NativeRankProvider(NativeRanks(ranks)))
    keys = np.arange(len(ranks), dtype=np.int32)
    with use_backend("numba_cpu"):
        output = provider(keys, dim_size=len(keys), num_workers=7)
        assert output.dtype == ranks.dtype
        assert np.shares_memory(output, ranks)
        layout = (
            ShardingConfig(type="dynamic", axis=("shard",), strategy=provider)
            .to_spec()
            .bind(DeviceMesh(axis_names=("shard",), device_shape=(7,)))
        )
        domain = RoutingDomain.from_layout(
            target=layout,
            axis_sizes=(len(keys),),
            axis_indices=(LiteralIndices(length=len(keys), values=keys),),
        )
        plan = TransportPlanner.compute_plan(src_local_rank=0, domain=domain)
        np.testing.assert_array_equal(np.asarray(plan.dest_devices).ravel(), ranks)
        np.testing.assert_array_equal(plan.send_counts, [2, 1, 1, 0, 0, 0, 1])


class ForeignDeviceRanks:
    def __dlpack_device__(self):
        return (2, 0)

    def __dlpack__(self, *args, **kwargs):
        raise AssertionError("device change must fail before DLPack consumption")


def test_native_output_rejects_device_change() -> None:
    provider = FunctionPartition(target=NativeRankProvider(ForeignDeviceRanks()))
    with use_backend("numba_cpu"), pytest.raises(ValueError, match="device domains"):
        provider(np.asarray([0]), dim_size=1, num_workers=1)


def test_native_output_rejects_unsupported_tensor() -> None:
    provider = FunctionPartition(target=NativeRankProvider(object()))
    with (
        use_backend("numba_cpu"),
        pytest.raises(TypeError, match="zero-copy conversion"),
    ):
        provider(np.asarray([0]), dim_size=1, num_workers=1)


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


def test_indexed_modulo_uses_common_backend_and_rejects_invalid_worker_count() -> None:
    from fluxway.common.types.sharding import IndexedModuloPartition

    strategy = IndexedModuloPartition()
    values = np.asarray([-5, -1, 0, 5, 8], dtype=np.int64)
    np.testing.assert_array_equal(
        np.asarray(strategy(values, dim_size=16, num_workers=4)),
        np.asarray([3, 3, 0, 1, 0], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="worker count must be positive"):
        strategy(values, dim_size=16, num_workers=0)
    with pytest.raises(TypeError, match="worker count must be an integer"):
        strategy(values, dim_size=16, num_workers=True)
