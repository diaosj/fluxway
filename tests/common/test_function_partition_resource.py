import numpy as np
import pytest
import ray.cloudpickle

from fluxway.common.mesh import DeviceMesh
from fluxway.common.resource_registry import (
    ResourceHandle,
    ResourceRegistry,
    bind_resource_registry,
)
from fluxway.common.tensor_backend import use_backend
from fluxway.common.types.sharding import (
    FunctionPartition,
    LiteralIndices,
    P,
    RoutingDomain,
    ShardingConfig,
    _parse_partition_strategy,
)
from fluxway.transport.planner import TransportPlanner


class NativeRanks:
    def __init__(self, ranks: np.ndarray) -> None:
        self.ranks = ranks

    def __dlpack__(self, *args, **kwargs):
        return self.ranks.__dlpack__(*args, **kwargs)

    def __dlpack_device__(self):
        return self.ranks.__dlpack_device__()


class NativeRankProvider:
    def __init__(self, result: object) -> None:
        self.result = result

    def __call__(self, values, *, partition_count: int, domain_size: int):
        del values, partition_count, domain_size
        return self.result


def test_function_partition_with_resource_handle_dict_coercion() -> None:
    partition = FunctionPartition(
        target={"namespace": "routing", "resource_id": "slot_0"},
        arguments={"offset": 2},
    )
    assert isinstance(partition.target, ResourceHandle)
    assert partition.target.namespace == "routing"
    assert partition.target.resource_id == "slot_0"
    assert partition.arguments["offset"] == 2


def test_function_partition_rejects_unsupported_target_type() -> None:
    with pytest.raises(TypeError, match="must be callable or ResourceHandle"):
        FunctionPartition(target=123)


@pytest.mark.parametrize("dtype", ("int32", "uint32", "int64", "uint64"))
def test_function_partition_resource_handle_dlpack_zero_copy(dtype: str) -> None:
    ranks = np.asarray([2, 6, 0, 1, 0], dtype=dtype)
    handle = ResourceHandle(namespace="routing", resource_id="dlpack_provider")
    partition = FunctionPartition(target=handle)

    registry = ResourceRegistry()
    registry.register(
        "routing", "dlpack_provider", NativeRankProvider(NativeRanks(ranks))
    )

    keys = np.arange(len(ranks), dtype=np.int32)
    with bind_resource_registry(registry), use_backend("numba_cpu"):
        output = partition(keys, dim_size=len(keys), num_workers=7)
        assert output.dtype == ranks.dtype
        assert np.shares_memory(output, ranks)

        layout = (
            ShardingConfig(type="dynamic", axis=["shard"], strategy=partition)
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


def test_function_partition_resource_handle_cloudpickle_roundtrip() -> None:
    handle = ResourceHandle(namespace="routing", resource_id="dynamic_modulo")
    partition = FunctionPartition(target=handle, arguments={"offset": 3})

    config = ShardingConfig(
        type="dynamic",
        axis=["shard"],
        strategy=partition,
    )
    registry = ResourceRegistry()

    def route_fn(values, *, partition_count: int, domain_size: int, offset: int):
        del domain_size
        return (values + offset) % partition_count

    registry.register("routing", "dynamic_modulo", route_fn)

    with bind_resource_registry(registry):
        spec = config.to_spec()

    # Serialization across process boundary (e.g., Driver -> Worker)
    restored_spec = ray.cloudpickle.loads(ray.cloudpickle.dumps(spec))
    assert restored_spec.partition == P("shard")
    assert restored_spec.is_dynamic
    assert isinstance(restored_spec.strategy, FunctionPartition)
    assert restored_spec.strategy.target == handle
    assert restored_spec.strategy.arguments["offset"] == 3

    values = np.asarray([0, 1, 7, 8], dtype=np.int32)
    with bind_resource_registry(registry), use_backend("numba_cpu"):
        routed = restored_spec.strategy(values, dim_size=16, num_workers=8)
        np.testing.assert_array_equal(np.asarray(routed), np.asarray([3, 4, 2, 3]))


def test_function_partition_resource_handle_unbound_registry_fails_only_on_execution() -> (
    None
):
    handle = ResourceHandle(namespace="routing", resource_id="slot_0")
    partition = FunctionPartition(target=handle)

    with (
        use_backend("numba_cpu"),
        pytest.raises(RuntimeError, match="No active ResourceRegistry bound"),
    ):
        partition(np.asarray([0]), dim_size=1, num_workers=1)


def test_function_partition_resource_handle_contract_allows_control_plane_binding() -> (
    None
):
    handle = ResourceHandle(namespace="routing", resource_id="worker_local")
    partition = FunctionPartition(target=handle)
    ShardingConfig(type="dynamic", axis=["shard"], strategy=partition).to_spec()


def test_function_partition_resource_handle_rejects_non_callable_resource() -> None:
    handle = ResourceHandle(namespace="routing", resource_id="not_a_func")
    partition = FunctionPartition(target=handle)

    registry = ResourceRegistry()
    registry.register("routing", "not_a_func", "non_callable_object")

    with (
        bind_resource_registry(registry),
        use_backend("numba_cpu"),
        pytest.raises(TypeError, match="not callable"),
    ):
        partition(np.asarray([0]), dim_size=1, num_workers=1)


def test_function_partition_resource_handle_missing_resource_raises() -> None:
    handle = ResourceHandle(namespace="routing", resource_id="unregistered")
    partition = FunctionPartition(target=handle)

    registry = ResourceRegistry()
    with (
        bind_resource_registry(registry),
        use_backend("numba_cpu"),
        pytest.raises(RuntimeError, match="not found"),
    ):
        partition(np.asarray([0]), dim_size=1, num_workers=1)


def test_parse_partition_strategy_function_with_resource_handle() -> None:
    raw = {
        "type": "function",
        "target": {"namespace": "routing", "resource_id": "slot_42"},
        "arguments": {"shift": 5},
    }
    strategy = _parse_partition_strategy(raw)
    assert isinstance(strategy, FunctionPartition)
    assert isinstance(strategy.target, ResourceHandle)
    assert strategy.target.namespace == "routing"
    assert strategy.target.resource_id == "slot_42"
    assert strategy.arguments["shift"] == 5
