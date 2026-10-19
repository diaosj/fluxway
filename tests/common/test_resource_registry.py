import threading
import pytest

from fluxway.common.resource_registry import (
    ResourceHandle,
    ResourceRegistry,
    bind_resource_registry,
    get_active_resource_registry,
    set_active_resource_registry,
)
from fluxway.common.logical_partition_binding import LogicalPartitionBinding
from fluxway.common.mesh import DeviceMesh


def test_registry_registration_and_resolution() -> None:
    registry = ResourceRegistry()
    resource = object()

    assert not registry.contains("routing", "slot_1")
    registry.register("routing", "slot_1", resource)

    assert registry.contains("routing", "slot_1")
    assert registry.resolve("routing", "slot_1") is resource


def test_registry_rejects_duplicate_registration() -> None:
    registry = ResourceRegistry()
    registry.register("routing", "slot_1", "initial")

    with pytest.raises(RuntimeError, match="already registered"):
        registry.register("routing", "slot_1", "duplicate")


def test_registry_resolve_missing_raises_runtime_error() -> None:
    registry = ResourceRegistry()

    with pytest.raises(RuntimeError, match="not found"):
        registry.resolve("routing", "non_existent")


def test_registry_validates_key_types() -> None:
    registry = ResourceRegistry()

    with pytest.raises(TypeError, match="non-empty string"):
        registry.register("", "slot_1", 123)

    with pytest.raises(TypeError, match="non-empty string"):
        registry.register("routing", "", 123)

    with pytest.raises(TypeError, match="non-empty string"):
        registry.resolve("", "slot_1")

    with pytest.raises(TypeError, match="non-empty string"):
        registry.resolve("routing", "")

    with pytest.raises(TypeError, match="non-empty string"):
        registry.unregister("", "slot_1")


def test_registry_unregister_and_clear() -> None:
    registry = ResourceRegistry()
    item = "payload"
    registry.register("routing", "slot_1", item)

    removed = registry.unregister("routing", "slot_1")
    assert removed == item
    assert not registry.contains("routing", "slot_1")

    with pytest.raises(RuntimeError, match="not found"):
        registry.unregister("routing", "slot_1")

    registry.register("ns_a", "id_1", 1)
    registry.register("ns_b", "id_2", 2)
    registry.clear()
    assert not registry.contains("ns_a", "id_1")
    assert not registry.contains("ns_b", "id_2")


def test_registry_thread_safety() -> None:
    registry = ResourceRegistry()
    num_threads = 16
    items_per_thread = 50

    def worker(thread_idx: int) -> None:
        for i in range(items_per_thread):
            key = f"t{thread_idx}_{i}"
            registry.register("threads", key, thread_idx * 1000 + i)
            val = registry.resolve("threads", key)
            assert val == thread_idx * 1000 + i

    threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for thread_idx in range(num_threads):
        for i in range(items_per_thread):
            key = f"t{thread_idx}_{i}"
            assert registry.contains("threads", key)
            assert registry.resolve("threads", key) == thread_idx * 1000 + i


def test_bind_resource_registry_context_manager() -> None:
    with pytest.raises(RuntimeError, match="No active ResourceRegistry bound"):
        get_active_resource_registry()

    scoped_registry = ResourceRegistry()
    scoped_registry.register("scoped", "item", 42)

    with bind_resource_registry(scoped_registry) as active:
        assert active is scoped_registry
        assert get_active_resource_registry() is scoped_registry
        assert get_active_resource_registry().resolve("scoped", "item") == 42

    with pytest.raises(RuntimeError, match="No active ResourceRegistry bound"):
        get_active_resource_registry()


def test_set_active_resource_registry_and_type_check() -> None:
    with pytest.raises(TypeError, match="ResourceRegistry"):
        set_active_resource_registry(object())  # type: ignore[arg-type]

    registry = ResourceRegistry()
    set_active_resource_registry(registry)
    assert get_active_resource_registry() is registry
    from fluxway.common.resource_registry import reset_active_resource_registry

    reset_active_resource_registry()


def test_resource_handle_schema() -> None:
    with pytest.raises(ValueError, match="resource identifier"):
        ResourceHandle(namespace="", resource_id="slot_0")

    with pytest.raises(ValueError, match="resource identifier"):
        ResourceHandle(namespace="routing", resource_id="   ")

    handle = ResourceHandle(namespace="routing", resource_id="slot_0")
    assert handle.namespace == "routing"
    assert handle.resource_id == "slot_0"

    with pytest.raises(Exception):
        handle.namespace = "mutated"  # type: ignore[misc]


def test_logical_partition_binding_projects_axes_and_is_immutable() -> None:
    binding = LogicalPartitionBinding(
        {index: f"value-{index}" for index in range(6)}, axes=("dp", "fsdp")
    )
    mesh = DeviceMesh(
        axis_names=("dp", "fsdp", "replica"),
        device_shape=(2, 3, 4),
    ).freeze()
    assert binding.resolve(0, logical_mesh=mesh) == "value-0"
    assert binding.resolve(3, logical_mesh=mesh) == "value-0"
    assert binding.resolve(5, logical_mesh=mesh) == "value-1"
    with pytest.raises(TypeError):
        binding.values[0] = "changed"  # type: ignore[index]
