import pytest

from fluxway.common.decorators import (
    _STATIC_CONTEXT_ATTR_NAMES,
    get_current_context,
    operator,
    operator_context,
)
from fluxway.common.operator_instances import OperatorInstanceContainer
from fluxway.common.mesh import DeviceMesh
from fluxway.spmd.worker.operator_manager import OperatorManager
from fluxway.spmd.worker.worker import SPMDWorker
from fluxway.resources import ResourceProfile
from tests.compiled_graph_fixtures import single_spmd_config


class TopologyAwareOperator:
    seen_in_init = None
    rank: int
    world_size: int

    def __init__(self):
        type(self).seen_in_init = (
            self.rank,
            self.world_size,
            getattr(self, "local_rank", None),
            getattr(self, "local_world_size", None),
            getattr(self, "node_rank", None),
            getattr(self, "num_nodes", None),
        )

    @operator(name="inspect")
    def inspect(self):
        return (
            self.rank,
            self.world_size,
            getattr(self, "local_rank", None),
            getattr(self, "local_world_size", None),
            getattr(self, "node_rank", None),
            getattr(self, "num_nodes", None),
        )


class RayComponentLimitsAwareOperator:
    seen_in_init = None
    ray_component_limits: object

    def __init__(self):
        type(self).seen_in_init = self.ray_component_limits

    @operator(name="inspect_ray_component_limits")
    def inspect_ray_component_limits(self):
        return self.ray_component_limits


def _clear_operator_instance():
    TopologyAwareOperator.seen_in_init = None


def _topology_config():
    operator_spec = TopologyAwareOperator.inspect._fluxway_operator_spec.updated(
        owner=TopologyAwareOperator
    )
    return single_spmd_config(
        name="operator-topology-context",
        spmd_id="spmd-a",
        mesh=DeviceMesh(axis_names=("dp",), device_shape=(8,)),
        operators={"inspect": operator_spec},
    )


def test_operator_context_exposes_topology_fields():
    with operator_context(
        static_args={},
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    ):
        ctx = get_current_context()
        assert ctx.local_rank == 1
        assert ctx.local_world_size == 4
        assert ctx.node_rank == 0
        assert ctx.num_nodes == 2


def test_operator_context_exposes_ray_component_limits_in_eager_execution():
    ray_component_limits = ResourceProfile(
        base={"CPU": 1}, ray_component_limits={"CPU": "0.5"}
    ).ray_component_limits

    with operator_context(
        static_args={},
        rank=0,
        world_size=1,
        ray_component_limits=ray_component_limits,
    ):
        assert get_current_context().ray_component_limits is ray_component_limits


def test_preinstantiate_operator_exposes_compiled_ray_component_limits():
    RayComponentLimitsAwareOperator.seen_in_init = None
    operator_spec = RayComponentLimitsAwareOperator.inspect_ray_component_limits._fluxway_operator_spec.updated(
        owner=RayComponentLimitsAwareOperator
    )
    profile = ResourceProfile(
        base={"CPU": 1}, ray_component_limits={"CPU": "0.5", "memory": 1024}
    )
    config = single_spmd_config(
        name="ray-component-limits-context",
        spmd_id="metrics",
        mesh=DeviceMesh(axis_names=("dp",), device_shape=(1,)),
        operators={"inspect_ray_component_limits": operator_spec},
        resource_profile=profile,
    )
    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    worker_cls = cls.modified_class if cls else SPMDWorker

    container = OperatorInstanceContainer()
    worker_cls._pre_instantiate_operators(
        config, 0, static_args={}, instance_container=container
    )

    assert RayComponentLimitsAwareOperator.seen_in_init is profile.ray_component_limits


def test_operator_wrapper_sets_topology_fields_on_instance():
    _clear_operator_instance()

    with operator_context(
        static_args={},
        rank=3,
        world_size=8,
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    ):
        result = TopologyAwareOperator.inspect()

    assert result == (3, 8, 1, 4, 0, 2)


def test_preinstantiate_operator_sets_topology_fields_in_init():
    _clear_operator_instance()
    config = _topology_config()
    container = OperatorInstanceContainer()

    cls = getattr(SPMDWorker, "__ray_metadata__", None)
    worker_cls = cls.modified_class if cls else SPMDWorker
    worker_cls._pre_instantiate_operators(  # type: ignore
        config,
        3,
        static_args={},
        instance_container=container,
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    )

    instance = container.require(TopologyAwareOperator)

    assert TopologyAwareOperator.seen_in_init == (3, 8, 1, 4, 0, 2)

    # Static attrs are plain instance attributes: they retain creation-time values
    # regardless of the active context, because rank/world_size/local_rank etc.
    # are fixed for the entire lifetime of a worker actor.
    assert instance.rank == 3
    assert instance.world_size == 8
    assert instance.local_rank == 1
    assert instance.local_world_size == 4
    assert instance.node_rank == 0
    assert instance.num_nodes == 2

    # Inside a different context: static attrs still return creation-time values;
    # only `ctx` (carrying per-step state) changes.
    with operator_context(
        static_args={},
        rank=99,
        world_size=1,
        local_rank=0,
        local_world_size=1,
        node_rank=0,
        num_nodes=1,
    ):
        assert instance.rank == 3  # plain attr, ignores context
        assert instance.local_rank == 1  # plain attr, ignores context


def test_operator_manager_execute_preserves_topology_fields():
    _clear_operator_instance()
    config = _topology_config()
    manager = OperatorManager(
        config,
        rank=3,
        spmd_id="spmd-a",
        static_args={},
        instance_container=OperatorInstanceContainer(),
        local_rank=1,
        local_world_size=4,
        node_rank=0,
        num_nodes=2,
    )

    result = manager.execute("inspect", step=0)

    assert result == (3, 8, 1, 4, 0, 2)


def test_operator_wrapper_resolves_local_classes():
    """
    Test that the _OperatorWrapper descriptor correctly resolves the owner class
    even if the class is defined inside a local scope (e.g. factory function),
    where importlib string-based reflection would normally fail.
    """

    def make_local_spmd():
        class LocalTopologyAwareOperator:
            @operator(name="inspect_local")
            def inspect_local(self):
                return getattr(self, "rank", -1)

        return LocalTopologyAwareOperator

    LocalClass = make_local_spmd()

    with operator_context(
        static_args={},
        rank=5,
        world_size=8,
    ):
        # Trigger __get__(None, LocalClass) on the descriptor.
        # This will resolve self._cls = LocalClass and successfully inject properties
        # without crashing due to '<locals>' in __qualname__.
        result = LocalClass.inspect_local()

    assert result == 5


def test_operator_wrapper_surfaces_descriptor_conflict_as_error():
    class ConflictingDescriptorOperator:
        @property
        def rank(self) -> int:
            return 999

        @operator(name="conflicting_op")
        def conflicting_op(self):
            return self.rank

    with pytest.raises(AttributeError):
        with operator_context(
            static_args={},
            rank=3,
            world_size=4,
        ):
            ConflictingDescriptorOperator.conflicting_op()


def test_operator_wrapper_supports_slotted_owner():
    seen = {}

    class SlottedOperator:
        __slots__ = tuple(_STATIC_CONTEXT_ATTR_NAMES) + ("extra",)

        def __init__(self):
            seen["rank"] = self.rank
            seen["world_size"] = self.world_size

        @operator(name="slotted_op")
        def slotted_op(self):
            return self.rank

    with operator_context(
        static_args={},
        rank=7,
        world_size=8,
    ):
        result = SlottedOperator.slotted_op()

    assert result == 7
    assert seen == {"rank": 7, "world_size": 8}
