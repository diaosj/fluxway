import pickle

import pytest

from fluxway.common.execution_layout import compile_execution_layout
from fluxway.common.mesh import DeviceMesh
from fluxway.common.types.sharding import P, ShardingSpec
from fluxway.spmd.worker.operator_manager.dispatcher import ExecutionDispatcher


def test_permuted_groups_share_context_routing_and_activity():
    groups = ((7, 0), (5, 2), (1, 6), (3, 4))
    mesh = DeviceMesh(("worker",), (8,))
    layout = compile_execution_layout(mesh, unit_size=2, groups=groups)
    dispatcher = ExecutionDispatcher(layout)
    source = dispatcher.get_effective_source_layout(
        mesh, ShardingSpec.chunk(P("worker"))
    )
    target = ShardingSpec.chunk(P("input")).bind(DeviceMesh(("input",), (2,)))
    for step in range(12):
        active = groups[step % len(groups)]
        for physical in range(8):
            assert dispatcher.should_execute(physical, 8, step) == (physical in active)
        for logical, physical in enumerate(active):
            context = dispatcher.get_logical_context(physical, 8, step)
            assert context.logical_rank == logical
            assert context.logical_world_size == 2
            assert dispatcher.resolve_active_producers(
                source, target, logical, step, 8
            ) == [physical]
    restored = pickle.loads(pickle.dumps(layout))
    assert restored == layout
    with pytest.raises(TypeError):
        layout.groups[0]._logical_by_physical[7] = 1


@pytest.mark.parametrize(
    "groups", [((0, 1), (1, 3)), ((0, 1),), ((0, 2), (3, 4)), ((0, 1, 2), (3,))]
)
def test_invalid_group_coverage_is_rejected(groups):
    with pytest.raises(ValueError):
        compile_execution_layout(
            DeviceMesh(("worker",), (4,)), unit_size=2, groups=groups
        )


def test_trailing_mesh_coordinates_are_logical():
    layout = compile_execution_layout(
        DeviceMesh(("data", "model"), (4, 2)),
        unit_size=4,
        groups=((6, 7, 0, 1), (2, 3, 4, 5)),
    )
    assert layout.logical_mesh.shape == (2, 2)
    assert layout.group_for_rank(6).logical_rank(6) == 0
    assert layout.group_for_step(0).physical_rank(2) == 0


def test_standard_execution_uses_the_same_layout_contract():
    layout = compile_execution_layout(DeviceMesh(("worker",), (4,)))
    dispatcher = ExecutionDispatcher(layout)
    assert layout.cycle == ((0, 1, 2, 3),)
    for physical in range(4):
        assert dispatcher.should_execute(physical, 4, 19)
        assert dispatcher.get_logical_context(physical, 4, 19).logical_rank == physical
    with pytest.raises(ValueError, match="world size"):
        dispatcher.should_execute(0, 8, 0)
