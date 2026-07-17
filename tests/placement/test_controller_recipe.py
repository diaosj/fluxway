from typing import Any
from unittest.mock import MagicMock

from fluxway import ResourceProfile, operator
from fluxway.common.decorators import spmd
from fluxway.control_plane.controller import ControllerActor
from fluxway.control_plane.graph.functional import FunctionalGraph
from fluxway.placement import PlacementRuntimeConfig

from .test_planner import scenario


@spmd(
    resource_profile=ResourceProfile(base={"CPU": 1}),
    mesh_axis_names=("worker",),
    mesh_axis_shapes=(1,),
)
class _ControllerWorker:
    @operator(
        input_sharding={"value": {"type": "static", "axis": []}},
        output_sharding={"value": {"type": "static", "axis": []}},
    )
    def step(self, value: Any = None) -> dict[str, Any]:
        return {"value": value}


def test_controller_exposes_resolved_recipe_allocation_report() -> None:
    _, recipe, _ = scenario(1)
    runtime = PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=20)
    with FunctionalGraph("controller-recipe") as static_graph:
        left = _ControllerWorker(name="left")
        right = _ControllerWorker(name="right")
        right.step(left.step())
        task_graph = static_graph.build()
    controller = ControllerActor.__ray_actor_class__(  # type: ignore[attr-defined]
        graph=task_graph,
        placement_recipe=recipe,
        placement_runtime=runtime,
    )
    assert controller.scheduler._placement_recipe == recipe
    resolved = MagicMock()
    resolved.model_dump.return_value = {
        "recipe_id": recipe.recipe_id,
    }
    controller.scheduler._resolved_plan = resolved

    summary = controller.get_resource_allocation_summary()

    assert summary == {"recipe_id": recipe.recipe_id}
    resolved.model_dump.assert_called_once_with(exclude={"recipe"})
