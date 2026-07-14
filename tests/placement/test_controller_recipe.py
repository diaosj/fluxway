from unittest.mock import MagicMock

from fluxway.control_plane.controller import ControllerActor
from fluxway.placement import PlacementRuntimeConfig

from .test_planner import scenario


def test_controller_exposes_resolved_recipe_allocation_report() -> None:
    graph, recipe, snapshot = scenario(1)
    runtime = PlacementRuntimeConfig(solve_timeout_s=10, pg_timeout_s=20)
    controller = ControllerActor.__ray_actor_class__(  # type: ignore[attr-defined]
        graph=graph,
        placement_recipe=recipe,
        placement_runtime=runtime,
    )
    plan = controller.scheduler._placement_recipe
    assert plan is recipe
    resolved = MagicMock()
    resolved.model_dump.return_value = {
        "recipe_id": recipe.recipe_id,
        "recipe_digest": recipe.digest,
    }
    controller.scheduler._resolved_plan = resolved

    summary = controller.get_resource_allocation_summary()

    assert summary == {
        "recipe_id": recipe.recipe_id,
        "recipe_digest": recipe.digest,
    }
    resolved.model_dump.assert_called_once_with(mode="json")
