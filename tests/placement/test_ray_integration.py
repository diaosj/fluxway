import asyncio
from contextlib import ExitStack

import pytest
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import Node, SPMDConfig
from fluxway.placement import (
    AFFINITY_FIRST,
    PlacementGroupTimeoutError,
    PlacementPlanner,
    PlacementRecipe,
    PlacementRuntimeConfig,
    ResourceProfile,
)
from fluxway.control_plane.scheduler import Scheduler
from fluxway.control_plane.scheduler.pg_manager import PGManager
from .test_planner import placement_graph, placement_inventory
from tests.ray_cluster_process import local_ray_cluster


@pytest.fixture
def ray_cluster():
    with ExitStack() as stack:
        yield lambda **options: stack.enter_context(local_ray_cluster(**options))


class RayIntegrationSPMD: ...


class MultiAcceleratorIntegrationSPMD: ...


@ray.remote
def report_node_id() -> str:
    return ray.get_runtime_context().get_node_id()


@ray.remote(num_cpus=1)
class CPUBlocker:
    def ready(self) -> bool:
        return True


def test_plain_ray_cluster_materializes_recipe(ray_cluster) -> None:
    ray_cluster(num_cpus=2)
    scheduler = None
    try:
        graph = placement_graph(
            nodes={
                "worker": Node(
                    name="worker",
                    spmd_config=SPMDConfig(
                        spmd_id="worker",
                        mesh=DeviceMesh(("data",), device_shape=(1,)),
                    ),
                    spmd_class=RayIntegrationSPMD,
                    resource_profile=ResourceProfile(base={"CPU": 1}),
                )
            }
        )
        recipe = PlacementRecipe(
            recipe_id="plain-ray",
            constraints=(),
            objective=AFFINITY_FIRST,
        )
        scheduler = Scheduler(
            graph=graph,
            placement_recipe=recipe,
            placement_runtime=PlacementRuntimeConfig(
                solve_timeout_s=10, pg_timeout_s=10
            ),
        )

        assert asyncio.run(scheduler.allocate_resources()) is None
        assert scheduler.allocation_report is not None
        assert scheduler.allocation_report.status == "PG_READY"
        worker_group = scheduler.pg_manager.get_worker_group("worker")
        assert worker_group.rank_to_bundle == (0,)
        node_id = ray.get(
            report_node_id.options(
                num_cpus=1,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=worker_group.placement_group,
                    placement_group_bundle_index=worker_group.rank_to_bundle[0],
                    placement_group_capture_child_tasks=False,
                ),
            ).remote()
        )
        worker_group.attest((node_id,))
    finally:
        if scheduler is not None and scheduler.allocation_report is not None:
            scheduler.pg_manager.release_job_pg()


def test_plain_ray_cluster_materializes_multiple_accelerator_resources(
    ray_cluster,
) -> None:
    ray_cluster(
        num_cpus=1,
        resources={
            "accelerator_type:H100": 1,
            "accelerator_type:TPU": 1,
        },
    )
    scheduler = None
    try:
        resources = {
            "accelerator_type:H100": 1,
            "accelerator_type:TPU": 1,
        }
        graph = placement_graph(
            nodes={
                "accelerated": Node(
                    name="accelerated",
                    spmd_config=SPMDConfig(
                        spmd_id="accelerated",
                        mesh=DeviceMesh(("data",), device_shape=(1,)),
                    ),
                    spmd_class=MultiAcceleratorIntegrationSPMD,
                    resource_profile=ResourceProfile(base=resources),
                )
            }
        )
        recipe = PlacementRecipe(
            recipe_id="plain-ray-multi-accelerator",
            constraints=(),
            objective=AFFINITY_FIRST,
        )
        scheduler = Scheduler(
            graph=graph,
            placement_recipe=recipe,
            placement_runtime=PlacementRuntimeConfig(
                solve_timeout_s=10, pg_timeout_s=10
            ),
        )

        assert asyncio.run(scheduler.allocate_resources()) is None
        worker_group = scheduler.pg_manager.get_worker_group("accelerated")
        node_id = ray.get(
            report_node_id.options(
                num_cpus=0,
                resources=resources,
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=worker_group.placement_group,
                    placement_group_bundle_index=0,
                    placement_group_capture_child_tasks=False,
                ),
            ).remote()
        )
        worker_group.attest((node_id,))
    finally:
        if scheduler is not None and scheduler.allocation_report is not None:
            scheduler.pg_manager.release_job_pg()


def test_plain_ray_cluster_surfaces_real_pg_timeout(ray_cluster) -> None:
    ray_cluster(num_cpus=1)
    blocker = None
    manager = PGManager()
    try:
        graph = placement_graph(
            nodes={
                "worker": Node(
                    name="worker",
                    spmd_config=SPMDConfig(
                        spmd_id="worker",
                        mesh=DeviceMesh(("data",), device_shape=(1,)),
                    ),
                    spmd_class=RayIntegrationSPMD,
                    resource_profile=ResourceProfile(base={"CPU": 1}),
                )
            }
        )
        recipe = PlacementRecipe(
            recipe_id="plain-ray-timeout",
            constraints=(),
            objective=AFFINITY_FIRST,
        )
        scheduler = Scheduler(
            graph=graph,
            placement_recipe=recipe,
            placement_runtime=PlacementRuntimeConfig(
                solve_timeout_s=10, pg_timeout_s=10
            ),
        )
        snapshot = scheduler._capture_snapshot()
        plan = PlacementPlanner(solve_timeout_s=10).resolve(
            placement_inventory(graph), recipe, snapshot
        )
        blocker = CPUBlocker.remote()  # type: ignore[attr-defined] # pylint: disable=no-member
        assert ray.get(blocker.ready.remote()) is True  # type: ignore[attr-defined]

        with pytest.raises(PlacementGroupTimeoutError):
            asyncio.run(
                manager.create_job_pg(
                    plan=plan,
                    snapshot=snapshot,
                    timeout_s=0.1,
                )
            )
    finally:
        if blocker is not None:
            ray.kill(blocker)  # type: ignore[arg-type]
