import asyncio

import pytest
import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from fluxway.common.mesh import DeviceMesh
from fluxway.common.types import Graph, Node, SPMDConfig
from fluxway.placement import (
    AFFINITY_FIRST_V1,
    PlacementGroupTimeoutError,
    PlacementPlanner,
    PlacementRecipe,
    PlacementRuntimeConfig,
    ResourceProfile,
)
from fluxway.control_plane.scheduler import Scheduler
from fluxway.control_plane.scheduler.pg_manager import PGManager


class RayIntegrationSPMD: ...


class MultiAcceleratorIntegrationSPMD: ...


@ray.remote
def report_node_id() -> str:
    return ray.get_runtime_context().get_node_id()


@ray.remote(num_cpus=1)
class CPUBlocker:
    def ready(self) -> bool:
        return True


def test_plain_ray_cluster_materializes_recipe() -> None:
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=2, include_dashboard=False)
    scheduler = None
    try:
        graph = Graph(
            nodes={
                "worker": Node(
                    name="worker",
                    spmd_config=SPMDConfig(
                        spmd_id="worker",
                        mesh=DeviceMesh(("data",), device_shape=(1,)),
                    ),
                    spmd_class_module=RayIntegrationSPMD.__module__,
                    spmd_class_qualname=RayIntegrationSPMD.__qualname__,
                )
            }
        )
        recipe = PlacementRecipe(
            recipe_id="plain-ray",
            version="1.0.0",
            profiles={RayIntegrationSPMD: ResourceProfile(base={"CPU": 1})},
            constraints=(),
            objective=AFFINITY_FIRST_V1,
        )
        scheduler = Scheduler(
            placement_recipe=recipe,
            placement_runtime=PlacementRuntimeConfig(
                solve_timeout_s=10, pg_timeout_s=10
            ),
        )

        assert asyncio.run(scheduler.allocate_resources(graph)) is None
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
        if started:
            ray.shutdown()


def test_plain_ray_cluster_materializes_multiple_accelerator_resources() -> None:
    started = not ray.is_initialized()
    if started:
        ray.init(
            num_cpus=1,
            resources={
                "accelerator_type:H100": 1,
                "accelerator_type:TPU": 1,
            },
            include_dashboard=False,
        )
    scheduler = None
    try:
        graph = Graph(
            nodes={
                "accelerated": Node(
                    name="accelerated",
                    spmd_config=SPMDConfig(
                        spmd_id="accelerated",
                        mesh=DeviceMesh(("data",), device_shape=(1,)),
                    ),
                    spmd_class_module=MultiAcceleratorIntegrationSPMD.__module__,
                    spmd_class_qualname=MultiAcceleratorIntegrationSPMD.__qualname__,
                )
            }
        )
        resources = {
            "accelerator_type:H100": 1,
            "accelerator_type:TPU": 1,
        }
        recipe = PlacementRecipe(
            recipe_id="plain-ray-multi-accelerator",
            version="1.0.0",
            profiles={MultiAcceleratorIntegrationSPMD: ResourceProfile(base=resources)},
            constraints=(),
            objective=AFFINITY_FIRST_V1,
        )
        scheduler = Scheduler(
            placement_recipe=recipe,
            placement_runtime=PlacementRuntimeConfig(
                solve_timeout_s=10, pg_timeout_s=10
            ),
        )

        assert asyncio.run(scheduler.allocate_resources(graph)) is None
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
        if started:
            ray.shutdown()


def test_plain_ray_cluster_surfaces_real_pg_timeout() -> None:
    started = not ray.is_initialized()
    if started:
        ray.init(num_cpus=1, include_dashboard=False)
    blocker = None
    manager = PGManager()
    try:
        graph = Graph(
            nodes={
                "worker": Node(
                    name="worker",
                    spmd_config=SPMDConfig(
                        spmd_id="worker",
                        mesh=DeviceMesh(("data",), device_shape=(1,)),
                    ),
                    spmd_class_module=RayIntegrationSPMD.__module__,
                    spmd_class_qualname=RayIntegrationSPMD.__qualname__,
                )
            }
        )
        recipe = PlacementRecipe(
            recipe_id="plain-ray-timeout",
            version="1.0.0",
            profiles={RayIntegrationSPMD: ResourceProfile(base={"CPU": 1})},
            constraints=(),
            objective=AFFINITY_FIRST_V1,
        )
        scheduler = Scheduler(
            placement_recipe=recipe,
            placement_runtime=PlacementRuntimeConfig(
                solve_timeout_s=10, pg_timeout_s=10
            ),
        )
        snapshot = scheduler._capture_snapshot()
        plan = PlacementPlanner(solve_timeout_s=10).resolve(graph, recipe, snapshot)
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
        if started:
            ray.shutdown()
