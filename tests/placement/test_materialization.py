import asyncio
from decimal import Decimal
import pytest

from fluxway.placement import (
    AFFINITY_FIRST_V1,
    ClusterNode,
    ClusterSnapshot,
    PlacementGroupTimeoutError,
    PlacementAttestationError,
    PlacementMaterializationError,
    PlacementPlanner,
    PlacementRecipe,
    PlacementRuntimeConfig,
    ResourceProfile,
    SPMDWorkerGroup,
)
from fluxway.control_plane.scheduler.pg_manager import PGManager
from fluxway.placement.worker_group import ray_actor_options, worker_tensor_device

from .test_planner import (
    graph_node,
    placement_graph,
    placement_inventory,
    scenario,
)


class MultiAcceleratorSPMD: ...


class FakePlacementGroup:
    id = "fake-placement-group"

    async def ready(self) -> None:
        return None


@pytest.mark.asyncio
async def test_cancelled_materialization_releases_group(monkeypatch):
    graph, recipe, snapshot = scenario(1)
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )
    entered = asyncio.Event()

    class PendingPlacementGroup:
        async def ready(self):
            entered.set()
            await asyncio.Event().wait()

    pg = PendingPlacementGroup()
    removed = []
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.placement_group", lambda **_: pg
    )
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.remove_placement_group",
        removed.append,
    )
    manager = PGManager()
    creation = asyncio.create_task(
        manager.create_job_pg(plan=plan, snapshot=snapshot, timeout_s=10)
    )
    await asyncio.wait_for(entered.wait(), 1)
    creation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await creation
    assert removed == [pg]
    assert not manager.has_job_pg


def test_heterogeneous_nodes_preserve_resources_and_identity(monkeypatch):
    profiles = {
        "cuda": {"CPU": 2, "GPU": 2, "accelerator_type:H100": Decimal("0.001")},
        "other": {"CPU": 4, "GPU": 1, "accelerator_type:V100": Decimal("0.001")},
    }
    graph = placement_graph(
        nodes={
            name: graph_node(
                name, MultiAcceleratorSPMD, 1, ResourceProfile(base=resources)
            )
            for name, resources in profiles.items()
        }
    )
    snapshot = ClusterSnapshot.from_nodes(
        tuple(
            ClusterNode(node_id=name, resources={**resources, "node:10.0.0.1": 1})
            for name, resources in profiles.items()
        )
    )
    recipe = PlacementRecipe(
        recipe_id="heterogeneous",
        version="1.0.0",
        constraints=(),
        objective=AFFINITY_FIRST_V1,
    )
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )
    captured = {}

    def create(**options):
        captured.update(options)
        return FakePlacementGroup()

    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.placement_group", create
    )
    asyncio.run(PGManager().create_job_pg(plan=plan, snapshot=snapshot, timeout_s=1))
    for selector, resources in zip(
        captured["bundle_label_selector"], captured["bundles"], strict=True
    ):
        assert resources == {
            key: float(value)
            for key, value in profiles[selector["ray.io/node-id"]].items()
        }


def test_pg_manager_materializes_one_pinned_job_placement_group(monkeypatch) -> None:
    graph, recipe, snapshot_without_markers = scenario(2)
    nodes = tuple(
        ClusterNode(
            node_id=node.node_id,
            resources={
                **node.resources,
                f"node:10.0.0.{index + 1}": Decimal("1"),
            },
        )
        for index, node in enumerate(snapshot_without_markers.nodes)
    )
    snapshot = ClusterSnapshot.from_nodes(nodes)
    from fluxway.placement import PlacementPlanner

    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )
    captured = {}

    def create_pg(*, bundles, strategy, bundle_label_selector):
        captured["bundles"] = bundles
        captured["strategy"] = strategy
        captured["bundle_label_selector"] = bundle_label_selector
        return FakePlacementGroup()

    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.placement_group", create_pg
    )
    manager = PGManager()

    pg = asyncio.run(manager.create_job_pg(plan=plan, snapshot=snapshot, timeout_s=1))

    assert pg.id == "fake-placement-group"
    assert captured["strategy"] == "STRICT_SPREAD"
    assert len(captured["bundles"]) == len(plan.bundle_resources())
    assert all(
        not any(key.startswith("node:") for key in bundle)
        for bundle in captured["bundles"]
    )
    assert captured["bundle_label_selector"] == [
        {"ray.io/node-id": node_id} for node_id in sorted(plan.bundle_resources())
    ]
    allocation = manager.get_worker_group("s1")
    assert allocation.placement_group is pg
    assert len(allocation.rank_to_bundle) == 12
    assert allocation.expected_node_ids == tuple(
        assignment.node_id
        for assignment in plan.assignments
        if assignment.spmd_id == "s1"
    )


def test_pg_manager_raises_typed_timeout_and_releases_pg(monkeypatch) -> None:
    graph, recipe, snapshot_without_markers = scenario(1)
    snapshot = ClusterSnapshot.from_nodes(
        tuple(
            ClusterNode(
                node_id=node.node_id,
                resources={
                    **node.resources,
                    f"node:10.0.1.{index + 1}": Decimal("1"),
                },
            )
            for index, node in enumerate(snapshot_without_markers.nodes)
        )
    )
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )
    removed = []

    class NeverReadyPlacementGroup:
        async def ready(self) -> None:
            await asyncio.Event().wait()

    pg = NeverReadyPlacementGroup()
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.placement_group",
        lambda **_: pg,
    )
    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.remove_placement_group",
        removed.append,
    )

    with pytest.raises(PlacementGroupTimeoutError, match="timed out"):
        asyncio.run(
            PGManager().create_job_pg(plan=plan, snapshot=snapshot, timeout_s=0.001)
        )

    assert removed == [pg]


def test_pg_manager_rejects_resolved_node_absent_from_snapshot() -> None:
    graph, recipe, snapshot = scenario(1)
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )

    absent = ClusterSnapshot.from_nodes(
        (ClusterNode(node_id="other", resources={"CPU": 16}),)
    )
    with pytest.raises(
        PlacementMaterializationError, match="absent from the Ray snapshot"
    ):
        asyncio.run(PGManager().create_job_pg(plan=plan, snapshot=absent, timeout_s=1))


def test_pg_manager_preserves_multiple_accelerator_resources(monkeypatch) -> None:
    graph = placement_graph(
        nodes={
            "accelerated": graph_node(
                "accelerated",
                MultiAcceleratorSPMD,
                1,
                ResourceProfile(
                    base={
                        "GPU": 1,
                        "accelerator_type:H100": Decimal("0.001"),
                    }
                ),
            )
        }
    )
    recipe = PlacementRecipe(
        recipe_id="multi-accelerator",
        version="1.0.0",
        constraints=(),
        objective=AFFINITY_FIRST_V1,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="gpu",
                resources={
                    "GPU": 1,
                    "accelerator_type:H100": Decimal("0.001"),
                    "node:10.0.2.1": 1,
                },
            ),
        )
    )
    plan = PlacementPlanner(solve_timeout_s=10).resolve(
        placement_inventory(graph), recipe, snapshot
    )
    captured = {}

    def create_pg(*, bundles, strategy, bundle_label_selector):
        captured["bundles"] = bundles
        captured["bundle_label_selector"] = bundle_label_selector
        return FakePlacementGroup()

    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.placement_group", create_pg
    )

    asyncio.run(PGManager().create_job_pg(plan=plan, snapshot=snapshot, timeout_s=1))

    assert captured["bundles"] == [
        {
            "GPU": 1.0,
            "accelerator_type:H100": 0.001,
        }
    ]
    assert captured["bundle_label_selector"] == [{"ray.io/node-id": "gpu"}]


def test_worker_group_attestation_rejects_rank_node_mismatch() -> None:
    group = SPMDWorkerGroup(
        spmd_id="attested",
        placement_group=object(),
        rank_to_bundle=(0, 1),
        expected_node_ids=("node-a", "node-b"),
        worker_resources={"CPU": Decimal("1")},
    )

    with pytest.raises(PlacementAttestationError, match="expected=.*actual"):
        group.attest(("node-b", "node-a"))


def test_ray_actor_options_preserve_explicit_zero_accelerators() -> None:
    options = ray_actor_options({"CPU": Decimal("0"), "GPU": Decimal("0")})

    assert options["num_cpus"] == 0
    assert options["num_gpus"] == 0


def test_worker_tensor_device_follows_materialized_gpu_resources() -> None:
    assert worker_tensor_device({"CPU": Decimal("1")}) == "cpu"
    assert worker_tensor_device({"CPU": Decimal("1"), "GPU": Decimal("0.25")}) == "cuda"


def test_default_runtime_uses_zero_resource_driver_actors() -> None:
    runtime = PlacementRuntimeConfig()

    assert runtime.driver_resources == {}
    assert ray_actor_options(runtime.driver_resources) == {
        "num_cpus": 0,
        "num_gpus": 0,
    }


def test_driver_resources_are_reserved_in_the_rank_zero_bundle(monkeypatch) -> None:
    graph = placement_graph(
        nodes={"driver": graph_node("driver", MultiAcceleratorSPMD, 1)}
    )
    recipe = PlacementRecipe(
        recipe_id="driver-resources",
        version="1.0.0",
        constraints=(),
        objective=AFFINITY_FIRST_V1,
    )
    snapshot = ClusterSnapshot.from_nodes(
        (
            ClusterNode(
                node_id="cpu",
                resources={"CPU": 2, "node:10.0.3.1": 1},
            ),
        )
    )
    plan = PlacementPlanner(
        solve_timeout_s=10,
        driver_resources={"CPU": Decimal("0.5")},
    ).resolve(placement_inventory(graph), recipe, snapshot)
    captured = {}

    def create_pg(*, bundles, strategy, bundle_label_selector):
        captured["bundles"] = bundles
        captured["bundle_label_selector"] = bundle_label_selector
        return FakePlacementGroup()

    monkeypatch.setattr(
        "fluxway.control_plane.scheduler.pg_manager.placement_group", create_pg
    )
    manager = PGManager()

    asyncio.run(manager.create_job_pg(plan=plan, snapshot=snapshot, timeout_s=1))

    assert captured["bundles"] == [{"CPU": 1.5}]
    assert captured["bundle_label_selector"] == [{"ray.io/node-id": "cpu"}]
    group = manager.get_worker_group("driver")
    assert group.driver_bundle_index == 0
    assert group.driver_resources == {"CPU": Decimal("0.5")}
