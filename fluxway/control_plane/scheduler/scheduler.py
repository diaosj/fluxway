import asyncio
import logging
from decimal import Decimal
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import ray

from fluxway.common.compiled_graph import CompiledGraph
from fluxway.common.types import Graph
from fluxway.control_plane.scheduler.pg_manager import PGManager
from fluxway.placement import (
    AllocationReport,
    ClusterNode,
    ClusterSnapshot,
    PlacementAttestationError,
    PlacementContractError,
    PlacementPlanner,
    PlacementRuntimeConfig,
    ResolvedPlacementPlan,
    PlacementRecipe,
)
from fluxway.placement.inventory import PlacementInventory

if TYPE_CHECKING:
    from fluxway.control_plane.spmd_manager import SPMDManager

logger = logging.getLogger(__name__)


def build_topology(graph: CompiledGraph) -> dict[str, list[str]]:
    topology: dict[str, list[str]] = {spmd.spmd_id: [] for spmd in graph.spmds}
    for dependency in graph.dependencies:
        edge = dependency.spec
        if edge.local_only:
            continue
        if (
            edge.source.spmd_id in topology
            and edge.target.spmd_id not in topology[edge.source.spmd_id]
        ):
            topology[edge.source.spmd_id].append(edge.target.spmd_id)
    return topology


class Scheduler:
    def __init__(
        self,
        spmd_manager: "SPMDManager | None" = None,
        *,
        graph: Graph,
        placement_recipe: PlacementRecipe,
        placement_runtime: PlacementRuntimeConfig | None = None,
    ):
        self._spmd_manager = spmd_manager
        self._pg_manager = PGManager()
        self._nodes = dict(graph.nodes)
        self._placement_inventory = PlacementInventory.from_compiled_graph(
            graph.require_compiled_graph()
        )
        if set(self._nodes) != set(self._placement_inventory.spmd_ids):
            raise PlacementContractError(
                "runtime nodes differ from compiled SPMD declarations"
            )
        self._placement_recipe = placement_recipe
        self._placement_runtime = placement_runtime or PlacementRuntimeConfig()
        self._resolved_plan: ResolvedPlacementPlan | None = None
        self._cluster_snapshot: ClusterSnapshot | None = None
        self._allocation_report: AllocationReport | None = None

    def set_spmd_manager(self, spmd_manager: "SPMDManager") -> None:
        self._spmd_manager = spmd_manager

    @property
    def spmd_manager(self) -> "SPMDManager | None":
        return self._spmd_manager

    @property
    def pg_manager(self) -> PGManager:
        return self._pg_manager

    @property
    def resolved_plan(self) -> ResolvedPlacementPlan | None:
        return self._resolved_plan

    @property
    def allocation_report(self) -> AllocationReport | None:
        return self._allocation_report

    async def allocate_resources(self) -> None:
        snapshot = self._capture_snapshot()
        planner = self._planner()
        try:
            plan = await asyncio.to_thread(
                planner.resolve,
                self._placement_inventory,
                self._placement_recipe,
                snapshot,
            )
        except Exception as error:
            self._record_failure(error, snapshot)
            raise
        self._allocation_report = AllocationReport.from_plan(
            plan, snapshot, status="PLANNED"
        )
        verified_snapshot = self._capture_snapshot()
        if self._placement_snapshot(verified_snapshot) != self._placement_snapshot(
            snapshot
        ):
            snapshot_error = PlacementContractError(
                "Ray cluster resources changed between solve and placement-group creation"
            )
            self._record_failure(snapshot_error, snapshot, plan)
            raise snapshot_error
        try:
            pg = await self._pg_manager.create_job_pg(
                plan=plan,
                snapshot=snapshot,
                timeout_s=self._placement_runtime.pg_timeout_s,
            )
        except Exception as error:
            self._record_failure(error, snapshot, plan)
            raise
        self._cluster_snapshot = snapshot
        self._resolved_plan = plan
        self._allocation_report = AllocationReport.from_plan(
            plan,
            snapshot,
            status="PG_READY",
            pg_id=str(pg.id),
        )

    def _capture_snapshot(self) -> ClusterSnapshot:
        nodes = tuple(
            sorted(
                (
                    ClusterNode(
                        node_id=node["NodeID"],
                        resources={
                            resource: Decimal(str(quantity))
                            for resource, quantity in node["Resources"].items()
                        },
                    )
                    for node in ray.nodes()
                    if node["Alive"]
                ),
                key=lambda node: node.node_id,
            )
        )
        available = {
            resource: Decimal(str(quantity))
            for resource, quantity in ray.available_resources().items()
        }
        return ClusterSnapshot(nodes=nodes, available=available)

    def _placement_snapshot(self, snapshot: ClusterSnapshot) -> ClusterSnapshot:
        requested_resources = {
            resource
            for resource, quantity in self._placement_runtime.driver_resources.items()
            if quantity > 0
        }
        requested_resources.update(self._placement_inventory.resource_names)
        projected_nodes = tuple(
            ClusterNode(
                node_id=node.node_id,
                resources={
                    resource: quantity
                    for resource, quantity in node.resources.items()
                    if resource in requested_resources
                },
            )
            for node in snapshot.nodes
        )
        projected_available = {
            resource: quantity
            for resource, quantity in snapshot.available.items()
            if resource in requested_resources
        }
        return ClusterSnapshot(
            nodes=projected_nodes,
            available=projected_available,
        )

    async def init_spmds(self) -> None:
        if self._spmd_manager is None:
            raise PlacementContractError("SPMD manager is not configured")
        tasks = [
            self._spmd_manager.start_spmd(
                spmd_id=spmd_id,
                config=node.spmd_config,
                worker_group=self._pg_manager.get_worker_group(spmd_id),
                checkpoint_id=None,
            )
            for spmd_id, node in self._nodes.items()
        ]
        try:
            handles = await asyncio.gather(*tasks)
            attested_plan, attested_node_ids = self._attest_resolved_plan(handles)
        except Exception as error:
            if self._cluster_snapshot is not None:
                self._record_failure(error, self._cluster_snapshot, self._resolved_plan)
            if self._pg_manager.has_job_pg:
                self._pg_manager.release_job_pg()
            raise
        report = self._allocation_report
        if report is None:
            raise PlacementContractError(
                "allocation report is missing before attestation"
            )
        self._resolved_plan = attested_plan
        self._set_allocation_report(
            AllocationReport.from_plan(
                attested_plan,
                self._require_snapshot(),
                status="READY",
                attested_node_ids=attested_node_ids,
                pg_id=report.pg_id,
            )
        )

    def abort_initialization(self, error: Exception) -> None:
        report = self._allocation_report
        if self._cluster_snapshot is not None and (
            report is None or report.status != "FAILED"
        ):
            self._record_failure(
                error,
                self._cluster_snapshot,
                self._resolved_plan,
            )
        if self._pg_manager.has_job_pg:
            self._pg_manager.release_job_pg()

    def _attest_resolved_plan(
        self, handles: list[Any]
    ) -> tuple[ResolvedPlacementPlan, dict[str, tuple[str, ...]]]:
        plan = self._resolved_plan
        if plan is None:
            raise PlacementContractError(
                "placement attestation requires a resolved Recipe plan"
            )
        attested_node_ids = {}
        for spmd_id, handle in zip(self._nodes, handles, strict=True):
            node_ids = handle.attested_node_ids
            if node_ids is None:
                raise PlacementAttestationError(
                    f"SPMD {spmd_id!r} returned no attested node IDs"
                )
            attested_node_ids[spmd_id] = tuple(node_ids)
        planner = self._planner()
        return (
            planner.attest(
                self._placement_inventory,
                self._placement_recipe,
                self._require_snapshot(),
                plan,
                attested_node_ids,
            ),
            attested_node_ids,
        )

    def _planner(self) -> PlacementPlanner:
        return PlacementPlanner(
            solve_timeout_s=self._placement_runtime.solve_timeout_s,
            driver_resources=self._placement_runtime.driver_resources,
            max_objective_components=self._placement_runtime.max_objective_components,
            max_sorting_comparators=self._placement_runtime.max_sorting_comparators,
            max_lexicographic_solves=self._placement_runtime.max_lexicographic_solves,
            max_model_variables=self._placement_runtime.max_model_variables,
            max_model_constraints=self._placement_runtime.max_model_constraints,
        )

    def _require_snapshot(self) -> ClusterSnapshot:
        if self._cluster_snapshot is None:
            raise PlacementContractError("cluster snapshot is not available")
        return self._cluster_snapshot

    def _set_allocation_report(self, report: AllocationReport) -> None:
        self._allocation_report = report
        if report.status not in ("READY", "FAILED"):
            return
        logger.info(
            "placement.summary status=%s recipe_id=%s quanta=%s "
            "occupancy=%s support=%s fragmentation=%s error=%s",
            report.status,
            report.recipe_id,
            report.allocated_quanta,
            report.occupancy,
            report.support,
            report.resource_fragmentation,
            report.error,
        )

    def _record_failure(
        self,
        error: Exception,
        snapshot: ClusterSnapshot,
        plan: ResolvedPlacementPlan | None = None,
    ) -> None:
        report = (
            AllocationReport.from_plan(
                plan,
                snapshot,
                status="FAILED",
                error=str(error),
            )
            if plan is not None
            else AllocationReport.from_failure(
                self._placement_recipe,
                snapshot,
                error,
            )
        )
        self._set_allocation_report(report)

    def establish_topology(self, topology: Mapping[str, list[str]]) -> None:
        if self._spmd_manager is None:
            raise PlacementContractError("SPMD manager is not configured")
        self._spmd_manager.establish_topology(dict(topology))

    def release_resources(self) -> None:
        self._pg_manager.release_job_pg()
        self._resolved_plan = None
        self._cluster_snapshot = None
        self._allocation_report = None
