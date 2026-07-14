import asyncio
import logging
from decimal import Decimal
from typing import cast

from ray.util.placement_group import (
    PlacementGroup,
    placement_group,
    remove_placement_group,
)

from fluxway.placement.models import (
    ClusterSnapshot,
    PlacementGroupTimeoutError,
    PlacementMaterializationError,
    ResolvedPlacementPlan,
)
from fluxway.placement.worker_group import SPMDWorkerGroup

logger = logging.getLogger(__name__)
RAY_NODE_MARKER_QUANTITY = Decimal("0.0001")


class PGManager:
    def __init__(self) -> None:
        self._job_pg: PlacementGroup | None = None
        self._worker_groups: dict[str, SPMDWorkerGroup] = {}

    @property
    def has_job_pg(self) -> bool:
        return self._job_pg is not None

    async def create_job_pg(
        self,
        *,
        plan: ResolvedPlacementPlan,
        snapshot: ClusterSnapshot,
        timeout_s: float,
    ) -> PlacementGroup:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self._job_pg is not None:
            raise PlacementMaterializationError("job placement group already exists")
        snapshot_by_id = {node.node_id: node for node in snapshot.nodes}
        bundle_resources = plan.bundle_resources()
        node_ids = tuple(sorted(bundle_resources))
        node_to_bundle = {node_id: index for index, node_id in enumerate(node_ids)}
        bundles = []
        for node_id in node_ids:
            try:
                node = snapshot_by_id[node_id]
            except KeyError as error:
                raise PlacementMaterializationError(
                    f"resolved node {node_id!r} is absent from the Ray snapshot"
                ) from error
            node_resources = [
                resource
                for resource in node.resources
                if resource.startswith("node:")
                and not resource.startswith("node:__internal_")
            ]
            if len(node_resources) != 1:
                raise PlacementMaterializationError(
                    f"Ray node {node_id!r} has {len(node_resources)} usable node resources"
                )
            bundle = {
                resource: float(quantity)
                for resource, quantity in bundle_resources[node_id].items()
                if quantity > 0
            }
            node_resource = node_resources[0]
            if node_resource in bundle:
                raise PlacementMaterializationError(
                    f"Recipe resources cannot request Ray node marker {node_resource!r}"
                )
            bundle[node_resource] = float(RAY_NODE_MARKER_QUANTITY)
            bundles.append(bundle)
        try:
            pg = placement_group(bundles=bundles, strategy="STRICT_SPREAD")
        except Exception as error:
            raise PlacementMaterializationError(
                f"Ray rejected placement group for recipe {plan.recipe_id}"
            ) from error
        logger.info(
            "placement.pg.waiting bundles=%s timeout_s=%s",
            len(bundles),
            timeout_s,
        )
        started = asyncio.get_running_loop().time()
        try:
            await asyncio.wait_for(pg.ready(), timeout=timeout_s)
            worker_groups = self._worker_groups_from_plan(
                plan,
                pg,
                node_to_bundle,
            )
        except TimeoutError as error:
            remove_placement_group(pg)
            raise PlacementGroupTimeoutError(
                f"placement group timed out after {timeout_s}s for recipe {plan.recipe_id}"
            ) from error
        except Exception as error:
            remove_placement_group(pg)
            if isinstance(error, PlacementMaterializationError):
                raise
            raise PlacementMaterializationError(
                f"Ray failed to materialize placement group for recipe {plan.recipe_id}"
            ) from error
        self._job_pg = pg
        self._worker_groups = worker_groups
        logger.info(
            "placement.pg.ready pg_id=%s bundles=%s wait_duration=%s",
            getattr(pg, "id", None),
            len(bundles),
            asyncio.get_running_loop().time() - started,
        )
        return cast(PlacementGroup, pg)

    def _worker_groups_from_plan(
        self,
        plan: ResolvedPlacementPlan,
        pg: PlacementGroup,
        node_to_bundle: dict[str, int],
    ) -> dict[str, SPMDWorkerGroup]:
        worker_groups = {}
        spmd_ids = sorted({assignment.spmd_id for assignment in plan.assignments})
        for spmd_id in spmd_ids:
            assignments = tuple(
                sorted(
                    (
                        assignment
                        for assignment in plan.assignments
                        if assignment.spmd_id == spmd_id
                    ),
                    key=lambda assignment: assignment.rank,
                )
            )
            resource_vectors = {
                tuple(sorted(assignment.resources.items()))
                for assignment in assignments
            }
            if len(resource_vectors) != 1:
                raise PlacementMaterializationError(
                    f"SPMD {spmd_id!r} has non-uniform per-worker resources"
                )
            rank_zero = assignments[0]
            if rank_zero.rank != 0:
                raise PlacementMaterializationError(
                    f"SPMD {spmd_id!r} has no rank-zero assignment"
                )
            worker_groups[spmd_id] = SPMDWorkerGroup(
                spmd_id=spmd_id,
                placement_group=pg,
                rank_to_bundle=tuple(
                    node_to_bundle[assignment.node_id] for assignment in assignments
                ),
                expected_node_ids=tuple(
                    assignment.node_id for assignment in assignments
                ),
                worker_resources=dict(next(iter(resource_vectors))),
                driver_bundle_index=node_to_bundle[rank_zero.node_id],
                driver_resources=dict(rank_zero.driver_resources),
            )
        return worker_groups

    def get_worker_group(self, spmd_id: str) -> SPMDWorkerGroup:
        try:
            return self._worker_groups[spmd_id]
        except KeyError as error:
            raise KeyError(
                f"SPMD {spmd_id!r} has no worker-group allocation"
            ) from error

    def release_job_pg(self) -> None:
        if self._job_pg is None:
            raise RuntimeError("job placement group does not exist")
        remove_placement_group(self._job_pg)
        self._job_pg = None
        self._worker_groups.clear()
