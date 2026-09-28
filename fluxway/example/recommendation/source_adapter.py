"""Canonical recommendation metadata planning and source activation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from fluxway.control_plane.components import DriverComponentContext
from fluxway.common.metadata import pack_metadata, metadata_digest
from fluxway.common.types.configs import SPMDConfig
from fluxway.common.ingestion import LayoutUnitDescriptor, ingestion_worker_id
from fluxway.control_plane.ingestion.compiler import (
    TailLayoutMode,
    LayoutCompileRequest,
    AssignmentPolicy,
    RankDemand,
    ReaderGroupDemand,
    ReaderTopologyDescriptor,
)
from fluxway.control_plane.ingestion.adapter import (
    CatalogCursor,
    CatalogViewObservation,
    IngestionSourceAdapter,
)
from fluxway.control_plane.ingestion.activation import (
    IngestionLayoutActivationComponent,
)


class RecommendationSourcePlanner:
    """Canonical planner for the ordered recommendation sample stream."""

    def __init__(self, *, source_identity: str, revision_id: int) -> None:
        if not source_identity or source_identity.strip() != source_identity:
            raise ValueError("recommendation source identity must be canonical")
        if (
            not isinstance(revision_id, int)
            or isinstance(revision_id, bool)
            or revision_id < 0
        ):
            raise ValueError("recommendation revision_id must be non-negative")
        self.source_identity = source_identity
        self.revision_id = revision_id

    def plan_units(self, start: int, end: int) -> tuple[LayoutUnitDescriptor, ...]:
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError("recommendation planning interval is invalid")
        revision = f"recommendation:{self.revision_id}"
        if start == end:
            return ()
        return (
            LayoutUnitDescriptor(
                unit_id=f"{self.source_identity}:{self.revision_id}:{start}:{end}",
                revision_identity=revision,
                cardinality=end - start,
                cardinality_metric="samples",
                split_policy="SplittableSpan",
                locality_key="recommendation",
                physical_identity=f"{self.source_identity}:{self.revision_id}",
                logical_start=start,
                opaque_token=pack_metadata({"revision_id": self.revision_id}),
            ),
        )


class RecommendationSourceAdapter(IngestionSourceAdapter):
    """Source-neutral catalog view adapter backed by the canonical planner."""

    def __init__(
        self, *, source_identity: str = "recommendation", revision_id: int = 0
    ) -> None:
        self._planner = RecommendationSourcePlanner(
            source_identity=source_identity, revision_id=revision_id
        )

    def capture_view(
        self, cursor: Mapping[str, Any], *, cardinality_limit: int
    ) -> CatalogViewObservation:
        if not isinstance(cursor, Mapping):
            raise TypeError("recommendation cursor must be a mapping")
        start = cursor["start"]
        end = cursor["end"]
        if (
            not isinstance(start, int)
            or not isinstance(end, int)
            or start < 0
            or end < start
        ):
            raise ValueError("recommendation cursor interval is invalid")
        if not isinstance(cardinality_limit, int) or cardinality_limit <= 0:
            raise ValueError("recommendation cardinality limit must be positive")
        units = self._planner.plan_units(start, end)
        if sum(unit.cardinality for unit in units) > cardinality_limit:
            raise ValueError("recommendation view exceeds cardinality limit")
        return CatalogViewObservation(
            source_identity=self._planner.source_identity,
            view_identity=f"{self._planner.source_identity}:{self._planner.revision_id}",
            schema_digest="recommendation-schema-v1",
            lineage_digest=f"recommendation-revision:{self._planner.revision_id}",
            units=units,
            terminal=bool(cursor.get("terminal", False)),
            cursor=CatalogCursor(cursor),
        )

    def encode_cursor(self, cursor: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(cursor, Mapping):
            raise TypeError("recommendation cursor must be a mapping")
        return dict(cursor)

    def decode_cursor(self, state: Mapping[str, Any]) -> Mapping[str, Any]:
        return self.encode_cursor(state)


class RecommendationSourceDescriptor:
    requires_source_control_runtime = True

    def build_reader_topology(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> ReaderTopologyDescriptor:
        worker_count = config.num_workers
        global_batch_size = config.static_args.get("global_batch_size", 24)
        for value in (worker_count, global_batch_size):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    "recommendation ingestion plan dimensions must be positive integers"
                )
        cycle = (
            config.require_graph_view().operator(operator_name).execution_layout.cycle
        )
        active_count = len(cycle[0])
        if (
            any(len(ranks) != active_count for ranks in cycle)
            or global_batch_size % active_count
        ):
            raise ValueError(
                "recommendation global batch must divide evenly over active readers"
            )
        return ReaderTopologyDescriptor(
            global_cardinality=global_batch_size,
            groups=tuple(
                ReaderGroupDemand(
                    ingestion_worker_id(spmd_id, operator_name, rank),
                    (
                        RankDemand(
                            ingestion_worker_id(spmd_id, operator_name, rank),
                            global_batch_size // active_count,
                        ),
                    ),
                )
                for rank in range(worker_count)
            ),
            execution_cycle=cycle,
        )

    def build_ingestion_plan(
        self,
        *,
        spmd_id: str,
        operator_name: str,
        config: SPMDConfig,
    ) -> LayoutCompileRequest:
        topology = self.build_reader_topology(
            spmd_id=spmd_id, operator_name=operator_name, config=config
        )
        num_samples = config.static_args.get("num_samples", 1000)
        num_epochs = config.static_args.get("num_epochs", 1)
        if num_samples <= 0 or num_epochs <= 0:
            raise ValueError("recommendation dataset dimensions must be positive")
        cardinality = num_samples * num_epochs
        revision = int(
            metadata_digest(
                {
                    key: value
                    for key, value in config.static_args.items()
                    if key not in {"global_batch_size", "audit_dir"}
                }
            )[:16],
            16,
        )
        observation = RecommendationSourceAdapter(revision_id=revision).capture_view(
            {"start": 0, "end": cardinality, "terminal": True},
            cardinality_limit=cardinality,
        )
        return observation.layout_request(
            topology=topology,
            tail_layout_mode=TailLayoutMode.DISCARD_REMAINDER,
            assignment_policy=AssignmentPolicy.default(),
        )

    def build_component(
        self, context: DriverComponentContext
    ) -> IngestionLayoutActivationComponent:
        return IngestionLayoutActivationComponent(context)


__all__ = [
    "RecommendationSourcePlanner",
    "RecommendationSourceAdapter",
    "RecommendationSourceDescriptor",
]
