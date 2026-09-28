"""Driver-owned delivery of compact ingestion layout descriptors."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Mapping
from typing import Any

from fluxway.common.driver_control import IngestionLayoutActivationRequest
from fluxway.common.ingestion.protocol import (
    IngestionLayoutActivation,
    SourceLayoutActivation,
)
from fluxway.common.ingestion.receiver import ingestion_worker_id
from fluxway.common.types import CheckpointCut, DriverState
from fluxway.control_plane.components import DriverComponentContext


class IngestionLayoutActivationComponent:
    def __init__(self, context: DriverComponentContext) -> None:
        if not isinstance(context, DriverComponentContext):
            raise TypeError("ingestion activation requires DriverComponentContext")
        if context.operator_spec.is_source is not True:
            raise ValueError("ingestion activation requires a source operator")
        self.component_id = context.component_id
        self._spmd_id = context.spmd_id
        self._operator_name = context.operator_name
        self._world_size = context.world_size
        self._worker_control = context.capabilities.worker_control
        self._applied_layouts: dict[int, str] = {}
        self._activation_lock = asyncio.Lock()

    async def on_state(self, state: Any) -> None:
        if not isinstance(state, DriverState):
            raise TypeError("ingestion activation component requires DriverState")

    async def dispatch(self, request: object) -> object:
        if not isinstance(request, SourceLayoutActivation):
            raise TypeError(
                "ingestion activation dispatch requires SourceLayoutActivation"
            )
        await self.activate(request)
        return None

    def checkpoint_state(self, cut: CheckpointCut) -> Mapping[str, Any]:
        if not isinstance(cut, CheckpointCut):
            raise TypeError("ingestion activation checkpoint requires CheckpointCut")
        return {}

    def restore_state(self, state: Mapping[str, Any], cut: CheckpointCut) -> None:
        if not isinstance(cut, CheckpointCut):
            raise TypeError("ingestion activation restore requires CheckpointCut")
        if not isinstance(state, Mapping) or state:
            raise ValueError("ingestion activation checkpoint fields do not match")
        self._applied_layouts.clear()

    async def activate(self, activation: SourceLayoutActivation) -> None:
        if not isinstance(activation, SourceLayoutActivation):
            raise TypeError("ingestion activation requires SourceLayoutActivation")
        expected = tuple(
            ingestion_worker_id(self._spmd_id, self._operator_name, rank)
            for rank in range(self._world_size)
        )
        if activation.epoch.worker_ids != expected:
            raise ValueError(
                "ingestion layout worker identities do not match Driver graph: "
                f"expected={expected!r} actual={activation.epoch.worker_ids!r}"
            )
        epoch = activation.epoch.epoch
        digest = activation.layout_digest
        async with self._activation_lock:
            previous = self._applied_layouts.get(epoch)
            if previous is not None:
                if previous != digest:
                    raise ValueError(
                        "ingestion activation replay is conflicting: "
                        f"epoch={epoch} previous={previous} current={digest}"
                    )
                return
            async with asyncio.TaskGroup() as tasks:
                for rank, descriptor in enumerate(activation.descriptors):
                    worker_activation = IngestionLayoutActivation(
                        epoch=epoch,
                        start_position=activation.epoch.start_position,
                        end_position=activation.epoch.end_position,
                        layout_digest=digest,
                        terminal=activation.epoch.terminal,
                        descriptor=descriptor,
                    )
                    request = IngestionLayoutActivationRequest(
                        rank=rank,
                        operator_name=self._operator_name,
                        activation=worker_activation,
                    )
                    result = self._worker_control.dispatch(request)
                    if not inspect.isawaitable(result):
                        raise TypeError("worker control dispatch must be awaitable")
                    tasks.create_task(result)
            self._applied_layouts[epoch] = digest


class DirectIngestionSourceDescriptor:
    requires_source_control_runtime = False

    def __init__(self, adapter_factory: Any | None = None) -> None:
        self.adapter_factory = adapter_factory

    def build_component(
        self, context: DriverComponentContext
    ) -> IngestionLayoutActivationComponent:
        return IngestionLayoutActivationComponent(context)


__all__ = [
    "DirectIngestionSourceDescriptor",
    "IngestionLayoutActivationComponent",
]
