from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, TYPE_CHECKING

from fluxway.common.compiled_graph import (
    CompiledDependency,
    CompiledLogicalTensor,
    CompiledTensorTransfer,
)
from fluxway.common.types.sharding import PhysicalLayout
from fluxway.transport.manifest import TensorEntryKind

if TYPE_CHECKING:
    from fluxway.spmd.worker.operator_manager.operator_manager import (
        ResolvedDependencyRoute,
    )


class AssemblyRouteResolver(Protocol):
    def get_predecessor_route(
        self,
        op_name: str,
        transfer_code: int,
    ) -> ResolvedDependencyRoute: ...

    def get_transfer_source_layout(
        self,
        op_name: str,
        transfer_code: int,
    ) -> PhysicalLayout: ...

    def get_transfer_target_layout(
        self,
        op_name: str,
        transfer_code: int,
    ) -> PhysicalLayout: ...

    def resolve_tensor_producers(
        self,
        op_name: str,
        transfer_code: int,
        current_rank: int,
        step: int = 0,
    ) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class CompiledAssemblePlan:
    op_name: str
    dependency: CompiledDependency
    transfer: CompiledTensorTransfer
    logical_tensor: CompiledLogicalTensor
    source_spmd_id: str
    source_layout: PhysicalLayout
    target_layout: PhysicalLayout

    @property
    def transfer_code(self) -> int:
        return self.transfer.transfer_code

    @property
    def assembles_to_dynamic(self) -> bool:
        return self.target_layout.is_content_addressable

    @property
    def has_dynamic_predecessor(self) -> bool:
        return self.source_layout.is_content_addressable

    @property
    def dynamic_to_static(self) -> bool:
        return self.has_dynamic_predecessor and not self.assembles_to_dynamic

    def filter_shards(
        self,
        *,
        op_manager: AssemblyRouteResolver,
        worker_rank: int,
        step: int,
        shards_dict: dict[Any, list[Any]],
        flatten_shards: Callable[[dict[Any, list[Any]], int], list[Any]],
    ) -> list[Any]:
        if self.dynamic_to_static:
            return _collect_source_shards(
                self.source_spmd_id,
                shards_dict,
                flatten_shards,
            )

        filtered_shards: list[Any] = []
        producer_ranks = op_manager.resolve_tensor_producers(
            self.op_name,
            self.transfer_code,
            worker_rank,
            step=step,
        )
        for rank in producer_ranks:
            key = (self.source_spmd_id, rank)
            if key in shards_dict:
                filtered_shards.extend(flatten_shards({key: shards_dict[key]}, 0))
        return filtered_shards

    def collect_arrived_shards(
        self,
        shards_dict: dict[Any, list[Any]],
        flatten_shards: Callable[[dict[Any, list[Any]], int], list[Any]],
    ) -> list[Any]:
        return _collect_source_shards(
            self.source_spmd_id,
            shards_dict,
            flatten_shards,
        )

    def collect_arrivals(
        self,
        arrivals_dict: dict[Any, dict[str, Any]],
    ) -> dict[Any, dict[str, Any]]:
        arrivals: dict[Any, dict[str, Any]] = {}
        for key, meta in arrivals_dict.items():
            source_spmd_id, _ = _require_shard_key(key)
            if source_spmd_id == self.source_spmd_id:
                arrivals[key] = meta
        return arrivals

    def completed_without_tensor(
        self,
        arrivals_dict: dict[Any, dict[str, Any]],
    ) -> bool:
        if not arrivals_dict:
            raise ValueError(
                f"Transfer code {self.transfer_code} has no arrival metadata."
            )
        completed_without_tensor = True
        for key, metadata in arrivals_dict.items():
            source_spmd_id, _ = _require_shard_key(key)
            if source_spmd_id != self.source_spmd_id:
                raise ValueError(
                    f"Transfer code {self.transfer_code} received unexpected source "
                    f"{source_spmd_id!r}; expected {self.source_spmd_id!r}."
                )
            raw_kind = metadata.get("tensor_kind")
            if raw_kind is None:
                raise ValueError(
                    f"Transfer code {self.transfer_code} arrival {key!r} lacks "
                    "tensor_kind."
                )
            tensor_kind = TensorEntryKind(int(raw_kind))
            if tensor_kind is TensorEntryKind.NOT_APPLICABLE:
                raise ValueError(
                    f"Tensor transfer code {self.transfer_code} cannot use "
                    "NOT_APPLICABLE."
                )
            if tensor_kind is not TensorEntryKind.COMPLETED_WITHOUT_TENSOR:
                completed_without_tensor = False
        return completed_without_tensor


def compile_assemble_plan(
    *,
    op_manager: AssemblyRouteResolver,
    op_name: str,
    transfer_code: int,
) -> CompiledAssemblePlan:
    route = op_manager.get_predecessor_route(op_name, transfer_code)
    return CompiledAssemblePlan(
        op_name=op_name,
        dependency=route.dependency,
        transfer=route.transfer,
        logical_tensor=route.logical_tensor,
        source_spmd_id=route.spmd_id,
        source_layout=op_manager.get_transfer_source_layout(op_name, transfer_code),
        target_layout=op_manager.get_transfer_target_layout(op_name, transfer_code),
    )


def _collect_source_shards(
    source_spmd_id: str,
    shards_dict: dict[Any, list[Any]],
    flatten_shards: Callable[[dict[Any, list[Any]], int], list[Any]],
) -> list[Any]:
    collected: list[Any] = []
    for key, rank_shards in shards_dict.items():
        shard_spmd_id, _ = _require_shard_key(key)
        if shard_spmd_id == source_spmd_id:
            collected.extend(flatten_shards({key: rank_shards}, 0))
    return collected


def _require_shard_key(key: Any) -> tuple[str, int]:
    if (
        not isinstance(key, tuple)
        or len(key) != 2
        or not isinstance(key[0], str)
        or not isinstance(key[1], int)
    ):
        raise ValueError(f"Invalid tensor shard key {key!r}.")
    return key
