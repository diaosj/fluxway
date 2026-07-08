from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AssembleProviderPlan:
    spmd_id: str
    dynamic_to_static: bool


@dataclass(frozen=True, slots=True)
class CompiledAssemblePlan:
    op_name: str
    tuuid: int
    input_layout: Any
    predecessor_topology: dict[str, Any]
    providers: tuple[AssembleProviderPlan, ...]
    has_dynamic_predecessor: bool

    @property
    def assembles_to_dynamic(self) -> bool:
        return self.input_layout is None or self.input_layout.is_content_addressable

    def filter_shards(
        self,
        *,
        op_manager: Any,
        worker_rank: int,
        step: int,
        shards_dict: dict[Any, list[Any]],
        flatten_shards: Callable[[dict[Any, list[Any]], int], list[Any]],
    ) -> list[Any]:
        filtered_shards: list[Any] = []
        for provider in self.providers:
            if provider.dynamic_to_static:
                filtered_shards.extend(
                    _collect_provider_shards(
                        provider.spmd_id,
                        shards_dict,
                        flatten_shards,
                    )
                )
                continue

            producer_ranks = op_manager.resolve_tensor_producers(
                self.op_name,
                self.tuuid,
                provider.spmd_id,
                worker_rank,
                step=step,
            )
            for rank in producer_ranks:
                key = (provider.spmd_id, rank)
                if key in shards_dict:
                    filtered_shards.extend(flatten_shards({key: shards_dict[key]}, 0))
        return filtered_shards

    def collect_arrived_shards(
        self,
        shards_dict: dict[Any, list[Any]],
        flatten_shards: Callable[[dict[Any, list[Any]], int], list[Any]],
    ) -> list[Any]:
        collected: list[Any] = []
        for provider in self.providers:
            collected.extend(
                _collect_provider_shards(
                    provider.spmd_id,
                    shards_dict,
                    flatten_shards,
                )
            )
        return collected

    def collect_arrivals(
        self,
        arrivals_dict: dict[Any, dict[str, Any]],
    ) -> dict[Any, dict[str, Any]]:
        provider_ids = {provider.spmd_id for provider in self.providers}
        arrivals: dict[Any, dict[str, Any]] = {}
        for key, meta in arrivals_dict.items():
            if not isinstance(key, tuple) or len(key) != 2:
                continue
            shard_spmd_id, _rank = key
            if shard_spmd_id in provider_ids:
                arrivals[key] = meta
        return arrivals


def compile_assemble_plan(
    *,
    op_manager: Any,
    op_name: str,
    tuuid: int,
) -> CompiledAssemblePlan:
    input_layout = op_manager._resolve_input_spec(op_name, tuuid)
    predecessor_topology = op_manager.get_predecessor_topology(op_name)
    providers: list[AssembleProviderPlan] = []
    has_dynamic_predecessor = False

    for pre_spmd_id, pre_info in predecessor_topology.items():
        producer_spec = pre_info.tuuid_sharding.get(tuuid)
        if producer_spec is None:
            continue
        producer_layout = op_manager._resolve_producer_spec(
            tuuid,
            provider_map=pre_info.tuuid_sharding,
        )
        producer_is_dynamic = bool(
            producer_layout is not None and producer_layout.is_content_addressable
        )
        has_dynamic_predecessor = has_dynamic_predecessor or producer_is_dynamic
        providers.append(
            AssembleProviderPlan(
                spmd_id=pre_spmd_id,
                dynamic_to_static=bool(
                    producer_is_dynamic
                    and input_layout is not None
                    and not input_layout.is_content_addressable
                ),
            )
        )

    return CompiledAssemblePlan(
        op_name=op_name,
        tuuid=tuuid,
        input_layout=input_layout,
        predecessor_topology=predecessor_topology,
        providers=tuple(providers),
        has_dynamic_predecessor=has_dynamic_predecessor,
    )


def _collect_provider_shards(
    spmd_id: str,
    shards_dict: dict[Any, list[Any]],
    flatten_shards: Callable[[dict[Any, list[Any]], int], list[Any]],
) -> list[Any]:
    collected: list[Any] = []
    for key, rank_shards in shards_dict.items():
        if not isinstance(key, tuple) or len(key) != 2:
            continue
        shard_spmd_id, _rank = key
        if shard_spmd_id != spmd_id:
            continue
        collected.extend(flatten_shards({key: rank_shards}, 0))
    return collected
