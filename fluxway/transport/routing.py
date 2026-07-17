from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np

from fluxway.common.tensor_backend import get_backend
from fluxway.common.tensor_components import (
    select_destination_content_index_source,
    select_source_geometry_index_source,
)
from fluxway.common.types.sharding import PhysicalLayout
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor
from fluxway.transport.planner import DestinationAssignment


class TransportRoutingMode(StrEnum):
    ALGEBRAIC = "algebraic"
    RECEIPT = "receipt"


class RoutingSourceKind(StrEnum):
    SOURCE_LAYOUT = "source_layout"
    ROUTING_INDICES = "routing_indices"
    ORIGIN_INDICES = "origin_indices"
    VALUES = "values"
    RECEIPT = "receipt"


@dataclass(frozen=True, slots=True)
class SelectedRoutingSource:
    kind: RoutingSourceKind
    value: Any | None
    entity_count: int

    @property
    def batch_key(self) -> Hashable:
        identity = (self.kind, self.entity_count)
        if self.value is None:
            return identity
        if self.kind is RoutingSourceKind.RECEIPT:
            receipt = self.value
            if not isinstance(receipt, SymmetryReceipt):
                raise TypeError(
                    "receipt routing source must be a SymmetryReceipt, got "
                    f"{type(receipt).__name__}."
                )
            if isinstance(receipt.sender_rank_tags, int | np.integer):
                return (
                    *identity,
                    receipt.world_size,
                    int(receipt.sender_rank_tags),
                )
        return (*identity, id(self.value))


def select_routing_source(
    *,
    tensor: TransportTensor,
    source: PhysicalLayout | None,
    target: PhysicalLayout | None,
    mode: TransportRoutingMode,
    expected_receipt_world_size: int | None = None,
) -> SelectedRoutingSource:
    if not isinstance(mode, TransportRoutingMode):
        raise TypeError("mode must be a TransportRoutingMode.")
    entity_count = tensor.routing_entity_count
    if mode is TransportRoutingMode.RECEIPT:
        receipt = tensor.receipt
        if receipt is None:
            raise ValueError("receipt routing requires TransportTensor.receipt.")
        if (
            expected_receipt_world_size is not None
            and receipt.world_size != expected_receipt_world_size
        ):
            raise ValueError(
                "receipt world size does not match the compiled target rank domain: "
                f"receipt={receipt.world_size} target={expected_receipt_world_size}."
            )
        return SelectedRoutingSource(RoutingSourceKind.RECEIPT, receipt, entity_count)

    source_is_content_addressable = bool(source and source.is_content_addressable)
    target_is_content_addressable = bool(target and target.is_content_addressable)
    source_indices, source_name = select_source_geometry_index_source(
        routing_indices=tensor.routing_indices,
        origin_indices=tensor.origin_indices,
        source_is_content_addressable=source_is_content_addressable,
        target_is_content_addressable=target_is_content_addressable,
    )
    if target_is_content_addressable:
        destination_values = select_destination_content_index_source(
            routing_indices=tensor.routing_indices,
            flat_values=tensor.flat_values,
        )
        kind = (
            RoutingSourceKind.ROUTING_INDICES
            if tensor.routing_indices is not None
            else RoutingSourceKind.VALUES
        )
        return SelectedRoutingSource(kind, destination_values, entity_count)
    if source_indices is not None:
        kind = (
            RoutingSourceKind.ORIGIN_INDICES
            if source_name == "origin_indices"
            else RoutingSourceKind.ROUTING_INDICES
        )
        return SelectedRoutingSource(kind, source_indices, entity_count)
    return SelectedRoutingSource(RoutingSourceKind.SOURCE_LAYOUT, None, entity_count)


def receipt_destination_assignment(
    receipt: SymmetryReceipt,
    *,
    entity_count: int,
    capacity: int,
) -> DestinationAssignment:
    receipt.validate_entity_count(entity_count)
    if capacity < entity_count:
        raise ValueError(
            f"Receipt routing capacity={capacity} is smaller than "
            f"entity_count={entity_count}."
        )
    if receipt.is_scalar():
        ranks = receipt.sender_rank_tags
    elif isinstance(receipt.sender_rank_tags, np.ndarray):
        ranks = receipt.sender_rank_tags
        padding_count = capacity - entity_count
        if padding_count:
            ranks = np.concatenate(
                [
                    ranks,
                    np.full(
                        (padding_count,),
                        receipt.world_size,
                        dtype=ranks.dtype,
                    ),
                ]
            )
    else:
        backend = get_backend()
        ranks = backend.adopt(receipt.sender_rank_tags)
        padding_count = capacity - entity_count
        if padding_count:
            padding = backend.full(
                (padding_count,),
                receipt.world_size,
                dtype=backend.dtype(ranks),
            )
            ranks = backend.cat([ranks, padding], dim=0)
    return DestinationAssignment(
        ranks=ranks,
        entity_count=entity_count,
        target_rank_count=receipt.world_size,
    )
