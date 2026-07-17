from dataclasses import dataclass
from typing import Any

from fluxway.spmd.worker.mailbox.mailbox import WorkerMailbox
from fluxway.transport.manifest import (
    PayloadRole,
    TensorEntryKind,
    TransportFrameBuilder,
)


@dataclass(frozen=True, slots=True)
class TensorFrameEntry:
    transfer_code: int
    payloads: tuple[tuple[PayloadRole, Any, int], ...]
    expected_count: int = 1
    tensor_kind: TensorEntryKind = TensorEntryKind.TENSOR
    meta_state: int = 0
    source_step: int | None = None
    global_shape: tuple[int, ...] | None = None
    compact_params: tuple[int, ...] | None = None


async def deposit_tensor_frame(
    mailbox: WorkerMailbox,
    op_name: str,
    sender_id: str,
    *,
    step: int,
    sender_rank: int,
    sender_world_size: int,
    entries: tuple[TensorFrameEntry, ...],
) -> int:
    builder = TransportFrameBuilder(
        step=step,
        sender_spmd_code=0,
        sender_rank=sender_rank,
        sender_world_size=sender_world_size,
    )
    for entry in entries:
        entry_index = builder.add_tensor_transfer_entry(
            transfer_code=entry.transfer_code,
            meta_state=entry.meta_state,
            tensor_kind=entry.tensor_kind,
            expected_count=entry.expected_count,
            source_step=step if entry.source_step is None else entry.source_step,
            compact_params=entry.compact_params,
            global_shape=entry.global_shape,
        )
        for role, payload, valid_count in entry.payloads:
            builder.add_payload(entry_index, role, payload, valid_count)
    return await mailbox.deposit_transport_frame(op_name, sender_id, builder.build())
