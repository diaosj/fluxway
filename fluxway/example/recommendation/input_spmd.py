from typing import Any
import asyncio

import numpy as np
import sparse_embedding as se  # type: ignore

from fluxway.common.tensor_backend import get_backend
from fluxway.common.decorators import spmd, operator
from fluxway.common.types.tensor import TransportTensor
from fluxway.common.index_descriptor import ExplicitIndex
from .data_types import UserBatch, InputNodeOutput, DPFeatures


def hash_routing(tensor, mesh):
    val = tensor.flat_values if isinstance(tensor, TransportTensor) else tensor
    if len(val.shape) > 1:
        val = val[:, 0]
    return np.mod(val, mesh.size)


async def _sparse_unique(
    tt: TransportTensor,
) -> tuple[TransportTensor, TransportTensor]:
    """Deduplicate ids via sparse_embedding.async_unique (C++ high-perf).

    Returns (unique_tt, inverse_tt) where:
    - unique_tt carries deduplicated keys with global scatter positions
    - inverse_tt maps each original element to its scatter position in the
      Dense static assembly buffer
    """
    ids_np = tt.flat_values
    total_elements = tt.global_shape[0]  # type: ignore

    unique_dlm, inv_dlm, first_index_dlm = await se.async_unique(
        ids_np.ravel(), return_inverse=True, return_first_indices=True
    )

    backend = get_backend()
    unique_np = backend.from_dlpack(unique_dlm).astype(np.int64)
    first_np = backend.from_dlpack(first_index_dlm).astype(np.int32)

    if tt.origin_indices is not None:
        origin = tt.origin_indices_materialized
        if origin is None:
            raise ValueError("origin_indices_materialized is None")
        global_positions = np.take(origin, first_np)
    else:
        global_positions = first_np

    inv_1d = backend.from_dlpack(inv_dlm).astype(np.int32).reshape(-1)
    # inv_1d is already a local 0-based index into unique_np for this rank.
    # global_positions encodes where each unique embedding lands in the global
    # assembly buffer — that belongs on unique_tt.origin_indices only.
    # DenseSPMD uses inv to index into the local emb shard, so keep it local.
    adjusted_inv = inv_1d

    # -- 数值正确性验证: unique + inv 还原 --
    # 还原: original_ids[i] == unique_ids[inv[i]] 对所有 i 成立
    original_ids = ids_np.ravel()
    np.testing.assert_array_equal(
        unique_np[inv_1d],
        original_ids,
        err_msg="[InputSPMD] unique+inv 还原失败: unique[inv] != original",
    )

    unique_tt = TransportTensor(
        flat_values=unique_np,
        origin_indices=ExplicitIndex(global_positions),
        global_shape=(total_elements,),
    )
    inv_tt = TransportTensor(
        flat_values=adjusted_inv.reshape(ids_np.shape),
        global_shape=tt.global_shape,
    )
    return unique_tt, inv_tt


@spmd(
    resource={"num_cpus": 1, "memory_gb": 1},
    mesh_axis_names=("dp",),
)
class InputSPMD:
    rank: int
    world_size: int
    ctx: "Any"
    """Deduplicates features via sparse_embedding.async_unique and routes to embedding tables."""

    def __init__(
        self,
        name: str = "",
        mesh: "Any" = None,
        runtime_env: "Any" = None,
        resource: "Any" = None,
        **kwargs: "Any",
    ):
        pass

    @operator(
        name="process",
        input_sharding={"data": {"type": "static", "axis": ["dp"]}},
        output_sharding={
            "unique_user_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "fn": "hash_routing",
            },
            "unique_item_ids": {
                "type": "dynamic",
                "axis": ["dp"],
                "fn": "hash_routing",
            },
            "user_inv": {"type": "static", "axis": ["dp"]},
            "item_inv": {"type": "static", "axis": ["dp"]},
            "dp_data": {"type": "static", "axis": ["dp"]},
        },
    )
    async def process(self, data: UserBatch) -> InputNodeOutput:
        user_task = _sparse_unique(data.active.user_id)
        item_task = _sparse_unique(data.passive.target_item_id)

        (u_user, inv_user), (u_item, inv_item) = await asyncio.gather(
            user_task, item_task
        )

        print(
            f"[InputSPMD.process] rank={self.rank} "
            f"unique_users={u_user.flat_values} user_inv={inv_user.flat_values} "
            f"unique_items={u_item.flat_values} item_inv={inv_item.flat_values}"
        )

        return InputNodeOutput(
            unique_user_ids=u_user,
            user_inv=inv_user,
            unique_item_ids=u_item,
            item_inv=inv_item,
            dp_data=DPFeatures(
                labels=data.labels, dense_stats=data.passive.target_item_stats
            ),
        )
