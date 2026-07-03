import pickle

import numpy as np

from fluxway.common.index_descriptor import ExplicitIndex
from fluxway.common.tensor_backend import use_backend
from fluxway.common.types.tensor import SymmetryReceipt, TransportTensor


def test_transport_tensor_pickle_preserves_receipt_for_ray_dataplane_cache():
    with use_backend("hybrid"):
        tensor = TransportTensor(
            np.array([10, 20, 30], dtype=np.int64),
            origin_indices=ExplicitIndex(np.array([2, 1, 0], dtype=np.int64)),
            global_shape=(64,),
            receipt=SymmetryReceipt(4, np.array([3, 2, 1], dtype=np.int32)),
        )

        restored = pickle.loads(pickle.dumps(tensor))

        assert restored.receipt is not None
        assert restored.receipt.world_size == 4
        assert restored.receipt.sender_rank_tags.tolist() == [3, 2, 1]
