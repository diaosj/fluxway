import numpy as np

from fluxway.common.coherence import CoherenceAction
from fluxway.spmd.worker.data_plane import build_coherence_frame
from fluxway.transport.coherence import (
    entry_coherence_action,
    entry_layout_state,
    entry_tensor_kind,
)
from fluxway.transport.manifest import (
    ENTRY_AUX,
    ENTRY_STATE_FLAGS,
    ENTRY_WIDTH,
    FRAME_FLAGS,
    FrameFlag,
    TensorEntryKind,
)


def test_compressed_entry_width_is_nine_i32_values() -> None:
    assert ENTRY_WIDTH == 9
    assert ENTRY_WIDTH * np.dtype(np.int32).itemsize == 36


def test_coherence_frame_uses_flat_entry_without_payload() -> None:
    header = build_coherence_frame(
        step=12,
        sender_spmd_code=3,
        sender_rank=1,
        sender_world_size=4,
        epoch=12,
        action=CoherenceAction.GRANT_RETURN,
        route_codes=(17, 18),
    )

    header.validate()
    state_flags = int(header.entry_i32[0, ENTRY_STATE_FLAGS])
    assert FrameFlag(int(header.frame_i32[FRAME_FLAGS])) is FrameFlag.COHERENCE
    assert entry_layout_state(state_flags) == 0
    assert (
        TensorEntryKind(entry_tensor_kind(state_flags))
        is TensorEntryKind.NOT_APPLICABLE
    )
    assert entry_coherence_action(state_flags) is CoherenceAction.GRANT_RETURN
    assert header.entry_i32.shape == (2, ENTRY_WIDTH)
    assert int(header.entry_i32[0, ENTRY_AUX]) == 12
    assert header.payload_i32.shape[0] == 0
