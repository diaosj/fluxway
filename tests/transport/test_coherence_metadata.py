import pytest

from fluxway.common.coherence import CoherenceAction
from fluxway.common.ragged import encode_ragged_level_count, ragged_level_count
from fluxway.transport.coherence import (
    entry_coherence_action,
    entry_layout_state,
    entry_tensor_kind,
    pack_entry_state_flags,
)
from fluxway.transport.manifest import TensorEntryKind


def test_entry_state_flags_round_trip_multilevel_ragged() -> None:
    layout_state = encode_ragged_level_count(1 | 2, 3)

    flags = pack_entry_state_flags(
        layout_state,
        TensorEntryKind.TENSOR,
        CoherenceAction.BUS_RD,
    )

    decoded_layout = entry_layout_state(flags)
    assert decoded_layout == layout_state
    assert ragged_level_count(decoded_layout) == 3
    assert TensorEntryKind(entry_tensor_kind(flags)) is TensorEntryKind.TENSOR
    assert entry_coherence_action(flags) is CoherenceAction.BUS_RD


def test_entry_state_flags_reject_reserved_bits() -> None:
    with pytest.raises(ValueError, match="reserved bits"):
        entry_layout_state(1 << 24)


def test_ownership_grant_action_round_trip() -> None:
    flags = pack_entry_state_flags(
        0,
        TensorEntryKind.TENSOR,
        CoherenceAction.OWNERSHIP_GRANT,
    )

    assert entry_coherence_action(flags) is CoherenceAction.OWNERSHIP_GRANT


def test_ownership_cancel_action_round_trip() -> None:
    flags = pack_entry_state_flags(
        0,
        TensorEntryKind.NOT_APPLICABLE,
        CoherenceAction.OWNERSHIP_CANCEL,
    )

    assert entry_coherence_action(flags) is CoherenceAction.OWNERSHIP_CANCEL


@pytest.mark.parametrize("reserved_action", (3, 6))
def test_removed_wire_actions_are_reserved(reserved_action: int) -> None:
    with pytest.raises(ValueError):
        entry_coherence_action(reserved_action << 20)


def test_entry_state_flags_reject_unknown_coherence_action() -> None:
    flags = 15 << 20

    with pytest.raises(ValueError):
        entry_coherence_action(flags)
