from __future__ import annotations

from fluxway.common.coherence import CoherenceAction
from fluxway.common.ragged import RAGGED_META_LEVEL_MASK, RAGGED_META_LEVEL_SHIFT


LAYOUT_FLAG_MASK = 0xFF
TENSOR_KIND_SHIFT = 16
TENSOR_KIND_MASK = 0x3
COHERENCE_ACTION_SHIFT = 20
COHERENCE_ACTION_MASK = 0xF
RESERVED_MASK = 0xFF000000
I32_MAX = (1 << 31) - 1


def pack_entry_state_flags(
    layout_state: int,
    tensor_kind: int,
    coherence_action: CoherenceAction,
) -> int:
    _require_i32(layout_state, "layout_state")
    allowed_layout_mask = LAYOUT_FLAG_MASK | (
        RAGGED_META_LEVEL_MASK << RAGGED_META_LEVEL_SHIFT
    )
    if layout_state & ~allowed_layout_mask:
        raise ValueError(f"layout_state contains unsupported bits: {layout_state:#x}.")
    _require_enum_code(tensor_kind, "tensor_kind", 3)
    if not isinstance(coherence_action, CoherenceAction):
        raise TypeError(
            "coherence_action must be CoherenceAction, "
            f"got {type(coherence_action).__name__}."
        )
    value = (
        layout_state
        | (tensor_kind << TENSOR_KIND_SHIFT)
        | (int(coherence_action) << COHERENCE_ACTION_SHIFT)
    )
    _require_i32(value, "entry_state_flags")
    return value


def entry_layout_state(state_flags: int) -> int:
    _validate_state_flags(state_flags)
    return state_flags & (
        LAYOUT_FLAG_MASK | (RAGGED_META_LEVEL_MASK << RAGGED_META_LEVEL_SHIFT)
    )


def entry_tensor_kind(state_flags: int) -> int:
    _validate_state_flags(state_flags)
    return (state_flags >> TENSOR_KIND_SHIFT) & TENSOR_KIND_MASK


def entry_coherence_action(state_flags: int) -> CoherenceAction:
    _validate_state_flags(state_flags)
    return CoherenceAction(
        (state_flags >> COHERENCE_ACTION_SHIFT) & COHERENCE_ACTION_MASK
    )


def _validate_state_flags(state_flags: int) -> None:
    _require_i32(state_flags, "state_flags")
    if state_flags & RESERVED_MASK:
        raise ValueError(f"state_flags has reserved bits set: {state_flags:#x}.")
    _require_enum_code(
        (state_flags >> TENSOR_KIND_SHIFT) & TENSOR_KIND_MASK,
        "tensor_kind",
        3,
    )
    CoherenceAction((state_flags >> COHERENCE_ACTION_SHIFT) & COHERENCE_ACTION_MASK)


def _require_i32(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0 or value > I32_MAX:
        raise OverflowError(f"{name}={value} is outside [0, {I32_MAX}].")


def _require_enum_code(value: int, name: str, maximum: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}.")
    if value < 0 or value > maximum:
        raise ValueError(f"{name}={value} is outside [0, {maximum}].")
