RAGGED_META_LEVEL_SHIFT = 8
RAGGED_META_LEVEL_MASK = 0xFF


def encode_ragged_level_count(meta_state: int, level_count: int) -> int:
    if level_count <= 0:
        raise ValueError(f"ragged level_count must be positive, got {level_count}")
    if level_count > RAGGED_META_LEVEL_MASK:
        raise ValueError(
            f"ragged level_count={level_count} exceeds encodable maximum "
            f"{RAGGED_META_LEVEL_MASK}"
        )
    return meta_state | (level_count << RAGGED_META_LEVEL_SHIFT)


def ragged_level_count(meta_state: int) -> int:
    if not (meta_state & 1):
        return 0
    encoded = (meta_state >> RAGGED_META_LEVEL_SHIFT) & RAGGED_META_LEVEL_MASK
    if encoded <= 0:
        return 1
    return encoded
