"""Native-facing communication table kernels."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class FrameTableCapacity:
    edge_count: int = 0
    payload_count: int = 0
    shape_count: int = 0
    global_shape_count: int = 0
    compact_count: int = 0
    segment_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "edge_count",
            "payload_count",
            "shape_count",
            "global_shape_count",
            "compact_count",
            "segment_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative int, got {value!r}.")


class FrameTableKernel:
    def empty_byte_buffer(self, slots: int, item_nbytes: int, name: str) -> bytearray:
        self._require_nonnegative(slots, name)
        self._require_positive(item_nbytes, f"{name}_item_nbytes")
        return bytearray(slots * item_nbytes)

    def grow_byte_buffer(
        self, buffer: bytearray, slots: int, item_nbytes: int, name: str
    ) -> bytearray:
        if not isinstance(buffer, bytearray):
            raise TypeError(f"{name} must be bytearray.")
        self._require_nonnegative(slots, name)
        self._require_positive(item_nbytes, f"{name}_item_nbytes")
        byte_count = slots * item_nbytes
        if byte_count <= len(buffer):
            return buffer
        grown = bytearray(byte_count)
        grown[: len(buffer)] = buffer
        return grown

    def i32_table_from_byte_buffer(
        self, buffer: bytearray, rows: int, width: int, name: str
    ) -> np.ndarray:
        self._require_nonnegative(rows, name)
        self._require_positive(width, f"{name}_width")
        slots = rows * width
        if slots * np.dtype(np.int32).itemsize > len(buffer):
            raise ValueError(
                f"{name} requested {slots} int32 slots from buffer with "
                f"{len(buffer) // np.dtype(np.int32).itemsize} slots."
            )
        return (
            np.frombuffer(buffer, dtype=np.int32, count=slots)
            .copy()
            .reshape(rows, width)
        )

    def i32_vector_from_byte_buffer(
        self, buffer: bytearray, size: int, name: str
    ) -> np.ndarray:
        self._require_nonnegative(size, name)
        if size * np.dtype(np.int32).itemsize > len(buffer):
            raise ValueError(
                f"{name} requested {size} int32 slots from buffer with "
                f"{len(buffer) // np.dtype(np.int32).itemsize} slots."
            )
        return np.frombuffer(buffer, dtype=np.int32, count=size).copy()

    def i64_vector_from_byte_buffer(
        self, buffer: bytearray, size: int, name: str
    ) -> np.ndarray:
        self._require_nonnegative(size, name)
        if size * np.dtype(np.int64).itemsize > len(buffer):
            raise ValueError(
                f"{name} requested {size} int64 slots from buffer with "
                f"{len(buffer) // np.dtype(np.int64).itemsize} slots."
            )
        return np.frombuffer(buffer, dtype=np.int64, count=size).copy()

    def _require_nonnegative(self, value: int, name: str) -> None:
        if not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative int, got {value!r}.")

    def _require_positive(self, value: int, name: str) -> None:
        if not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}.")


FRAME_TABLE_KERNEL = FrameTableKernel()
