"""Flat transport manifest ABI for data-plane frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import struct
from typing import Any

import numpy as np

from fluxway.transport.kernels import FRAME_TABLE_KERNEL, FrameTableCapacity


INT32_MAX = np.iinfo(np.int32).max
INT64_MAX = np.iinfo(np.int64).max
INT32_NBYTES = np.dtype(np.int32).itemsize
INT64_NBYTES = np.dtype(np.int64).itemsize
PROTOCOL_VERSION = 1


class EmptyKind(IntEnum):
    DATA_EMPTY_ALLOWED = 0
    PROTOCOL_EMPTY = 1
    DATA_REQUIRED = 2


class PayloadRole(IntEnum):
    VALUES = 0
    RAGGED_LENGTHS = 1
    ORIGIN = 2


class PayloadDType(IntEnum):
    INT32 = 1
    INT64 = 2
    UINT32 = 3
    UINT64 = 4
    FLOAT16 = 5
    FLOAT32 = 6
    FLOAT64 = 7
    BFLOAT16 = 8


class SenderEmissionMode(IntEnum):
    STATIC_PRODUCERS_ONLY = 0
    ALL_SENDERS_WITH_PROTOCOL_EMPTY = 1


FRAME_VERSION = 0
FRAME_STEP = 1
FRAME_SENDER_SPMD_CODE = 2
FRAME_SENDER_RANK = 3
FRAME_SENDER_WORLD_SIZE = 4
FRAME_EDGE_COUNT = 5
FRAME_PAYLOAD_COUNT = 6
FRAME_FLAGS = 7
FRAME_WIDTH = 8

EDGE_TUUID = 0
EDGE_META_STATE = 1
EDGE_EMPTY_KIND = 2
EDGE_PAYLOAD_BEGIN = 3
EDGE_PAYLOAD_COUNT = 4
EDGE_MIN_ROWS = 5
EDGE_EXPECTED_COUNT = 6
EDGE_COMPACT_BEGIN = 7
EDGE_COMPACT_COUNT = 8
EDGE_GLOBAL_SHAPE_BEGIN = 9
EDGE_GLOBAL_SHAPE_COUNT = 10
EDGE_WIDTH = 11

PAYLOAD_ROLE = 0
PAYLOAD_DTYPE = 1
PAYLOAD_NDIM = 2
PAYLOAD_SHAPE_BEGIN = 3
PAYLOAD_SHAPE_COUNT = 4
PAYLOAD_VALID_COUNT = 5
PAYLOAD_SEGMENT_ARG_INDEX = 6
PAYLOAD_FLAGS = 7
PAYLOAD_WIDTH = 8

COMPACT_PARAM_COUNT = 4


@dataclass(frozen=True, slots=True)
class TransportPlanManifest:
    expected_senders_per_dst: tuple[int, ...]
    sender_has_data_per_dst: tuple[bool, ...]
    emission_mode: SenderEmissionMode

    def __post_init__(self) -> None:
        if len(self.expected_senders_per_dst) != len(self.sender_has_data_per_dst):
            raise ValueError(
                "TransportPlanManifest expected_senders_per_dst and "
                "sender_has_data_per_dst must have the same length."
            )
        for value in self.expected_senders_per_dst:
            _as_i32(value, "expected_senders_per_dst entry")
        object.__setattr__(
            self, "emission_mode", SenderEmissionMode(int(self.emission_mode))
        )

    @property
    def target_world_size(self) -> int:
        return len(self.expected_senders_per_dst)

    def expected_count_for(self, dst_rank: int) -> int:
        if dst_rank < 0 or dst_rank >= self.target_world_size:
            raise IndexError(
                f"dst_rank={dst_rank} is outside manifest target world size {self.target_world_size}."
            )
        return self.expected_senders_per_dst[dst_rank]

    def should_emit_to(self, dst_rank: int) -> bool:
        if dst_rank < 0 or dst_rank >= self.target_world_size:
            raise IndexError(
                f"dst_rank={dst_rank} is outside manifest target world size {self.target_world_size}."
            )
        if self.expected_senders_per_dst[dst_rank] <= 0:
            return False
        if self.emission_mode is SenderEmissionMode.ALL_SENDERS_WITH_PROTOCOL_EMPTY:
            return True
        return self.sender_has_data_per_dst[dst_rank]


def _as_integral(value: int, field: str) -> int:
    if isinstance(value, bool) or isinstance(value, np.bool_):
        raise TypeError(f"{field} must be integer scalar, got bool.")
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be integer scalar, got {type(value).__name__}.")
    return value


def _as_i32(value: int, field: str) -> int:
    int_value = _as_integral(value, field)
    if int_value < 0 or int_value > INT32_MAX:
        raise OverflowError(
            f"{field}={int_value} is outside int32 range [0, {INT32_MAX}]."
        )
    return int_value


def _as_i64(value: int, field: str) -> int:
    int_value = _as_integral(value, field)
    if int_value < 0 or int_value > INT64_MAX:
        raise OverflowError(
            f"{field}={int_value} is outside int64 range [0, {INT64_MAX}]."
        )
    return int_value


def dtype_code(dtype: Any) -> int:
    dt = np.dtype(dtype)
    if dt == np.dtype(np.int32):
        return int(PayloadDType.INT32)
    if dt == np.dtype(np.int64):
        return int(PayloadDType.INT64)
    if dt == np.dtype(np.uint32):
        return int(PayloadDType.UINT32)
    if dt == np.dtype(np.uint64):
        return int(PayloadDType.UINT64)
    if dt == np.dtype(np.float16):
        return int(PayloadDType.FLOAT16)
    if dt == np.dtype(np.float32):
        return int(PayloadDType.FLOAT32)
    if dt == np.dtype(np.float64):
        return int(PayloadDType.FLOAT64)
    if str(dt) == "bfloat16":
        return int(PayloadDType.BFLOAT16)
    raise TypeError(f"Unsupported transport payload dtype {dt}.")


def dtype_from_code(code: int) -> np.dtype:
    payload_dtype = PayloadDType(code)
    if payload_dtype is PayloadDType.INT32:
        return np.dtype(np.int32)
    if payload_dtype is PayloadDType.INT64:
        return np.dtype(np.int64)
    if payload_dtype is PayloadDType.UINT32:
        return np.dtype(np.uint32)
    if payload_dtype is PayloadDType.UINT64:
        return np.dtype(np.uint64)
    if payload_dtype is PayloadDType.FLOAT16:
        return np.dtype(np.float16)
    if payload_dtype is PayloadDType.FLOAT32:
        return np.dtype(np.float32)
    if payload_dtype is PayloadDType.FLOAT64:
        return np.dtype(np.float64)
    if payload_dtype is PayloadDType.BFLOAT16:
        try:
            return np.dtype("bfloat16")
        except TypeError as exc:
            raise TypeError(
                "bfloat16 dtype is not available in this NumPy runtime."
            ) from exc
    raise ValueError(f"Unsupported transport payload dtype code {code}.")


def _require_nonnegative_i32_table(name: str, table: np.ndarray) -> None:
    if table.size and np.any(table < 0):
        first = np.argwhere(table < 0)[0]
        raise ValueError(
            f"{name} contains negative int32 metadata at index {tuple(int(x) for x in first)}."
        )


def _validate_enum(enum_type: type[IntEnum], value: int, field: str) -> None:
    try:
        enum_type(value)
    except ValueError as exc:
        raise ValueError(
            f"{field}={value} is not a valid {enum_type.__name__}."
        ) from exc


def _require_host_metadata_array(
    name: str,
    table: Any,
    *,
    dtype: Any,
    ndim: int,
    width: int | None = None,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    if not isinstance(table, np.ndarray):
        raise TypeError(
            f"{name} must be a host numpy metadata array, got {type(table).__name__}."
        )
    if hasattr(table, "__cuda_array_interface__"):
        raise TypeError(f"{name} must be host metadata, not a CUDA array.")
    expected_dtype = np.dtype(dtype)
    if table.dtype != expected_dtype:
        raise ValueError(
            f"{name} must have dtype={expected_dtype}, got dtype={table.dtype}."
        )
    if table.ndim != ndim:
        raise ValueError(f"{name} must have ndim={ndim}, got ndim={table.ndim}.")
    if width is not None and (table.ndim != 2 or table.shape[1] != width):
        raise ValueError(f"{name} width must be {width}, got shape={table.shape}.")
    if shape is not None and table.shape != shape:
        raise ValueError(f"{name} must have shape={shape}, got shape={table.shape}.")
    return table


@dataclass(frozen=True, slots=True)
class TransportFrameHeader:
    frame_i32: np.ndarray
    edge_i32: np.ndarray
    payload_i32: np.ndarray
    shape_i32: np.ndarray
    global_shape_i64: np.ndarray
    compact_i64: np.ndarray

    def validate(self) -> None:
        _require_host_metadata_array(
            "frame_i32", self.frame_i32, dtype=np.int32, ndim=1, shape=(FRAME_WIDTH,)
        )
        _require_host_metadata_array(
            "edge_i32", self.edge_i32, dtype=np.int32, ndim=2, width=EDGE_WIDTH
        )
        _require_host_metadata_array(
            "payload_i32",
            self.payload_i32,
            dtype=np.int32,
            ndim=2,
            width=PAYLOAD_WIDTH,
        )
        _require_host_metadata_array(
            "shape_i32", self.shape_i32, dtype=np.int32, ndim=1
        )
        _require_host_metadata_array(
            "global_shape_i64", self.global_shape_i64, dtype=np.int64, ndim=1
        )
        _require_host_metadata_array(
            "compact_i64", self.compact_i64, dtype=np.int64, ndim=1
        )
        _require_nonnegative_i32_table("frame_i32", self.frame_i32)
        _require_nonnegative_i32_table("edge_i32", self.edge_i32)
        _require_nonnegative_i32_table("payload_i32", self.payload_i32)
        _require_nonnegative_i32_table("shape_i32", self.shape_i32)
        if int(self.frame_i32[FRAME_VERSION]) != PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported transport frame version {int(self.frame_i32[FRAME_VERSION])}; "
                f"expected {PROTOCOL_VERSION}."
            )
        edge_count = int(self.frame_i32[FRAME_EDGE_COUNT])
        payload_count = int(self.frame_i32[FRAME_PAYLOAD_COUNT])
        if edge_count != self.edge_i32.shape[0]:
            raise ValueError(
                f"frame edge_count={edge_count} does not match edge_i32 rows={self.edge_i32.shape[0]}."
            )
        if payload_count != self.payload_i32.shape[0]:
            raise ValueError(
                f"frame payload_count={payload_count} does not match payload_i32 rows={self.payload_i32.shape[0]}."
            )
        for row_idx, row in enumerate(self.edge_i32):
            _validate_enum(
                EmptyKind, int(row[EDGE_EMPTY_KIND]), f"edge row {row_idx} empty_kind"
            )
            payload_begin = int(row[EDGE_PAYLOAD_BEGIN])
            payload_count_row = int(row[EDGE_PAYLOAD_COUNT])
            compact_begin = int(row[EDGE_COMPACT_BEGIN])
            compact_count = int(row[EDGE_COMPACT_COUNT])
            global_shape_begin = int(row[EDGE_GLOBAL_SHAPE_BEGIN])
            global_shape_count = int(row[EDGE_GLOBAL_SHAPE_COUNT])
            if payload_begin + payload_count_row > payload_count:
                raise ValueError(
                    f"edge row {row_idx} payload range [{payload_begin}, "
                    f"{payload_begin + payload_count_row}) exceeds payload_count={payload_count}."
                )
            if compact_begin + compact_count > self.compact_i64.shape[0]:
                raise ValueError(
                    f"edge row {row_idx} compact range [{compact_begin}, "
                    f"{compact_begin + compact_count}) exceeds compact_i64 size={self.compact_i64.shape[0]}."
                )
            if global_shape_begin + global_shape_count > self.global_shape_i64.shape[0]:
                raise ValueError(
                    f"edge row {row_idx} global shape range [{global_shape_begin}, "
                    f"{global_shape_begin + global_shape_count}) exceeds "
                    f"global_shape_i64 size={self.global_shape_i64.shape[0]}."
                )
        for row_idx, row in enumerate(self.payload_i32):
            _validate_enum(
                PayloadRole, int(row[PAYLOAD_ROLE]), f"payload row {row_idx} role"
            )
            _validate_enum(
                PayloadDType, int(row[PAYLOAD_DTYPE]), f"payload row {row_idx} dtype"
            )
            shape_begin = int(row[PAYLOAD_SHAPE_BEGIN])
            shape_count = int(row[PAYLOAD_SHAPE_COUNT])
            if shape_begin + shape_count > self.shape_i32.shape[0]:
                raise ValueError(
                    f"payload row {row_idx} shape range [{shape_begin}, "
                    f"{shape_begin + shape_count}) exceeds shape_i32 size={self.shape_i32.shape[0]}."
                )

    def to_frame(self, payload_segments: tuple[Any, ...]) -> TransportFrame:
        frame = TransportFrame(
            frame_i32=self.frame_i32,
            edge_i32=self.edge_i32,
            payload_i32=self.payload_i32,
            shape_i32=self.shape_i32,
            global_shape_i64=self.global_shape_i64,
            compact_i64=self.compact_i64,
            payload_segments=payload_segments,
        )
        frame.validate()
        return frame


@dataclass(frozen=True)
class TransportFrame:
    frame_i32: np.ndarray
    edge_i32: np.ndarray
    payload_i32: np.ndarray
    shape_i32: np.ndarray
    global_shape_i64: np.ndarray
    compact_i64: np.ndarray
    payload_segments: tuple[Any, ...]

    @property
    def header(self) -> TransportFrameHeader:
        return TransportFrameHeader(
            frame_i32=self.frame_i32,
            edge_i32=self.edge_i32,
            payload_i32=self.payload_i32,
            shape_i32=self.shape_i32,
            global_shape_i64=self.global_shape_i64,
            compact_i64=self.compact_i64,
        )

    def validate(self) -> None:
        self.header.validate()
        for row_idx, row in enumerate(self.payload_i32):
            segment_idx = int(row[PAYLOAD_SEGMENT_ARG_INDEX])
            if segment_idx >= len(self.payload_segments):
                raise ValueError(
                    f"payload row {row_idx} segment index {segment_idx} exceeds "
                    f"payload segment count {len(self.payload_segments)}."
                )
            segment = self.payload_segments[segment_idx]
            if not hasattr(segment, "dtype"):
                raise TypeError(
                    f"payload segment {segment_idx} must expose dtype metadata, got "
                    f"{type(segment).__name__}."
                )
            actual_dtype_code = dtype_code(segment.dtype)
            declared_dtype_code = int(row[PAYLOAD_DTYPE])
            if actual_dtype_code != declared_dtype_code:
                raise ValueError(
                    f"payload row {row_idx} dtype_code={declared_dtype_code} does not "
                    f"match segment dtype_code={actual_dtype_code}."
                )


class TransportFrameBuilder:
    _EDGE_STRUCT = struct.Struct("=" + "i" * EDGE_WIDTH)
    _PAYLOAD_STRUCT = struct.Struct("=" + "i" * PAYLOAD_WIDTH)
    _I32_SCALAR_STRUCT = struct.Struct("=i")

    def __init__(
        self,
        step: int,
        sender_spmd_code: int,
        sender_rank: int,
        sender_world_size: int,
        flags: int = 0,
        capacity: FrameTableCapacity | None = None,
    ):
        self._step = _as_i32(step, "step")
        self._sender_spmd_code = _as_i32(sender_spmd_code, "sender_spmd_code")
        self._sender_rank = _as_i32(sender_rank, "sender_rank")
        self._sender_world_size = _as_i32(sender_world_size, "sender_world_size")
        self._flags = _as_i32(flags, "flags")
        self._capacity = capacity or FrameTableCapacity()
        self._edge_i32_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.edge_count * EDGE_WIDTH,
            INT32_NBYTES,
            "edge_i32",
        )
        self._payload_i32_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.payload_count * PAYLOAD_WIDTH,
            INT32_NBYTES,
            "payload_i32",
        )
        self._shape_i32_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.shape_count, INT32_NBYTES, "shape_i32"
        )
        self._global_shape_i64_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.global_shape_count,
            INT64_NBYTES,
            "global_shape_i64",
        )
        self._compact_i64_buffer = FRAME_TABLE_KERNEL.empty_byte_buffer(
            self._capacity.compact_count,
            INT64_NBYTES,
            "compact_i64",
        )
        self._segments = [None] * self._capacity.segment_count
        self._edge_capacity = self._capacity.edge_count
        self._payload_capacity = self._capacity.payload_count
        self._shape_capacity = self._capacity.shape_count
        self._global_shape_capacity = self._capacity.global_shape_count
        self._compact_capacity = self._capacity.compact_count
        self._segment_capacity = self._capacity.segment_count
        self._edge_count = 0
        self._payload_count = 0
        self._shape_count = 0
        self._global_shape_count = 0
        self._compact_count = 0
        self._segment_count = 0

    def add_edge(
        self,
        *,
        tuuid: int,
        meta_state: int,
        empty_kind: EmptyKind,
        min_rows: int,
        expected_count: int,
        compact_params: list[int] | tuple[int, ...] | None = None,
        global_shape: tuple[int, ...] | None = None,
    ) -> int:
        payload_begin = self._payload_count
        compact_begin = self.add_compact_params(compact_params)
        compact_count = 0 if compact_params is None else COMPACT_PARAM_COUNT
        global_shape_begin = self._global_shape_count
        global_shape_count = 0
        if global_shape is not None:
            global_shape_values = tuple(
                _as_i64(value, f"global_shape[{i}]")
                for i, value in enumerate(global_shape)
            )
            self._ensure_global_shape_capacity(
                self._global_shape_count + len(global_shape_values)
            )
            struct.pack_into(
                "=" + "q" * len(global_shape_values),
                self._global_shape_i64_buffer,
                self._global_shape_count * INT64_NBYTES,
                *global_shape_values,
            )
            global_shape_count = len(global_shape)
            self._global_shape_count += global_shape_count
        edge_index = self._edge_count
        self._ensure_edge_capacity(edge_index + 1)
        self._EDGE_STRUCT.pack_into(
            self._edge_i32_buffer,
            edge_index * self._EDGE_STRUCT.size,
            _as_i32(tuuid, "tuuid"),
            _as_i32(meta_state, "meta_state"),
            _as_i32(int(empty_kind), "empty_kind"),
            _as_i32(payload_begin, "payload_begin"),
            0,
            _as_i32(min_rows, "min_rows"),
            _as_i32(expected_count, "expected_count"),
            _as_i32(compact_begin, "compact_begin"),
            _as_i32(compact_count, "compact_count"),
            _as_i32(global_shape_begin, "global_shape_begin"),
            _as_i32(global_shape_count, "global_shape_count"),
        )
        self._edge_count += 1
        return edge_index

    def add_payload(
        self, edge_index: int, role: PayloadRole, segment: Any, valid_count: int
    ) -> None:
        if edge_index < 0 or edge_index >= self._edge_count:
            raise IndexError(f"edge_index {edge_index} is out of range.")
        if not hasattr(segment, "dtype"):
            raise TypeError(
                f"payload segment must expose dtype metadata, got {type(segment).__name__}."
            )
        shape = tuple(int(x) for x in getattr(segment, "shape", ()))
        shape_begin = self._shape_count
        shape_values = tuple(
            _as_i32(value, f"payload shape dim {dim}")
            for dim, value in enumerate(shape)
        )
        self._ensure_shape_capacity(self._shape_count + len(shape_values))
        struct.pack_into(
            "=" + "i" * len(shape_values),
            self._shape_i32_buffer,
            self._shape_count * INT32_NBYTES,
            *shape_values,
        )
        self._shape_count += len(shape_values)

        segment_idx = self._segment_count
        self._ensure_segment_capacity(segment_idx + 1)
        self._segments[segment_idx] = segment
        self._segment_count += 1

        payload_index = self._payload_count
        self._ensure_payload_capacity(payload_index + 1)
        self._PAYLOAD_STRUCT.pack_into(
            self._payload_i32_buffer,
            payload_index * self._PAYLOAD_STRUCT.size,
            _as_i32(int(role), "payload role"),
            _as_i32(dtype_code(segment.dtype), "dtype_code"),
            _as_i32(len(shape), "ndim"),
            _as_i32(shape_begin, "shape_begin"),
            _as_i32(len(shape), "shape_count"),
            _as_i32(valid_count, "valid_count"),
            _as_i32(segment_idx, "segment_arg_index"),
            0,
        )
        self._payload_count += 1
        payload_count_byte = (
            edge_index * EDGE_WIDTH + EDGE_PAYLOAD_COUNT
        ) * INT32_NBYTES
        payload_count = (
            self._I32_SCALAR_STRUCT.unpack_from(
                self._edge_i32_buffer, payload_count_byte
            )[0]
            + 1
        )
        self._I32_SCALAR_STRUCT.pack_into(
            self._edge_i32_buffer, payload_count_byte, payload_count
        )
        _as_i32(payload_count, "payload_count")

    def add_compact_params(self, params: list[int] | tuple[int, ...] | None) -> int:
        if params is None:
            return 0
        if len(params) != COMPACT_PARAM_COUNT:
            raise ValueError(
                f"compact params must have {COMPACT_PARAM_COUNT} entries, got {len(params)}."
            )
        begin = self._compact_count
        compact_values = tuple(
            _as_i64(value, f"compact[{i}]") for i, value in enumerate(params)
        )
        self._ensure_compact_capacity(self._compact_count + COMPACT_PARAM_COUNT)
        struct.pack_into(
            "=" + "q" * len(compact_values),
            self._compact_i64_buffer,
            self._compact_count * INT64_NBYTES,
            *compact_values,
        )
        self._compact_count += COMPACT_PARAM_COUNT
        return begin

    def build(self) -> TransportFrame:
        edge_count = _as_i32(self._edge_count, "edge_count")
        payload_count = _as_i32(self._payload_count, "payload_count")
        frame_i32 = np.array(
            [
                PROTOCOL_VERSION,
                self._step,
                self._sender_spmd_code,
                self._sender_rank,
                self._sender_world_size,
                edge_count,
                payload_count,
                self._flags,
            ],
            dtype=np.int32,
        )
        edge_i32 = FRAME_TABLE_KERNEL.i32_table_from_byte_buffer(
            self._edge_i32_buffer, edge_count, EDGE_WIDTH, "edge_i32"
        )
        payload_i32 = FRAME_TABLE_KERNEL.i32_table_from_byte_buffer(
            self._payload_i32_buffer, payload_count, PAYLOAD_WIDTH, "payload_i32"
        )
        shape_i32 = FRAME_TABLE_KERNEL.i32_vector_from_byte_buffer(
            self._shape_i32_buffer, self._shape_count, "shape_i32"
        )
        global_shape_i64 = FRAME_TABLE_KERNEL.i64_vector_from_byte_buffer(
            self._global_shape_i64_buffer,
            self._global_shape_count,
            "global_shape_i64",
        )
        compact_i64 = FRAME_TABLE_KERNEL.i64_vector_from_byte_buffer(
            self._compact_i64_buffer, self._compact_count, "compact_i64"
        )
        frame = TransportFrame(
            frame_i32=frame_i32,
            edge_i32=edge_i32,
            payload_i32=payload_i32,
            shape_i32=shape_i32,
            global_shape_i64=global_shape_i64,
            compact_i64=compact_i64,
            payload_segments=tuple(self._segments[: self._segment_count]),
        )
        frame.validate()
        return frame

    def _ensure_edge_capacity(self, needed: int) -> None:
        if needed <= self._edge_capacity:
            return
        next_capacity = self._next_capacity(self._edge_capacity, needed)
        self._edge_i32_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._edge_i32_buffer,
            next_capacity * EDGE_WIDTH,
            INT32_NBYTES,
            "edge_i32",
        )
        self._edge_capacity = next_capacity

    def _ensure_payload_capacity(self, needed: int) -> None:
        if needed <= self._payload_capacity:
            return
        next_capacity = self._next_capacity(self._payload_capacity, needed)
        self._payload_i32_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._payload_i32_buffer,
            next_capacity * PAYLOAD_WIDTH,
            INT32_NBYTES,
            "payload_i32",
        )
        self._payload_capacity = next_capacity

    def _ensure_shape_capacity(self, needed: int) -> None:
        if needed <= self._shape_capacity:
            return
        next_capacity = self._next_capacity(self._shape_capacity, needed)
        self._shape_i32_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._shape_i32_buffer,
            next_capacity,
            INT32_NBYTES,
            "shape_i32",
        )
        self._shape_capacity = next_capacity

    def _ensure_global_shape_capacity(self, needed: int) -> None:
        if needed <= self._global_shape_capacity:
            return
        next_capacity = self._next_capacity(self._global_shape_capacity, needed)
        self._global_shape_i64_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._global_shape_i64_buffer,
            next_capacity,
            INT64_NBYTES,
            "global_shape_i64",
        )
        self._global_shape_capacity = next_capacity

    def _ensure_compact_capacity(self, needed: int) -> None:
        if needed <= self._compact_capacity:
            return
        next_capacity = self._next_capacity(self._compact_capacity, needed)
        self._compact_i64_buffer = FRAME_TABLE_KERNEL.grow_byte_buffer(
            self._compact_i64_buffer,
            next_capacity,
            INT64_NBYTES,
            "compact_i64",
        )
        self._compact_capacity = next_capacity

    def _ensure_segment_capacity(self, needed: int) -> None:
        if needed <= self._segment_capacity:
            return
        next_capacity = self._next_capacity(self._segment_capacity, needed)
        self._segments.extend([None] * (next_capacity - self._segment_capacity))
        self._segment_capacity = next_capacity

    def _next_capacity(self, current: int, needed: int) -> int:
        if needed <= current:
            return current
        if current == 0:
            return needed
        return max(needed, current * 2)


def metadata_nbytes(frame: TransportFrame) -> int:
    return (
        frame.frame_i32.nbytes
        + frame.edge_i32.nbytes
        + frame.payload_i32.nbytes
        + frame.shape_i32.nbytes
        + frame.global_shape_i64.nbytes
        + frame.compact_i64.nbytes
    )


def legacy_metadata_nbytes_as_i64(frame: TransportFrame) -> int:
    return (
        (
            frame.frame_i32.size
            + frame.edge_i32.size
            + frame.payload_i32.size
            + frame.shape_i32.size
        )
        * np.dtype(np.int64).itemsize
        + frame.global_shape_i64.nbytes
        + frame.compact_i64.nbytes
    )
