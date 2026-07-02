"""Flat transport manifest ABI for data-plane frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np


INT32_MAX = np.iinfo(np.int32).max
INT64_MAX = np.iinfo(np.int64).max
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


@dataclass(frozen=True)
class TransportFrame:
    frame_i32: np.ndarray
    edge_i32: np.ndarray
    payload_i32: np.ndarray
    shape_i32: np.ndarray
    global_shape_i64: np.ndarray
    compact_i64: np.ndarray
    payload_segments: tuple[Any, ...]

    def validate(self) -> None:
        if self.frame_i32.dtype != np.int32 or self.frame_i32.shape != (FRAME_WIDTH,):
            raise ValueError(
                f"frame_i32 must have dtype=int32 shape=({FRAME_WIDTH},), "
                f"got dtype={self.frame_i32.dtype} shape={self.frame_i32.shape}."
            )
        if self.edge_i32.dtype != np.int32 or self.edge_i32.ndim != 2:
            raise ValueError("edge_i32 must be a 2-D int32 array.")
        if self.edge_i32.shape[1] != EDGE_WIDTH:
            raise ValueError(
                f"edge_i32 width must be {EDGE_WIDTH}, got {self.edge_i32.shape[1]}."
            )
        if self.payload_i32.dtype != np.int32 or self.payload_i32.ndim != 2:
            raise ValueError("payload_i32 must be a 2-D int32 array.")
        if self.payload_i32.shape[1] != PAYLOAD_WIDTH:
            raise ValueError(
                f"payload_i32 width must be {PAYLOAD_WIDTH}, got {self.payload_i32.shape[1]}."
            )
        if self.shape_i32.dtype != np.int32 or self.shape_i32.ndim != 1:
            raise ValueError("shape_i32 must be a 1-D int32 array.")
        if self.global_shape_i64.dtype != np.int64 or self.global_shape_i64.ndim != 1:
            raise ValueError("global_shape_i64 must be a 1-D int64 array.")
        if self.compact_i64.dtype != np.int64 or self.compact_i64.ndim != 1:
            raise ValueError("compact_i64 must be a 1-D int64 array.")
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
            segment_idx = int(row[PAYLOAD_SEGMENT_ARG_INDEX])
            if shape_begin + shape_count > self.shape_i32.shape[0]:
                raise ValueError(
                    f"payload row {row_idx} shape range [{shape_begin}, "
                    f"{shape_begin + shape_count}) exceeds shape_i32 size={self.shape_i32.shape[0]}."
                )
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
    def __init__(
        self,
        step: int,
        sender_spmd_code: int,
        sender_rank: int,
        sender_world_size: int,
        flags: int = 0,
    ):
        self._step = _as_i32(step, "step")
        self._sender_spmd_code = _as_i32(sender_spmd_code, "sender_spmd_code")
        self._sender_rank = _as_i32(sender_rank, "sender_rank")
        self._sender_world_size = _as_i32(sender_world_size, "sender_world_size")
        self._flags = _as_i32(flags, "flags")
        self._edges: list[list[int]] = []
        self._payloads: list[list[int]] = []
        self._shape: list[int] = []
        self._global_shape: list[int] = []
        self._compact: list[int] = []
        self._segments: list[Any] = []

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
        payload_begin = len(self._payloads)
        compact_begin = self.add_compact_params(compact_params)
        compact_count = 0 if compact_params is None else COMPACT_PARAM_COUNT
        global_shape_begin = len(self._global_shape)
        global_shape_count = 0
        if global_shape is not None:
            for i, value in enumerate(global_shape):
                self._global_shape.append(_as_i64(value, f"global_shape[{i}]"))
            global_shape_count = len(global_shape)
        edge_index = len(self._edges)
        self._edges.append(
            [
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
            ]
        )
        return edge_index

    def add_payload(
        self, edge_index: int, role: PayloadRole, segment: Any, valid_count: int
    ) -> None:
        if edge_index < 0 or edge_index >= len(self._edges):
            raise IndexError(f"edge_index {edge_index} is out of range.")
        if not hasattr(segment, "dtype"):
            raise TypeError(
                f"payload segment must expose dtype metadata, got {type(segment).__name__}."
            )
        shape = tuple(int(x) for x in getattr(segment, "shape", ()))
        shape_begin = len(self._shape)
        for dim, value in enumerate(shape):
            self._shape.append(_as_i32(value, f"payload shape dim {dim}"))
        segment_idx = len(self._segments)
        self._segments.append(segment)
        self._payloads.append(
            [
                _as_i32(int(role), "payload role"),
                _as_i32(dtype_code(segment.dtype), "dtype_code"),
                _as_i32(len(shape), "ndim"),
                _as_i32(shape_begin, "shape_begin"),
                _as_i32(len(shape), "shape_count"),
                _as_i32(valid_count, "valid_count"),
                _as_i32(segment_idx, "segment_arg_index"),
                0,
            ]
        )
        self._edges[edge_index][EDGE_PAYLOAD_COUNT] += 1
        _as_i32(self._edges[edge_index][EDGE_PAYLOAD_COUNT], "payload_count")

    def add_compact_params(self, params: list[int] | tuple[int, ...] | None) -> int:
        if params is None:
            return 0
        if len(params) != COMPACT_PARAM_COUNT:
            raise ValueError(
                f"compact params must have {COMPACT_PARAM_COUNT} entries, got {len(params)}."
            )
        begin = len(self._compact)
        for i, value in enumerate(params):
            self._compact.append(_as_i64(value, f"compact[{i}]"))
        return begin

    def build(self) -> TransportFrame:
        edge_count = _as_i32(len(self._edges), "edge_count")
        payload_count = _as_i32(len(self._payloads), "payload_count")
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
        edge_i32 = np.asarray(self._edges, dtype=np.int32).reshape(
            edge_count, EDGE_WIDTH
        )
        payload_i32 = np.asarray(self._payloads, dtype=np.int32).reshape(
            payload_count, PAYLOAD_WIDTH
        )
        shape_i32 = np.asarray(self._shape, dtype=np.int32)
        global_shape_i64 = np.asarray(self._global_shape, dtype=np.int64)
        compact_i64 = np.asarray(self._compact, dtype=np.int64)
        frame = TransportFrame(
            frame_i32=frame_i32,
            edge_i32=edge_i32,
            payload_i32=payload_i32,
            shape_i32=shape_i32,
            global_shape_i64=global_shape_i64,
            compact_i64=compact_i64,
            payload_segments=tuple(self._segments),
        )
        frame.validate()
        return frame


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
