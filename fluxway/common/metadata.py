from __future__ import annotations

import hashlib
from collections.abc import Mapping
from enum import Enum
from typing import Any

from fluxway.common import msgpack_codec


def _canonical_metadata(value: Any) -> Any:
    if isinstance(value, Enum):
        return _canonical_metadata(value.value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("metadata mappings require string keys")
        return {key: _canonical_metadata(value[key]) for key in sorted(value)}
    if isinstance(value, (tuple, list)):
        return [_canonical_metadata(item) for item in value]
    return value


def pack_metadata(value: Any) -> bytes:
    return msgpack_codec.packb(
        _canonical_metadata(value), use_bin_type=True, strict_types=True
    )


def unpack_metadata(payload: bytes) -> Any:
    if not isinstance(payload, bytes):
        raise TypeError("metadata payload must be bytes")
    try:
        value = msgpack_codec.unpackb(payload, raw=False)
        if pack_metadata(value) != payload:
            raise ValueError("noncanonical encoding")
    except (TypeError, ValueError) as error:
        raise ValueError("metadata is not canonical MessagePack") from error
    return value


def metadata_digest(value: Any) -> str:
    return hashlib.sha256(pack_metadata(value)).hexdigest()
