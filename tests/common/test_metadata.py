import msgpack
import pytest

from fluxway.common.metadata import metadata_digest, pack_metadata, unpack_metadata


def test_nested_metadata_has_stable_encoding_and_preserves_bytes():
    left = {"z": [{"b": b"\x00\xff", "a": 7}], "a": (True, None, "value")}
    right = {"a": [True, None, "value"], "z": [{"a": 7, "b": b"\x00\xff"}]}
    assert pack_metadata(left) == pack_metadata(right)
    assert metadata_digest(left) == metadata_digest(right)
    assert unpack_metadata(pack_metadata(left)) == right
    assert metadata_digest(right) != metadata_digest(dict(right, extra=1))


@pytest.mark.parametrize(
    "payload",
    [
        b'{"a":1}',
        b"\x81\xa1a",
        b"\x82\xa1a\x01\xa1a\x02",
        msgpack.packb({"z": 1, "a": 2}, use_bin_type=True),
        msgpack.packb({"a": 1}, use_bin_type=True) + b"\x00",
        msgpack.packb({1: "value"}, use_bin_type=True),
    ],
)
def test_metadata_rejects_invalid_or_noncanonical_payloads(payload):
    with pytest.raises(ValueError, match="canonical MessagePack"):
        unpack_metadata(payload)


def test_metadata_rejects_unsupported_objects_and_nonstring_keys():
    with pytest.raises(TypeError):
        pack_metadata({"value": object()})
    with pytest.raises(TypeError, match="string keys"):
        pack_metadata({"nested": {1: "value"}})
    with pytest.raises(TypeError, match="must be bytes"):
        unpack_metadata("value")
