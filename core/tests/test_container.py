"""Tests for full-file framing, recovery, and integrity checks."""

import hashlib
import random

import pytest

from core.container import (
    ContainerError,
    HEADER_SIZE,
    protect_bytes,
    recover_bytes,
)
from core.encoder import encode


def test_empty_file():
    protected = protect_bytes(b"")

    assert len(protected) == HEADER_SIZE
    assert recover_bytes(protected) == b""


def test_round_trip_at_block_boundaries():
    for size in (1, 222, 223, 224, 446, 447, 1000):
        data = bytes(i % 256 for i in range(size))

        assert recover_bytes(protect_bytes(data)) == data


def test_genuine_trailing_zeros_are_preserved():
    data = b"HELLO" + bytes(300)

    assert recover_bytes(protect_bytes(data)) == data


def test_expected_container_size():
    data = bytes(224)

    # 224 bytes require two blocks of 223 data bytes each.
    assert len(protect_bytes(data)) == HEADER_SIZE + 2 * 255


def test_custom_parameters():
    data = b"Multiple small blocks!"

    protected = protect_bytes(data, n=9, k=5)

    assert recover_bytes(protected) == data


def test_maximum_errors_in_every_block():
    rng = random.Random(42)
    data = bytes(rng.randrange(256) for _ in range(700))
    protected = bytearray(protect_bytes(data))

    for start in range(HEADER_SIZE, len(protected), 255):
        for position in rng.sample(range(255), 16):
            protected[start + position] ^= rng.randrange(1, 256)

    assert recover_bytes(bytes(protected)) == data


def test_header_damage_is_rejected():
    protected = bytearray(protect_bytes(b"HELLO"))
    protected[10] ^= 1

    with pytest.raises(ContainerError, match="Header checksum"):
        recover_bytes(bytes(protected))


def test_header_checksum_damage_is_rejected():
    protected = bytearray(protect_bytes(b"HELLO"))
    protected[HEADER_SIZE - 1] ^= 1

    with pytest.raises(ContainerError, match="Header checksum"):
        recover_bytes(bytes(protected))


def test_truncated_payload_is_rejected():
    protected = protect_bytes(b"HELLO")

    with pytest.raises(ContainerError, match="length"):
        recover_bytes(protected[:-1])


def test_extra_bytes_are_rejected():
    protected = protect_bytes(b"HELLO")

    with pytest.raises(ContainerError, match="length"):
        recover_bytes(protected + b"\x00")


def test_too_short_header_is_rejected():
    with pytest.raises(ContainerError, match="too short"):
        recover_bytes(b"RF")


def test_different_valid_codeword_fails_file_hash():
    protected = bytearray(protect_bytes(b"HELLO", n=9, k=5))

    # This replacement passes Reed-Solomon parity checks,
    # but it must fail the original file's stored hash.
    protected[HEADER_SIZE:] = encode(b"WORLD", n=9, k=5)

    with pytest.raises(ContainerError, match="SHA-256"):
        recover_bytes(bytes(protected))


def test_unsupported_version_is_rejected():
    protected = bytearray(protect_bytes(b"HELLO"))
    protected[:8] = b"RSFILE99"

    # Recompute the checksum to reach the version validation.
    metadata_size = HEADER_SIZE - 32
    protected[metadata_size:HEADER_SIZE] = hashlib.sha256(
        protected[:metadata_size]
    ).digest()

    with pytest.raises(ContainerError, match="version"):
        recover_bytes(bytes(protected))


def test_invalid_parameters():
    for n, k in ((255, 255), (256, 223), (9, 0)):
        with pytest.raises(ValueError):
            protect_bytes(b"HELLO", n=n, k=k)

    with pytest.raises(TypeError):
        protect_bytes(b"HELLO", n=9.0, k=5)


def test_wrong_input_types():
    with pytest.raises(TypeError):
        protect_bytes("HELLO")

    with pytest.raises(TypeError):
        recover_bytes("HELLO")