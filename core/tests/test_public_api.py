"""Tests for the installed ResilientFile public interface."""

import pytest
import resilientfile as rf

from resilientfile.encoder import generator_polynomial


def test_public_block_recovery():
    original = b"HELLO"
    encoded = rf.encode(original, n=9, k=5)

    damaged = bytearray(encoded)
    damaged[0] ^= 9
    damaged[8] ^= 200

    result = rf.decode(bytes(damaged), n=9, k=5)

    assert isinstance(result, rf.DecodeResult)
    assert result.data == original
    assert result.corrected_positions == [0, 8]


def test_public_erasure_recovery():
    original = b"HELLO"
    encoded = rf.encode(original, n=9, k=5)

    positions = [0, 2, 5, 8]
    damaged = bytearray(encoded)

    for position in positions:
        damaged[position] = 0

    assert rf.recover_erasures(
        bytes(damaged), positions, n=9, k=5
    ) == original


def test_public_file_recovery():
    original = b"ResilientFile public API.\n" * 40
    protected = rf.protect_bytes(original)

    assert rf.recover_bytes(protected) == original


def test_generator_cache_cannot_be_modified_by_caller():
    first = generator_polynomial(4)
    first[0] = 255

    assert generator_polynomial(4) == [1, 15, 54, 120, 64]


def test_cached_generator_still_validates_inputs():
    generator_polynomial(1)

    with pytest.raises(TypeError):
        generator_polynomial(True)

    with pytest.raises(ValueError):
        generator_polynomial(0)


def test_public_exception_can_be_caught():
    with pytest.raises(rf.ContainerError):
        rf.recover_bytes(b"invalid file")