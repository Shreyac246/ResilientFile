"""Tests for known-position erasure recovery."""

import random

import pytest

from core.encoder import encode
from core.erasure_decoder import RecoveryError, recover


def damage(encoded, positions, placeholder=0):
    """Replace bytes without removing their positions."""
    result = bytearray(encoded)

    for position in positions:
        result[position] = placeholder

    return bytes(result)


def test_undamaged_block():
    encoded = encode(b"HELLO", n=9, k=5)

    assert recover(encoded, [], n=9, k=5) == b"HELLO"


def test_each_single_position():
    encoded = encode(b"HELLO", n=9, k=5)

    for position in range(9):
        damaged = damage(encoded, [position])

        assert recover(
            damaged, [position], n=9, k=5
        ) == b"HELLO"


def test_multiple_data_erasures():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [0, 2, 4]

    assert recover(
        damage(encoded, positions), positions, n=9, k=5
    ) == b"HELLO"


def test_only_parity_erased():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [5, 6, 7, 8]

    assert recover(
        damage(encoded, positions), positions, n=9, k=5
    ) == b"HELLO"


def test_maximum_mixed_erasures():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [0, 2, 5, 8]

    assert recover(
        damage(encoded, positions), positions, n=9, k=5
    ) == b"HELLO"


def test_unsorted_positions():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [8, 0, 5, 2]

    assert recover(
        damage(encoded, positions), positions, n=9, k=5
    ) == b"HELLO"


def test_placeholder_values_are_ignored():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [0, 2, 5, 8]

    for placeholder in (0, 1, 127, 255):
        assert recover(
            damage(encoded, positions, placeholder),
            positions,
            n=9,
            k=5,
        ) == b"HELLO"


def test_original_zero_bytes():
    data = bytes([0, 0, 1, 0, 2])
    encoded = encode(data, n=9, k=5)
    positions = [0, 1, 3, 7]

    assert recover(
        damage(encoded, positions, 255), positions, n=9, k=5
    ) == data


def test_all_zero_block():
    encoded = encode(bytes(5), n=9, k=5)
    positions = [0, 2, 5, 8]

    assert recover(
        damage(encoded, positions, 255), positions, n=9, k=5
    ) == bytes(5)


def test_default_rs_255_223_at_limit():
    data = bytes(range(223))
    encoded = encode(data)
    positions = list(range(32))

    assert recover(
        damage(encoded, positions), positions
    ) == data


def test_random_recovery():
    rng = random.Random(42)

    for n, k in ((5, 3), (9, 5), (15, 9), (31, 23)):
        for _ in range(20):
            data = bytes(rng.randrange(256) for _ in range(k))
            encoded = encode(data, n=n, k=k)

            for count in (0, 1, n - k):
                positions = rng.sample(range(n), count)
                damaged = bytearray(encoded)

                for position in positions:
                    damaged[position] ^= rng.randrange(1, 256)

                assert recover(
                    bytes(damaged), positions, n=n, k=k
                ) == data


def test_too_many_erasures():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [0, 1, 2, 3, 4]

    with pytest.raises(RecoveryError):
        recover(
            damage(encoded, positions), positions, n=9, k=5
        )


def test_unreported_single_error_is_detected():
    damaged = bytearray(encode(b"HELLO", n=9, k=5))
    damaged[0] ^= 1

    with pytest.raises(RecoveryError):
        recover(bytes(damaged), [], n=9, k=5)


def test_one_erasure_and_unreported_error():
    damaged = bytearray(encode(b"HELLO", n=9, k=5))
    damaged[0] = 0
    damaged[1] ^= 1

    with pytest.raises(RecoveryError):
        recover(bytes(damaged), [0], n=9, k=5)


def test_duplicate_positions():
    encoded = encode(b"HELLO", n=9, k=5)

    with pytest.raises(ValueError):
        recover(encoded, [0, 0], n=9, k=5)


def test_invalid_positions():
    encoded = encode(b"HELLO", n=9, k=5)

    for position in (-1, 9):
        with pytest.raises(ValueError):
            recover(encoded, [position], n=9, k=5)

    for position in ("0", 1.5, True):
        with pytest.raises(TypeError):
            recover(encoded, [position], n=9, k=5)


def test_wrong_block_length():
    with pytest.raises(ValueError):
        recover(b"HELLO", [], n=9, k=5)


def test_invalid_parameters():
    for n, k in ((9, 9), (256, 5), (9, 0)):
        with pytest.raises(ValueError):
            recover(bytes(9), [], n=n, k=k)

    with pytest.raises(TypeError):
        recover(bytes(9), [], n=9.0, k=5)


def test_wrong_input_type():
    with pytest.raises(TypeError):
        recover("123456789", [], n=9, k=5)


def test_positions_are_not_modified():
    encoded = encode(b"HELLO", n=9, k=5)
    positions = [8, 0, 2]
    original = positions[:]

    recover(
        damage(encoded, positions), positions, n=9, k=5
    )

    assert positions == original