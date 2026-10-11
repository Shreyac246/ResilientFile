"""Tests for unknown-position error correction."""

import random

import pytest

from core.encoder import encode
from core.error_decoder import DecodeResult, decode
from core.erasure_decoder import RecoveryError


def corrupt(encoded, positions, differences):
    """Change bytes with nonzero XOR differences."""
    damaged = bytearray(encoded)

    for position, difference in zip(positions, differences):
        damaged[position] ^= difference

    return bytes(damaged)


def test_undamaged_block():
    result = decode(encode(b"HELLO", n=9, k=5), n=9, k=5)

    assert isinstance(result, DecodeResult)
    assert result.data == b"HELLO"
    assert result.corrected_positions == []


def test_every_single_error_position_and_value():
    encoded = encode(b"HELLO", n=9, k=5)

    for position in range(9):
        for difference in range(1, 256):
            damaged = corrupt(encoded, [position], [difference])
            result = decode(damaged, n=9, k=5)

            assert result.data == b"HELLO"
            assert result.corrected_positions == [position]


def test_two_unknown_errors():
    encoded = encode(b"HELLO", n=9, k=5)
    damaged = corrupt(encoded, [1, 7], [77, 123])

    result = decode(damaged, n=9, k=5)

    assert result.data == b"HELLO"
    assert result.corrected_positions == [1, 7]


def test_only_parity_errors():
    encoded = encode(b"HELLO", n=9, k=5)
    damaged = corrupt(encoded, [5, 8], [37, 91])

    result = decode(damaged, n=9, k=5)

    assert result.data == b"HELLO"
    assert result.corrected_positions == [5, 8]


def test_first_and_last_positions():
    encoded = encode(b"HELLO", n=9, k=5)
    damaged = corrupt(encoded, [0, 8], [12, 234])

    result = decode(damaged, n=9, k=5)

    assert result.data == b"HELLO"
    assert result.corrected_positions == [0, 8]


def test_zero_first_syndrome_with_actual_errors():
    # Equal error magnitudes cancel in the first syndrome.
    # The decoder must still detect the remaining syndromes.
    encoded = encode(b"HELLO", n=9, k=5)
    damaged = corrupt(encoded, [0, 8], [73, 73])

    result = decode(damaged, n=9, k=5)

    assert result.data == b"HELLO"
    assert result.corrected_positions == [0, 8]


def test_all_zero_original():
    encoded = encode(bytes(5), n=9, k=5)
    damaged = corrupt(encoded, [0, 6], [255, 32])

    result = decode(damaged, n=9, k=5)

    assert result.data == bytes(5)
    assert result.corrected_positions == [0, 6]


def test_leading_zero_original():
    data = bytes([0, 0, 1, 2, 3])
    encoded = encode(data, n=9, k=5)
    damaged = corrupt(encoded, [0, 1], [42, 99])

    result = decode(damaged, n=9, k=5)

    assert result.data == data
    assert result.corrected_positions == [0, 1]


def test_default_rs_255_223_at_limit():
    data = bytes(range(223))
    encoded = encode(data)

    positions = list(range(0, 256, 16))
    differences = list(range(1, 17))
    damaged = corrupt(encoded, positions, differences)

    result = decode(damaged)

    assert result.data == data
    assert result.corrected_positions == positions


def test_random_errors_up_to_limit():
    rng = random.Random(42)

    for n, k in ((5, 3), (9, 5), (15, 9), (31, 23), (10, 5)):
        limit = (n - k) // 2

        for _ in range(15):
            data = bytes(rng.randrange(256) for _ in range(k))
            encoded = encode(data, n=n, k=k)

            for count in range(limit + 1):
                positions = rng.sample(range(n), count)
                differences = [
                    rng.randrange(1, 256)
                    for _ in positions
                ]

                damaged = corrupt(encoded, positions, differences)
                result = decode(damaged, n=n, k=k)

                assert result.data == data
                assert result.corrected_positions == sorted(positions)


def test_one_parity_symbol_cannot_correct_single_error():
    encoded = encode(b"HI", n=3, k=2)
    damaged = corrupt(encoded, [0], [1])

    with pytest.raises(RecoveryError):
        decode(damaged, n=3, k=2)


def test_encoding_recovered_data_restores_codeword():
    encoded = encode(b"HELLO", n=9, k=5)
    damaged = corrupt(encoded, [2, 6], [101, 202])

    result = decode(damaged, n=9, k=5)

    assert encode(result.data, n=9, k=5) == encoded


def test_input_is_not_modified():
    encoded = encode(b"HELLO", n=9, k=5)
    damaged = corrupt(encoded, [0], [7])
    original_input = bytes(damaged)

    decode(damaged, n=9, k=5)

    assert damaged == original_input


def test_wrong_length():
    with pytest.raises(ValueError):
        decode(b"HELLO", n=9, k=5)


def test_wrong_type():
    with pytest.raises(TypeError):
        decode("123456789", n=9, k=5)


def test_invalid_parameters():
    for n, k in ((9, 9), (256, 5), (9, 0)):
        with pytest.raises(ValueError):
            decode(bytes(9), n=n, k=k)

    for n, k in ((9.0, 5), (9, "5"), (True, 5)):
        with pytest.raises(TypeError):
            decode(bytes(9), n=n, k=k)