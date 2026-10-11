"""Tests for single-block Reed-Solomon encoding."""

import random

import pytest

from core import gf256 as gf
from core import polynomial as poly
from core.encoder import encode, generator_polynomial


def test_known_generator_polynomials():
    assert generator_polynomial(1) == [1, 1]
    assert generator_polynomial(2) == [1, 3, 2]
    assert generator_polynomial(4) == [1, 15, 54, 120, 64]


def test_generator_degree_and_leading_coefficient():
    for parity_count in (1, 2, 4, 8, 32, 254):
        generator = generator_polynomial(parity_count)

        assert len(generator) == parity_count + 1
        assert generator[0] == 1


def test_generator_roots():
    for parity_count in (1, 2, 4, 8, 32):
        generator = generator_polynomial(parity_count)

        for i in range(parity_count):
            root = gf.power(gf.GENERATOR, i)
            assert poly.evaluate(generator, root) == 0


def test_invalid_parity_count():
    for invalid in (0, -1, 255):
        with pytest.raises(ValueError):
            generator_polynomial(invalid)

    for invalid in (2.5, "4", True):
        with pytest.raises(TypeError):
            generator_polynomial(invalid)


def test_known_small_codeword():
    encoded = encode(bytes([1, 2, 3]), n=5, k=3)

    assert encoded == bytes([1, 2, 3, 4, 4])


def test_known_hello_codeword():
    encoded = encode(b"HELLO", n=9, k=5)

    assert encoded == b"HELLO" + bytes([142, 148, 89, 1])


def test_output_type_length_and_data_preservation():
    data = b"ResilientFile"
    encoded = encode(data, n=len(data) + 8, k=len(data))

    assert isinstance(encoded, bytes)
    assert len(encoded) == len(data) + 8
    assert encoded[:len(data)] == data


def test_default_rs_255_223():
    data = bytes(range(223))
    encoded = encode(data)

    assert len(encoded) == 255
    assert encoded[:223] == data

    for i in range(32):
        assert poly.evaluate(
            list(encoded), gf.power(gf.GENERATOR, i)
        ) == 0


def test_all_zero_data():
    assert encode(bytes(5), n=9, k=5) == bytes(9)


def test_leading_zero_data_is_preserved():
    data = bytes([0, 0, 1, 2, 3])
    encoded = encode(data, n=9, k=5)

    assert len(encoded) == 9
    assert encoded[:5] == data


def test_zero_parity_keeps_required_length():
    # With one parity symbol, [1, 1] has zero remainder.
    assert encode(bytes([1, 1]), n=3, k=2) == bytes([1, 1, 0])


def test_extreme_valid_parameters():
    for n, k in ((2, 1), (255, 1), (255, 254)):
        data = bytes([7]) * k
        encoded = encode(data, n=n, k=k)

        assert len(encoded) == n
        assert encoded[:k] == data

        generator = generator_polynomial(n - k)
        _, remainder = poly.divide(list(encoded), generator)
        assert remainder == [0]


def test_codewords_are_divisible_by_generator():
    rng = random.Random(42)

    for n, k in ((5, 3), (15, 9), (31, 23), (255, 223)):
        generator = generator_polynomial(n - k)

        for _ in range(10):
            data = bytes(rng.randrange(256) for _ in range(k))
            encoded = encode(data, n=n, k=k)

            _, remainder = poly.divide(list(encoded), generator)

            assert remainder == [0]


def test_codewords_vanish_at_generator_roots():
    rng = random.Random(123)

    for n, k in ((9, 5), (15, 11), (31, 23)):
        for _ in range(10):
            data = bytes(rng.randrange(256) for _ in range(k))
            encoded = encode(data, n=n, k=k)

            for i in range(n - k):
                root = gf.power(gf.GENERATOR, i)
                assert poly.evaluate(list(encoded), root) == 0


def test_encoding_is_linear():
    a = bytes([1, 2, 3, 4, 5])
    b = bytes([9, 8, 7, 6, 5])

    combined_data = bytes(x ^ y for x, y in zip(a, b))

    encoded_a = encode(a, n=9, k=5)
    encoded_b = encode(b, n=9, k=5)

    combined_encoded = bytes(
        x ^ y for x, y in zip(encoded_a, encoded_b)
    )

    assert encode(combined_data, n=9, k=5) == combined_encoded


def test_invalid_parameters():
    for n, k in ((5, 5), (4, 5), (256, 5), (5, 0), (-1, 1)):
        with pytest.raises(ValueError):
            encode(b"HELLO", n=n, k=k)

    for n, k in ((9.0, 5), (9, "5"), (True, 5)):
        with pytest.raises(TypeError):
            encode(b"HELLO", n=n, k=k)


def test_wrong_input_length():
    for data in (b"", b"HI", b"TOO LONG"):
        with pytest.raises(ValueError):
            encode(data, n=9, k=5)


def test_wrong_input_type():
    for data in ("HELLO", [1, 2, 3, 4, 5], bytearray(b"HELLO")):
        with pytest.raises(TypeError):
            encode(data, n=9, k=5)