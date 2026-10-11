"""Tests for ResilientFile's GF(256) arithmetic."""

import random

import pytest

from core.gf256 import (
    GENERATOR,
    add,
    subtract,
    multiply,
    inverse,
    divide,
    power,
)


def reference_multiply(a, b):
    """Independent shift-and-reduce implementation for comparison."""
    result = 0

    while b:
        if b & 1:
            result ^= a

        b >>= 1
        a <<= 1

        if a & 0x100:
            a ^= 0x11D

    return result


def test_addition_is_xor():
    assert add(5, 7) == 2
    assert add(255, 255) == 0


def test_subtraction_equals_addition():
    for a in range(256):
        assert subtract(a, 73) == add(a, 73)


def test_additive_identity():
    for a in range(256):
        assert add(a, 0) == a


def test_self_addition_is_zero():
    for a in range(256):
        assert add(a, a) == 0


def test_multiply_by_zero():
    for a in range(256):
        assert multiply(a, 0) == 0
        assert multiply(0, a) == 0


def test_multiplicative_identity():
    for a in range(256):
        assert multiply(a, 1) == a


def test_known_products():
    assert multiply(0x80, 2) == 0x1D
    assert multiply(0x53, 0xCA) == 0x8F


def test_all_products_match_reference():
    for a in range(256):
        for b in range(256):
            assert multiply(a, b) == reference_multiply(a, b)


def test_multiplication_is_commutative():
    for a in range(256):
        for b in range(256):
            assert multiply(a, b) == multiply(b, a)


def test_distributivity():
    rng = random.Random(42)

    for _ in range(2000):
        a, b, c = [rng.randrange(256) for _ in range(3)]

        assert multiply(a, add(b, c)) == add(
            multiply(a, b),
            multiply(a, c),
        )


def test_multiplication_is_associative():
    rng = random.Random(123)

    for _ in range(2000):
        a, b, c = [rng.randrange(256) for _ in range(3)]

        assert multiply(multiply(a, b), c) == multiply(
            a, multiply(b, c)
        )


def test_every_nonzero_element_has_inverse():
    for a in range(1, 256):
        assert multiply(a, inverse(a)) == 1


def test_division_round_trip():
    for a in range(256):
        for b in range(1, 256):
            assert multiply(divide(a, b), b) == a


def test_zero_divided_by_nonzero():
    for b in range(1, 256):
        assert divide(0, b) == 0


def test_division_by_zero_fails():
    with pytest.raises(ZeroDivisionError):
        divide(25, 0)


def test_inverse_of_zero_fails():
    with pytest.raises(ZeroDivisionError):
        inverse(0)


def test_generator_is_primitive():
    value = 1
    visited = set()

    for _ in range(255):
        assert value not in visited
        visited.add(value)
        value = reference_multiply(value, GENERATOR)

    assert visited == set(range(1, 256))
    assert value == 1


def test_power_matches_repeated_multiplication():
    for a in range(256):
        expected = 1

        for exponent in range(12):
            assert power(a, exponent) == expected
            expected = reference_multiply(expected, a)


def test_negative_power():
    for a in range(1, 256):
        assert power(a, -1) == inverse(a)


def test_zero_power_cases():
    assert power(0, 0) == 1
    assert power(0, 5) == 0

    with pytest.raises(ZeroDivisionError):
        power(0, -1)


def test_invalid_field_values():
    for invalid in (-1, 256):
        with pytest.raises(ValueError):
            add(invalid, 1)

    for invalid in ("5", 2.5, True):
        with pytest.raises(TypeError):
            multiply(invalid, 1)


def test_invalid_exponent():
    with pytest.raises(TypeError):
        power(2, 1.5)