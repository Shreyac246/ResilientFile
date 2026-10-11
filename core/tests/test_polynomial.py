"""Tests for GF(256) polynomial operations."""

import random

import pytest

from core import gf256 as gf
from core import polynomial as poly


def test_normalize():
    assert poly.normalize([0, 0, 3, 5]) == [3, 5]
    assert poly.normalize([0, 0, 0]) == [0]
    assert poly.normalize([7]) == [7]


def test_invalid_polynomials():
    with pytest.raises(ValueError):
        poly.normalize([])

    for invalid in (-1, 256):
        with pytest.raises(ValueError):
            poly.normalize([1, invalid])

    for invalid in (2.5, "3", True):
        with pytest.raises(TypeError):
            poly.normalize([1, invalid])


def test_add_same_degree():
    assert poly.add([1, 2, 3], [4, 5, 6]) == [5, 7, 5]


def test_add_different_degrees():
    # Align [4, 5] with the final two coefficients.
    assert poly.add([1, 2, 3], [4, 5]) == [1, 6, 6]


def test_add_self_and_zero():
    assert poly.add([3, 5, 7], [3, 5, 7]) == [0]
    assert poly.add([3, 5, 7], [0]) == [3, 5, 7]


def test_subtraction_equals_addition():
    p = [3, 5, 7]
    q = [8, 2]

    assert poly.subtract(p, q) == poly.add(p, q)


def test_scale():
    assert poly.scale([1, 2, 3], 2) == [2, 4, 6]
    assert poly.scale([128, 1], 2) == [29, 2]
    assert poly.scale([3, 5, 7], 0) == [0]
    assert poly.scale([3, 5, 7], 1) == [3, 5, 7]


def test_multiply_known_polynomials():
    # (x + 1)^2 = x^2 + 1 in characteristic 2.
    assert poly.multiply([1, 1], [1, 1]) == [1, 0, 1]

    # (x + 2)(x + 3) = x^2 + x + 6.
    assert poly.multiply([1, 2], [1, 3]) == [1, 1, 6]


def test_multiply_zero_and_one():
    assert poly.multiply([3, 5, 7], [0]) == [0]
    assert poly.multiply([3, 5, 7], [1]) == [3, 5, 7]


def test_evaluate():
    assert poly.evaluate([1, 2, 3], 0) == 3
    assert poly.evaluate([1, 2, 3], 1) == 0
    assert poly.evaluate([1, 2, 3], 2) == 3
    assert poly.evaluate([9], 25) == 9
    assert poly.evaluate([0], 25) == 0


def test_invalid_scalar_and_evaluation_point():
    with pytest.raises(ValueError):
        poly.scale([1, 2], 256)

    with pytest.raises(TypeError):
        poly.scale([1, 2], "2")

    with pytest.raises(ValueError):
        poly.evaluate([9], 256)


def test_exact_division():
    quotient, remainder = poly.divide([1, 0, 1], [1, 1])

    assert quotient == [1, 1]
    assert remainder == [0]


def test_division_with_remainder():
    # x^2 + x + 1 = x(x + 1) + 1.
    quotient, remainder = poly.divide([1, 1, 1], [1, 1])

    assert quotient == [1, 0]
    assert remainder == [1]


def test_division_by_non_monic_polynomial():
    # (2x^2 + 6x + 4) / (2x + 2) = x + 2.
    quotient, remainder = poly.divide([2, 6, 4], [2, 2])

    assert quotient == [1, 2]
    assert remainder == [0]


def test_dividend_smaller_than_divisor():
    assert poly.divide([3, 4], [1, 2, 3]) == ([0], [3, 4])


def test_division_by_constant():
    assert poly.divide([2, 4, 6], [2]) == ([1, 2, 3], [0])


def test_zero_dividend():
    assert poly.divide([0], [1, 2]) == ([0], [0])


def test_zero_divisor_fails():
    with pytest.raises(ZeroDivisionError):
        poly.divide([1, 2, 3], [0, 0])


def test_inputs_are_not_modified():
    p = [0, 1, 2, 3]
    q = [0, 2, 5]

    original_p = p[:]
    original_q = q[:]

    poly.normalize(p)
    poly.add(p, q)
    poly.subtract(p, q)
    poly.scale(p, 7)
    poly.multiply(p, q)
    poly.evaluate(p, 7)
    poly.divide(p, q)

    assert p == original_p
    assert q == original_q


def test_random_division_reconstruction():
    rng = random.Random(42)

    for _ in range(500):
        dividend = [
            rng.randrange(256)
            for _ in range(rng.randint(1, 15))
        ]

        # A nonzero first coefficient ensures a nonzero divisor.
        divisor = [rng.randrange(1, 256)] + [
            rng.randrange(256)
            for _ in range(rng.randint(0, 6))
        ]

        quotient, remainder = poly.divide(dividend, divisor)

        reconstructed = poly.add(
            poly.multiply(divisor, quotient),
            remainder,
        )

        assert reconstructed == poly.normalize(dividend)
        assert remainder == [0] or len(remainder) < len(divisor)


def test_random_product_evaluation():
    rng = random.Random(123)

    for _ in range(200):
        p = [rng.randrange(256) for _ in range(5)]
        q = [rng.randrange(256) for _ in range(4)]
        x = rng.randrange(256)

        # (p*q)(x) must equal p(x)*q(x).
        assert poly.evaluate(poly.multiply(p, q), x) == gf.multiply(
            poly.evaluate(p, x),
            poly.evaluate(q, x),
        )


def test_random_distributivity():
    rng = random.Random(99)

    for _ in range(200):
        p = [rng.randrange(256) for _ in range(4)]
        q = [rng.randrange(256) for _ in range(3)]
        r = [rng.randrange(256) for _ in range(5)]

        assert poly.multiply(p, poly.add(q, r)) == poly.add(
            poly.multiply(p, q),
            poly.multiply(p, r),
        )