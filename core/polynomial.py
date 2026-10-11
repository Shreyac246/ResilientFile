"""Polynomial operations over GF(256).

Coefficients are ordered highest degree first.
Example: [3, 5, 7] represents 3*x^2 + 5*x + 7.

Zero is represented as [0].
Empty polynomials are rejected.
Functions return new lists without modifying their inputs.
"""

from . import gf256 as gf


def normalize(poly: list[int]) -> list[int]:
    """Validate coefficients and remove unnecessary leading zeros."""
    coefficients = list(poly)

    if not coefficients:
        raise ValueError("A polynomial must contain at least one coefficient.")

    for coefficient in coefficients:
        if type(coefficient) is not int:
            raise TypeError("Polynomial coefficients must be integers.")

        if not 0 <= coefficient <= 255:
            raise ValueError("Polynomial coefficients must be from 0 to 255.")

    first = 0

    while first < len(coefficients) - 1 and coefficients[first] == 0:
        first += 1

    return coefficients[first:]


def add(p: list[int], q: list[int]) -> list[int]:
    """Add polynomials by aligning coefficients of equal degree."""
    p = normalize(p)
    q = normalize(q)

    result = [0] * max(len(p), len(q))

    p_offset = len(result) - len(p)
    q_offset = len(result) - len(q)

    for i, coefficient in enumerate(p):
        result[i + p_offset] ^= coefficient

    for i, coefficient in enumerate(q):
        result[i + q_offset] ^= coefficient

    return normalize(result)


def subtract(p: list[int], q: list[int]) -> list[int]:
    """Subtraction equals addition over GF(256)."""
    return add(p, q)


def scale(poly: list[int], scalar: int) -> list[int]:
    """Multiply every coefficient by a field element."""
    poly = normalize(poly)

    result = [
        gf.multiply(coefficient, scalar)
        for coefficient in poly
    ]

    return normalize(result)


def multiply(p: list[int], q: list[int]) -> list[int]:
    """Multiply polynomials using GF(256) coefficient arithmetic."""
    p = normalize(p)
    q = normalize(q)

    result = [0] * (len(p) + len(q) - 1)

    for i, p_coefficient in enumerate(p):
        for j, q_coefficient in enumerate(q):
            result[i + j] ^= gf.multiply(
                p_coefficient, q_coefficient
            )

    return normalize(result)


def evaluate(poly: list[int], x: int) -> int:
    """Evaluate a polynomial using Horner's method."""
    poly = normalize(poly)

    # Validate x even when the polynomial is constant.
    gf.add(x, 0)

    result = poly[0]

    for coefficient in poly[1:]:
        result = gf.add(
            gf.multiply(result, x),
            coefficient,
        )

    return result


def divide(
    dividend: list[int],
    divisor: list[int],
) -> tuple[list[int], list[int]]:
    """Return (quotient, remainder) using polynomial long division.

    Supports divisors whose leading coefficient is not 1.

    dividend = divisor * quotient + remainder
    """
    dividend = normalize(dividend)
    divisor = normalize(divisor)

    if divisor == [0]:
        raise ZeroDivisionError("Cannot divide by the zero polynomial.")

    if len(dividend) < len(divisor):
        return [0], dividend

    remainder = dividend[:]
    quotient_length = len(dividend) - len(divisor) + 1
    quotient = [0] * quotient_length

    for i in range(quotient_length):
        factor = gf.divide(remainder[i], divisor[0])
        quotient[i] = factor

        if factor == 0:
            continue

        for j, coefficient in enumerate(divisor):
            remainder[i + j] ^= gf.multiply(factor, coefficient)

    # The processed leading coefficients are now zero.
    remainder = remainder[quotient_length:]

    return normalize(quotient), normalize(remainder or [0])