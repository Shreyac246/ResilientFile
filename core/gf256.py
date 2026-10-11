"""Byte arithmetic in GF(256) for ResilientFile.

Field polynomial: x^8 + x^4 + x^3 + x^2 + 1 (0x11D).
Primitive element: 2.

Addition and subtraction use XOR.
Multiplication and division use logarithm/exponent tables.
"""

PRIMITIVE_POLYNOMIAL = 0x11D
GENERATOR = 2


def _validate(value: int) -> None:
    """Accept only integer field elements from 0 to 255."""
    if type(value) is not int:
        raise TypeError("A field element must be an integer.")

    if not 0 <= value <= 255:
        raise ValueError("A field element must be between 0 and 255.")


def _build_tables():
    """Build tables from successive powers of the element 2."""
    exp_table = [0] * 510
    log_table = [0] * 256

    value = 1
    seen = set()

    for exponent in range(255):
        if value in seen:
            raise RuntimeError("Generator repeated before visiting all elements.")

        seen.add(value)
        exp_table[exponent] = value
        log_table[value] = exponent

        # Multiply by 2, then reduce using the field polynomial.
        value <<= 1
        if value & 0x100:
            value ^= PRIMITIVE_POLYNOMIAL

    if value != 1 or len(seen) != 255:
        raise RuntimeError("Invalid primitive-element cycle.")

    # Repeat the table to avoid modulo in multiplication.
    for exponent in range(255, 510):
        exp_table[exponent] = exp_table[exponent - 255]

    return tuple(exp_table), tuple(log_table)


_EXP, _LOG = _build_tables()


def add(a: int, b: int) -> int:
    """Add two field elements."""
    _validate(a)
    _validate(b)
    return a ^ b


def subtract(a: int, b: int) -> int:
    """Subtraction equals addition in this field."""
    return add(a, b)


def multiply(a: int, b: int) -> int:
    """Multiply two field elements."""
    _validate(a)
    _validate(b)

    # Zero has no multiplicative logarithm.
    if a == 0 or b == 0:
        return 0

    return _EXP[_LOG[a] + _LOG[b]]


def inverse(a: int) -> int:
    """Return the element that multiplies with a to produce 1."""
    _validate(a)

    if a == 0:
        raise ZeroDivisionError("Zero has no multiplicative inverse.")

    return _EXP[255 - _LOG[a]]


def divide(a: int, b: int) -> int:
    """Divide a by a nonzero field element b."""
    _validate(a)
    _validate(b)

    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero.")

    if a == 0:
        return 0

    return _EXP[(_LOG[a] - _LOG[b]) % 255]


def power(a: int, exponent: int) -> int:
    """Raise a field element to an integer power.

    This API defines a**0 as 1, including 0**0.
    Negative powers require a nonzero base.
    """
    _validate(a)

    if type(exponent) is not int:
        raise TypeError("The exponent must be an integer.")

    if exponent == 0:
        return 1

    if a == 0:
        if exponent < 0:
            raise ZeroDivisionError("Zero cannot have a negative power.")
        return 0

    return _EXP[(_LOG[a] * exponent) % 255]