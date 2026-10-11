"""Reed-Solomon correction of unknown-position byte errors.

Matches encoder.py:
- Field polynomial 0x11D
- Primitive element 2
- First generator root 2**0
- Codeword coefficients highest degree first

Internal locator/evaluator polynomials use ascending order.
That difference is explicit in _evaluate_ascending().
"""

from dataclasses import dataclass

from . import gf256 as gf
from . import polynomial as poly
from .erasure_decoder import RecoveryError


@dataclass
class DecodeResult:
    """Original data and zero-based corrected codeword positions."""

    data: bytes
    corrected_positions: list[int]


def _syndromes(codeword, parity_count):
    """Evaluate the received polynomial at every generator root."""
    return [
        poly.evaluate(codeword, gf.power(gf.GENERATOR, i))
        for i in range(parity_count)
    ]


def _evaluate_ascending(coefficients, x):
    """Evaluate coefficients stored constant term first."""
    return poly.evaluate(list(reversed(coefficients)), x)


def _find_locator(syndromes):
    """Berlekamp-Massey: find the error locator polynomial.

    Returns coefficients in ascending order:
        [1, lambda_1, lambda_2, ...]
    """
    count = len(syndromes)

    locator = [1] + [0] * count
    previous = [1] + [0] * count

    degree = 0
    shift = 1
    last_discrepancy = 1

    for step in range(count):
        discrepancy = syndromes[step]

        for j in range(1, degree + 1):
            discrepancy ^= gf.multiply(
                locator[j], syndromes[step - j]
            )

        if discrepancy == 0:
            shift += 1
            continue

        old_locator = locator[:]
        factor = gf.divide(discrepancy, last_discrepancy)

        for j in range(count + 1 - shift):
            locator[j + shift] ^= gf.multiply(
                factor, previous[j]
            )

        if 2 * degree <= step:
            degree = step + 1 - degree
            previous = old_locator
            last_discrepancy = discrepancy
            shift = 1
        else:
            shift += 1

    return locator[:degree + 1]


def _find_positions(locator, n):
    """Chien search over the positions present in this block."""
    positions = []

    for position in range(n):
        inverse_location = gf.power(
            gf.GENERATOR, -(n - 1 - position)
        )

        if _evaluate_ascending(locator, inverse_location) == 0:
            positions.append(position)

    return positions


def _find_magnitudes(syndromes, locator, positions, n):
    """Forney's formula for our first-generator-root convention."""
    parity_count = len(syndromes)

    # Omega(z) = syndrome(z) * locator(z) modulo z**parity_count.
    # Both operands are represented in ascending order.
    evaluator = [0] * parity_count

    for i, coefficient in enumerate(locator):
        for j in range(parity_count - i):
            evaluator[i + j] ^= gf.multiply(
                coefficient, syndromes[j]
            )

    # Formal derivative in characteristic 2:
    # even powers disappear, odd powers retain their coefficient.
    derivative = [
        locator[i] if i % 2 else 0
        for i in range(1, len(locator))
    ]

    magnitudes = []

    for position in positions:
        location = gf.power(gf.GENERATOR, n - 1 - position)
        inverse_location = gf.inverse(location)

        numerator = _evaluate_ascending(
            evaluator, inverse_location
        )
        denominator = _evaluate_ascending(
            derivative, inverse_location
        )

        if denominator == 0:
            raise RecoveryError("Invalid error locator derivative.")

        # The location factor is required because our roots start at 2**0.
        magnitude = gf.divide(
            gf.multiply(location, numerator),
            denominator,
        )

        magnitudes.append(magnitude)

    return magnitudes


def decode(
    encoded: bytes,
    n: int = 255,
    k: int = 223,
) -> DecodeResult:
    """Correct up to floor((n-k)/2) unknown byte errors.

    The block must retain its original length and byte positions.
    This function does not accept erasure-position hints.

    Beyond the correction bound, decoding may raise RecoveryError
    or produce another valid codeword. Separate integrity checking
    is needed before trusting data outside the guaranteed bound.
    """
    if type(n) is not int or type(k) is not int:
        raise TypeError("n and k must be integers.")

    if not 1 <= k < n <= 255:
        raise ValueError("Parameters must satisfy 1 <= k < n <= 255.")

    if not isinstance(encoded, bytes):
        raise TypeError("encoded must be bytes.")

    if len(encoded) != n:
        raise ValueError(
            f"Expected exactly {n} encoded bytes, received {len(encoded)}."
        )

    parity_count = n - k
    received = list(encoded)

    # Stage 1: compute syndromes.
    syndromes = _syndromes(received, parity_count)

    if not any(syndromes):
        return DecodeResult(
            data=encoded[:k],
            corrected_positions=[],
        )

    # Stage 2: find the locator polynomial.
    locator = _find_locator(syndromes)
    error_count = len(locator) - 1

    if error_count == 0 or 2 * error_count > parity_count:
        raise RecoveryError(
            "Cannot correct this block within the error limit."
        )

    # Stage 3: locate the errors.
    positions = _find_positions(locator, n)

    if len(positions) != error_count:
        raise RecoveryError(
            "The error locator does not match valid block positions."
        )

    # Stage 4: calculate and apply corrections.
    magnitudes = _find_magnitudes(
        syndromes, locator, positions, n
    )

    for position, magnitude in zip(positions, magnitudes):
        received[position] ^= magnitude

    if any(_syndromes(received, parity_count)):
        raise RecoveryError("The corrected block failed parity checks.")

    return DecodeResult(
        data=bytes(received[:k]),
        corrected_positions=positions,
    )