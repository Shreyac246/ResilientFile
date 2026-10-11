"""Combined Reed-Solomon error-and-erasure decoding."""

from . import gf256 as gf
from . import polynomial as poly

from .error_decoder import (
    DecodeResult,
    _syndromes,
    _find_locator,
    _find_positions,
    _find_magnitudes,
)
from .erasure_decoder import RecoveryError


def decode_joint(
    encoded: bytes,
    erasure_positions=None,
    n: int = 255,
    k: int = 223,
) -> DecodeResult:
    """Correct errors and erasures when 2*errors + erasures <= n-k."""

    if type(n) is not int or type(k) is not int:
        raise TypeError("n and k must be integers.")

    if not 1 <= k < n <= 255:
        raise ValueError("Invalid Reed-Solomon parameters.")

    if not isinstance(encoded, bytes) or len(encoded) != n:
        raise ValueError("Encoded input must contain exactly n bytes.")

    erasures = list(
        [] if erasure_positions is None else erasure_positions
    )

    if any(
        type(p) is not int or not 0 <= p < n
        for p in erasures
    ):
        raise ValueError("Invalid erasure position.")

    if len(set(erasures)) != len(erasures):
        raise ValueError("Duplicate erasure positions.")

    parity_count = n - k

    if len(erasures) > parity_count:
        raise RecoveryError("Too many erasures.")

    syndromes = _syndromes(list(encoded), parity_count)

    if not any(syndromes):
        return DecodeResult(encoded[:k], [])

    # Construct erasure locator and modified syndromes.
    erasure_locator = [1]
    modified_syndromes = syndromes[:]

    for position in erasures:
        location = gf.power(
            gf.GENERATOR, n - 1 - position
        )

        erasure_locator = list(reversed(poly.multiply(
            list(reversed(erasure_locator)),
            [location, 1],
        )))

        modified_syndromes = [
            gf.multiply(modified_syndromes[j], location)
            ^ modified_syndromes[j + 1]
            for j in range(len(modified_syndromes) - 1)
        ]

    # Berlekamp-Massey finds unknown errors.
    unknown_locator = _find_locator(modified_syndromes)
    unknown_count = len(unknown_locator) - 1

    if 2 * unknown_count + len(erasures) > parity_count:
        raise RecoveryError("Correction capacity exceeded.")

    # Combine unknown-error and known-erasure locations.
    locator = list(reversed(poly.multiply(
        list(reversed(unknown_locator)),
        list(reversed(erasure_locator)),
    )))

    positions = _find_positions(locator, n)

    if (
        len(positions) != len(locator) - 1
        or not set(erasures).issubset(positions)
    ):
        raise RecoveryError("Could not locate all damaged symbols.")

    # Compute correction values using Forney's algorithm.
    magnitudes = _find_magnitudes(
        syndromes, locator, positions, n
    )

    repaired = bytearray(encoded)

    for position, magnitude in zip(positions, magnitudes):
        repaired[position] ^= magnitude

    # Verify that the repaired codeword is algebraically valid.
    if any(_syndromes(list(repaired), parity_count)):
        raise RecoveryError("Repaired block failed parity checks.")

    return DecodeResult(
        data=bytes(repaired[:k]),
        corrected_positions=positions,
    )