"""Recover a Reed-Solomon block from known erasures.

Matches encoder.py:
- GF(256), polynomial 0x11D, primitive element 2
- Generator roots begin at 2**0
- Highest-degree coefficient first
- Original data followed by parity

Positions are zero-based indexes into the encoded block.
Values at erased positions are ignored.
"""

from . import gf256 as gf
from . import polynomial as poly


class RecoveryError(ValueError):
    """Recovery failed or the supplied block is inconsistent."""


def _solve(
    matrix: list[list[int]],
    rhs: list[int],
) -> list[int]:
    """Solve a square linear system using GF(256) elimination."""
    size = len(rhs)

    rows = [
        row[:] + [value]
        for row, value in zip(matrix, rhs)
    ]

    for column in range(size):
        pivot = next(
            (
                row
                for row in range(column, size)
                if rows[row][column] != 0
            ),
            None,
        )

        if pivot is None:
            raise RecoveryError("The recovery equations are singular.")

        rows[column], rows[pivot] = rows[pivot], rows[column]

        # Make the pivot coefficient equal to 1.
        factor = rows[column][column]
        rows[column] = [
            gf.divide(value, factor)
            for value in rows[column]
        ]

        # Eliminate this variable from every other row.
        for row in range(size):
            if row == column:
                continue

            factor = rows[row][column]

            if factor == 0:
                continue

            rows[row] = [
                gf.subtract(a, gf.multiply(factor, b))
                for a, b in zip(rows[row], rows[column])
            ]

    return [row[-1] for row in rows]


def recover(
    encoded: bytes,
    erasure_positions: list[int],
    n: int = 255,
    k: int = 223,
) -> bytes:
    """Recover the original k data bytes from an n-byte block.

    Up to n-k erased positions are supported, provided all
    other bytes are correct.

    Erased positions may contain any placeholder value.
    Their positions must remain present in the input block.
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

    positions = list(erasure_positions)

    for position in positions:
        if type(position) is not int:
            raise TypeError("Erasure positions must be integers.")

        if not 0 <= position < n:
            raise ValueError(f"Erasure position {position} is out of range.")

    if len(set(positions)) != len(positions):
        raise ValueError("Erasure positions must not contain duplicates.")

    parity_count = n - k

    if len(positions) > parity_count:
        raise RecoveryError(
            f"Cannot recover {len(positions)} erasures "
            f"with only {parity_count} parity bytes."
        )

    received = list(encoded)

    # Ignore the supplied values at all erased positions.
    for position in positions:
        received[position] = 0

    erasure_count = len(positions)

    if erasure_count:
        matrix = []
        rhs = []

        # Every valid codeword evaluates to zero at each
        # generator root. Use erasure_count such equations.
        for i in range(erasure_count):
            root = gf.power(gf.GENERATOR, i)

            matrix.append([
                gf.power(root, n - 1 - position)
                for position in positions
            ])

            # Known contribution equals missing contribution
            # because subtraction is XOR in this field.
            rhs.append(poly.evaluate(received, root))

        recovered_values = _solve(matrix, rhs)

        for position, value in zip(positions, recovered_values):
            received[position] = value

    # Check every parity equation, including any unused ones.
    for i in range(parity_count):
        root = gf.power(gf.GENERATOR, i)

        if poly.evaluate(received, root) != 0:
            raise RecoveryError(
                "Parity checks failed. There may be additional "
                "damage outside the supplied erasure positions."
            )

    return bytes(received[:k])