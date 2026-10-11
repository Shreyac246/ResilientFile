"""Systematic Reed-Solomon encoding over GF(256).

Conventions:
- Field polynomial: 0x11D
- Primitive element: 2
- First generator root: 2**0
- Highest-degree coefficient first
- Codeword layout: data followed by parity
"""

from functools import lru_cache

from . import gf256 as gf
from . import polynomial as poly


@lru_cache(maxsize=254)
def _cached_generator(parity_symbols: int) -> tuple[int, ...]:
    """Build and retain an immutable generator polynomial."""
    generator = [1]

    for i in range(parity_symbols):
        root = gf.power(gf.GENERATOR, i)
        generator = poly.multiply(generator, [1, root])

    return tuple(generator)


def generator_polynomial(parity_symbols: int) -> list[int]:
    """Return a fresh copy of the cached generator polynomial."""
    if type(parity_symbols) is not int:
        raise TypeError("The parity count must be an integer.")

    if not 1 <= parity_symbols <= 254:
        raise ValueError("The parity count must be between 1 and 254.")

    return list(_cached_generator(parity_symbols))


def encode(data: bytes, n: int = 255, k: int = 223) -> bytes:
    """Encode exactly k data bytes into n bytes.

    Requires 1 <= k < n <= 255.
    The first k output bytes equal the original data.
    The remaining n-k bytes contain parity.
    """
    if type(n) is not int or type(k) is not int:
        raise TypeError("n and k must be integers.")

    if not 1 <= k < n <= 255:
        raise ValueError("Parameters must satisfy 1 <= k < n <= 255.")

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")

    if len(data) != k:
        raise ValueError(
            f"Expected exactly {k} data bytes, received {len(data)}."
        )

    parity_count = n - k
    generator = generator_polynomial(parity_count)

    shifted_message = list(data) + [0] * parity_count
    _, remainder = poly.divide(shifted_message, generator)

    # Restore leading zeros removed by polynomial normalization.
    parity = [0] * (parity_count - len(remainder)) + remainder

    return data + bytes(parity)