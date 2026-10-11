"""A versioned container for complete Reed-Solomon protected files.

This module operates only on bytes. File access belongs to the CLI.

Version 1:
- 50-byte metadata header
- 32-byte SHA-256 checksum of the header
- Zero or more n-byte Reed-Solomon codewords

The header is checked but is not Reed-Solomon protected.
"""

import hashlib
import struct

from .encoder import encode
from .error_decoder import decode
from .erasure_decoder import RecoveryError


MAGIC = b"RSFILE01"
_HEADER = struct.Struct(">8sBBQ32s")
HEADER_SIZE = _HEADER.size + 32


class ContainerError(ValueError):
    """Invalid container or unsuccessful integrity verification."""


def protect_bytes(
    data: bytes,
    n: int = 255,
    k: int = 223,
) -> bytes:
    """Protect an arbitrary-length byte sequence."""
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")

    if type(n) is not int or type(k) is not int:
        raise TypeError("n and k must be integers.")

    if not 1 <= k < n <= 255:
        raise ValueError("Parameters must satisfy 1 <= k < n <= 255.")

    original_digest = hashlib.sha256(data).digest()

    header = _HEADER.pack(
        MAGIC,
        n,
        k,
        len(data),
        original_digest,
    )

    header_digest = hashlib.sha256(header).digest()
    protected = bytearray(header + header_digest)

    for offset in range(0, len(data), k):
        block = data[offset:offset + k]

        # Only the final block may need padding.
        padded = block.ljust(k, b"\x00")

        protected.extend(encode(padded, n=n, k=k))

    return bytes(protected)


def recover_bytes(protected: bytes) -> bytes:
    """Recover and verify the original bytes.

    Parameters n and k are read from the checked header.
    Returns data only after its SHA-256 matches the stored digest.
    """
    if not isinstance(protected, bytes):
        raise TypeError("protected must be bytes.")

    if len(protected) < HEADER_SIZE:
        raise ContainerError("File is too short to contain an RF header.")

    header = protected[:_HEADER.size]
    stored_header_digest = protected[_HEADER.size:HEADER_SIZE]

    if hashlib.sha256(header).digest() != stored_header_digest:
        raise ContainerError("Header checksum failed; metadata is damaged.")

    magic, n, k, original_length, original_digest = _HEADER.unpack(header)

    if magic != MAGIC:
        raise ContainerError("Unsupported file format or version.")

    if not 1 <= k < n <= 255:
        raise ContainerError("Invalid coding parameters in header.")

    block_count = (original_length + k - 1) // k
    expected_length = HEADER_SIZE + block_count * n

    if len(protected) != expected_length:
        raise ContainerError(
            "Container length is incorrect; the file may be truncated "
            "or contain extra bytes."
        )

    recovered = bytearray()

    for block_index in range(block_count):
        start = HEADER_SIZE + block_index * n
        codeword = protected[start:start + n]

        try:
            result = decode(codeword, n=n, k=k)
        except RecoveryError as error:
            raise ContainerError(
                f"Could not recover block {block_index}: {error}"
            ) from error

        recovered.extend(result.data)

    # Slice by recorded length: genuine trailing zero bytes are preserved.
    original = bytes(recovered[:original_length])

    if hashlib.sha256(original).digest() != original_digest:
        raise ContainerError(
            "Recovered data failed SHA-256 verification. "
            "The original file could not be verified."
        )

    return original