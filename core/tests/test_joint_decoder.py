import random
import pytest

from core.encoder import encode
from core.joint_decoder import decode_joint
from core.erasure_decoder import RecoveryError


def test_mixed_errors_and_erasures():
    original = b"HELLO"
    encoded = bytearray(encode(original, n=9, k=5))

    encoded[0] ^= 17  # Known erasure
    encoded[2] ^= 29  # Known erasure
    encoded[7] ^= 43  # Unknown error

    result = decode_joint(
        bytes(encoded),
        erasure_positions=[0, 2],
        n=9,
        k=5,
    )

    assert result.data == original
    assert result.corrected_positions == [0, 2, 7]


def test_randomized_mixed_recovery():
    rng = random.Random(42)

    for _ in range(100):
        original = bytes(rng.randrange(256) for _ in range(223))
        encoded = bytearray(encode(original))

        erasure_count = rng.randrange(33)
        error_count = rng.randrange((32 - erasure_count) // 2 + 1)

        positions = rng.sample(
            range(255), erasure_count + error_count
        )

        for position in positions:
            encoded[position] ^= rng.randrange(1, 256)

        result = decode_joint(
            bytes(encoded),
            erasure_positions=positions[:erasure_count],
        )

        assert result.data == original


def test_too_many_known_erasures():
    encoded = encode(bytes(range(223)))

    with pytest.raises(RecoveryError):
        decode_joint(
            encoded,
            erasure_positions=list(range(33)),
        )

@pytest.mark.parametrize(
    "errors,erasures",
    [
        (16, 0),
        (12, 8),
        (8, 16),
        (4, 24),
        (0, 32),
    ],
)
def test_exact_correction_boundary(errors, erasures):
    rng = random.Random(1000 + errors)

    for _ in range(20):
        original = bytes(
            rng.randrange(256) for _ in range(223)
        )

        encoded = bytearray(encode(original))

        positions = rng.sample(
            range(255), errors + erasures
        )

        for position in positions:
            encoded[position] ^= rng.randrange(1, 256)

        result = decode_joint(
            bytes(encoded),
            erasure_positions=positions[:erasures],
        )

        assert result.data == original
        assert result.corrected_positions == sorted(positions)
        