"""Tests for the mesh-loss integration harness (Layer 3).

These tests call core.encoder / core.erasure_decoder only through the
harness's own encode_packets/decode_packets functions, the same path
run_mesh_demo.py itself uses -- never core.container -- because Layer
3 is supposed to prove Layer 1 works standalone, without the file
container from Layer 2.
"""

import random

import pytest

from core.erasure_decoder import RecoveryError
from mesh_harness.channel_model import GilbertElliottChannel
from mesh_harness.run_mesh_demo import (
    decode_packets,
    deinterleave,
    encode_packets,
    interleave,
    run_trial,
)

def test_invalid_packet_inputs_are_rejected():
    with pytest.raises(ValueError):
        encode_packets(b"HELLO", n=10, k=0, packet_size=50)

    with pytest.raises(ValueError):
        encode_packets(b"HELLO", n=10, k=6, packet_size=0)

    with pytest.raises(ValueError):
        decode_packets(
            [], [], n=10, k=6,
            original_length=5, packet_size=50,
        )

    with pytest.raises(ValueError):
        decode_packets(
            [b"x"] * 10, [True] * 10,
            n=10, k=6, original_length=5, packet_size=50,
        )

    with pytest.raises(ValueError):
        deinterleave([b"a", b"b", b"c"], n=2, segment_count=1)
def _random_bytes(length, seed):
    rng = random.Random(seed)
    return bytes(rng.randrange(256) for _ in range(length))


def test_round_trip_no_loss():
    data = _random_bytes(4000, seed=1)
    n, k, packet_size = 10, 6, 50
    packets = encode_packets(data, n, k, packet_size)
    present = [True] * len(packets)
    restored = decode_packets(packets, present, n, k, len(data), packet_size)
    assert restored == data


def test_round_trip_at_exact_erasure_limit():
    """Every segment loses exactly n-k packets: must still recover."""
    data = _random_bytes(4000, seed=2)
    n, k, packet_size = 10, 6, 50
    packets = encode_packets(data, n, k, packet_size)
    segment_count = len(packets) // n
    present = [True] * len(packets)
    for segment in range(segment_count):
        for offset in range(n - k):  # drop the first (n-k) packets of each segment
            present[segment * n + offset] = False
    restored = decode_packets(packets, present, n, k, len(data), packet_size)
    assert restored == data


def test_one_loss_past_limit_raises_cleanly():
    data = _random_bytes(4000, seed=3)
    n, k, packet_size = 10, 6, 50
    packets = encode_packets(data, n, k, packet_size)
    present = [True] * len(packets)
    for offset in range(n - k + 1):  # one more than the budget
        present[offset] = False
    with pytest.raises(RecoveryError):
        decode_packets(packets, present, n, k, len(data), packet_size)


def test_encode_packets_is_systematic():
    """The first k packets of each segment must equal the raw input."""
    data = _random_bytes(1000, seed=4)
    n, k, packet_size = 10, 6, 50
    packets = encode_packets(data, n, k, packet_size)
    padded = data + b"\x00" * ((-len(data)) % (k * packet_size))
    for i in range(k):
        assert packets[i] == padded[i * packet_size : (i + 1) * packet_size]


def test_interleave_round_trips():
    packets = [bytes([i]) for i in range(24)]  # 4 segments of n=6
    n = 6
    mixed = interleave(packets, n)
    restored = deinterleave(mixed, n, segment_count=4)
    assert restored == packets


def test_interleave_rejects_non_multiple_length():
    with pytest.raises(ValueError):
        interleave([b"a", b"b", b"c"], n=2)


def test_run_trial_reports_consistent_totals():
    data = _random_bytes(4000, seed=5)
    channel = GilbertElliottChannel(seed=123)
    result = run_trial(data, n=10, k=6, channel=channel, packet_size=50)
    assert result.packets_total > 0
    assert 0 <= result.packets_lost <= result.packets_total
    assert isinstance(result.recovered, bool)


class FixedLossChannel:
    """Produce an exact loss pattern for a deterministic test."""

    def __init__(self, lost_positions):
        self.lost_positions = set(lost_positions)

    def send(self, packet_count):
        return [
            position in self.lost_positions
            for position in range(packet_count)
        ]


def test_interleaving_recovers_a_consecutive_burst():
    # Three segments, each containing 4 data packets of 2 bytes.
    data = bytes(range(24))

    plain = run_trial(
        data, 6, 4, FixedLossChannel([0, 1, 2]),
        packet_size=2, use_interleaving=False,
    )

    interleaved = run_trial(
        data, 6, 4, FixedLossChannel([0, 1, 2]),
        packet_size=2, use_interleaving=True,
    )

    assert not plain.recovered
    assert interleaved.recovered


def test_interleaving_can_concentrate_a_different_loss_pattern():
    data = bytes(range(24))

    plain = run_trial(
        data, 6, 4, FixedLossChannel([0, 6, 12]),
        packet_size=2, use_interleaving=False,
    )

    interleaved = run_trial(
        data, 6, 4, FixedLossChannel([0, 6, 12]),
        packet_size=2, use_interleaving=True,
    )

    assert plain.recovered
    assert not interleaved.recovered