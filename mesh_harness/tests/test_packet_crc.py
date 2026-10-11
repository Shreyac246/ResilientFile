
"""Tests for CRC32-framed packet recovery."""

from mesh_harness.packet_crc import (
    frame_packet,
    verify_packet,
    recover_framed_packets,
)
from mesh_harness.run_mesh_demo import encode_packets
from core.erasure_decoder import RecoveryError

import pytest


def test_valid_packet():
    payload = bytes(range(200))
    frame = frame_packet(3, payload)

    assert len(frame) == 208
    assert verify_packet(frame, 3, 200) == payload


def test_corrupted_payload_rejected():
    payload = bytes(range(200))
    frame = bytearray(frame_packet(0, payload))

    frame[15] ^= 0xFF

    assert verify_packet(bytes(frame), 0, 200) is None


def test_corrupted_crc_rejected():
    payload = bytes(range(200))
    frame = bytearray(frame_packet(0, payload))

    frame[-1] ^= 1

    assert verify_packet(bytes(frame), 0, 200) is None


def test_wrong_packet_index_rejected():
    payload = bytes(range(200))
    frame = frame_packet(4, payload)

    assert verify_packet(frame, 5, 200) is None


def test_recovery_with_missing_and_corrupted_packets():
    original = bytes(range(256)) * 25

    packets = encode_packets(
        original, n=40, k=32, packet_size=200
    )

    frames = [
        frame_packet(i, packet)
        for i, packet in enumerate(packets)
    ]

    # Three missing packets.
    for index in [0, 1, 2]:
        frames[index] = None

    # Five received but corrupted packets.
    for index in [3, 4, 5, 6, 7]:
        damaged = bytearray(frames[index])
        damaged[15] ^= 0xFF
        frames[index] = bytes(damaged)

    recovered = recover_framed_packets(
        frames,
        n=40,
        k=32,
        original_length=len(original),
    )

    assert recovered == original


def test_recovery_fails_beyond_erasure_limit():
    original = bytes(range(256)) * 25

    packets = encode_packets(
        original, n=40, k=32, packet_size=200
    )

    frames = [
        frame_packet(i, packet)
        for i, packet in enumerate(packets)
    ]

    # Nine erasures exceed the RS(40,32) budget of eight.
    for index in range(9):
        frames[index] = None

    with pytest.raises(RecoveryError):
        recover_framed_packets(
            frames,
            n=40,
            k=32,
            original_length=len(original),
        )
