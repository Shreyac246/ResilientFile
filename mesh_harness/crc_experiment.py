
"""Reproducible CRC-assisted mesh recovery experiments."""

import random
from dataclasses import dataclass

from core.erasure_decoder import RecoveryError
from .channel_model import GilbertElliottChannel
from .run_mesh_demo import (
    encode_packets,
    interleave,
    deinterleave,
    PACKET_SIZE,
)
from .packet_crc import (
    frame_packet,
    verify_packet,
    recover_framed_packets,
)


@dataclass
class CRCTrialResult:
    packets_total: int
    packets_dropped: int
    packets_corrupted: int
    packets_crc_rejected: int
    effective_erasures: int
    recovered: bool
    reason: str = ""

    @property
    def recovery_rate(self):
        return float(self.recovered)


def run_crc_trial(
    data: bytes,
    n: int = 40,
    k: int = 32,
    packet_size: int = PACKET_SIZE,
    corruption_probability: float = 0.03,
    seed: int = 42,
    use_interleaving: bool = False,
    channel_params: dict | None = None,
) -> CRCTrialResult:
    """Simulate packet loss, CRC-detected corruption, and recovery."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")

    if not 0 <= corruption_probability <= 1:
        raise ValueError("Invalid corruption probability.")

    if type(seed) is not int:
        raise TypeError("Seed must be an integer.")

    # 1. Reed-Solomon packet encoding.
    packets = encode_packets(
        data, n=n, k=k, packet_size=packet_size
    )

    segment_count = len(packets) // n

    # 2. Add original packet indexes and CRC32.
    frames = [
        frame_packet(i, packet)
        for i, packet in enumerate(packets)
    ]

    # 3. Change transmission order if requested.
    if use_interleaving:
        wire_frames = interleave(frames, n)
    else:
        wire_frames = frames

    # 4. Configure the burst-loss channel.
    if channel_params is None:
        channel_params = dict(
            p_good_to_bad=0.03,
            p_bad_to_good=0.35,
            loss_in_good=0.01,
            loss_in_bad=0.75,
        )

    channel = GilbertElliottChannel(
        seed=seed,
        **channel_params,
    )

    losses = channel.send(len(wire_frames))

    # Separate seeded generator for corruption.
    rng = random.Random(seed + 1000000)

    received_wire = []
    corrupted_count = 0

    for frame, lost in zip(wire_frames, losses):

        if lost:
            received_wire.append(None)
            continue

        # Simulate corruption only in arrived packets.
        if rng.random() < corruption_probability:
            damaged = bytearray(frame)

            # Modify one byte of the packet payload.
            byte_index = rng.randrange(4, 4 + packet_size)
            damaged[byte_index] ^= rng.randrange(1, 256)

            received_wire.append(bytes(damaged))
            corrupted_count += 1

        else:
            received_wire.append(frame)

    # 5. Restore original packet-slot order.
    if use_interleaving:
        received = deinterleave(
            received_wire, n, segment_count
        )
    else:
        received = received_wire

    # 6. Count packets rejected by CRC32.
    crc_rejected = 0

    for index, frame in enumerate(received):
        if frame is not None:
            if verify_packet(
                frame, index, packet_size
            ) is None:
                crc_rejected += 1

    dropped_count = sum(losses)
    effective_erasures = dropped_count + crc_rejected

    # 7. Recover the file through the existing adapter.
    try:
        restored = recover_framed_packets(
            received,
            n=n,
            k=k,
            original_length=len(data),
            packet_size=packet_size,
        )

        success = restored == data

        reason = (
            ""
            if success
            else "Recovered content does not match original."
        )

    except RecoveryError as error:
        success = False
        reason = str(error)

    return CRCTrialResult(
        packets_total=len(packets),
        packets_dropped=dropped_count,
        packets_corrupted=corrupted_count,
        packets_crc_rejected=crc_rejected,
        effective_erasures=effective_erasures,
        recovered=success,
        reason=reason,
    )


def run_crc_sweep(
    data: bytes,
    trials: int = 100,
    corruption_probability: float = 0.03,
    use_interleaving: bool = False,
    channel_params: dict | None = None,
):
    """Repeat experiments and calculate aggregate metrics."""

    if type(trials) is not int or trials <= 0:
        raise ValueError("trials must be a positive integer.")

    results = []

    for trial in range(trials):
        result = run_crc_trial(
            data=data,
            corruption_probability=corruption_probability,
            seed=1000 + trial,
            use_interleaving=use_interleaving,
            channel_params=channel_params,
        )
        results.append(result)

    return {
        "trials": trials,
        "successes": sum(r.recovered for r in results),
        "recovery_rate": (
            sum(r.recovered for r in results) / trials
        ),
        "mean_loss_rate": sum(
            r.packets_dropped / r.packets_total
            if r.packets_total else 0
            for r in results
        ) / trials,
        "mean_crc_rejections": sum(
            r.packets_crc_rejected for r in results
        ) / trials,
        "mean_effective_erasures": sum(
            r.effective_erasures for r in results
        ) / trials,
    }
