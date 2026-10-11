"""Send a file's bytes through a simulated lossy mesh link and prove
core.encoder / core.erasure_decoder recover it byte-perfect.

This is Layer 3. It calls Layer 1 (core.encoder.encode,
core.erasure_decoder.recover) directly, exactly the way an outside
project embedding resilientfile_core would -- there is no file
container, no CLI, and no dependency on cli/ anywhere in this module.
That is what makes the "standalone library" claim in the pitch true
rather than aspirational.

Packet-level erasure coding, concretely:
    A "segment" is k packets of packet_size bytes each. For every
    byte offset within the packets, the k bytes at that offset across
    the k data packets are treated as one k-symbol Reed-Solomon
    message and encoded into n symbols. Doing this for every offset
    produces n packets: the first k are the original data
    (systematic), the remaining n-k are parity packets. This is the
    standard way to turn a byte-oriented RS coder into a
    packet-oriented erasure code -- interleave across packets, one
    RS codeword per byte column.

    A dropped packet on a real network is a known-position erasure
    (the transport layer's sequence numbers tell you exactly which
    packet never arrived), not a random bit flip. So this harness
    always feeds erasure positions to core.erasure_decoder.recover,
    the same call encoder.py's own test suite already validates at
    up to n-k simultaneous erasures.

Run directly:
    python -m mesh_harness.run_mesh_demo samples/original.txt
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

from core.encoder import encode
from core.erasure_decoder import RecoveryError, recover

PACKET_SIZE = 200  # synthetic application payload size; not radio-calibrated


@dataclass
class TrialResult:
    """Outcome of sending one file once through one channel instance."""

    n: int
    k: int
    packet_size: int
    packets_total: int
    packets_lost: int
    recovered: bool
    reason: str = ""

    @property
    def loss_rate(self) -> float:
        if self.packets_total == 0:
            return 0.0
        return self.packets_lost / self.packets_total


@dataclass
class SweepResult:
    """Aggregate outcome across repeated trials at one channel setting."""

    label: str
    trials: list[TrialResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self.trials:
            return 0.0
        return sum(t.recovered for t in self.trials) / len(self.trials)

    @property
    def mean_loss_rate(self) -> float:
        if not self.trials:
            return 0.0
        return statistics.mean(t.loss_rate for t in self.trials)

def _validate_coding_parameters(n, k, packet_size):
    if any(type(value) is not int for value in (n, k, packet_size)):
        raise TypeError("n, k and packet_size must be integers.")

    if not 1 <= k < n <= 255:
        raise ValueError("Require 1 <= k < n <= 255.")

    if packet_size <= 0:
        raise ValueError("packet_size must be positive.")
    
def _pad_to_segment(data: bytes, k: int, packet_size: int) -> bytes:
    segment_size = k * packet_size
    remainder = len(data) % segment_size
    if remainder:
        data = data + b"\x00" * (segment_size - remainder)
    return data


def encode_packets(
    data: bytes, n: int, k: int, packet_size: int = PACKET_SIZE
) -> list[bytes]:
    """Turn raw bytes into a flat, ordered list of n-packet segments.

    Every group of n consecutive packets in the returned list is one
    independently recoverable segment. Packets are packet_size bytes
    each; the final data packet in the final segment may be zero
    padded if the input did not divide evenly.
    """
    _validate_coding_parameters(n, k, packet_size)
    
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes.")
    padded = _pad_to_segment(data, k, packet_size)
    packets: list[bytearray] = []
    for segment_start in range(0, len(padded), k * packet_size):
        segment = padded[segment_start : segment_start + k * packet_size]
        data_packets = [
            segment[i * packet_size : (i + 1) * packet_size] for i in range(k)
        ]
        out_packets = [bytearray(packet_size) for _ in range(n)]
        for column in range(packet_size):
            message = bytes(packet[column] for packet in data_packets)
            codeword = encode(message, n=n, k=k)
            for row in range(n):
                out_packets[row][column] = codeword[row]
        packets.extend(bytes(p) for p in out_packets)
    return packets


def decode_packets(
    packets: list[bytes],
    present: list[bool],
    n: int,
    k: int,
    original_length: int,
    packet_size: int = PACKET_SIZE,
) -> bytes:
    """Recover original bytes from the packets that arrived.

    present[i] is True when packets[i] survived.
    Missing packet contents are ignored.

    Supports up to n-k missing packets per segment, provided
    all surviving packets are correct.
    """
    _validate_coding_parameters(n, k, packet_size)

    if type(original_length) is not int or original_length < 0:
        raise ValueError(
            "original_length must be a nonnegative integer."
        )

    if len(packets) != len(present):
        raise ValueError(
            "packets and present must be the same length."
        )

    if any(type(value) is not bool for value in present):
        raise TypeError("present must contain booleans.")

    segment_size = k * packet_size
    expected_segments = (
        original_length + segment_size - 1
    ) // segment_size

    if len(packets) != expected_segments * n:
        raise ValueError(
            "Packet count does not match the original file length."
        )

    for packet, arrived in zip(packets, present):
        if arrived:
            if not isinstance(packet, bytes):
                raise TypeError("Received packets must be bytes.")

            if len(packet) != packet_size:
                raise ValueError(
                    "Received packet has an incorrect size."
                )

    recovered = bytearray()

    for segment_index in range(expected_segments):
        start = segment_index * n

        segment_packets = packets[start:start + n]
        segment_present = present[start:start + n]

        erased_positions = [
            i for i, arrived in enumerate(segment_present)
            if not arrived
        ]

        if len(erased_positions) > n - k:
            raise RecoveryError(
                f"Segment {segment_index} lost "
                f"{len(erased_positions)} packets; "
                f"the recovery limit is {n - k}."
            )

        data_packets = [
            bytearray(packet_size)
            for _ in range(k)
        ]

        for column in range(packet_size):
            codeword = bytearray(n)

            for row in range(n):
                if segment_present[row]:
                    codeword[row] = segment_packets[row][column]

            message = recover(
                bytes(codeword),
                erased_positions,
                n=n,
                k=k,
            )

            for row in range(k):
                data_packets[row][column] = message[row]

        for packet in data_packets:
            recovered.extend(packet)

    return bytes(recovered[:original_length])

def interleave(packets: list[bytes], n: int) -> list[bytes]:
    """Reorder packets so a channel burst hits many segments a little,
    instead of one segment a lot.

    Bursty loss (Gilbert-Elliott, or any real link) concentrates
    consecutive losses close together in send order. If packets are
    sent segment-by-segment, a single burst can knock out far more
    than one segment's n-k budget even while the file-wide loss rate
    looks safe. A block interleaver spreads each segment's packets
    across the whole transmission -- send column 0 of every segment,
    then column 1 of every segment, and so on -- so a contiguous
    burst lands on many different segments instead of hammering one.
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer.")
    segment_count = len(packets) // n
    if segment_count * n != len(packets):
        raise ValueError("packets length must be a multiple of n.")
    interleaved = []
    for column in range(n):
        for segment in range(segment_count):
            interleaved.append(packets[segment * n + column])
    return interleaved


def deinterleave(items: list, n: int, segment_count: int) -> list:
    """Invert interleave() for packets or their per-packet metadata."""
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer.")

    if type(segment_count) is not int or segment_count < 0:
        raise ValueError("segment_count must be a nonnegative integer.")

    if len(items) != n * segment_count:
        raise ValueError("Item count does not match the interleaver dimensions.")
    restored = [None] * (segment_count * n)
    index = 0
    for column in range(n):
        for segment in range(segment_count):
            restored[segment * n + column] = items[index]
            index += 1
    return restored


def run_trial(
    data: bytes,
    n: int,
    k: int,
    channel,
    packet_size: int = PACKET_SIZE,
    use_interleaving: bool = False,
) -> TrialResult:
    """Encode data, send it through channel once, and try to recover it."""
    packets = encode_packets(data, n, k, packet_size)
    segment_count = len(packets) // n
    if use_interleaving:
        wire_packets = interleave(packets, n)
    else:
        wire_packets = packets
    losses = channel.send(len(wire_packets))
    wire_present = [not lost for lost in losses]
    if use_interleaving:
        present = deinterleave(wire_present, n, segment_count)
    else:
        present = wire_present
    try:
        restored = decode_packets(
            packets, present, n, k, len(data), packet_size
        )
        recovered = restored == data
        reason = "" if recovered else "recovered bytes did not match the original"
    except RecoveryError as error:
        recovered = False
        reason = str(error)
    return TrialResult(
        n=n,
        k=k,
        packet_size=packet_size,
        packets_total=len(packets),
        packets_lost=sum(losses),
        recovered=recovered,
        reason=reason,
    )


def run_sweep(
    data: bytes,
    n: int,
    k: int,
    settings: list[tuple[str, dict]],
    trials_per_setting: int = 20,
    base_seed: int = 1000,
    packet_size: int = PACKET_SIZE,
    use_interleaving: bool = False,
) -> list[SweepResult]:
    """Run several channel configurations, each repeated trials_per_setting times."""
    from .channel_model import GilbertElliottChannel

    results = []
    seed = base_seed
    for label, params in settings:
        sweep = SweepResult(label=label)
        for _ in range(trials_per_setting):
            channel = GilbertElliottChannel(seed=seed, **params)
            seed += 1
            sweep.trials.append(
                run_trial(data, n, k, channel, packet_size, use_interleaving)
            )
        results.append(sweep)
    return results


def _print_report(results: list[SweepResult]) -> None:
    header = f"{'Channel':<26}{'Mean loss':>12}{'Recovered':>12}{'Trials':>9}"
    print(header)
    print("-" * len(header))
    for sweep in results:
        print(
            f"{sweep.label:<26}"
            f"{sweep.mean_loss_rate * 100:>10.1f}% "
            f"{sweep.success_rate * 100:>10.1f}% "
            f"{len(sweep.trials):>8}"
        )


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("Usage: python -m mesh_harness.run_mesh_demo <input file>", file=sys.stderr)
        return 1
    input_path = Path(argv[0])
    data = input_path.read_bytes()
    n, k = 40, 32  # (n,k): survives up to 8 lost packets per 40-packet segment
    settings = [
        ("Mild (occasional drop)", dict(p_good_to_bad=0.01, p_bad_to_good=0.5,
                                         loss_in_good=0.005, loss_in_bad=0.4)),
        ("Moderate burst", dict(p_good_to_bad=0.03, p_bad_to_good=0.35,
                                       loss_in_good=0.01, loss_in_bad=0.75)),
        ("Severe burst", dict(p_good_to_bad=0.08, p_bad_to_good=0.2,
                                     loss_in_good=0.02, loss_in_bad=0.9)),
    ]
    print(f"Input: {input_path} ({len(data)} bytes)")
    print(f"Code: packet-level RS({n},{k}), packet size {PACKET_SIZE} bytes "
          f"(survives up to {n - k} lost packets per {n}-packet segment)\n")

    print("Without interleaving (segment-sequential transmission):")
    results_plain = run_sweep(data, n, k, settings, trials_per_setting=50,
                               use_interleaving=False)
    _print_report(results_plain)

    print("\nWith block interleaving (packets spread across segments):")
    results_interleaved = run_sweep(data, n, k, settings, trials_per_setting=50,
                                     use_interleaving=True)
    _print_report(results_interleaved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())