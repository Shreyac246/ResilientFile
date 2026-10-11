"""Gilbert-Elliott bursty packet-loss channel simulator.

Real LoRa and Bluetooth mesh links do not lose packets uniformly at
random. Interference, distance, and obstacles cause loss to cluster in
bursts, then clear up for a while. A flat "drop X% of packets
independently" model misses this and understates how a real link
behaves. The Gilbert-Elliott model is a two-state Markov chain -- a
GOOD state with low loss and a BAD state with high loss -- that
switches between the two with fixed probabilities, producing bursty
loss patterns much closer to a real mesh link.

This module knows nothing about Reed-Solomon, files, or packet
contents. It only decides, for a sequence of packets numbered in
order, which ones are lost. A lost packet in this model is a dropped
packet, not a corrupted one: on a real transport, a missing sequence
number is an erasure at a known position, which is exactly what
core.erasure_decoder.recover is built to fix.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class GilbertElliottChannel:
    """A two-state bursty-loss channel.

    p_good_to_bad / p_bad_to_good control how often the channel
    switches states (and therefore how long a burst tends to last).
    loss_in_good / loss_in_bad control the loss probability within
    each state. Defaults describe a link that is usually fine but
    occasionally suffers short, lossy bursts.
    """

    p_good_to_bad: float = 0.03
    p_bad_to_good: float = 0.35
    loss_in_good: float = 0.01
    loss_in_bad: float = 0.75
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("p_good_to_bad", "p_bad_to_good", "loss_in_good", "loss_in_bad"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1, got {value}.")
        self._rng = random.Random(self.seed)
        self._state = "good"

    def reset(self) -> None:
        """Return the channel to its initial state and RNG seed."""
        self._rng = random.Random(self.seed)
        self._state = "good"

    def send(self, packet_count: int) -> list[bool]:
        """Simulate sending packet_count packets in sequence.

        Returns a list of booleans the same length as packet_count,
        where True means that packet was lost. The channel's internal
        state carries over between calls, so calling this repeatedly
        on the same instance continues the same burst process.
        """
        if packet_count < 0:
            raise ValueError("packet_count must not be negative.")
        losses = []
        for _ in range(packet_count):
            if self._state == "good":
                if self._rng.random() < self.p_good_to_bad:
                    self._state = "bad"
            else:
                if self._rng.random() < self.p_bad_to_good:
                    self._state = "good"
            loss_probability = (
                self.loss_in_bad if self._state == "bad" else self.loss_in_good
            )
            losses.append(self._rng.random() < loss_probability)
        return losses