"""Tests for the Gilbert-Elliott bursty-loss channel model."""

import pytest

from mesh_harness.channel_model import GilbertElliottChannel


def test_deterministic_with_seed():
    a = GilbertElliottChannel(seed=42).send(500)
    b = GilbertElliottChannel(seed=42).send(500)
    assert a == b


def test_different_seeds_usually_differ():
    a = GilbertElliottChannel(seed=1).send(200)
    b = GilbertElliottChannel(seed=2).send(200)
    assert a != b


def test_send_length_matches_request():
    channel = GilbertElliottChannel(seed=1)
    assert len(channel.send(0)) == 0
    assert len(channel.send(37)) == 37


def test_negative_packet_count_rejected():
    with pytest.raises(ValueError):
        GilbertElliottChannel(seed=1).send(-1)


def test_state_carries_over_between_calls():
    """Two calls totaling N should not equal one call of N in general,
    but both must be internally consistent (deterministic replay)."""
    a = GilbertElliottChannel(seed=5)
    first = a.send(100)
    second = a.send(100)
    combined_a = first + second

    b = GilbertElliottChannel(seed=5)
    combined_b = b.send(200)

    assert combined_a == combined_b


def test_reset_replays_same_sequence():
    channel = GilbertElliottChannel(seed=9)
    first_run = channel.send(150)
    channel.reset()
    second_run = channel.send(150)
    assert first_run == second_run


def test_out_of_range_probability_rejected():
    with pytest.raises(ValueError):
        GilbertElliottChannel(loss_in_bad=1.5)
    with pytest.raises(ValueError):
        GilbertElliottChannel(p_good_to_bad=-0.1)


def test_all_good_state_never_enters_bad():
    """If the good-to-bad transition probability is zero, loss should
    stay near the low, good-state rate across a long run."""
    channel = GilbertElliottChannel(
        seed=3, p_good_to_bad=0.0, loss_in_good=0.02, loss_in_bad=0.9
    )
    losses = channel.send(5000)
    rate = sum(losses) / len(losses)
    assert rate < 0.05


def test_always_bad_state_uses_bad_loss_rate():
    channel = GilbertElliottChannel(
        seed=4, p_good_to_bad=1.0, p_bad_to_good=0.0,
        loss_in_good=0.0, loss_in_bad=1.0,
    )
    losses = channel.send(50)
    # First packet may still see the good-state roll before transitioning;
    # from the second packet on the channel should be fully in the bad
    # state with loss probability 1.0.
    assert all(losses[1:])


def test_bursts_cluster_more_than_uniform_would():
    """A channel with a sticky bad state should produce longer average
    loss runs than an equivalent-rate uniform random loss process."""
    channel = GilbertElliottChannel(
        seed=6, p_good_to_bad=0.02, p_bad_to_good=0.1,
        loss_in_good=0.01, loss_in_bad=0.8,
    )
    losses = channel.send(20000)

    def mean_run_length(sequence):
        runs = []
        run = 0
        for value in sequence:
            if value:
                run += 1
            elif run:
                runs.append(run)
                run = 0
        if run:
            runs.append(run)
        return sum(runs) / len(runs) if runs else 0

    observed_rate = sum(losses) / len(losses)
    import random as _random

    rng = _random.Random(6)
    uniform = [
        rng.random() < observed_rate
        for _ in range(20000)
    ]
    assert mean_run_length(losses) > mean_run_length(uniform)