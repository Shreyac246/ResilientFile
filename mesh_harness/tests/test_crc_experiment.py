
"""Tests for the CRC-assisted transmission experiment."""

import pytest

from mesh_harness.crc_experiment import (
    run_crc_trial,
    run_crc_sweep,
)


NO_LOSS = {
    "p_good_to_bad": 0.0,
    "p_bad_to_good": 1.0,
    "loss_in_good": 0.0,
    "loss_in_bad": 0.0,
}


def test_perfect_channel():
    data = b"ResilientFile" * 100

    result = run_crc_trial(
        data,
        corruption_probability=0.0,
        channel_params=NO_LOSS,
    )

    assert result.recovered
    assert result.packets_dropped == 0
    assert result.packets_corrupted == 0
    assert result.packets_crc_rejected == 0


def test_every_packet_corrupted():
    data = b"ResilientFile" * 100

    result = run_crc_trial(
        data,
        corruption_probability=1.0,
        channel_params=NO_LOSS,
    )

    assert not result.recovered
    assert result.packets_corrupted == result.packets_total
    assert result.packets_crc_rejected == result.packets_total


def test_interleaving_with_perfect_channel():
    data = b"ResilientFile" * 1000

    result = run_crc_trial(
        data,
        corruption_probability=0.0,
        use_interleaving=True,
        channel_params=NO_LOSS,
    )

    assert result.recovered


def test_reproducible_experiment():
    data = b"ResilientFile" * 200

    a = run_crc_trial(
        data,
        seed=123,
        corruption_probability=0.1,
    )

    b = run_crc_trial(
        data,
        seed=123,
        corruption_probability=0.1,
    )

    assert a == b


def test_sweep_statistics():
    data = b"ResilientFile" * 100

    report = run_crc_sweep(
        data,
        trials=5,
        corruption_probability=0.0,
        channel_params=NO_LOSS,
    )

    assert report["trials"] == 5
    assert report["successes"] == 5
    assert report["recovery_rate"] == 1.0
    assert report["mean_effective_erasures"] == 0.0


def test_invalid_trial_count():
    with pytest.raises(ValueError):
        run_crc_sweep(b"data", trials=0)
