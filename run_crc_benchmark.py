
from mesh_harness.crc_experiment import run_crc_sweep

data = bytes(range(256)) * 75

settings = [
    (
        "Mild",
        {
            "p_good_to_bad": 0.01,
            "p_bad_to_good": 0.50,
            "loss_in_good": 0.005,
            "loss_in_bad": 0.40,
        },
        0.01,
    ),
    (
        "Moderate",
        {
            "p_good_to_bad": 0.03,
            "p_bad_to_good": 0.35,
            "loss_in_good": 0.01,
            "loss_in_bad": 0.75,
        },
        0.03,
    ),
    (
        "Severe",
        {
            "p_good_to_bad": 0.08,
            "p_bad_to_good": 0.20,
            "loss_in_good": 0.02,
            "loss_in_bad": 0.90,
        },
        0.05,
    ),
]

print(
    f"{'Scenario':<12} {'Mode':<14} "
    f"{'Recovery':>10} {'Loss':>10} "
    f"{'CRC rejects':>13}"
)
print("-" * 65)

for name, channel, corruption in settings:
    for interleaved in (False, True):
        result = run_crc_sweep(
            data,
            trials=50,
            corruption_probability=corruption,
            use_interleaving=interleaved,
            channel_params=channel,
        )

        print(
            f"{name:<12} "
            f"{'Interleaved' if interleaved else 'Sequential':<14} "
            f"{result['recovery_rate'] * 100:>9.1f}% "
            f"{result['mean_loss_rate'] * 100:>9.1f}% "
            f"{result['mean_crc_rejections']:>13.2f}"
        )
