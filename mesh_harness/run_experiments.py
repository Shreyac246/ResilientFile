"""Paired packet-loss experiments with CSV results and run metadata."""

import argparse
import csv
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

from core.erasure_decoder import RecoveryError
from .channel_model import GilbertElliottChannel
from .run_mesh_demo import (
    encode_packets,
    decode_packets,
    deinterleave,
)


SETTINGS = [
    (
        "Mild",
        dict(
            p_good_to_bad=0.01,
            p_bad_to_good=0.5,
            loss_in_good=0.005,
            loss_in_bad=0.4,
        ),
    ),
    (
        "Moderate",
        dict(
            p_good_to_bad=0.03,
            p_bad_to_good=0.35,
            loss_in_good=0.01,
            loss_in_bad=0.75,
        ),
    ),
    (
        "Severe",
        dict(
            p_good_to_bad=0.08,
            p_bad_to_good=0.2,
            loss_in_good=0.02,
            loss_in_bad=0.9,
        ),
    ),
]


def evaluate_trace(
    data, packets, losses, n, k, packet_size, interleaved
):
    """Evaluate one fixed loss trace under one transmission order."""
    if len(losses) != len(packets):
        raise ValueError("Loss trace and packet counts must match.")

    if any(type(value) is not bool for value in losses):
        raise TypeError("Loss trace must contain booleans.")

    segment_count = len(packets) // n
    present = [not lost for lost in losses]

    if interleaved:
        present = deinterleave(present, n, segment_count)

    segment_losses = [
        sum(not ok for ok in present[start:start + n])
        for start in range(0, len(present), n)
    ]

    within_budget = all(
        count <= n - k for count in segment_losses
    )

    reason = ""

    try:
        restored = decode_packets(
            packets, present, n, k, len(data), packet_size
        )

        recovered = restored == data

        if not recovered:
            reason = "Decoded bytes differ from the original."

    except RecoveryError as error:
        recovered = False
        reason = str(error)

    return {
        "mode": "interleaved" if interleaved else "sequential",
        "packets_total": len(packets),
        "packets_lost": sum(losses),
        "loss_rate": (
            sum(losses) / len(packets) if packets else 0.0
        ),
        "segment_losses": json.dumps(segment_losses),
        "max_segment_loss": max(segment_losses, default=0),
        "segments_over_budget": sum(
            count > n - k for count in segment_losses
        ),
        "within_budget": within_budget,
        "recovered": recovered,
        "reason": reason,
    }


def source_hashes():
    """Record which source versions were used in the experiment."""
    root = Path(__file__).resolve().parent.parent

    paths = list((root / "core").glob("*.py"))
    paths += [
        Path(__file__),
        Path(__file__).with_name("channel_model.py"),
        Path(__file__).with_name("run_mesh_demo.py"),
    ]

    return {
        str(path.relative_to(root)).replace("\\", "/"):
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(set(paths))
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--packet-size", type=int, default=200)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        if args.trials < 1:
            raise ValueError("Trials must be positive.")

        if (
            not 1 <= args.k < args.n <= 255
            or args.packet_size < 1
        ):
            raise ValueError(
                "Invalid coding parameters or packet size."
            )

        data = args.input.read_bytes()

        if not data:
            raise ValueError("Use a nonempty input file.")

        # Refuse to overwrite an earlier experiment directory.
        args.out.mkdir(parents=True, exist_ok=False)

    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")

    metadata = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "input_name": args.input.name,
        "input_bytes": len(data),
        "input_sha256": hashlib.sha256(data).hexdigest(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "n": args.n,
        "k": args.k,
        "packet_size": args.packet_size,
        "trials_per_setting": args.trials,
        "base_seed": args.seed,
        "seed_rule": (
            "base_seed + trial_index * 3 + setting_index "
            "(zero-based)"
        ),
        "settings": [
            {"label": label, **params}
            for label, params in SETTINGS
        ],
        "initial_state": (
            "good; channel transitions before each loss decision"
        ),
        "assumptions": [
            "Metadata and sequence indexes are reliably available.",
            "Surviving packets are correct; loss only, no retransmission.",
            "Synthetic settings; no radio headers or airtime modeled.",
            "Both modes use FEC and the same loss trace for each pair.",
        ],
        "source_sha256": source_hashes(),
    }

    metadata_path = args.out / "metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("Encoding input once for all trials...", flush=True)

    packets = encode_packets(
        data, args.n, args.k, args.packet_size
    )

    segments = len(packets) // args.n
    metadata["segments"] = segments
    metadata["packets_per_transmission"] = len(packets)

    print(
        f"Segments: {segments}; "
        f"packets per transmission: {len(packets)}",
        flush=True,
    )

    if segments == 1:
        print(
            "Only one segment: interleaving cannot change its ordering."
        )

    fields = [
        "setting",
        "trial",
        "seed",
        "trace_sha256",
        "mode",
        "packets_total",
        "packets_lost",
        "loss_rate",
        "segment_losses",
        "max_segment_loss",
        "segments_over_budget",
        "within_budget",
        "recovered",
        "reason",
    ]

    rows = []

    with (args.out / "trials.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for setting_index, (label, params) in enumerate(SETTINGS):
            for trial in range(args.trials):
                seed = (
                    args.seed
                    + trial * len(SETTINGS)
                    + setting_index
                )

                channel = GilbertElliottChannel(
                    seed=seed, **params
                )
                losses = channel.send(len(packets))

                trace_hash = hashlib.sha256(
                    bytes(losses)
                ).hexdigest()

                # Both modes receive the exact same loss trace.
                for interleaved in (False, True):
                    result = evaluate_trace(
                        data,
                        packets,
                        losses,
                        args.n,
                        args.k,
                        args.packet_size,
                        interleaved,
                    )

                    row = {
                        "setting": label,
                        "trial": trial + 1,
                        "seed": seed,
                        "trace_sha256": trace_hash,
                        **result,
                    }

                    writer.writerow(row)
                    rows.append(row)

                if (
                    (trial + 1) % 25 == 0
                    or trial + 1 == args.trials
                ):
                    handle.flush()
                    print(
                        f"{label}: {trial + 1}/{args.trials} "
                        "paired trials",
                        flush=True,
                    )

    summary = []

    for label, _ in SETTINGS:
        for mode in ("sequential", "interleaved"):
            subset = [
                row for row in rows
                if row["setting"] == label and row["mode"] == mode
            ]

            successes = sum(row["recovered"] for row in subset)

            mean_loss = (
                sum(row["packets_lost"] for row in subset)
                / sum(row["packets_total"] for row in subset)
            )

            summary.append({
                "setting": label,
                "mode": mode,
                "trials": len(subset),
                "successes": successes,
                "recovery_rate": successes / len(subset),
                "mean_loss_rate": mean_loss,
                "failures_within_budget": sum(
                    row["within_budget"] and not row["recovered"]
                    for row in subset
                ),
            })

            print(
                f"{label:9} {mode:12} "
                f"recovered {successes}/{len(subset)}; "
                f"mean loss {mean_loss:.2%}"
            )

    with (args.out / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0])
        )
        writer.writeheader()
        writer.writerows(summary)

    anomalies = sum(
        row["failures_within_budget"] for row in summary
    )

    metadata["status"] = (
        "completed" if anomalies == 0
        else "completed_with_anomalies"
    )
    metadata["failures_within_budget"] = anomalies
    metadata["finished_utc"] = (
        datetime.now(timezone.utc).isoformat()
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved results in: {args.out}")

    if anomalies:
        print(
            "Unexpected failures occurred within the recovery budget; "
            "investigate before reporting."
        )

    return 1 if anomalies else 0


if __name__ == "__main__":
    raise SystemExit(main())