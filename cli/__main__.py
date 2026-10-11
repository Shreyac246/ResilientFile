"""Command-line file protection and recovery.

Run from the project root:
    python -m cli protect input output
    python -m cli recover input output
"""

import argparse
import sys
from pathlib import Path

from core.container import protect_bytes, recover_bytes


def main():
    parser = argparse.ArgumentParser(
        description="Protect and recover files with Reed-Solomon coding."
    )

    commands = parser.add_subparsers(dest="command", required=True)

    protect_parser = commands.add_parser(
        "protect", help="Create a protected .rf file."
    )
    protect_parser.add_argument("input", type=Path)
    protect_parser.add_argument("output", type=Path)
    protect_parser.add_argument("--n", type=int, default=255)
    protect_parser.add_argument("--k", type=int, default=223)

    recover_parser = commands.add_parser(
        "recover", help="Recover and verify a protected file."
    )
    recover_parser.add_argument("input", type=Path)
    recover_parser.add_argument("output", type=Path)

    args = parser.parse_args()

    try:
        if args.output.exists():
            raise FileExistsError(
                "Output already exists. Choose a different output filename."
            )

        source = args.input.read_bytes()

        if args.command == "protect":
            result = protect_bytes(source, n=args.n, k=args.k)
        else:
            result = recover_bytes(source)

        # Exclusive creation also prevents overwriting if a file
        # appears after the earlier existence check.
        with args.output.open("xb") as output_file:
            output_file.write(result)

    except (OSError, ValueError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    if args.command == "protect":
        print(f"Protected: {args.input} -> {args.output}")
        print(f"Original bytes: {len(source)}")
        print(f"Protected bytes: {len(result)}")
        print(
            f"Code: RS({args.n},{args.k}); "
            f"up to {(args.n - args.k) // 2} unknown byte errors "
            "per encoded block."
        )
    else:
        print(f"Recovered: {args.output}")
        print(f"Recovered bytes: {len(result)}")
        print("SHA-256 verification: PASSED")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())