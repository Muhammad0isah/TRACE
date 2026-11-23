"""Organizer-only envelope decryption. Never publish the private key."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
from .crypto import atomic_write, configured_benchmark, decrypt_bytes, read_limited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--private-key", type=Path)
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--purpose", choices=["submission", "ground_truth"], default="submission")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    private_pem = args.private_key.read_bytes() if args.private_key else os.environ.get("SUBMISSION_PRIVATE_KEY", "").encode()
    if not private_pem:
        parser.error("Provide --private-key or the SUBMISSION_PRIVATE_KEY environment secret")
    payload = decrypt_bytes(read_limited(args.input), private_pem,
                            args.benchmark_id or configured_benchmark(), args.purpose)
    atomic_write(args.output, payload, overwrite=args.overwrite, mode=0o600)
    print(f"Decrypted payload written to {args.output}")


if __name__ == "__main__":
    main()
