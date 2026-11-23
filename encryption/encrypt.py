"""Encrypt a predictions archive using the competition's public key."""
from __future__ import annotations
import argparse
from pathlib import Path
from .crypto import MAX_PAYLOAD_BYTES, atomic_write, configured_benchmark, encrypt_bytes, read_limited


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--public-key", type=Path, default=Path(__file__).resolve().with_name("public_key.pem"))
    parser.add_argument("--benchmark-id", default=None)
    parser.add_argument("--purpose", choices=["submission", "ground_truth"], default="submission")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    encrypted = encrypt_bytes(read_limited(args.input, MAX_PAYLOAD_BYTES), args.public_key.read_bytes(),
                              args.benchmark_id or configured_benchmark(), args.purpose)
    atomic_write(args.output, encrypted, overwrite=args.overwrite)
    print(f"Encrypted submission written to {args.output}")


if __name__ == "__main__":
    main()
