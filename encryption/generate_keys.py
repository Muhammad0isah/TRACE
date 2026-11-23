"""Generate an independent organizer key pair; do not commit the private key."""
from __future__ import annotations
import argparse
from pathlib import Path
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from .crypto import atomic_write


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-out", type=Path, default=Path("evaluation-private/keys/submission_private_key.pem"))
    parser.add_argument("--public-out", type=Path, default=Path("encryption/public_key.pem"))
    parser.add_argument("--bits", type=int, choices=[2048, 3072, 4096], default=3072)
    args = parser.parse_args()
    if args.private_out.resolve() == args.public_out.resolve():
        parser.error("The private and public output paths must differ")
    if args.private_out.exists() or args.public_out.exists():
        parser.error("An output key already exists; preserve the existing pair or select new paths")
    key = rsa.generate_private_key(public_exponent=65537, key_size=args.bits)
    private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
    public = key.public_key().public_bytes(serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo)
    atomic_write(args.private_out, private, mode=0o600)
    atomic_write(args.public_out, public)
    print(f"Public key: {args.public_out}\nPrivate organizer key: {args.private_out} (keep secret)")


if __name__ == "__main__":
    main()
