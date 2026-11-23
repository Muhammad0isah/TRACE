"""Versioned RSA-OAEP/SHA-256 + Fernet envelopes.

RSA protects a fresh symmetric key. Fernet authenticates the encrypted payload,
including the benchmark identifier, purpose and recipient-key fingerprint.
Only the public key belongs in the competition repository.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile

from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa

MAGIC = b"VCB1"
MAX_ENVELOPE_BYTES = 64 * 1024 * 1024
MAX_PAYLOAD_BYTES = 40 * 1024 * 1024
PURPOSES = {"submission", "ground_truth"}


class EnvelopeError(ValueError):
    """An invalid, unauthenticated or incorrectly addressed envelope."""


def _context(benchmark_id: str, purpose: str) -> None:
    if not isinstance(benchmark_id, str) or not benchmark_id or len(benchmark_id) > 160:
        raise EnvelopeError("A nonempty benchmark identifier of at most 160 characters is required")
    if purpose not in PURPOSES:
        raise EnvelopeError("Unsupported envelope purpose")


def _fingerprint(public_key: rsa.RSAPublicKey) -> str:
    der = public_key.public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    return hashlib.sha256(der).hexdigest()


def _oaep() -> padding.OAEP:
    return padding.OAEP(mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=MAGIC)


def encrypt_bytes(payload: bytes, public_pem: bytes, benchmark_id: str, purpose: str = "submission") -> bytes:
    _context(benchmark_id, purpose)
    if not isinstance(payload, bytes) or len(payload) > MAX_PAYLOAD_BYTES:
        raise EnvelopeError("Payload must be bytes and at most 40 MiB")
    key = serialization.load_pem_public_key(public_pem)
    if not isinstance(key, rsa.RSAPublicKey) or key.key_size < 2048:
        raise EnvelopeError("An RSA public key of at least 2048 bits is required")
    symmetric_key = Fernet.generate_key()
    wrapped = key.encrypt(symmetric_key, _oaep())
    packet = json.dumps({"version": 1, "benchmark_id": benchmark_id, "purpose": purpose,
                         "key_sha256": _fingerprint(key),
                         "payload_b64": base64.b64encode(payload).decode("ascii")},
                        separators=(",", ":"), sort_keys=True).encode("utf-8")
    encrypted = MAGIC + struct.pack(">H", len(wrapped)) + wrapped + Fernet(symmetric_key).encrypt(packet)
    if len(encrypted) > MAX_ENVELOPE_BYTES:
        raise EnvelopeError("Encrypted envelope exceeds 64 MiB")
    return encrypted


def decrypt_bytes(envelope: bytes, private_pem: bytes, benchmark_id: str, purpose: str = "submission") -> bytes:
    _context(benchmark_id, purpose)
    if not isinstance(envelope, bytes) or not 7 <= len(envelope) <= MAX_ENVELOPE_BYTES:
        raise EnvelopeError("Invalid envelope size (maximum 64 MiB)")
    if envelope[:4] != MAGIC:
        raise EnvelopeError("Unsupported envelope version")
    wrapped_size = struct.unpack(">H", envelope[4:6])[0]
    key = serialization.load_pem_private_key(private_pem, password=None)
    if not isinstance(key, rsa.RSAPrivateKey) or key.key_size < 2048:
        raise EnvelopeError("An RSA private key of at least 2048 bits is required")
    if wrapped_size != (key.key_size + 7) // 8 or len(envelope) <= 6 + wrapped_size:
        raise EnvelopeError("Invalid wrapped-key length")
    try:
        symmetric_key = key.decrypt(envelope[6:6 + wrapped_size], _oaep())
        packet = json.loads(Fernet(symmetric_key).decrypt(envelope[6 + wrapped_size:]))
        if (not isinstance(packet, dict) or packet.get("version") != 1
                or packet.get("benchmark_id") != benchmark_id or packet.get("purpose") != purpose
                or packet.get("key_sha256") != _fingerprint(key.public_key())):
            raise EnvelopeError("Envelope benchmark, purpose or recipient does not match")
        payload = base64.b64decode(packet["payload_b64"], validate=True)
        if len(payload) > MAX_PAYLOAD_BYTES:
            raise EnvelopeError("Decrypted payload exceeds 40 MiB")
        return payload
    except (InvalidToken, ValueError, TypeError, KeyError, UnicodeError) as exc:
        raise EnvelopeError("Envelope authentication or context validation failed") from exc


def read_limited(path: Path, limit: int = MAX_ENVELOPE_BYTES) -> bytes:
    with Path(path).open("rb") as handle:
        contents = handle.read(limit + 1)
    if len(contents) > limit:
        raise EnvelopeError(f"File exceeds the {limit}-byte limit")
    return contents


def atomic_write(path: Path, contents: bytes, overwrite: bool = False, mode: int = 0o644) -> None:
    """Publish a complete file atomically, refusing an existing path by default."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def configured_benchmark(root: Path | None = None) -> str:
    root = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    config = json.loads((root / "competition/config.json").read_text(encoding="utf-8"))
    benchmark_id = config["benchmark_id"]
    _context(benchmark_id, "submission")
    return benchmark_id
