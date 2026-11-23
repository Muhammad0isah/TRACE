#!/usr/bin/env python3
"""Validate organizer truth and publish encrypted evaluation assets atomically.

Run in an organizer checkout after data preparation and key generation. The
output directory contains ciphertext, a public image manifest, and hashes only.
Plaintext annotations and private keys are never copied into evaluation assets.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

# Support both `python scripts/seal_ground_truth.py` and module invocation.
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from competition import task
from cryptography.hazmat.primitives import serialization
from encryption.crypto import MAX_PAYLOAD_BYTES, encrypt_bytes, read_limited


def _json(raw: bytes) -> dict:
    def pairs_hook(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_hook,
                       parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number")
    return float(value)


def _validate_detection_truth(raw, manifest, benchmark_id):
    truth = _json(raw)
    if truth.get("benchmark_id", truth.get("info", {}).get("description")) != benchmark_id:
        raise ValueError("Ground-truth benchmark identity does not match config")
    truth_images = truth.get("images")
    categories = truth.get("categories")
    annotations = truth.get("annotations")
    if not isinstance(truth_images, list) or not isinstance(categories, list):
        raise ValueError("Ground truth requires COCO images and categories")
    if not isinstance(annotations, list) or not annotations:
        raise ValueError("Ground truth requires nonempty COCO annotations")
    expected = {row["image_id"]: row for row in manifest["images"]}
    actual = {}
    for row in truth_images:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or row["id"] in actual:
            raise ValueError("Invalid or duplicate ground-truth image ID")
        actual[row["id"]] = row
    if set(actual) != set(expected):
        raise ValueError("Ground truth and test manifest image IDs differ")
    for image_id, row in actual.items():
        if any(type(row.get(field)) is not int or row[field] != expected[image_id][field]
               for field in ("width", "height")):
            raise ValueError("Ground truth and test manifest dimensions differ")
    actual_categories = {}
    for row in categories:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] in actual_categories:
            raise ValueError("Invalid or duplicate ground-truth category")
        actual_categories[row["id"]] = row
    expected_categories = {row["id"]: row for row in manifest["categories"]}
    if actual_categories != expected_categories:
        raise ValueError("Ground truth and test manifest categories differ")
    annotation_ids = set()
    for row in annotations:
        if not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] in annotation_ids:
            raise ValueError("Invalid or duplicate ground-truth annotation ID")
        annotation_ids.add(row["id"])
        image_id, category_id = row.get("image_id"), row.get("category_id")
        if not isinstance(image_id, str) or image_id not in expected:
            raise ValueError("Unknown annotation image ID")
        if type(category_id) is not int or category_id not in expected_categories:
            raise ValueError("Unknown annotation category ID")
        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Ground-truth bbox must contain x,y,width,height")
        x, y, width, height = [_number(value, "bbox") for value in bbox]
        image = expected[image_id]
        if (min(x, y) < 0 or min(width, height) <= 0
                or x + width > image["width"] + 1e-6 or y + height > image["height"] + 1e-6):
            raise ValueError("Ground-truth box lies outside its native image")
        if _number(row.get("area"), "area") <= 0:
            raise ValueError("Ground-truth area must be positive")
        if type(row.get("iscrowd")) is not int or row["iscrowd"] not in (0, 1):
            raise ValueError("Ground-truth iscrowd must be 0 or 1")


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def seal_ground_truth(root=REPOSITORY_ROOT, *, truth_path=None, manifest_path=None,
                      public_key_path=None, output_path=None):
    """Return public sealing metadata, refusing any existing output directory."""
    root = Path(root).resolve()
    config = _json(read_limited(root / "competition/config.json", 1024 * 1024))
    benchmark_id, task_name = config.get("benchmark_id"), config.get("task")
    if task_name not in ("object_detection", "semantic_segmentation"):
        raise ValueError("Unsupported task in competition/config.json")
    extension = ".json" if task_name == "object_detection" else ".zip"
    truth_path = Path(truth_path) if truth_path is not None else root / f"evaluation-private/ground_truth{extension}"
    manifest_path = Path(manifest_path) if manifest_path is not None else root / "data/public/test_manifest.json"
    public_key_path = Path(public_key_path) if public_key_path is not None else root / "encryption/public_key.pem"
    output_path = Path(output_path) if output_path is not None else root / "evaluation"
    output_path = output_path.absolute()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError("Evaluation output already exists; use a new output directory")
    if truth_path.resolve().is_relative_to((root / "data/public").resolve()):
        raise ValueError("Organizer truth must not be stored in public participant data")
    manifest = task.load_manifest(manifest_path)
    if manifest["benchmark_id"] != benchmark_id:
        raise ValueError("Manifest benchmark does not match competition config")
    payload = read_limited(truth_path, MAX_PAYLOAD_BYTES)
    manifest_raw = read_limited(manifest_path, 1024 * 1024)
    if task_name == "semantic_segmentation":
        task.validate_submission(truth_path, manifest_path)
    else:
        _validate_detection_truth(payload, manifest, benchmark_id)
    public_pem = read_limited(public_key_path, 64 * 1024)
    encrypted = encrypt_bytes(payload, public_pem, benchmark_id, purpose="ground_truth")
    public_der = serialization.load_pem_public_key(public_pem).public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    metadata = {"version": 1, "benchmark_id": benchmark_id, "purpose": "ground_truth",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "image_count": len(manifest["images"]), "public_key_sha256": _digest(public_der),
        "files": {"ground_truth.enc": {"sha256": _digest(encrypted), "bytes": len(encrypted)},
                  "test_manifest.json": {"sha256": _digest(manifest_raw), "bytes": len(manifest_raw)}}}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A cooperating sealer cannot race a second process into the same output.
    lock_path = output_path.with_name(f".{output_path.name}.seal.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    stage = None
    try:
        os.close(descriptor)
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError("Evaluation output already exists")
        stage = Path(tempfile.mkdtemp(prefix=f".{output_path.name}.seal-", dir=output_path.parent))
        (stage / "ground_truth.enc").write_bytes(encrypted)
        (stage / "test_manifest.json").write_bytes(manifest_raw)
        (stage / "seal.json").write_text(json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        for file in stage.iterdir():
            file.chmod(0o644)
            with file.open("rb") as stream:
                os.fsync(stream.fileno())
        stage.chmod(0o755)
        if output_path.exists() or output_path.is_symlink():
            raise FileExistsError("Evaluation output appeared while sealing")
        stage.rename(output_path)
        stage = None
    finally:
        if stage is not None:
            shutil.rmtree(stage)
        lock_path.unlink()
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--truth", type=Path, help="Private prepared ground_truth.json or ground_truth.zip")
    parser.add_argument("--manifest", type=Path, help="Prepared public test_manifest.json")
    parser.add_argument("--public-key", type=Path, help="Repository RSA public key PEM")
    parser.add_argument("--output", type=Path, help="New evaluation directory (default: ROOT/evaluation)")
    args = parser.parse_args()
    try:
        metadata = seal_ground_truth(args.root, truth_path=args.truth, manifest_path=args.manifest,
            public_key_path=args.public_key, output_path=args.output)
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Sealing failed: {exc}\n")
    print(json.dumps(metadata, sort_keys=True, indent=2))
    print("Sealed evaluation assets created. Keep organizer plaintext and private keys private.")


if __name__ == "__main__":
    main()
