"""Prepare the complete official Laboro Tomato release without extracting ZIP paths.

Usage: python scripts/prepare_data.py --source downloads/laboro_tomato.zip
The CLI always enforces the full official release counts. It never subsamples.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import stat
import sys
import tempfile
import zipfile

import numpy as np
from PIL import Image, ImageOps, __version__ as PILLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from competition.task import CONFIG, BENCHMARK_ID, MAX_DETECTIONS

SOURCE_URL = "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip"
NAMES = {"b_fully_ripened", "b_half_ripened", "b_green", "l_fully_ripened",
         "l_half_ripened", "l_green"}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def safe_name(value: str) -> str:
    if not isinstance(value, str) or "\\" in value:
        raise ValueError("Source paths must be relative POSIX paths")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Unsafe source path: {value}")
    return path.as_posix()


class Source:
    """Read folder or archive members; no member is ever extracted to disk."""
    def __init__(self, source: Path):
        self.path = source.resolve()
        self.archive = None
        if self.path.is_file():
            self.archive = zipfile.ZipFile(self.path)
            names = []
            for entry in self.archive.infolist():
                name = safe_name(entry.filename)
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("ZIP symlinks are not accepted")
                if not entry.is_dir():
                    names.append(name)
            if len(names) != len(set(names)):
                raise ValueError("ZIP contains duplicate member names")
            self.names = set(names)
            candidates = [name[:-len("annotations/train.json")] for name in names
                          if name.endswith("annotations/train.json") and
                          name[:-len("annotations/train.json")] + "annotations/test.json"
                          in self.names]
            if len(candidates) != 1:
                raise ValueError("Expected exactly one annotations/train.json + test.json pair")
            self.prefix = candidates[0]
        elif self.path.is_dir():
            candidates = [self.path, self.path / "laboro_tomato"]
            roots = [p for p in candidates if (p / "annotations/train.json").is_file()
                     and (p / "annotations/test.json").is_file()]
            if len(roots) != 1:
                raise ValueError("Pass laboro_tomato/ or the folder immediately above it")
            self.path = roots[0].resolve()
        else:
            raise ValueError("Source must be the official ZIP or its existing extracted folder")

    def read(self, name: str) -> bytes:
        name = safe_name(name)
        if self.archive:
            entry = self.archive.getinfo(self.prefix + name)
            if entry.file_size > 256 * 1024 * 1024:
                raise ValueError("Source member exceeds 256 MiB safety limit")
            return self.archive.read(entry)
        path = (self.path / name).resolve()
        if not path.is_relative_to(self.path):
            raise ValueError("Source symlink escapes the dataset folder")
        return path.read_bytes()

    def image(self, split: str, file_name: str) -> bytes:
        name = safe_name(file_name)
        # Official COCO versions can use a basename or a split-relative filename.
        if name.startswith(split + "/"):
            return self.read(name)
        return self.read(split + "/" + name)

    def close(self):
        if self.archive:
            self.archive.close()


def corrupt(image: Image.Image, image_id: str, seed: int = 25) -> Image.Image:
    """Native geometry, independent zero-mean RGB noise, lossless output."""
    seed_bytes = hashlib.sha256(f"{seed}:{image_id}".encode()).digest()[:8]
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed_bytes, "big")))
    pixels = np.asarray(image.convert("RGB")).copy()
    # Row blocks bound temporary memory for the native 12-megapixel photographs.
    for start in range(0, pixels.shape[0], 128):
        block = pixels[start:start + 128]
        noise = rng.standard_normal(block.shape, dtype=np.float32) * np.float32(3.0)
        block[:] = np.clip(np.rint(block.astype(np.float32) + noise), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def normalize_source_geometry(image: Image.Image, annotation: dict) -> tuple[Image.Image, dict]:
    """Decode the JPEG's declared display orientation into the COCO coordinate plane.

    The official release's Orientation=6 JPEGs store a landscape raster but its
    COCO boxes/dimensions refer to the displayed portrait image. Materializing
    EXIF orientation changes no COCO coordinates and introduces no interpolation.
    An unexplained mismatch fails instead of guessing a transform from dimensions.
    """
    orientation = image.getexif().get(274, 1)
    if type(orientation) is not int or orientation not in range(1, 9):
        raise ValueError(f"Invalid source EXIF orientation: {annotation['file_name']}")
    stored_width, stored_height = image.size
    normalized = ImageOps.exif_transpose(image)
    if normalized.size != (annotation["width"], annotation["height"]):
        raise ValueError(f"Source dimensions disagree with COCO after EXIF normalization: "
                         f"{annotation['file_name']} (stored={image.size}, "
                         f"orientation={orientation}, displayed={normalized.size}, "
                         f"COCO={(annotation['width'], annotation['height'])})")
    return normalized, {"stored_width": stored_width, "stored_height": stored_height,
                        "exif_orientation": orientation,
                        "exif_transpose_applied": orientation != 1}


def _validate_native(coco: dict, split: str, expected: dict | None) -> None:
    for key in ("images", "annotations", "categories"):
        if not isinstance(coco.get(key), list) or not coco[key]:
            raise ValueError(f"{split}: missing/nonempty COCO {key}")
    if expected:
        if len(coco["images"]) != expected[f"expected_{split}_images"]:
            raise ValueError(f"{split}: expected complete official image count")
        if len(coco["annotations"]) != expected[f"expected_{split}_boxes"]:
            raise ValueError(f"{split}: expected complete official annotation count")
        if len(coco["categories"]) != 6 or {c.get("name") for c in coco["categories"]} != NAMES:
            raise ValueError("Expected the six native tomato_mixed categories")
    categories = {c["id"] for c in coco["categories"]}
    if len(categories) != len(coco["categories"]) or any(type(c) is not int for c in categories):
        raise ValueError("Invalid native category IDs")
    images = {i["id"]: i for i in coco["images"]}
    if len(images) != len(coco["images"]):
        raise ValueError("Duplicate native image IDs")
    file_names = [i.get("file_name") for i in coco["images"]]
    if len(set(file_names)) != len(file_names):
        raise ValueError("Duplicate source filenames")
    for image in images.values():
        safe_name(image["file_name"])
        if any(type(image.get(k)) is not int or image[k] <= 0 for k in ("width", "height")):
            raise ValueError("Invalid source image dimensions")
    ids, counts = set(), Counter()
    for ann in coco["annotations"]:
        if ann["id"] in ids or ann["image_id"] not in images or ann["category_id"] not in categories:
            raise ValueError("Invalid source annotation reference")
        ids.add(ann["id"])
        bbox = ann.get("bbox")
        if (not isinstance(bbox, list) or len(bbox) != 4 or any(
                isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
                for v in bbox)):
            raise ValueError("Invalid native bbox")
        x, y, w, h = bbox
        image = images[ann["image_id"]]
        if (min(x, y) < 0 or min(w, h) <= 0 or x + w > image["width"] + 1e-6 or
                y + h > image["height"] + 1e-6):
            raise ValueError("Native bbox outside image; audit rather than silently clip")
        if ann.get("iscrowd", 0) not in (0, False):
            raise ValueError("Unexpected crowd annotation; audit before benchmark release")
        counts[ann["image_id"]] += 1
    if max(counts.values(), default=0) > MAX_DETECTIONS:
        raise ValueError(f"Native image exceeds published detection cap {MAX_DETECTIONS}; "
                         "audit and version the metric before release, do not drop objects")


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare(source_path: Path, public_path: Path, organizer_path: Path,
            *, expected: dict | None = CONFIG) -> dict:
    """Library entry point. expected=None exists only for synthetic unit fixtures."""
    public_path, organizer_path = public_path.resolve(), organizer_path.resolve()
    if public_path == organizer_path or public_path in organizer_path.parents:
        raise ValueError("Organizer files must be outside the public data directory")
    if public_path.exists() and any(public_path.iterdir()):
        raise ValueError("Public output is not empty; use a fresh output directory")
    for name in ("ground_truth.json", "provenance.json"):
        if (organizer_path / name).exists():
            raise ValueError(f"Organizer {name} already exists; use fresh outputs")
    source = Source(Path(source_path))
    public_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".tomato-prepare-", dir=public_path.parent))
    provenance = {"benchmark_id": BENCHMARK_ID, "images": []}
    metadata = {"benchmark_id": BENCHMARK_ID, "version": 1,
                "dataset": "Laboro Tomato tomato_mixed (complete official release)",
                "source_url": SOURCE_URL,
                "source_documentation": "https://github.com/laboroai/LaboroTomato",
                "license": "CC BY-NC-SA 4.0",
                "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
                "attribution": "Laboro.AI, Laboro Tomato dataset",
                "modification": "EXIF-normalized native display geometry; independent RGB Gaussian noise; PNG output",
                "coordinate_policy": "Materialize source EXIF display orientation; keep native COCO boxes and dimensions unchanged",
                "corruption": CONFIG["corruption"], "seed": CONFIG["seed"],
                "numpy_version": np.__version__, "pillow_version": PILLOW_VERSION,
                "counts": {}, "hashes": {}, "source_exif_orientation_counts": {},
                "split_policy": "All official train/test images retained in their original partitions",
                "security_note": "Public-source labels can be reconstructed; opaque IDs do not ensure secrecy",
                "max_detections_per_image": MAX_DETECTIONS}
    test_coco = None
    try:
        if Path(source_path).is_file():
            metadata["source_archive_sha256"] = file_digest(Path(source_path))
        categories = None
        for split in ("train", "test"):
            native_bytes = source.read(f"annotations/{split}.json")
            native = json.loads(native_bytes)
            _validate_native(native, split, expected)
            current_categories = sorted(native["categories"], key=lambda c: c["id"])
            if categories is not None and current_categories != categories:
                raise ValueError("Train and test category metadata differ")
            categories = current_categories
            metadata[f"source_{split}_annotations_sha256"] = digest(native_bytes)
            output = {"info": {"description": BENCHMARK_ID, "version": "1"},
                      "images": [], "annotations": [], "categories": categories}
            manifest = {"benchmark_id": BENCHMARK_ID, "images": [], "categories": categories}
            id_mapping = {}
            out_dir = temporary / split / "images"
            out_dir.mkdir(parents=True)
            sorted_images = sorted(native["images"], key=lambda i: i["file_name"])
            exif_counts = Counter()
            for index, image in enumerate(sorted_images, 1):
                opaque_id = f"{split}_{index:06d}"
                file_name = f"{split}/images/{opaque_id}.png"
                raw_bytes = source.image(split, image["file_name"])
                with Image.open(io.BytesIO(raw_bytes)) as raw:
                    normalized, geometry = normalize_source_geometry(raw, image)
                    exif_counts[str(geometry["exif_orientation"])] += 1
                    modified = corrupt(normalized, opaque_id, CONFIG["seed"])
                    modified.save(temporary / file_name, format="PNG", compress_level=6)
                id_mapping[image["id"]] = opaque_id
                output["images"].append({"id": opaque_id, "file_name": file_name,
                                         "width": image["width"], "height": image["height"]})
                manifest["images"].append({"image_id": opaque_id, "file_name": file_name,
                                           "width": image["width"], "height": image["height"]})
                metadata["hashes"][file_name] = file_digest(temporary / file_name)
                provenance["images"].append({"image_id": opaque_id, "official_split": split,
                    "source_image_id": image["id"], "source_file_name": image["file_name"],
                    "source_sha256": digest(raw_bytes), **geometry})
            for ann in native["annotations"]:
                # Native boxes/classes/area retained. Masks are unused by this box-detection task.
                output["annotations"].append({"id": ann["id"],
                    "image_id": id_mapping[ann["image_id"]], "category_id": ann["category_id"],
                    "bbox": ann["bbox"], "area": ann.get("area", ann["bbox"][2] * ann["bbox"][3]),
                    "iscrowd": ann.get("iscrowd", 0)})
            metadata["counts"][split] = {"images": len(output["images"]),
                "boxes": len(output["annotations"]),
                "boxes_by_category": dict(Counter(str(a["category_id"]) for a in output["annotations"])),
                "max_boxes_per_image": max(Counter(a["image_id"] for a in output["annotations"]).values())}
            metadata["source_exif_orientation_counts"][split] = dict(exif_counts)
            if split == "train":
                write_json(temporary / "train" / "annotations.json", output)
                metadata["hashes"]["train/annotations.json"] = file_digest(temporary / "train" / "annotations.json")
                write_json(temporary / "train_manifest.json", manifest)
            else:
                test_coco = output
                write_json(temporary / "test_manifest.json", manifest)
        metadata["categories"] = categories
        metadata["hashes"]["train_manifest.json"] = file_digest(temporary / "train_manifest.json")
        metadata["hashes"]["test_manifest.json"] = file_digest(temporary / "test_manifest.json")
        write_json(temporary / "dataset.json", metadata)
        organizer_path.mkdir(parents=True, exist_ok=True)
        organizer_path.chmod(0o700)
        write_json(organizer_path / "ground_truth.json", test_coco)
        write_json(organizer_path / "provenance.json", provenance)
        (organizer_path / "ground_truth.json").chmod(0o600)
        (organizer_path / "provenance.json").chmod(0o600)
        if public_path.exists():
            public_path.rmdir()  # Empty directories only; never remove user files.
        temporary.rename(public_path)
    finally:
        source.close()
        if temporary.exists():
            shutil.rmtree(temporary)  # Only our own temporary preparation directory.
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "data/public")
    parser.add_argument("--organizer", type=Path, default=ROOT / "evaluation-private")
    args = parser.parse_args()
    result = prepare(args.source, args.output, args.organizer)
    print(json.dumps({"benchmark_id": BENCHMARK_ID, "counts": result["counts"],
                      "public": str(args.output), "private": str(args.organizer)}, indent=2))


if __name__ == "__main__":
    main()
