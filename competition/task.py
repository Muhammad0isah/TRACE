"""Strict prediction validation and deterministic detection F1 scoring."""
from __future__ import annotations

from collections import Counter
import json
import math
from pathlib import Path

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text())
BENCHMARK_ID = CONFIG["benchmark_id"]
MAX_DETECTIONS = CONFIG["max_detections_per_image"]
MAX_SUBMISSION_BYTES = 32 * 1024 * 1024


def _json(path: str | Path, limit: int | None = None) -> dict:
    path = Path(path)
    if limit is not None and path.stat().st_size > limit:
        raise ValueError(f"JSON file exceeds {limit} bytes")
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    def bad_constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    value = json.loads(path.read_text(), object_pairs_hook=unique_pairs,
                       parse_constant=bad_constant)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _number(value, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(value)


def load_manifest(path: str | Path) -> dict:
    manifest = _json(path)
    if manifest.get("benchmark_id") != BENCHMARK_ID:
        raise ValueError("Wrong benchmark_id in manifest")
    images, categories = manifest.get("images"), manifest.get("categories")
    if not isinstance(images, list) or not images:
        raise ValueError("Manifest images must be a nonempty list")
    if not isinstance(categories, list) or not categories:
        raise ValueError("Manifest categories must be a nonempty list")
    image_ids, category_ids = set(), set()
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("image_id"), str):
            raise ValueError("Invalid manifest image")
        if image["image_id"] in image_ids:
            raise ValueError("Duplicate manifest image_id")
        image_ids.add(image["image_id"])
        for field in ("width", "height"):
            if type(image.get(field)) is not int or image[field] <= 0:
                raise ValueError(f"Invalid manifest {field}")
    for category in categories:
        if not isinstance(category, dict) or type(category.get("id")) is not int:
            raise ValueError("Invalid manifest category")
        if category["id"] in category_ids:
            raise ValueError("Duplicate manifest category")
        category_ids.add(category["id"])
    return manifest


def validate_submission(path: str | Path, manifest_path: str | Path) -> dict:
    """Return a parsed submission, rejecting invalid or incomplete predictions."""
    manifest = load_manifest(manifest_path)
    submission = _json(path, MAX_SUBMISSION_BYTES)
    if set(submission) != {"benchmark_id", "image_ids", "predictions"}:
        raise ValueError("Expected only benchmark_id, image_ids and predictions")
    if submission["benchmark_id"] != BENCHMARK_ID:
        raise ValueError("Submission benchmark_id mismatch")
    images = {item["image_id"]: item for item in manifest["images"]}
    categories = {item["id"] for item in manifest["categories"]}
    image_ids = submission["image_ids"]
    if not isinstance(image_ids, list) or any(not isinstance(i, str) for i in image_ids):
        raise ValueError("image_ids must be a list of strings")
    if len(image_ids) != len(set(image_ids)) or set(image_ids) != set(images):
        raise ValueError("image_ids must cover every test image exactly once")
    predictions = submission["predictions"]
    if not isinstance(predictions, list):
        raise ValueError("predictions must be a list")
    if len(predictions) > MAX_DETECTIONS * len(images):
        raise ValueError("Too many predictions")
    counts = Counter()
    for detection in predictions:
        if not isinstance(detection, dict) or set(detection) != {
                "image_id", "category_id", "bbox", "score"}:
            raise ValueError("Each prediction requires image_id, category_id, bbox, score")
        image_id = detection["image_id"]
        if not isinstance(image_id, str) or image_id not in images:
            raise ValueError("Unknown prediction image_id")
        category_id = detection["category_id"]
        if type(category_id) is not int or category_id not in categories:
            raise ValueError("Unknown category_id")
        score = _number(detection["score"], "score")
        if not 0 <= score <= 1:
            raise ValueError("score must lie in [0, 1]")
        bbox = detection["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("bbox must be [x, y, width, height]")
        x, y, width, height = [_number(v, "bbox") for v in bbox]
        image = images[image_id]
        if (x < 0 or y < 0 or width <= 0 or height <= 0 or
                x + width > image["width"] + 1e-6 or
                y + height > image["height"] + 1e-6):
            raise ValueError("bbox must have positive size and lie inside the native image")
        counts[image_id] += 1
        if counts[image_id] > MAX_DETECTIONS:
            raise ValueError(f"At most {MAX_DETECTIONS} detections per image")
    return submission


def evaluate(predictions_path: str | Path, truth_path: str | Path,
             manifest_path: str | Path) -> dict:
    """Compute micro detection F1 at fixed score and IoU thresholds."""
    submission = validate_submission(predictions_path, manifest_path)
    manifest = load_manifest(manifest_path)
    truth = _json(truth_path)
    images = {item["image_id"]: item for item in manifest["images"]}
    truth_images = {item["id"]: item for item in truth.get("images", [])}
    if len(truth_images) != len(truth.get("images", [])) or set(truth_images) != set(images):
        raise ValueError("Ground truth and manifest image IDs differ")
    categories = {item["id"] for item in manifest["categories"]}
    if {item["id"] for item in truth.get("categories", [])} != categories:
        raise ValueError("Ground truth and manifest categories differ")
    if not truth.get("annotations"):
        raise ValueError("Ground truth has no annotations")
    for image_id, image in images.items():
        if any(truth_images[image_id].get(k) != image[k] for k in ("width", "height")):
            raise ValueError("Ground truth and manifest dimensions differ")
    score_threshold = float(CONFIG["score_threshold"])
    iou_threshold = float(CONFIG["iou_threshold"])
    truth_by_key: dict[tuple[str, int], list[list[float]]] = {}
    for annotation in truth["annotations"]:
        if annotation.get("image_id") not in images or annotation.get("category_id") not in categories:
            raise ValueError("Ground truth contains an unknown image or category")
        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError("Ground truth contains an invalid bbox")
        truth_by_key.setdefault((annotation["image_id"], annotation["category_id"]), []).append(bbox)

    def iou(first: list[float], second: list[float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[0] + first[2], second[0] + second[2])
        bottom = min(first[1] + first[3], second[1] + second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        union = first[2] * first[3] + second[2] * second[3] - intersection
        return intersection / union if union else 0.0

    predictions_by_key: dict[tuple[str, int], list[dict]] = {}
    for detection in submission["predictions"]:
        if detection["score"] >= score_threshold:
            predictions_by_key.setdefault((detection["image_id"], detection["category_id"]), []).append(detection)
    true_positive = false_positive = 0
    for key, detections in predictions_by_key.items():
        targets = truth_by_key.get(key, [])
        matched = set()
        for detection in sorted(detections, key=lambda row: row["score"], reverse=True):
            best_index, best_iou = None, iou_threshold
            for index, target in enumerate(targets):
                overlap = iou(detection["bbox"], target)
                if index not in matched and overlap >= best_iou:
                    best_index, best_iou = index, overlap
            if best_index is None:
                false_positive += 1
            else:
                matched.add(best_index)
                true_positive += 1
    false_negative = sum(len(targets) for targets in truth_by_key.values()) - true_positive
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = (2 * true_positive / denominator) if denominator else 1.0
    return {"score": f1, "metrics": {"f1_score": f1}, "primary_metric": "f1_score"}
