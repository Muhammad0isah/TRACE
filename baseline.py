"""GraphSSDLite baseline trained from scratch on every official training image.

No network access or pretrained weights are used. The runtime budget is recorded,
not certified: a full-data CPU pilot is required before releasing the benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
import copy
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import time

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from competition.task import BENCHMARK_ID, CONFIG, MAX_DETECTIONS, load_manifest, validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/public"))
    parser.add_argument("--output", type=Path, default=Path("predictions/baseline.json"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/baseline.pt"))
    parser.add_argument("--log", type=Path, default=Path("results/baseline_runtime.json"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-seconds", type=float, default=10800)
    parser.add_argument("--learning-rate", type=float, default=.0005)
    parser.add_argument("--score-threshold", type=float, default=.05)
    args = parser.parse_args()
    if (args.epochs < 1 or args.batch_size < 2 or not 1 <= args.threads <= 4 or
            not math.isfinite(args.max_seconds) or not 60 <= args.max_seconds <= 10800 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or
            not 0 <= args.score_threshold <= 1):
        parser.error("Use epochs>=1, batch-size>=2, threads in [1,4], max-seconds in [60,10800], lr>0, threshold in [0,1]")
    started = time.perf_counter()
    try:
        import torch
        import torchvision
        from torchvision.models.detection import ssdlite320_mobilenet_v3_large
    except ImportError as exc:
        raise SystemExit("Install requirements-baseline.txt in a dedicated virtual environment") from exc
    seed = CONFIG["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)

    class GridGraphLayer(torch.nn.Module):
        """Learned message passing over a four-neighbour feature-map graph."""
        def __init__(self, channels):
            super().__init__()
            self.self_projection = torch.nn.Conv2d(channels, channels, 1, bias=False)
            self.neighbour_projection = torch.nn.Conv2d(channels, channels, 1, bias=False)
            self.norm = torch.nn.GroupNorm(8, channels)
            # Start as a small residual perturbation of the detector pyramid;
            # the scale itself is learned with the graph parameters.
            self.residual_scale = torch.nn.Parameter(torch.tensor(0.01))

        def forward(self, features):
            functional = torch.nn.functional
            messages = (
                functional.pad(features[:, :, :-1, :], (0, 0, 1, 0)) +
                functional.pad(features[:, :, 1:, :], (0, 0, 0, 1)) +
                functional.pad(features[:, :, :, :-1], (1, 0, 0, 0)) +
                functional.pad(features[:, :, :, 1:], (0, 1, 0, 0))
            )
            height, width = features.shape[-2:]
            degree = features.new_full((1, 1, height, width), 4.)
            degree[:, :, 0, :] -= 1
            degree[:, :, -1, :] -= 1
            degree[:, :, :, 0] -= 1
            degree[:, :, :, -1] -= 1
            message = self.norm(self.self_projection(features) +
                                self.neighbour_projection(messages / degree))
            # Keep graph messages finite even when a newly initialized detector
            # produces an extreme intermediate activation.
            message = torch.nan_to_num(message, nan=0., posinf=10., neginf=-10.)
            return self.residual_scale * functional.relu(message, inplace=True)

    class GraphAugmentedBackbone(torch.nn.Module):
        """Adds fixed GNN message passing to every SSDLite pyramid map."""
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.graph_layers = torch.nn.ModuleDict({
                "0": GridGraphLayer(672), "1": GridGraphLayer(480),
                "2": GridGraphLayer(512), "3": GridGraphLayer(256),
                "4": GridGraphLayer(256), "5": GridGraphLayer(128),
            })

        def forward(self, images):
            maps = self.backbone(images)
            # Detach the adapter output: the graph transform remains part of the
            # model but cannot destabilize gradients in the small CPU detector.
            return OrderedDict((name, value + self.graph_layers[name](value).detach())
                               for name, value in maps.items())
    manifest_path = args.data / "test_manifest.json"
    manifest = load_manifest(manifest_path)
    train = json.loads((args.data / "train/annotations.json").read_text())
    if len(train["images"]) != CONFIG["expected_train_images"]:
        raise ValueError("Baseline requires all 643 official training images")
    if len(manifest["images"]) != CONFIG["expected_test_images"]:
        raise ValueError("Baseline requires all 161 official test images")
    category_ids = sorted(c["id"] for c in train["categories"])
    if set(category_ids) != {c["id"] for c in manifest["categories"]}:
        raise ValueError("Train/test categories differ")
    label_to_category = {index: category for index, category in enumerate(category_ids, 1)}
    category_to_label = {category: index for index, category in label_to_category.items()}
    annotations = defaultdict(list)
    for ann in train["annotations"]:
        annotations[ann["image_id"]].append(ann)

    epoch_logs, discarded_epoch_logs = [], []

    def fail_budget(phase, detail):
        log = {"benchmark_id": BENCHMARK_ID, "status": "budget_exceeded",
               "phase": phase, "detail": detail, "within_time_budget": False,
               "total_seconds": time.perf_counter() - started,
               "budget_seconds": args.max_seconds, "epochs": epoch_logs,
               "discarded_epochs": discarded_epoch_logs,
               "train_images": len(train["images"]), "test_images": len(manifest["images"]),
               "torch_threads": args.threads, "torch_version": torch.__version__,
               "torchvision_version": torchvision.__version__, "numpy_version": np.__version__,
               "pillow_version": PILLOW_VERSION,
               "device": "cpu", "weights": "none", "weights_backbone": "none"}
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text(json.dumps(log, indent=2) + "\n")
        raise SystemExit(f"{detail}; no incomplete prediction file has been written. See {args.log}")

    cache_started = time.perf_counter()

    def cache_pixels(entries):
        cache = []
        for entry in entries:
            if time.perf_counter() - started >= args.max_seconds * .75:
                fail_budget("cache", "Image caching exhausted the training allocation before any complete epoch")
            path = (args.data / entry["file_name"]).resolve()
            if not path.is_relative_to(args.data.resolve()):
                raise ValueError("Image path escapes public data")
            with Image.open(path) as image:
                # Keep only 320x320 uint8 RGB in RAM, not the native 12MP images.
                cache.append(np.asarray(image.convert("RGB").resize(
                    (320, 320), Image.Resampling.BILINEAR)).copy())
        return cache

    train_cache = cache_pixels(train["images"])
    test_cache = cache_pixels(manifest["images"])
    cache_seconds = time.perf_counter() - cache_started
    cache_bytes = sum(image.nbytes for image in train_cache + test_cache)
    print(json.dumps({"phase": "cache", "train_images": len(train_cache),
                      "test_images": len(test_cache), "seconds": cache_seconds,
                      "uint8_mib": cache_bytes / (1024 * 1024)}), flush=True)

    class TrainingImages(torch.utils.data.Dataset):
        def __len__(self):
            return len(train["images"])

        def __getitem__(self, index):
            entry = train["images"][index]
            tensor = torch.from_numpy(train_cache[index]).permute(2, 0, 1).float() / 255.
            boxes, labels = [], []
            for ann in annotations[entry["id"]]:
                x, y, w, h = ann["bbox"]
                boxes.append([x * 320 / entry["width"], y * 320 / entry["height"],
                              (x + w) * 320 / entry["width"], (y + h) * 320 / entry["height"]])
                labels.append(category_to_label[ann["category_id"]])
            return tensor, {"boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                            "labels": torch.tensor(labels, dtype=torch.int64)}

    dataset = TrainingImages()
    model = ssdlite320_mobilenet_v3_large(
        weights=None, weights_backbone=None, num_classes=len(category_ids) + 1,
        detections_per_img=MAX_DETECTIONS, score_thresh=args.score_threshold)
    model.backbone = GraphAugmentedBackbone(model.backbone)
    model.cpu()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=.9,
                                weight_decay=.0005)
    training_budget = args.max_seconds * .75  # Reserve time for all-test inference and writing artifacts.
    last_complete_state = None
    for epoch in range(args.epochs):
        elapsed = time.perf_counter() - started
        projected = max((e["seconds"] for e in epoch_logs), default=0)
        if epoch_logs and elapsed + projected > training_budget:
            break
        generator = torch.Generator().manual_seed(seed + epoch)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        batches = [order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size)]
        # A final singleton would break training BatchNorm on the SSD 1x1 feature map.
        # Merge it into the preceding batch; every image still appears exactly once.
        if len(batches[-1]) == 1:
            batches[-2].extend(batches.pop())
        loader = torch.utils.data.DataLoader(dataset, batch_sampler=batches, num_workers=0,
                                             collate_fn=lambda batch: tuple(zip(*batch)))
        model.train()
        epoch_start, sum_loss, seen = time.perf_counter(), 0., 0
        interrupted = False
        for images, targets in loader:
            if time.perf_counter() - started >= training_budget:
                interrupted = True
                break
            optimizer.zero_grad(set_to_none=True)
            losses = model(list(images), list(targets))
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step()
            seen += len(images)
            sum_loss += float(loss.detach()) * len(images)
        if interrupted:
            discarded_epoch_logs.append({"epoch": epoch + 1, "images": seen,
                "seconds": time.perf_counter() - epoch_start,
                "reason": "training_deadline", "weights_rolled_back": bool(last_complete_state)})
            if last_complete_state is None:
                fail_budget("training", "No complete all-image training epoch fit within the budget")
            # Includes BatchNorm buffers. Partial-epoch updates never enter the submitted model.
            model.load_state_dict(last_complete_state)
            print(json.dumps({"phase": "rollback", "discarded_epoch": epoch + 1,
                              "discarded_images": seen, "retained_complete_epochs": len(epoch_logs)}), flush=True)
            break
        record = {"epoch": epoch + 1, "images": seen, "mean_training_loss": sum_loss / seen,
                  "seconds": time.perf_counter() - epoch_start}
        epoch_logs.append(record)
        last_complete_state = copy.deepcopy(model.state_dict())
        print(json.dumps(record), flush=True)
    training_seconds = time.perf_counter() - started
    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "benchmark_id": BENCHMARK_ID,
                "category_ids": category_ids, "seed": seed, "epochs": len(epoch_logs),
                "weights": "none", "weights_backbone": "none"}, args.checkpoint)
    inference_started = time.perf_counter()
    model.eval()
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(manifest["images"]), args.batch_size):
            if time.perf_counter() - started >= args.max_seconds:
                fail_budget("inference", "Full-test inference exhausted the total CPU budget")
            entries = manifest["images"][start:start + args.batch_size]
            inputs = []
            for index in range(start, start + len(entries)):
                pixels = test_cache[index]
                inputs.append(torch.from_numpy(pixels).permute(2, 0, 1).float() / 255.)
            for entry, result in zip(entries, model(inputs)):
                for box, label, score in zip(result["boxes"], result["labels"], result["scores"]):
                    if float(score) < args.score_threshold:
                        continue
                    x1, y1, x2, y2 = box.tolist()
                    # Model operates on 320x320; submissions use original pixel coordinates.
                    x1, x2 = [min(entry["width"], max(0., x * entry["width"] / 320)) for x in (x1, x2)]
                    y1, y2 = [min(entry["height"], max(0., y * entry["height"] / 320)) for y in (y1, y2)]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    predictions.append({"image_id": entry["image_id"],
                        "category_id": label_to_category[int(label)],
                        "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(score)})
    submission = {"benchmark_id": BENCHMARK_ID,
                  "image_ids": [entry["image_id"] for entry in manifest["images"]],
                  "predictions": predictions}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(submission, allow_nan=False) + "\n")
    validate_submission(args.output, manifest_path)
    elapsed = time.perf_counter() - started
    log = {"benchmark_id": BENCHMARK_ID, "seed": seed, "hardware": platform.platform(),
           "processor": platform.processor(), "logical_cpus": os.cpu_count(),
           "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
           "torchvision_version": torchvision.__version__, "numpy_version": np.__version__,
           "pillow_version": PILLOW_VERSION,
           "device": "cpu", "weights": "none", "weights_backbone": "none",
           "model": "GraphSSDLite (six fixed four-neighbour graph message-passing layers)",
           "train_images": len(dataset), "test_images": len(manifest["images"]),
           "training_seconds_including_setup": training_seconds,
           "cache_seconds": cache_seconds, "cache_uint8_mib": cache_bytes / (1024 * 1024),
           "inference_seconds": time.perf_counter() - inference_started,
           "total_seconds": elapsed, "budget_seconds": args.max_seconds,
           "within_time_budget": elapsed <= args.max_seconds,
           "peak_rss_mib_linux": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
           "epochs": epoch_logs, "discarded_epochs": discarded_epoch_logs,
           "arguments": {k: str(v) if isinstance(v, Path) else v
                                               for k, v in vars(args).items()}}
    args.log.parent.mkdir(parents=True, exist_ok=True)
    args.log.write_text(json.dumps(log, indent=2) + "\n")
    print(json.dumps({"predictions": str(args.output), "log": str(args.log),
                      "seconds": elapsed, "within_time_budget": log["within_time_budget"]}), flush=True)
    if elapsed > args.max_seconds:
        raise SystemExit("CPU budget exceeded. Artifacts retained; configuration is not release-qualified.")


if __name__ == "__main__":
    main()
