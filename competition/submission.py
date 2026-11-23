"""Record the first valid submission per participant and team, then render scores."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
from pathlib import Path
import re

from encryption.crypto import atomic_write, configured_benchmark
from .render_leaderboard import render

TEAM_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,39}\Z")
ACTOR_PATTERN = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?\Z")
DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


class SubmissionError(ValueError):
    """A submission cannot be entered on this leaderboard."""


def _text(value: object, name: str, limit: int, required: bool = True) -> str:
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise SubmissionError(f"{name} must be {'nonempty ' if required else ''}text of at most {limit} characters")
    if "\x00" in value:
        raise SubmissionError(f"{name} contains a null character")
    return value


def parse_metadata(data: object) -> dict:
    if not isinstance(data, dict):
        raise SubmissionError("Metadata must be a JSON object")
    if set(data) - {"team", "type", "model", "notes", "run", "runlog"}:
        raise SubmissionError("Unsupported metadata fields")
    if not isinstance(data.get("team"), str) or not TEAM_PATTERN.fullmatch(data["team"]):
        raise SubmissionError("Metadata team must match [A-Za-z0-9][A-Za-z0-9_-]{0,39}")
    if data.get("type") not in {"human", "llm", "human+llm"}:
        raise SubmissionError("Metadata type must be human, llm or human+llm")
    result = {"team": data["team"], "type": data["type"], "model": _text(data.get("model"), "model", 160),
              "notes": _text(data.get("notes", ""), "notes", 4000, required=False)}
    if "runlog" in data:
        result["runlog"] = _text(data["runlog"], "runlog", 2000, required=False)
    if "run" in data:
        if not isinstance(data["run"], dict):
            raise SubmissionError("run metadata must be a JSON object")
        try:
            encoded = json.dumps(data["run"], allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise SubmissionError("run metadata must contain finite JSON values") from exc
        if len(encoded) > 8000:
            raise SubmissionError("run metadata exceeds 8,000 characters")
        result["run"] = data["run"]
    return result


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise SubmissionError(f"{name} must be a finite number")
    return float(value)


def validated_result(result: object, benchmark_id: str) -> dict:
    if not isinstance(result, dict) or set(result) - {"score", "metrics", "primary_metric", "benchmark_id"}:
        raise SubmissionError("Result must contain only score, metrics, primary_metric and optional benchmark_id")
    if result.get("benchmark_id", benchmark_id) != benchmark_id:
        raise SubmissionError("Result benchmark does not match the competition")
    score = _number(result.get("score"), "score")
    if not 0 <= score <= 1:
        raise SubmissionError("The primary score must be between zero and one")
    metrics = result.get("metrics", {})
    if not isinstance(metrics, dict) or len(metrics) > 100:
        raise SubmissionError("metrics must be an object with at most 100 numeric entries")
    checked = {}
    for name, value in metrics.items():
        _text(name, "metric name", 120)
        checked[name] = _number(value, name)
    primary = _text(result.get("primary_metric", "score"), "primary_metric", 120)
    return {"score": score, "metrics": checked, "primary_metric": primary}


def record_result(result: dict, metadata: dict, actor: str, team: str, pr_number: int,
                  submission_digest: str, benchmark_id: str, root: Path) -> dict:
    """Commit a score once; exact workflow retries are safely idempotent.

    Invalid submissions never enter this function's ledger. A participant cannot
    enter again under a different team, metadata type or model in this version.
    Callers must authenticate actor, team and PR against the hosting platform.
    """
    root = Path(root)
    if not isinstance(team, str) or not TEAM_PATTERN.fullmatch(team):
        raise SubmissionError("Team must match [A-Za-z0-9][A-Za-z0-9_-]{0,39}")
    if isinstance(pr_number, bool) or not isinstance(pr_number, int) or pr_number <= 0:
        raise SubmissionError("PR number must be a positive integer")
    if not isinstance(submission_digest, str) or not DIGEST_PATTERN.fullmatch(submission_digest):
        raise SubmissionError("submission_digest must be a lowercase SHA-256 hex digest")
    if benchmark_id != configured_benchmark(root):
        raise SubmissionError("Benchmark does not match the authoritative configuration")
    checked_result = validated_result(result, benchmark_id)
    checked_metadata = parse_metadata(metadata)
    if checked_metadata["team"].casefold() != team.casefold():
        raise SubmissionError("Metadata team must match the submission directory")
    directory = root / "leaderboard"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".record.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        ledger = directory / "records.json"
        records = json.loads(ledger.read_text(encoding="utf-8")) if ledger.exists() else []
        if not isinstance(records, list):
            raise SubmissionError("Leaderboard ledger is malformed")
        for existing in records:
            if existing["benchmark_id"] != benchmark_id:
                continue
            if existing["team"].casefold() == team.casefold():
                if (existing["pr_number"] == pr_number and existing["submission_sha256"] == submission_digest
                        and existing["team"].casefold() == team.casefold()):
                    render(root, records)
                    return existing
                raise SubmissionError("Only the first valid submission per team is accepted in this benchmark version")
        entry = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
                 "benchmark_id": benchmark_id, "team": team,
                 **checked_metadata, **checked_result, "pr_number": pr_number,
                 "submission_sha256": submission_digest}
        records.append(entry)
        atomic_write(ledger, (json.dumps(records, indent=2, allow_nan=False) + "\n").encode(), overwrite=True)
        render(root, records)
        return entry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--actor", required=True)
    parser.add_argument("--team", required=True)
    parser.add_argument("--pr-number", type=int, required=True)
    parser.add_argument("--submission-sha256", required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    entry = record_result(json.loads(args.result.read_text()), json.loads(args.metadata.read_text()),
                          args.actor, args.team, args.pr_number, args.submission_sha256,
                          configured_benchmark(args.root), args.root)
    print(json.dumps(entry, allow_nan=False))


if __name__ == "__main__":
    main()
