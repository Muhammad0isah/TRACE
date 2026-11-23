"""Trusted-base scoring and isolated leaderboard publication for GitHub Actions.

Pull request code is never imported, checked out or executed. Only two validated
regular files are read from its immutable head SHA through the GitHub API.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request

from competition.submission import ACTOR_PATTERN, TEAM_PATTERN, SubmissionError, parse_metadata, record_result, validated_result
from encryption.crypto import (MAX_ENVELOPE_BYTES, EnvelopeError, atomic_write,
                               configured_benchmark, decrypt_bytes, read_limited)

REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
SHA = re.compile(r"[0-9a-f]{40}\Z")
DIGEST = re.compile(r"[0-9a-f]{64}\Z")
SUBMISSION_PATH = re.compile(r"submissions/([A-Za-z0-9][A-Za-z0-9_-]{0,39})/(submission\.enc|meta\.json)\Z")
MAX_METADATA_BYTES = 16 * 1024
MAX_JSON_BYTES = 4 * 1024 * 1024
BUNDLE_KEYS = {"version", "repository", "pr_number", "head_sha", "actor", "team", "benchmark_id",
               "submission_sha256", "config_sha256", "manifest_sha256", "ground_truth_sha256", "metadata", "result"}
PUBLIC_PATHS = ["leaderboard/records.json", "leaderboard/leaderboard.csv", "leaderboard/leaderboard.md",
                "docs/leaderboard.csv", "docs/leaderboard.json", "docs/index.html"]


class CIError(ValueError):
    """A safe-to-display workflow rejection; never contains submitted data."""


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise CIError("Unexpected GitHub API redirect")


class GitHubAPI:
    def __init__(self, token: str):
        if not token:
            raise CIError("GITHUB_TOKEN is missing")
        self.token = token
        self.opener = urllib.request.build_opener(NoRedirect())

    def request(self, route: str, *, raw: bool = False, limit: int = MAX_JSON_BYTES,
                method: str = "GET", payload: dict | None = None):
        if not route.startswith("/repos/") or any(character in route for character in ("\r", "\n", "#")):
            raise CIError("Invalid GitHub API route")
        headers = {"Authorization": f"Bearer {self.token}", "User-Agent": "vision-coding-benchmark",
                   "X-GitHub-Api-Version": "2022-11-28",
                   "Accept": ("application/vnd.github.raw+json" if raw else
                              "application/vnd.github.object+json" if "/contents/" in route else "application/vnd.github+json")}
        body = json.dumps(payload).encode() if payload is not None else None
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request("https://api.github.com" + route, data=body, headers=headers, method=method)
        try:
            with self.opener.open(request, timeout=30) as response:
                data = response.read(limit + 1)
        except urllib.error.HTTPError as exc:
            raise CIError(f"GitHub API request failed (HTTP {exc.code})") from None
        except (urllib.error.URLError, TimeoutError, OSError):
            raise CIError("GitHub API request failed") from None
        if len(data) > limit:
            raise CIError("GitHub API response exceeds the allowed size")
        if raw:
            return data
        try:
            return json.loads(data)
        except (ValueError, UnicodeError):
            raise CIError("GitHub API returned invalid JSON") from None


def repository_name(value: object) -> str:
    if not isinstance(value, str) or not REPOSITORY.fullmatch(value):
        raise CIError("Invalid repository identifier")
    return value


def sha(value: object) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise CIError("Invalid Git commit or object identifier")
    return value


def pr_number(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CIError("Invalid pull request number")
    return value


def parse_event(event: dict, expected_repository: str) -> dict:
    try:
        pull = event["pull_request"]
        repository = repository_name(event["repository"]["full_name"])
        if repository.casefold() != repository_name(expected_repository).casefold():
            raise CIError("Event repository does not match this workflow")
        if repository_name(pull["base"]["repo"]["full_name"]).casefold() != repository.casefold():
            raise CIError("Pull request base repository does not match this workflow")
        actor = pull["user"]["login"]
        if not isinstance(actor, str) or not ACTOR_PATTERN.fullmatch(actor):
            raise CIError("Invalid pull request author")
        return {"repository": repository, "head_repository": repository_name(pull["head"]["repo"]["full_name"]),
                "head_sha": sha(pull["head"]["sha"]), "actor": actor,
                "pr_number": pr_number(event.get("number", pull.get("number")))}
    except (KeyError, TypeError):
        raise CIError("Pull request event is incomplete") from None


def check_live_pull(api: GitHubAPI, identity: dict) -> dict:
    pull = api.request(f"/repos/{identity['repository']}/pulls/{identity['pr_number']}")
    try:
        if (pull["state"] != "open" or pull["head"]["sha"] != identity["head_sha"]
                or pull["user"]["login"].casefold() != identity["actor"].casefold()
                or repository_name(pull["base"]["repo"]["full_name"]).casefold() != identity["repository"].casefold()):
            raise CIError("Pull request changed or no longer matches the submitted event")
        if "head_repository" in identity and repository_name(pull["head"]["repo"]["full_name"]).casefold() != identity["head_repository"].casefold():
            raise CIError("Pull request head repository changed")
    except (KeyError, TypeError, AttributeError):
        raise CIError("Live pull request metadata is incomplete") from None
    return pull


def select_files(files: object) -> tuple[str, dict[str, dict]]:
    if not isinstance(files, list) or len(files) != 2:
        raise CIError("A submission PR must change exactly submission.enc and meta.json for one team")
    selected, teams = {}, set()
    for entry in files:
        if not isinstance(entry, dict):
            raise CIError("Invalid pull request file metadata")
        path = entry.get("filename")
        match = SUBMISSION_PATH.fullmatch(path) if isinstance(path, str) else None
        if not match or entry.get("status") not in {"added", "modified"} or entry.get("previous_filename"):
            raise CIError("Submission PR contains an unsupported path or file change")
        team, name = match.groups()
        teams.add(team)
        if name in selected:
            raise CIError("Submission files must be unique")
        sha(entry.get("sha"))
        selected[name] = entry
    if len(teams) != 1 or set(selected) != {"submission.enc", "meta.json"}:
        raise CIError("Both submission files must belong to the same team")
    return teams.pop(), selected


def fetch_files(api: GitHubAPI, identity: dict) -> tuple[str, bytes, dict]:
    pull = check_live_pull(api, identity)
    if pull.get("changed_files") != 2:
        raise CIError("A submission PR must change exactly two files")
    files = api.request(f"/repos/{identity['repository']}/pulls/{identity['pr_number']}/files?per_page=100&page=1")
    team, selected = select_files(files)
    head_repository, head_sha = identity["head_repository"], identity["head_sha"]
    commit = api.request(f"/repos/{head_repository}/git/commits/{head_sha}")
    try:
        tree_sha = sha(commit["tree"]["sha"])
    except (KeyError, TypeError):
        raise CIError("GitHub commit tree is missing") from None
    # Walk exact paths instead of using a recursive tree that can be truncated.
    for component in ("submissions", team):
        tree = api.request(f"/repos/{head_repository}/git/trees/{tree_sha}")
        matches = [item for item in tree.get("tree", []) if item.get("path") == component]
        if tree.get("truncated") or len(matches) != 1 or matches[0].get("mode") != "040000" or matches[0].get("type") != "tree":
            raise CIError("Submission directories must be ordinary Git trees")
        tree_sha = sha(matches[0].get("sha"))
    final_tree = api.request(f"/repos/{head_repository}/git/trees/{tree_sha}")
    if final_tree.get("truncated"):
        raise CIError("Submission directory tree is truncated")
    contents = {}
    for name, entry in selected.items():
        matches = [item for item in final_tree.get("tree", []) if item.get("path") == name]
        if (len(matches) != 1 or matches[0].get("type") != "blob" or matches[0].get("mode") not in {"100644", "100755"}
                or sha(matches[0].get("sha")) != entry["sha"]):
            raise CIError("Submission files must be regular Git blobs; symlinks are forbidden")
        limit = MAX_METADATA_BYTES if name == "meta.json" else MAX_ENVELOPE_BYTES
        size = matches[0].get("size")
        if isinstance(size, bool) or not isinstance(size, int) or not 0 < size <= limit:
            raise CIError("Submission file exceeds its size limit or is empty")
        route = f"/repos/{head_repository}/contents/{urllib.parse.quote(entry['filename'], safe='/')}?ref={head_sha}"
        information = api.request(route)
        if (not isinstance(information, dict) or information.get("type") != "file"
                or information.get("sha") != entry["sha"] or information.get("size") != size):
            raise CIError("Submission content metadata does not match the immutable Git blob")
        raw = api.request(route, raw=True, limit=limit)
        blob_sha = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        if len(raw) != size or blob_sha != entry["sha"]:
            raise CIError("Downloaded submission content does not match its Git blob")
        contents[name] = raw
    try:
        metadata = parse_metadata(json.loads(contents["meta.json"]))
    except (ValueError, TypeError, UnicodeError):
        raise CIError("meta.json does not satisfy the public submission metadata schema") from None
    if metadata["team"].casefold() != team.casefold():
        raise CIError("meta.json team must match the submission directory")
    return metadata["team"], contents["submission.enc"], metadata


def file_digest(path: Path, limit: int) -> str:
    return hashlib.sha256(read_limited(path, limit)).hexdigest()


def release_fingerprints(root: Path, manifest: Path, truth: Path) -> dict:
    if not manifest.is_file() or not truth.is_file():
        raise CIError("Organizer setup is incomplete: publish evaluation/test_manifest.json and evaluation/ground_truth.enc before scoring")
    return {"config_sha256": file_digest(root / "competition/config.json", MAX_JSON_BYTES),
            "manifest_sha256": file_digest(manifest, MAX_JSON_BYTES),
            "ground_truth_sha256": file_digest(truth, MAX_ENVELOPE_BYTES)}


def score_submission(root: Path, event_path: Path, output: Path, api: GitHubAPI,
                     expected_repository: str, private_pem: bytes, manifest: Path | None = None) -> dict:
    root = Path(root)
    manifest = manifest or root / "evaluation/test_manifest.json"
    truth = root / "evaluation/ground_truth.enc"
    fingerprints = release_fingerprints(root, manifest, truth)
    if not private_pem:
        raise CIError("Organizer setup is incomplete: configure the SUBMISSION_PRIVATE_KEY Actions secret")
    try:
        event = json.loads(read_limited(event_path, MAX_JSON_BYTES))
    except (ValueError, UnicodeError):
        raise CIError("Workflow event JSON is invalid") from None
    identity = parse_event(event, expected_repository)
    team, envelope, metadata = fetch_files(api, identity)
    benchmark_id = configured_benchmark(root)
    # Authenticated contexts prevent an encrypted truth file being submitted as predictions.
    predictions = decrypt_bytes(envelope, private_pem, benchmark_id, purpose="submission")
    ground_truth = decrypt_bytes(read_limited(truth), private_pem, benchmark_id, purpose="ground_truth")
    from competition import task
    with tempfile.TemporaryDirectory(prefix="competition-score-") as temporary:
        predictions_path, truth_path = Path(temporary) / "predictions.bin", Path(temporary) / "truth.bin"
        atomic_write(predictions_path, predictions, mode=0o600)
        atomic_write(truth_path, ground_truth, mode=0o600)
        # Evaluation code is imported exclusively from the trusted base checkout.
        # Suppress library progress output, which must not reveal hidden annotations.
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            task.validate_submission(predictions_path, manifest)
            result = validated_result(task.evaluate(predictions_path, truth_path, manifest), benchmark_id)
    bundle = {"version": 1, **{key: identity[key] for key in ("repository", "pr_number", "head_sha", "actor")},
              "team": team, "benchmark_id": benchmark_id,
              "submission_sha256": hashlib.sha256(envelope).hexdigest(), **fingerprints,
              "metadata": metadata, "result": result}
    atomic_write(output, (json.dumps(bundle, indent=2, allow_nan=False) + "\n").encode(), overwrite=True)
    return bundle


def validate_bundle(bundle: object, root: Path, expected_repository: str) -> dict:
    if not isinstance(bundle, dict) or set(bundle) != BUNDLE_KEYS or bundle.get("version") != 1:
        raise CIError("Result artifact schema is invalid")
    if repository_name(bundle["repository"]).casefold() != repository_name(expected_repository).casefold():
        raise CIError("Result artifact belongs to another repository")
    pr_number(bundle["pr_number"])
    sha(bundle["head_sha"])
    if not isinstance(bundle["actor"], str) or not ACTOR_PATTERN.fullmatch(bundle["actor"]):
        raise CIError("Result artifact participant is invalid")
    if not isinstance(bundle["team"], str) or not TEAM_PATTERN.fullmatch(bundle["team"]):
        raise CIError("Result artifact team is invalid")
    benchmark_id = configured_benchmark(root)
    if bundle["benchmark_id"] != benchmark_id:
        raise CIError("Result artifact belongs to another benchmark version")
    for key in ("submission_sha256", "config_sha256", "manifest_sha256", "ground_truth_sha256"):
        if not isinstance(bundle[key], str) or not DIGEST.fullmatch(bundle[key]):
            raise CIError("Result artifact digest is invalid")
    fingerprints = release_fingerprints(root, root / "evaluation/test_manifest.json", root / "evaluation/ground_truth.enc")
    if any(bundle[key] != value for key, value in fingerprints.items()):
        raise CIError("The benchmark release changed after scoring; rerun the workflow")
    parse_metadata(bundle["metadata"])
    validated_result(bundle["result"], benchmark_id)
    return bundle


def git(root: Path, arguments: list[str], token: str, *, check: bool = True) -> subprocess.CompletedProcess:
    environment = os.environ.copy()
    credential = base64.b64encode(("x-access-token:" + token).encode()).decode()
    environment.update({"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                        "GIT_CONFIG_VALUE_0": "AUTHORIZATION: basic " + credential, "GIT_TERMINAL_PROMPT": "0"})
    result = subprocess.run(["git", *arguments], cwd=root, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=90)
    if check and result.returncode:
        raise CIError("Leaderboard Git operation failed")
    return result


def comment_result(api: GitHubAPI, bundle: dict) -> None:
    score = float(bundle["result"]["score"])
    body = ("Verified competition submission recorded.\n\n"
            f"Team: `{bundle['team']}`. Score: **{score:.6f}**.\n\n"
            "The first valid submission per team is final for this benchmark version.")
    api.request(f"/repos/{bundle['repository']}/issues/{bundle['pr_number']}/comments", method="POST", payload={"body": body})


def publish_result(root: Path, bundle_path: Path, api: GitHubAPI, expected_repository: str,
                   token: str, branch: str) -> dict:
    # A clean Actions checkout is disposable. Refuse local worktrees before reset.
    if os.environ.get("GITHUB_ACTIONS") != "true" or Path(os.environ.get("GITHUB_WORKSPACE", "")).resolve() != root.resolve():
        raise CIError("Leaderboard publication runs only in an isolated GitHub Actions workspace")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/-]*", branch) or any(part in branch for part in ("..", "//", "@{")):
        raise CIError("Invalid publication branch")
    try:
        bundle = json.loads(read_limited(bundle_path, 32 * 1024))
    except (ValueError, UnicodeError):
        raise CIError("Result artifact is invalid JSON") from None
    for attempt in range(3):
        git(root, ["fetch", "--no-tags", "--depth=1", "origin", branch], token)
        git(root, ["reset", "--hard", "FETCH_HEAD"], token)
        validate_bundle(bundle, root, expected_repository)
        check_live_pull(api, bundle)
        entry = record_result(bundle["result"], bundle["metadata"], bundle["actor"], bundle["team"],
                              bundle["pr_number"], bundle["submission_sha256"], bundle["benchmark_id"], root)
        git(root, ["add", "--", *PUBLIC_PATHS], token)
        changes = git(root, ["diff", "--cached", "--quiet"], token, check=False)
        if changes.returncode == 0:
            return entry
        if changes.returncode != 1:
            raise CIError("Unable to inspect leaderboard changes")
        git(root, ["-c", "user.name=github-actions[bot]", "-c", "user.email=41898282+github-actions[bot]@users.noreply.github.com",
                   "commit", "-m", "Record verified competition submission"], token)
        pushed = git(root, ["push", "origin", f"HEAD:refs/heads/{branch}"], token, check=False)
        if pushed.returncode == 0:
            try:
                comment_result(api, bundle)
            except CIError:
                print("Leaderboard updated; the optional pull request comment could not be posted.")
            return entry
    raise CIError("Leaderboard changed concurrently; rerun this workflow to publish safely")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("--event", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    publish = commands.add_parser("publish")
    publish.add_argument("--bundle", type=Path, required=True)
    publish.add_argument("--branch", required=True)
    args = parser.parse_args()
    try:
        token = os.environ.get("GITHUB_TOKEN", "")
        api = GitHubAPI(token)
        repository = os.environ.get("GITHUB_REPOSITORY", "")
        if args.command == "score":
            score_submission(args.root, args.event, args.output, api, repository,
                             os.environ.get("SUBMISSION_PRIVATE_KEY", "").encode())
            print("Submission evaluated; public result artifact is ready.")
        else:
            publish_result(args.root, args.bundle, api, repository, token, args.branch)
            print("Verified leaderboard publication completed.")
    except (CIError, EnvelopeError, SubmissionError) as exc:
        print(f"Competition workflow: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception:
        # Native parsers or validators can include submitted values in errors.
        # Do not put plaintext predictions, labels or private-key details in logs.
        print("Competition workflow failed: invalid submission or organizer configuration. Check the documented schema and setup.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
