"""Explicitly download the publisher's complete original ZIP; does not prepare data."""
import argparse
import hashlib
from pathlib import Path
import sys
import urllib.request

URL = "http://assets.laboro.ai.s3.amazonaws.com/laborotomato/laboro_tomato.zip"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("downloads/laboro_tomato.zip"))
    parser.add_argument("--download", action="store_true", help="Required opt-in for the large download")
    parser.add_argument("--sha256", help="Optional organizer-published expected checksum")
    args = parser.parse_args()
    if not args.download:
        parser.error("Read README.md and pass --download to begin the complete official download")
    if args.output.exists():
        parser.error("Output already exists; will not overwrite it")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(args.output.suffix + ".part")
    if partial.exists():
        parser.error("Partial download exists; remove or rename it explicitly before retrying")
    hasher = hashlib.sha256()
    total = 0
    try:
        with urllib.request.urlopen(URL, timeout=120) as response, partial.open("xb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                hasher.update(chunk)
                total += len(chunk)
                if total % (128 * 1024 * 1024) == 0:
                    print(f"Downloaded {total // (1024 * 1024)} MiB", file=sys.stderr)
        checksum = hasher.hexdigest()
        if args.sha256 and checksum.lower() != args.sha256.lower():
            raise ValueError("SHA256 mismatch; incomplete/untrusted file kept as .part")
        partial.rename(args.output)
    except Exception:
        print(f"Download failed; any partial file remains at {partial}", file=sys.stderr)
        raise
    print(f"{checksum}  {args.output}")
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(f"{checksum}  {args.output.name}\n")


if __name__ == "__main__":
    main()
