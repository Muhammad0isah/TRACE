"""Validate plain predictions before encrypting them."""
import argparse
import json
from .task import validate_submission


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    result = validate_submission(args.predictions, args.manifest)
    print(json.dumps({"valid": True, "images": len(result["image_ids"]),
                      "detections": len(result["predictions"])}))


if __name__ == "__main__":
    main()
