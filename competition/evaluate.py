"""Organizer CLI: python -m competition.evaluate PREDICTIONS TRUTH --manifest MANIFEST."""
import argparse
import json
from .task import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions")
    parser.add_argument("truth")
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.predictions, args.truth, args.manifest), indent=2))


if __name__ == "__main__":
    main()
