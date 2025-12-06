"""Main entry point to program."""

# coding: utf-8

import argparse
import sys

from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true", default=False)
    args = parser.parse_args()

    run_experiments(verbose=args.verbose, interactive=args.interactive)

    if not args.interactive:
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
