"""Main entry point for experiments with (H0, H1)-smooth Frank-Wolfe."""

import argparse
import os
import sys

if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)

try:
    from .experiments import run_experiments
except ImportError:
    from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Experiments for Frank-Wolfe under (H0, H1)-smoothness"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        default=False,
        help="Run in interactive mode with plot display",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable verbose output",
    )
    args = parser.parse_args()

    run_experiments(verbose=args.verbose, interactive=args.interactive)

    if not args.interactive:
        print("Done.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
