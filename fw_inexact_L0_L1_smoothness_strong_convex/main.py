"""Main entry point for experiments with mixed inexact Frank-Wolfe."""

import argparse
import os
import sys

# Add parent directory to path if running directly
if __name__ == "__main__" and __package__ is None:
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)

try:
    # Try relative import first (when run as module)
    from .experiments import run_experiments
except ImportError:
    # Fall back to absolute import (when run directly)
    from experiments import run_experiments


def main():
    parser = argparse.ArgumentParser(
        description="Experiments for Frank-Wolfe with mixed inexactness and (L0, L1)-smoothness"
    )
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Run in interactive mode with plot display")
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="Enable verbose output")
    args = parser.parse_args()

    run_experiments(verbose=args.verbose, interactive=args.interactive)

    if not args.interactive:
        print("Done.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())