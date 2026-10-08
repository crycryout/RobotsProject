"""Regenerate deterministic statistics from an existing episode table."""

import argparse
from pathlib import Path

from robots_project.evaluation.artifacts import summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    summary = summarize(args.run_dir, plots=not args.no_plots)
    print(f"{args.run_dir / 'summary.json'}: complete={summary['complete']}, "
          f"macro_success_once={summary['macro_success_once']}")
    return 0 if summary["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
