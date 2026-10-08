"""Validate all completed episodes and their manifest checksums."""

import argparse
from pathlib import Path

from robots_project.data.validation import validate_run
from robots_project.utils import atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    report = validate_run(args.run_dir)
    atomic_json(args.run_dir / "validation.json", report)
    print(f"Validated {len(report['episodes'])} episodes; errors={len(report['errors'])}; "
          f"complete_run={report['complete_run']}")
    return 0 if report["valid"] and report["complete_run"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
