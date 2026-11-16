#!/usr/bin/env python3
"""Generate the deterministic R187 synthetic profiles and manifest only."""
from __future__ import annotations

import argparse
from pathlib import Path

import run as experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    experiment.ROOT = args.output_dir.expanduser().resolve()
    manifest = experiment.setup()
    print(f"Generated {len(manifest['profiles'])} profiles in {experiment.ROOT}")


if __name__ == "__main__":
    main()
