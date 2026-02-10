#!/usr/bin/env python3
"""Wrapper script to run the SIT pipeline.

Usage:
    python scripts/sit_run.py --config configs/smoke.yaml
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sit.cli import cli

if __name__ == "__main__":
    cli(["run"] + sys.argv[1:])
