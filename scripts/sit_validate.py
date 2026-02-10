#!/usr/bin/env python3
"""Wrapper script to validate a SIT run.

Usage:
    python scripts/sit_validate.py --run-id <run-id>
"""

import sys
import os

# Ensure the project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sit.cli import cli

if __name__ == "__main__":
    cli(["validate"] + sys.argv[1:])
