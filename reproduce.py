#!/usr/bin/env python3
"""Reviewer-facing reproduction interface."""

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from exact_feedback.reproduction.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
