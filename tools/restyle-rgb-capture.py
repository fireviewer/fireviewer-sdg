#!/usr/bin/env python3
"""Prepare, run or admit one composition-locked RGB restyle capture."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from fireviewer_sdg.rgb_restyle import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
