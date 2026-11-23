#!/usr/bin/env python3
# Copyright 2026 The SenseNova-RoboRSI Authors.
# Licensed under the Apache License, Version 2.0 (the "License");

"""Repository entry point for online WET-OP evaluation (no local simulator)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roborsi.evaluation.wetop_online import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
