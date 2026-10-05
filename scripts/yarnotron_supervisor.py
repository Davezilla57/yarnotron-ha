#!/usr/bin/env python3
"""Run the Mac supervisor: python3 scripts/yarnotron_supervisor.py --config supervisor.toml"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yarnotron.supervisor.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
