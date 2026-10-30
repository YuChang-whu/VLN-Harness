"""Convenience launcher for the optional GPT-6 Astra server overlay."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


HARNESS_ROOT = Path(__file__).resolve().parents[2]
if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))


if __name__ == "__main__":
    runpy.run_module("realworld_agent.server.gpt6_astra_server", run_name="__main__")
