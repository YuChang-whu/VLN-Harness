#!/usr/bin/env python3
"""CLI entrypoint for the subgoal-aware LiveVLN HTTP server."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from realworld_agent.server.subgoal_livevln_server import main


if __name__ == "__main__":
    main()
