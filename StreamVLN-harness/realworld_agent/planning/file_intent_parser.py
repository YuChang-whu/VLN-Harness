"""Intent parser backed by a precomputed Qwen JSON file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from realworld_agent.planning.intent_parser import IntentPlan, intent_from_payload


class FileIntentParser:
    """Load structured VLN intent from a JSON file exported by Qwen."""

    def __init__(self, intent_json_path: Path):
        self.intent_json_path = Path(intent_json_path)

    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        payload = json.loads(self.intent_json_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"intent JSON must be an object: {self.intent_json_path}")
        raw_output = str(payload.get("raw_output") or "")
        instruction = str(payload.get("instruction") or instruction)
        intent = intent_from_payload(instruction, payload)
        intent.parser_name = f"Qwen intent JSON file ({self.intent_json_path.name})"
        intent.raw_output = raw_output
        return intent
