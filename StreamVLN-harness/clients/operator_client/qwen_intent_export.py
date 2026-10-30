#!/usr/bin/env python3
"""Export Qwen3-8B VLN intent parsing result as JSON."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from realworld_agent.planning.intent_parser import QwenIntentParser
from realworld_agent.policy.qwen_client import DEFAULT_QWEN_MODEL_PATH, QwenGenerationConfig, QwenIntentClient


DEFAULT_INSTRUCTION = (
    "Move toward first, ,turn left when you are very close to the plants, "
    "go through the door, and stop immediately after entering the corridor."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Qwen VLN intent JSON")
    parser.add_argument("instruction", nargs="?", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument(
        "--qwen-torch-dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
    )
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    client = QwenIntentClient(
        QwenGenerationConfig(
            model_path=args.qwen_model_path,
            device_map=args.qwen_device_map,
            torch_dtype=args.qwen_torch_dtype,
            max_new_tokens=args.max_new_tokens,
        )
    )
    parser = QwenIntentParser(model_client=client, allow_fallback=False)
    intent = parser.parse(args.instruction)
    payload = asdict(intent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"\nWROTE_INTENT_JSON={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
