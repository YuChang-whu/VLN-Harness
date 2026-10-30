#!/usr/bin/env python3
"""Run Qwen-planned subgoals with StreamVLN STOP-based switching.

This demo implements the first closed-loop policy:
1. Qwen parses the full instruction into semantic subgoals.
2. The current subgoal description is sent to StreamVLN as the instruction.
3. Non-STOP actions keep the current subgoal active.
4. STOP marks the current subgoal complete and switches to the next subgoal.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from realworld_agent.execution.subgoal_executor import SubgoalExecutorConfig, StreamVLNSubgoalExecutor
from realworld_agent.planning.intent_parser import IntentPlan, QwenIntentParser, RuleFallbackIntentParser, intent_from_payload
from realworld_agent.planning.manager import PlanningManager
from realworld_agent.policy.qwen_client import DEFAULT_QWEN_MODEL_PATH, QwenGenerationConfig, QwenIntentClient
from realworld_agent.policy.streamvln_client import (
    DEFAULT_MODEL_PATH,
    DEFAULT_STREAMVLN_ROOT,
    StreamVLNGenerationConfig,
    StreamVLNModelClient,
)
from realworld_agent.session.task_session import TaskSessionManager
from realworld_agent.terminal.renderer import TerminalRenderer


DEFAULT_INSTRUCTION = (
    "Move toward first, ,turn left when you are very close to the plants, "
    "go through the door, and stop immediately after entering the corridor."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Qwen plan + StreamVLN subgoal execution demo")
    parser.add_argument("instruction", nargs="?", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--planner", choices=("qwen", "rule"), default="qwen")
    parser.add_argument("--intent-json-path", type=Path, help="load a Qwen-parsed intent JSON file instead of running planner model")
    parser.add_argument("--strict-planner", action="store_true", help="raise Qwen parser errors instead of fallback")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--delay", type=float, default=0.2)
    parser.add_argument("--max-steps-per-subgoal", type=int, default=0, help="0 means no debug step limit")
    parser.add_argument("--run-model-every", type=int, default=1)

    parser.add_argument("--qwen-model-path", type=Path, default=DEFAULT_QWEN_MODEL_PATH)
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument(
        "--qwen-torch-dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=512)
    parser.add_argument("--qwen-enable-thinking", action="store_true")

    parser.add_argument("--streamvln-model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--streamvln-root", type=Path, default=DEFAULT_STREAMVLN_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--streamvln-max-new-tokens", type=int, default=256)
    parser.add_argument("--num-frames", type=int, default=32)
    parser.add_argument("--num-history", type=int, default=8)
    parser.add_argument("--skip-load-check", action="store_true")
    return parser


class FileIntentParser:
    def __init__(self, intent_json_path: Path):
        self.intent_json_path = intent_json_path

    def parse(self, instruction: str, observation=None) -> IntentPlan:
        payload = json.loads(self.intent_json_path.read_text(encoding="utf-8"))
        raw_output = payload.get("raw_output", "") if isinstance(payload, dict) else ""
        if isinstance(payload, dict) and "instruction" in payload and "subgoals" in payload:
            instruction = str(payload.get("instruction") or instruction)
        intent = intent_from_payload(instruction, payload)
        intent.parser_name = f"Qwen intent JSON file ({self.intent_json_path.name})"
        intent.raw_output = raw_output
        return intent


def build_planner(args: argparse.Namespace) -> PlanningManager:
    if args.intent_json_path is not None:
        return PlanningManager(intent_parser=FileIntentParser(args.intent_json_path))

    if args.planner == "rule":
        return PlanningManager(intent_parser=RuleFallbackIntentParser())

    qwen_client = QwenIntentClient(
        QwenGenerationConfig(
            model_path=args.qwen_model_path,
            device_map=args.qwen_device_map,
            torch_dtype=args.qwen_torch_dtype,
            max_new_tokens=args.qwen_max_new_tokens,
            enable_thinking=args.qwen_enable_thinking,
        )
    )
    return PlanningManager(
        intent_parser=QwenIntentParser(
            model_client=qwen_client,
            allow_fallback=not args.strict_planner,
        )
    )


def build_streamvln_client(args: argparse.Namespace) -> StreamVLNModelClient:
    config = StreamVLNGenerationConfig(
        model_path=args.streamvln_model_path,
        streamvln_root=args.streamvln_root,
        device=args.device,
        attn_implementation=args.attn_implementation,
        max_new_tokens=args.streamvln_max_new_tokens,
        num_frames=args.num_frames,
        num_history=args.num_history,
        verify_realworld_step=not args.skip_load_check,
    )
    return StreamVLNModelClient(config=config)


def main() -> int:
    args = build_parser().parse_args()
    planner = build_planner(args)
    streamvln_client = build_streamvln_client(args)
    session = TaskSessionManager(instruction=args.instruction, planner=planner)
    renderer = TerminalRenderer(use_color=not args.no_color)

    session.start()
    renderer.print(session)

    executor = StreamVLNSubgoalExecutor(
        session=session,
        streamvln_client=streamvln_client,
        renderer=renderer,
        config=SubgoalExecutorConfig(
            max_steps_per_subgoal=args.max_steps_per_subgoal,
            delay=args.delay,
            run_model_every=args.run_model_every,
        ),
    )
    executor.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
