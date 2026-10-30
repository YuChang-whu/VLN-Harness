#!/usr/bin/env python3
"""Terminal demo for model-driven task/plan flow visualization.

Run:
    python clients/operator_client/plan_demo.py "go forward, enter the room, stop near the table"
    python clients/operator_client/plan_demo.py --fallback-only "go forward to enter the room"
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from realworld_agent.planning.intent_parser import QwenIntentParser, RuleFallbackIntentParser, StreamVLNIntentParser
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


STOP_HINTS = ("stop", "wait", "停止", "停下", "停在", "等待", "终点")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-world VLN planning terminal demo")
    parser.add_argument(
        "instruction",
        nargs="?",
        default="go forward to enter the room, then stop near the table",
        help="navigation instruction to plan and display",
    )
    parser.add_argument("--interactive", action="store_true", help="manually drive plan updates")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    parser.add_argument("--delay", type=float, default=0.7, help="demo delay between updates")
    parser.add_argument(
        "--parser",
        choices=("qwen", "streamvln", "rule"),
        default="qwen",
        help="intent parser backend; qwen is the default semantic parser",
    )
    parser.add_argument(
        "--fallback-only",
        action="store_true",
        help="backward-compatible alias for --parser rule",
    )
    parser.add_argument(
        "--strict-model",
        action="store_true",
        help="raise model loading/parsing errors instead of falling back",
    )
    parser.add_argument(
        "--qwen-model-path",
        type=Path,
        default=DEFAULT_QWEN_MODEL_PATH,
        help="local Qwen model path for semantic intent parsing",
    )
    parser.add_argument("--qwen-device-map", default="auto", help="Qwen device_map setting")
    parser.add_argument(
        "--qwen-torch-dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
        help="Qwen torch dtype",
    )
    parser.add_argument(
        "--qwen-enable-thinking",
        action="store_true",
        help="enable Qwen thinking mode if the tokenizer supports it",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="StreamVLN real-world checkpoint path",
    )
    parser.add_argument(
        "--streamvln-root",
        type=Path,
        default=DEFAULT_STREAMVLN_ROOT,
        help="StreamVLN source root containing streamvln/model/stream_video_vln.py",
    )
    parser.add_argument("--device", default="cuda:0", help="device for StreamVLN inference")
    parser.add_argument("--attn-implementation", default="eager", help="StreamVLN attention implementation")
    parser.add_argument("--max-new-tokens", type=int, default=768, help="max tokens for intent JSON generation")
    parser.add_argument("--num-frames", type=int, default=32, help="StreamVLN num_frames runtime setting")
    parser.add_argument("--num-history", type=int, default=8, help="StreamVLN num_history runtime setting")
    parser.add_argument(
        "--skip-load-check",
        action="store_true",
        help="skip the original evaluator.step load/inference check before intent parsing",
    )
    return parser


def build_planner(args: argparse.Namespace) -> PlanningManager:
    parser_backend = "rule" if args.fallback_only else args.parser
    if parser_backend == "rule":
        return PlanningManager(intent_parser=RuleFallbackIntentParser())

    if parser_backend == "streamvln":
        config = StreamVLNGenerationConfig(
            model_path=args.model_path,
            streamvln_root=args.streamvln_root,
            device=args.device,
            attn_implementation=args.attn_implementation,
            max_new_tokens=args.max_new_tokens,
            num_frames=args.num_frames,
            num_history=args.num_history,
            verify_realworld_step=not args.skip_load_check,
        )
        model_client = StreamVLNModelClient(config=config)
        return PlanningManager(
            intent_parser=StreamVLNIntentParser(
                model_client=model_client,
                allow_fallback=not args.strict_model,
            )
        )

    qwen_config = QwenGenerationConfig(
        model_path=args.qwen_model_path,
        device_map=args.qwen_device_map,
        torch_dtype=args.qwen_torch_dtype,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.qwen_enable_thinking,
    )
    qwen_client = QwenIntentClient(config=qwen_config)
    return PlanningManager(
        intent_parser=QwenIntentParser(
            model_client=qwen_client,
            allow_fallback=not args.strict_model,
        )
    )


def run_auto_demo(session: TaskSessionManager, renderer: TerminalRenderer, delay: float) -> None:
    session.start()
    renderer.print(session)
    time.sleep(delay)

    scripted_updates = [
        ("observe", "RGB-D 和 odom 已同步，初始观测稳定。", None),
        ("done", "已由意图解析器生成 goal / stop condition / landmarks / subgoals。", None),
        ("observe", "当前前方可通行，准备推进到第一个语义子目标。", None),
        ("done", "初始观测稳定，可以开始执行子目标。", None),
    ]

    for kind, message, actions in scripted_updates:
        if kind == "observe":
            session.observe(message)
        elif kind == "actions":
            session.apply_actions(actions, source="demo_policy")
        elif kind == "done":
            session.complete_current_step(message)
        renderer.print(session, clear=True)
        time.sleep(delay)

    while not session.is_done():
        current = session.require_plan().current_step
        if current is None:
            session.apply_actions([0], source="demo_policy")
            session.require_plan().stop_reason = "所有子目标完成，到达终点后 STOP。"
        elif current.step_id == "final_stop":
            session.apply_actions([0], source="demo_policy")
        elif any(hint in current.detail.lower() or hint in current.detail for hint in STOP_HINTS):
            session.observe(f"围绕 {current.title} 获取终点观测。")
            session.apply_actions([0], source="demo_policy")
            session.require_plan().stop_reason = "终点子目标已到达，执行 STOP。"
        else:
            session.observe(f"围绕 {current.title} 获取新观测。")
            session.apply_actions([1], source="demo_policy")
            session.complete_current_step(f"{current.title} 已到达，子目标打勾。")

        renderer.print(session, clear=True)
        time.sleep(delay)


def run_interactive(session: TaskSessionManager, renderer: TerminalRenderer) -> None:
    session.start()
    renderer.print(session)
    print("\nCommands: show | observe <note> | actions <ids/text> | done <evidence> | stop [reason] | quit")

    while True:
        try:
            command = input("vln-agent> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not command:
            continue
        if command in {"quit", "exit"}:
            break
        if command == "show":
            renderer.print(session)
            continue
        if command.startswith("observe "):
            session.observe(command[len("observe ") :].strip())
        elif command.startswith("actions "):
            session.apply_actions(command[len("actions ") :].strip(), source="operator")
        elif command.startswith("done"):
            evidence = command[len("done") :].strip()
            session.complete_current_step(evidence or "人工确认当前子目标已完成。")
        elif command.startswith("stop"):
            reason = command[len("stop") :].strip()
            session.stop(reason or "人工请求 STOP。")
        else:
            print("Unknown command. Use: show | observe <note> | actions <ids/text> | done <evidence> | stop [reason] | quit")
            continue

        renderer.print(session)
        if session.is_done():
            print("Task is stopped. Use quit to exit.")


def main() -> int:
    args = build_parser().parse_args()
    planner = build_planner(args)
    session = TaskSessionManager(instruction=args.instruction, planner=planner)
    renderer = TerminalRenderer(use_color=not args.no_color)

    if args.interactive:
        run_interactive(session, renderer)
    else:
        run_auto_demo(session, renderer, args.delay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
