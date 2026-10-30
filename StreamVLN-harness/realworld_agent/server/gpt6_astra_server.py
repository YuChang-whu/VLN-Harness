"""Optional GPT-6 Astra entry point for the existing subgoal LiveVLN server.

Without ``--GPT`` this module delegates directly to the original server. With
``--GPT`` it preserves the original HTTP/session/VQA contracts while replacing
intent parsing and per-request action prediction with GPT-6 Astra.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from dataclasses import asdict
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any

from realworld_agent.planning.astra_intent_parser import AstraIntentParser
from realworld_agent.planning.manager import PlanningManager
from realworld_agent.policy.astra_client import AstraClient, AstraConfig, DEFAULT_ASTRA_MODEL
from realworld_agent.server import subgoal_livevln_server as legacy


def _astra_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--GPT",
        "--gpt",
        dest="use_gpt",
        action="store_true",
        help="use GPT-6 Astra for intent parsing and one-step RGB action prediction",
    )
    parser.add_argument(
        "--astra-model",
        default=os.getenv("OPENAI_MODEL", "").strip() or DEFAULT_ASTRA_MODEL,
    )
    parser.add_argument(
        "--astra-base-url",
        default=os.getenv("OPENAI_BASE_URL", "").strip(),
        help="optional OpenAI-compatible base URL; blank uses the official SDK default",
    )
    parser.add_argument("--astra-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--astra-timeout", type=float, default=30.0)
    parser.add_argument("--astra-max-retries", type=int, default=2)
    parser.add_argument(
        "--astra-reasoning-effort",
        choices=("low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    parser.add_argument(
        "--astra-image-detail",
        choices=("low", "high", "auto"),
        default="high",
    )
    parser.add_argument("--astra-max-output-tokens", type=int, default=1024)
    return parser


def parse_args() -> argparse.Namespace:
    """Add Astra options while continuing to use the original argument parser."""
    original_argv = list(sys.argv)
    astra_args, legacy_argv = _astra_parser().parse_known_args(original_argv[1:])
    try:
        sys.argv = [original_argv[0], *legacy_argv]
        args = legacy.parse_args()
    finally:
        sys.argv = original_argv
    for key, value in vars(astra_args).items():
        setattr(args, key, value)
    return args


class AstraSubgoalRuntime(legacy.SubgoalRuntime):
    """Astra policy overlay; the inherited VQA implementation is unchanged."""

    def __init__(self, args: argparse.Namespace, streamvln_server_module):
        super().__init__(args, streamvln_server_module, evaluator=None)
        self.astra_client = AstraClient(
            AstraConfig(
                model=args.astra_model,
                base_url=args.astra_base_url,
                api_key_env=args.astra_api_key_env,
                timeout=args.astra_timeout,
                max_retries=args.astra_max_retries,
                reasoning_effort=args.astra_reasoning_effort,
                image_detail=args.astra_image_detail,
                max_output_tokens=args.astra_max_output_tokens,
            )
        )

    def _requires_online_qwen(self, metadata: dict[str, Any]) -> bool:
        del metadata
        return False

    def _build_planner(self, metadata: dict[str, Any]) -> PlanningManager:
        del metadata
        return PlanningManager(intent_parser=AstraIntentParser(self.astra_client))

    def _complete_preflight_steps(self) -> None:
        assert self.session is not None
        while True:
            plan = self.session.require_plan()
            current = plan.current_step
            if current is None:
                return
            if current.step_id.startswith("subgoal_") or current.step_id == "final_stop":
                return
            if current.step_id == "understand_instruction":
                evidence = "GPT-6 Astra 已输出 goal / landmarks / constraints / subgoals。"
            elif current.step_id == "stabilize_observation":
                evidence = "真实 RGB 请求已到达，进入 GPT-6 Astra 子目标执行循环。"
            else:
                evidence = "进入 GPT-6 Astra 子目标执行循环前完成准备步骤。"
            self.session.complete_current_step(evidence)

    def step(self, rgb, depth, intrinsic, metadata: dict[str, Any]) -> dict[str, Any]:
        # The original handler has already decoded/saved RGB-D before this call.
        # Astra intentionally receives only the original RGB bytes.
        del rgb, depth, intrinsic
        if self.session is None:
            instruction = (
                metadata.get("prompt")
                or metadata.get("instruction")
                or metadata.get("task")
                or self.args.default_instruction
            )
            self.reset(instruction, metadata)

        assert self.session is not None
        plan = self.session.require_plan()
        current = plan.current_step
        if self.terminate or current is None:
            self.terminate = True
            return self._action_payload([0], 0.0, self.last_llm_output, "", "", False)

        image_path = self.latest_saved_artifacts.get("image", "")
        if not image_path:
            raise RuntimeError("saved RGB image path is unavailable for Astra inference")
        image_bytes = Path(image_path).read_bytes()

        active_instruction = current.streamvln_instruction or self._format_streamvln_instruction(current)
        completion_condition = current.completion_condition or "Astra outputs STOP for this subgoal"
        self.session.observe(
            f"真实小车 RGB 请求: 使用当前子目标 instruction='{active_instruction}'；"
            f"完成条件='{completion_condition}'。"
        )

        result = self.astra_client.predict_action(
            image_bytes=image_bytes,
            full_instruction=self.instruction,
            active_instruction=active_instruction,
            completion_condition=completion_condition,
            previous_actions=self.session.last_actions,
            step_id=current.step_id,
        )
        raw_actions = result.actions
        self.action_seq = list(raw_actions)
        self.last_llm_output = result.raw_output
        full_actions, action_e, guard_g, _ = legacy.split_livevln_actions(raw_actions)

        self.session.append_history_event(
            "gpt6_astra_inference_completed",
            {
                "model": self.args.astra_model,
                "response_id": result.response_id,
                "action_sequence": full_actions,
                "active_instruction": active_instruction,
                "completion_condition": completion_condition,
                "rgb_only": True,
            },
            request_id=self.latest_request_id or None,
        )

        # Astra produces one action. Any STOP immediately completes the current
        # subgoal, independent of the reason text and stop-confirmations setting.
        subgoal_completed = action_e == 0
        if subgoal_completed:
            self.subgoal_stop_count = 1
            self.session.append_history_event(
                "gpt6_astra_stop_detected",
                {
                    "action_e": action_e,
                    "guard_g": guard_g,
                    "action_sequence": full_actions,
                    "active_instruction": active_instruction,
                    "completion_condition": completion_condition,
                },
                request_id=self.latest_request_id or None,
            )
            self.session.apply_subgoal_policy_actions(
                [0],
                source="gpt6_astra_subgoal_policy",
                request_id=self.latest_request_id or None,
            )
            self.action_seq = []
            if plan.current_step is None:
                self.terminate = True
        else:
            self.subgoal_stop_count = 0
            self.session.apply_subgoal_policy_actions(
                full_actions,
                source="gpt6_astra_subgoal_policy",
                request_id=self.latest_request_id or None,
            )

        return self._action_payload(
            raw_actions,
            result.generate_time,
            result.raw_output,
            active_instruction,
            completion_condition,
            subgoal_completed,
        )


class AstraLiveVLNHandler(legacy.SubgoalLiveVLNHandler):
    server_version = "SubgoalGPT6Astra/1.0"

    def _print_eval_summary(
        self,
        session,
        request_id: str,
        payload: dict[str, Any],
        active_before: dict[str, Any] | None,
    ) -> None:
        if payload.get("hold"):
            return super()._print_eval_summary(session, request_id, payload, active_before)

        active_id = active_before.get("step_id") if active_before else "no_active_subgoal"
        print(
            f"[REQ {request_id}] {active_id} 进行中 | "
            f"action_e={self._action_label(payload, 'action_e')} "
            f"guard_g={self._action_label(payload, 'guard_g')} | "
            f"completed={str(payload.get('subgoal_completed', False)).lower()} | "
            f"terminate={str(payload.get('terminate', False)).lower()} | "
            f"elapsed_ms={payload.get('elapsed_ms')}",
            flush=True,
        )
        if payload.get("subgoal_completed"):
            print(f"[SUBGOAL] {active_id} 已完成", flush=True)
            print("  reason: GPT-6 Astra output STOP", flush=True)
            next_subgoal = payload.get("active_subgoal")
            if next_subgoal:
                print(f"[SUBGOAL] {next_subgoal.get('step_id')} 进行中", flush=True)
                print(f"  instruction: {next_subgoal.get('instruction')}", flush=True)
            elif payload.get("terminate"):
                print("[TASK] 所有子目标已完成，任务结束", flush=True)
        elif payload.get("terminate"):
            print("[TASK] 所有子目标已完成，任务结束", flush=True)


def main() -> None:
    args = parse_args()
    if not args.use_gpt:
        legacy.main()
        return

    legacy.configure_huggingface_cache(offline=not args.allow_hf_network)
    streamvln_server = legacy.load_streamvln_server_module(args.streamvln_deploy_root)
    server = ThreadingHTTPServer((args.host, args.port), AstraLiveVLNHandler)
    server.args = args
    server.streamvln_server = streamvln_server
    server.evaluator = None
    server.runtime = AstraSubgoalRuntime(args, streamvln_server)
    server.infer_lock = threading.Lock()

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    endpoint_label = args.astra_base_url or "OpenAI SDK default"
    print("Listening on http://%s:%d" % (args.host, args.port), flush=True)
    print("POST endpoint: /eval_vln", flush=True)
    print("Service: subgoal_livevln_server + GPT-6 Astra", flush=True)
    print("Intent/action backend:", args.astra_model, flush=True)
    print("Astra base URL:", endpoint_label, flush=True)
    print("Astra API key env:", args.astra_api_key_env, flush=True)
    print("Astra visual input: RGB only", flush=True)
    print("VQA backend unchanged:", args.vqa_backend, flush=True)
    print("StreamVLN/Qwen model weights: not loaded in --GPT mode", flush=True)
    print("Saving session history to", args.save_dir, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
