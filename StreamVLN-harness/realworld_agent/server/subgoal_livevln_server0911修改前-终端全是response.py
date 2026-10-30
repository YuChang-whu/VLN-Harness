"""LiveVLN-compatible server with Qwen subgoal tracking.

The robot POST interface intentionally mirrors the original Limo LiveVLN server:
- multipart/form-data
- JSON metadata field named ``json``
- file fields ``image`` and optional ``depth``
- response fields ``action_e``, ``guard_g``, ``revisable_tail`` and
  ``action_sequence``.

The difference is that this server keeps a task session. The full instruction
is first converted to semantic subgoals, then only the current subgoal text is
sent to StreamVLN. A StreamVLN immediate STOP means the current subgoal is
complete; the next request will use the next subgoal as instruction.
"""

from __future__ import annotations

import argparse
import cgi
import gc
import importlib.util
import json
import sys
import threading
import time
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HARNESS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STREAMVLN_DEPLOY_ROOT = Path("/root/autodl-tmp/yc_code/StreamVLN-harness")
DEFAULT_STREAMVLN_MODEL_PATH = Path(
    "/root/autodl-tmp/yc_code/StreamVLN/data/pretrained_models/"
    "StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"
)
DEFAULT_QWEN_MODEL_PATH = HARNESS_ROOT / "models" / "Qwen3-8B"
DEFAULT_SAVE_DIR = HARNESS_ROOT / "outputs" / "sessions"
DEFAULT_HF_HOME = Path("/root/autodl-tmp/system_cache/huggingface")
DEFAULT_HF_HUB_CACHE = DEFAULT_HF_HOME / "hub"


def configure_huggingface_cache(offline: bool = True) -> None:
    """Keep StreamVLN/SigLIP loading on the data disk and avoid flaky network calls."""
    import os

    os.environ.setdefault("HF_HOME", str(DEFAULT_HF_HOME))
    os.environ.setdefault("HF_HUB_CACHE", str(DEFAULT_HF_HUB_CACHE))
    os.environ.setdefault("TRANSFORMERS_CACHE", str(DEFAULT_HF_HUB_CACHE))
    os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


if str(HARNESS_ROOT) not in sys.path:
    sys.path.insert(0, str(HARNESS_ROOT))

from realworld_agent.hitl.vqa_manager import COMPLETE_SUBGOAL, CONTINUE, STOP, VQAManager
from realworld_agent.policy.vqa_evaluator import VQAEvaluator, VQAEvaluatorConfig
from realworld_agent.planning.file_intent_parser import FileIntentParser
from realworld_agent.planning.intent_parser import QwenIntentParser, RuleFallbackIntentParser
from realworld_agent.planning.manager import PlanningManager
from realworld_agent.planning.qwen_subprocess_parser import QwenSubprocessIntentParser
from realworld_agent.planning.types import TaskPlan
from realworld_agent.policy.qwen_client import QwenGenerationConfig, QwenIntentClient
from realworld_agent.session.task_session import TaskSessionManager


def load_streamvln_server_module(streamvln_deploy_root: Path):
    root = Path(streamvln_deploy_root)
    server_path = root / "limo" / "limo_realworld_server.py"
    if not server_path.exists():
        raise FileNotFoundError(f"missing StreamVLN real-world server: {server_path}")
    for path in (root / "limo", root, root / "streamvln"):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    spec = importlib.util.spec_from_file_location("harness_limo_realworld_server", server_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import StreamVLN server module from {server_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def split_livevln_actions(actions: list[int]) -> tuple[list[int], int, int, list[int]]:
    full_actions = [int(action) for action in actions]
    if not full_actions:
        full_actions = [0]
    action_e = full_actions[0]
    guard_g = full_actions[1] if len(full_actions) > 1 else 0
    revisable_tail = full_actions[2:]
    return full_actions, action_e, guard_g, revisable_tail


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json_body(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("content-length", "0") or 0)
    if length <= 0:
        return {}
    body = handler.rfile.read(length)
    if not body:
        return {}
    value = json.loads(body.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON body must be an object")
    return value


def write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_safe(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_session_request_inputs(
    session: TaskSessionManager,
    request_id: str,
    image_bytes: bytes,
    depth_bytes: bytes,
    metadata: dict[str, Any],
    request_payload: dict[str, Any],
) -> dict[str, str]:
    request_dir = session.request_dir(request_id)
    saved: dict[str, str] = {}

    image_path = request_dir / "rgb.jpg"
    image_path.write_bytes(image_bytes)
    saved["image"] = str(image_path)

    if depth_bytes:
        depth_path = request_dir / "depth.png"
        depth_path.write_bytes(depth_bytes)
        saved["depth"] = str(depth_path)

    metadata_path = request_dir / "metadata.json"
    write_json_file(metadata_path, metadata)
    saved["metadata"] = str(metadata_path)

    request_path = request_dir / "request.json"
    request_payload = dict(request_payload)
    request_payload["saved"] = saved
    write_json_file(request_path, request_payload)
    saved["request"] = str(request_path)
    return saved


def write_session_response(
    session: TaskSessionManager,
    request_id: str,
    response_payload: dict[str, Any],
) -> str:
    response_path = session.request_dir(request_id) / "response.json"
    write_json_file(response_path, response_payload)
    return str(response_path)


def plan_to_payload(plan: TaskPlan) -> dict[str, Any]:
    current = plan.current_step
    subgoal_steps = [step for step in plan.steps if step.step_id.startswith("subgoal_") or step.step_id == "final_stop"]
    current_subgoal_index = 0
    for index, step in enumerate(subgoal_steps, start=1):
        if current is not None and step.step_id == current.step_id:
            current_subgoal_index = index
            break
    return {
        "progress_ratio": plan.progress_ratio,
        "completed_count": plan.completed_count,
        "total_steps": len(plan.steps),
        "current_step": None if current is None else asdict(current),
        "active_subgoal": None if current is None else {
            "index": current_subgoal_index,
            "count": len(subgoal_steps),
            "step_id": current.step_id,
            "title": current.title,
            "instruction": current.streamvln_instruction or current.detail,
            "description": current.detail,
            "completion_condition": current.completion_condition,
            "subgoal_type": current.subgoal_type,
            "status": current.status,
        },
        "subgoals": [
            {
                "index": index,
                "step_id": step.step_id,
                "title": step.title,
                "instruction": step.streamvln_instruction or step.detail,
                "description": step.detail,
                "completion_condition": step.completion_condition,
                "subgoal_type": step.subgoal_type,
                "status": step.status,
                "evidence": step.evidence,
            }
            for index, step in enumerate(subgoal_steps, start=1)
        ],
        "intent": None if plan.intent is None else {
            "goal": plan.intent.goal,
            "stop_condition": plan.intent.stop_condition,
            "landmarks": plan.intent.landmarks,
            "constraints": plan.intent.constraints,
            "route_hints": plan.intent.route_hints,
            "parser_name": plan.intent.parser_name,
            "fallback": plan.intent.fallback,
            "fallback_reason": plan.intent.fallback_reason,
        },
        "stopped": plan.stopped,
        "stop_reason": plan.stop_reason,
        "recent_events": [asdict(event) for event in plan.events[-8:]],
    }


class SubgoalRuntime:
    def __init__(self, args: argparse.Namespace, streamvln_server_module, evaluator=None):
        self.args = args
        self.streamvln_server = streamvln_server_module
        self.evaluator = evaluator
        self.session: TaskSessionManager | None = None
        self.instruction = ""
        self.action_seq: list[int] = []
        self.last_llm_output = ""
        self.terminate = False
        self.subgoal_stop_count = 0
        self.latest_request_id = ""
        self.latest_saved_artifacts: dict[str, str] = {}
        self.latest_plan_payload: dict[str, Any] = {}
        self.vqa_manager = VQAManager(
            VQAEvaluator(
                VQAEvaluatorConfig(
                    backend=self.args.vqa_backend,
                    model_path=Path(self.args.vqa_model_path),
                    streamvln_deploy_root=Path(self.args.streamvln_deploy_root),
                    device=self.args.vqa_device,
                    attn_implementation=self.args.vqa_attn_implementation,
                    model_max_length=self.args.model_max_length,
                    max_new_tokens=self.args.vqa_max_new_tokens,
                    num_future_steps=self.args.num_future_steps,
                    num_frames=self.args.num_frames,
                    num_history=0,
                    rgb_height=self.args.rgb_height,
                    warmup=self.args.vqa_warmup,
                )
            )
        )

    def reset(self, instruction: str, metadata: dict[str, Any]) -> None:
        if self._requires_online_qwen(metadata):
            self._release_evaluator_for_intent_parsing()
        self.instruction = instruction
        self.session = TaskSessionManager(
            instruction=instruction,
            planner=self._build_planner(metadata),
            history_root=Path(self.args.save_dir),
        )
        print("Parsing instruction into subgoals now...", flush=True)
        self.session.start()
        self._complete_preflight_steps()
        if self.evaluator is not None:
            self.evaluator.reset_memory()
        self.terminate = False
        self.action_seq = []
        self.last_llm_output = ""
        self.subgoal_stop_count = 0
        print("Reset Subgoal LiveVLN session", flush=True)
        self._print_plan_summary()

    def _requires_online_qwen(self, metadata: dict[str, Any]) -> bool:
        intent_json_path = metadata.get("intent_json_path") or self.args.intent_json_path
        return not intent_json_path and self.args.planner == "qwen"

    def _release_evaluator_for_intent_parsing(self) -> None:
        if self.evaluator is None:
            return
        print("Releasing StreamVLN model before Qwen intent parsing to avoid GPU OOM...", flush=True)
        evaluator = self.evaluator
        self.evaluator = None
        try:
            if hasattr(evaluator, "reset_memory"):
                evaluator.reset_memory()
        except Exception as exc:
            print("Warning: StreamVLN reset_memory before release failed:", repr(exc), flush=True)
        del evaluator
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("CUDA cache cleared before Qwen intent parsing.", flush=True)
        except Exception as exc:
            print("Warning: CUDA cache clear before Qwen parsing failed:", repr(exc), flush=True)

    def _build_planner(self, metadata: dict[str, Any]) -> PlanningManager:
        intent_json_path = metadata.get("intent_json_path") or self.args.intent_json_path
        if intent_json_path:
            return PlanningManager(intent_parser=FileIntentParser(Path(intent_json_path)))
        if self.args.planner == "rule":
            return PlanningManager(intent_parser=RuleFallbackIntentParser())
        if self.args.qwen_parser_mode == "subprocess":
            return PlanningManager(
                intent_parser=QwenSubprocessIntentParser(
                    python_bin=self.args.qwen_python,
                    model_path=Path(self.args.qwen_model_path),
                    device_map=self.args.qwen_device_map,
                    torch_dtype=self.args.qwen_torch_dtype,
                    max_new_tokens=self.args.qwen_max_new_tokens,
                    timeout=self.args.qwen_timeout,
                    cuda_visible_devices=self.args.qwen_cuda_visible_devices,
                )
            )
        qwen_client = QwenIntentClient(
            QwenGenerationConfig(
                model_path=Path(self.args.qwen_model_path),
                device_map=self.args.qwen_device_map,
                torch_dtype=self.args.qwen_torch_dtype,
                max_new_tokens=self.args.qwen_max_new_tokens,
                enable_thinking=self.args.qwen_enable_thinking,
            )
        )
        return PlanningManager(
            intent_parser=QwenIntentParser(
                model_client=qwen_client,
                allow_fallback=not self.args.strict_planner,
            )
        )

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
                evidence = "意图解析器已输出 goal / landmarks / constraints / subgoals。"
            elif current.step_id == "stabilize_observation":
                evidence = "真实 RGB-D 请求已到达，进入 StreamVLN 子目标执行循环。"
            else:
                evidence = "进入真实世界子目标执行循环前完成准备步骤。"
            self.session.complete_current_step(evidence)

    def _print_plan_summary(self) -> None:
        if self.session is None:
            return
        plan = self.session.require_plan()
        intent = plan.intent
        print("Subgoal plan:", flush=True)
        if intent is not None:
            print("  goal:", intent.goal, flush=True)
            print("  stop_condition:", intent.stop_condition, flush=True)
            print("  parser:", intent.parser_name, flush=True)
        for step in plan.steps:
            if step.step_id.startswith("subgoal_") or step.step_id == "final_stop":
                print(f"  - {step.step_id}: {step.detail}", flush=True)
                print(f"    completion_condition: {step.completion_condition or 'not specified'}", flush=True)
                print(f"    streamvln_instruction: {step.streamvln_instruction or step.detail}", flush=True)

    def is_vqa_active(self) -> bool:
        return bool(self.vqa_manager.active_checkpoint and self.vqa_manager.active_checkpoint.is_active)

    def remember_saved_request(
        self,
        request_id: str,
        saved_artifacts: dict[str, str],
        plan_payload: dict[str, Any],
    ) -> None:
        self.latest_request_id = request_id
        self.latest_saved_artifacts = dict(saved_artifacts)
        self.latest_plan_payload = dict(plan_payload)

    def hold_payload(self, reason: str = "hitl_vqa_active") -> dict[str, Any]:
        plan_payload = plan_to_payload(self.session.require_plan()) if self.session is not None else {}
        return {
            "action": [0],
            "action_e": 0,
            "guard_g": 0,
            "revisable_tail": [],
            "action_sequence": [0],
            "active_instruction": "",
            "active_completion_condition": "",
            "subgoal_completed": False,
            "terminate": False,
            "generate_time": 0.0,
            "llm_output": self.last_llm_output,
            "plan": plan_payload,
            "hold": True,
            "hold_reason": reason,
        }

    def start_vqa_checkpoint(self, reason: str = "operator requested visual QA") -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("cannot start VQA before a task session exists")
        if not self.latest_request_id or not self.latest_saved_artifacts:
            raise RuntimeError("cannot start VQA before the first saved robot request")
        plan_payload = plan_to_payload(self.session.require_plan())
        checkpoint = self.vqa_manager.start_checkpoint(
            session=self.session,
            request_id=self.latest_request_id,
            saved_artifacts=self.latest_saved_artifacts,
            reason=reason,
            plan_payload=plan_payload,
            last_llm_output=self.last_llm_output,
        )
        return self.vqa_manager.status() | {"checkpoint_id": checkpoint.checkpoint_id}

    def ask_vqa(self, checkpoint_id: str, question: str) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("no active task session")
        answer = self.vqa_manager.ask_question(self.session, checkpoint_id, question)
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id or self.vqa_manager.status().get("checkpoint", {}).get("checkpoint_id"),
            "answer": answer.answer,
            "backend": answer.backend,
            "raw_output": answer.raw_output,
        }

    def decide_vqa(self, checkpoint_id: str, decision: str, reason: str) -> dict[str, Any]:
        if self.session is None:
            raise RuntimeError("no active task session")
        checkpoint = self.vqa_manager.active_checkpoint
        decision_request_id = checkpoint.request_id if checkpoint is not None else ""
        payload = self.vqa_manager.record_decision(self.session, checkpoint_id, decision, reason)
        if decision == COMPLETE_SUBGOAL:
            self.session.complete_current_subgoal_by_human_vqa(
                evidence=reason or "Human VQA confirmed current subgoal is complete.",
                request_id=decision_request_id,
            )
            if self.evaluator is not None:
                self.evaluator.reset_memory()
            self.action_seq = []
            self.subgoal_stop_count = 0
            if self.session.require_plan().current_step is None:
                self.session.stop("Human VQA confirmed all subgoals are complete.")
                self.terminate = True
        elif decision == STOP:
            self.session.stop(reason or "Human VQA requested global STOP.")
            self.terminate = True
        elif decision == CONTINUE:
            self.session.append_history_event(
                "hitl_vqa_continue",
                {"reason": reason or "Human chose to continue current subgoal."},
            )
            self.session.write_latest_state()
        return {
            "ok": True,
            "decision": asdict(payload),
            "vqa": self.vqa_manager.status(),
            "plan": plan_to_payload(self.session.require_plan()),
            "terminate": self.terminate,
        }

    def _ensure_evaluator_loaded(self) -> None:
        if self.evaluator is not None:
            return
        print("Loading StreamVLN model after intent parsing from", self.args.model_path, flush=True)
        self.evaluator = self.streamvln_server.load_streamvln(self.args)

    def step(self, rgb, depth, intrinsic, metadata: dict[str, Any]) -> dict[str, Any]:
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

        self._ensure_evaluator_loaded()
        assert self.evaluator is not None

        active_instruction = current.streamvln_instruction or self._format_streamvln_instruction(current)
        completion_condition = current.completion_condition or "StreamVLN outputs STOP for this subgoal"
        generate_time = 0.0
        llm_output = self.last_llm_output
        self.session.observe(
            f"真实小车请求: 使用当前子目标 instruction='{active_instruction}'；"
            f"完成条件='{completion_condition}'。"
        )

        for _ in range(self.args.steps_per_request):
            run_model = self.evaluator.step_id % self.args.run_every == 0
            return_action, step_generate_time, return_llm_output = self.evaluator.step(
                0,
                rgb,
                active_instruction,
                run_model=run_model,
                depth=depth,
                intrinsic=intrinsic,
            )
            if step_generate_time > 0:
                generate_time = float(step_generate_time)
            if return_llm_output is not None:
                llm_output = return_llm_output
                self.last_llm_output = return_llm_output
            if return_action is not None:
                self.action_seq = [int(action) for action in return_action]
            self.evaluator.step_id += 1

        raw_actions = self.action_seq or [0]
        full_actions, action_e, guard_g, _ = split_livevln_actions(raw_actions)
        explicit_guard_stop = len(full_actions) > 1 and guard_g == 0
        committed_stop = action_e == 0 or explicit_guard_stop
        subgoal_completed = False
        if committed_stop:
            self.subgoal_stop_count += 1
            if self.subgoal_stop_count >= max(1, self.args.stop_confirmations):
                stop_source = "action_e" if action_e == 0 else "guard_g"
                subgoal_completed = True
                self.session.append_history_event(
                    "streamvln_committed_stop_detected",
                    {
                        "stop_source": stop_source,
                        "action_e": action_e,
                        "guard_g": guard_g,
                        "action_sequence": full_actions,
                        "active_instruction": active_instruction,
                        "completion_condition": completion_condition,
                    },
                    request_id=self.latest_request_id or None,
                )
                self.session.apply_subgoal_policy_actions([0], source=f"streamvln_livevln_subgoal_{stop_source}")
                if self.evaluator is not None:
                    self.evaluator.reset_memory()
                self.action_seq = []
                if plan.current_step is None:
                    self.terminate = True
        else:
            self.subgoal_stop_count = 0
            self.session.apply_subgoal_policy_actions(full_actions, source="streamvln_livevln_subgoal")

        return self._action_payload(
            raw_actions,
            generate_time,
            llm_output,
            active_instruction,
            completion_condition,
            subgoal_completed,
        )

    def _format_streamvln_instruction(self, step) -> str:
        if step.completion_condition:
            return f"{step.detail}. Stop when {step.completion_condition}."
        return step.detail

    def _action_payload(
        self,
        raw_actions: list[int],
        generate_time: float,
        llm_output: str,
        active_instruction: str,
        completion_condition: str,
        subgoal_completed: bool,
    ) -> dict[str, Any]:
        full_actions, action_e, guard_g, revisable_tail = split_livevln_actions(raw_actions)
        if self.terminate:
            full_actions, action_e, guard_g, revisable_tail = [0], 0, 0, []
        plan_payload = plan_to_payload(self.session.require_plan()) if self.session is not None else {}
        return {
            "action": full_actions,
            "action_e": action_e,
            "guard_g": guard_g,
            "revisable_tail": revisable_tail,
            "action_sequence": full_actions,
            "active_instruction": active_instruction,
            "active_completion_condition": completion_condition,
            "subgoal_completed": subgoal_completed,
            "terminate": self.terminate,
            "generate_time": generate_time,
            "llm_output": llm_output,
            "plan": plan_payload,
        }


class SubgoalLiveVLNHandler(BaseHTTPRequestHandler):
    server_version = "SubgoalLiveVLN/1.0"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def do_GET(self):
        streamvln_server = self.server.streamvln_server
        path = urlparse(self.path).path
        if path == "/hitl/vqa/status":
            payload = {"ok": True, "vqa": self.server.runtime.vqa_manager.status()}
            if self.server.runtime.session is not None:
                session = self.server.runtime.session
                payload["session_id"] = session.session_id
                payload["session_dir"] = str(session.session_dir)
                payload["plan"] = plan_to_payload(session.require_plan())
                payload["latest_request_id"] = self.server.runtime.latest_request_id
                payload["latest_saved"] = self.server.runtime.latest_saved_artifacts
            streamvln_server.json_response(self, 200, payload)
            return
        if path not in ("/", "/eval_vln", "/status"):
            streamvln_server.json_response(self, 404, {"ok": False, "error": "not found"})
            return
        payload = {
            "ok": True,
            "service": "subgoal_livevln_server",
            "endpoint": "/eval_vln",
            "model_path": self.server.args.model_path,
            "save_dir": self.server.args.save_dir,
            "action_semantics": streamvln_server.ACTION_NAMES,
            "response_fields": [
                "action_e",
                "guard_g",
                "revisable_tail",
                "action_sequence",
                "active_subgoal",
                "active_completion_condition",
                "subgoal_completed",
                "plan",
            ],
        }
        payload["vqa"] = self.server.runtime.vqa_manager.status()
        if self.server.runtime.session is not None:
            session = self.server.runtime.session
            payload["session_id"] = session.session_id
            payload["session_dir"] = str(session.session_dir)
            payload["plan"] = plan_to_payload(session.require_plan())
        streamvln_server.json_response(self, 200, payload)

    def do_POST(self):
        streamvln_server = self.server.streamvln_server
        path = urlparse(self.path).path
        if path.startswith("/hitl/vqa/"):
            self._handle_vqa_post(path)
            return
        if path != "/eval_vln":
            streamvln_server.json_response(self, 404, {"ok": False, "error": "not found"})
            return

        started = time.time()
        request_id: str | None = None
        session: TaskSessionManager | None = None
        saved_artifacts: dict[str, str] = {}
        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            streamvln_server.json_response(self, 400, {"ok": False, "error": "expected multipart/form-data"})
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("content-length", "0"),
                },
            )
            metadata = streamvln_server.parse_json_field(form)
            image_field, image_bytes = streamvln_server.read_file_field(form, "image")
            depth_field, depth_bytes = streamvln_server.read_file_field(form, "depth")
            if image_field is None or not image_bytes:
                streamvln_server.json_response(self, 400, {"ok": False, "error": "missing image file field"})
                return

            instruction = (
                metadata.get("prompt")
                or metadata.get("instruction")
                or metadata.get("task")
                or self.server.args.default_instruction
            )
            if not instruction:
                streamvln_server.json_response(self, 400, {"ok": False, "error": "missing prompt/instruction in json"})
                return

            rgb = streamvln_server.decode_rgb(image_bytes)
            depth = streamvln_server.decode_depth(depth_bytes, metadata, rgb.shape)
            intrinsic = streamvln_server.parse_intrinsic(metadata)
            reset = bool(metadata.get("reset", False))

            with self.server.infer_lock:
                if reset or self.server.runtime.session is None or instruction != self.server.runtime.instruction:
                    self.server.runtime.reset(instruction, metadata)

                session = self.server.runtime.session
                assert session is not None
                request_id = session.next_request_id()
                request_payload = {
                    "session_id": session.session_id,
                    "request_id": request_id,
                    "instruction": instruction,
                    "reset": reset,
                    "metadata": metadata,
                    "received": {
                        "image": {
                            "filename": getattr(image_field, "filename", None),
                            "bytes": len(image_bytes),
                            "shape": list(rgb.shape),
                        },
                        "depth": {
                            "present": bool(depth_bytes),
                            "filename": getattr(depth_field, "filename", None) if depth_field is not None else None,
                            "bytes": len(depth_bytes),
                            "shape": list(depth.shape) if depth is not None else None,
                        },
                    },
                }
                session.append_history_event(
                    "robot_request_received",
                    {
                        "instruction": instruction,
                        "reset": reset,
                        "has_rgb": bool(image_bytes),
                        "has_depth": bool(depth_bytes),
                    },
                    request_id=request_id,
                )
                saved_artifacts = save_session_request_inputs(
                    session,
                    request_id,
                    image_bytes,
                    depth_bytes,
                    metadata,
                    request_payload,
                )
                session.append_history_event(
                    "observation_saved",
                    {
                        "saved": saved_artifacts,
                        "rgb_shape": list(rgb.shape),
                        "depth_shape": list(depth.shape) if depth is not None else None,
                    },
                    request_id=request_id,
                )
                active_before = None
                plan_before = session.require_plan()
                if plan_before.current_step is not None:
                    active_before = asdict(plan_before.current_step)
                current_plan_payload = plan_to_payload(session.require_plan())
                self.server.runtime.remember_saved_request(
                    request_id=request_id,
                    saved_artifacts=saved_artifacts,
                    plan_payload=current_plan_payload,
                )
                if self.server.runtime.is_vqa_active():
                    session.append_history_event(
                        "hitl_vqa_hold_response",
                        {
                            "reason": "hitl_vqa_active",
                            "active_step": active_before,
                        },
                        request_id=request_id,
                    )
                    result = self.server.runtime.hold_payload("hitl_vqa_active")
                else:
                    session.append_history_event(
                        "inference_started",
                        {
                            "active_step": active_before,
                        },
                        request_id=request_id,
                    )
                    result = self.server.runtime.step(rgb, depth, intrinsic, metadata)

            assert session is not None
            assert request_id is not None
            full_actions = result["action_sequence"]
            payload = {
                "ok": True,
                "session_id": session.session_id,
                "session_dir": str(session.session_dir),
                "request_id": request_id,
                "action": result["action"],
                "action_names": streamvln_server.action_names(result["action"]),
                "action_e": result["action_e"],
                "action_e_name": streamvln_server.action_names([result["action_e"]])[0],
                "guard_g": result["guard_g"],
                "guard_g_name": streamvln_server.action_names([result["guard_g"]])[0],
                "revisable_tail": result["revisable_tail"],
                "revisable_tail_names": streamvln_server.action_names(result["revisable_tail"]),
                "action_sequence": full_actions,
                "action_sequence_names": streamvln_server.action_names(full_actions),
                "instruction": instruction,
                "active_instruction": result["active_instruction"],
                "active_completion_condition": result["active_completion_condition"],
                "active_subgoal": result["plan"].get("active_subgoal"),
                "subgoal_completed": result["subgoal_completed"],
                "plan": result["plan"],
                "reset": reset,
                "terminate": result["terminate"],
                "hold": result.get("hold", False),
                "hold_reason": result.get("hold_reason", ""),
                "generate_time": result["generate_time"],
                "elapsed_ms": round((time.time() - started) * 1000.0, 2),
                "received": {
                    "image": {
                        "filename": getattr(image_field, "filename", None),
                        "bytes": len(image_bytes),
                        "shape": list(rgb.shape),
                    },
                    "depth": {
                        "present": bool(depth_bytes),
                        "filename": getattr(depth_field, "filename", None) if depth_field is not None else None,
                        "bytes": len(depth_bytes),
                        "shape": list(depth.shape) if depth is not None else None,
                    },
                },
            }
            if self.server.args.include_llm_output:
                payload["llm_output"] = result["llm_output"]
            saved_artifacts["response"] = str(session.request_dir(request_id) / "response.json")
            payload["saved"] = saved_artifacts
            write_session_response(session, request_id, payload)
            session.append_history_event(
                "hitl_vqa_hold_returned" if result.get("hold") else "inference_completed",
                {
                    "generate_time": result["generate_time"],
                    "elapsed_ms": payload["elapsed_ms"],
                    "terminate": result["terminate"],
                    "subgoal_completed": result["subgoal_completed"],
                    "hold": result.get("hold", False),
                    "hold_reason": result.get("hold_reason", ""),
                },
                request_id=request_id,
            )
            session.append_history_event(
                "action_proposed",
                {
                    "action": result["action"],
                    "action_e": result["action_e"],
                    "guard_g": result["guard_g"],
                    "revisable_tail": result["revisable_tail"],
                    "action_sequence": full_actions,
                    "active_instruction": result["active_instruction"],
                    "active_completion_condition": result["active_completion_condition"],
                },
                request_id=request_id,
            )
            session.write_latest_state()

            print("=" * 72, flush=True)
            print("Received Subgoal LiveVLN POST /eval_vln", flush=True)
            print("instruction:", instruction, flush=True)
            print("active_instruction:", result["active_instruction"], flush=True)
            print("active_completion_condition:", result["active_completion_condition"], flush=True)
            print("rgb_shape:", rgb.shape, "depth_shape:", None if depth is None else depth.shape, flush=True)
            print("session_id:", session.session_id, "request_id:", request_id, flush=True)
            print("action_e:", payload["action_e"], "guard_g:", payload["guard_g"], flush=True)
            print("subgoal_completed:", result["subgoal_completed"], "terminate:", result["terminate"], flush=True)
            print("sequence:", full_actions, flush=True)
            print("elapsed_ms:", payload["elapsed_ms"], flush=True)
            streamvln_server.json_response(self, 200, payload)
        except Exception as exc:
            print("ERROR handling Subgoal LiveVLN /eval_vln:", repr(exc), flush=True)
            if session is None:
                session = self.server.runtime.session
            error_payload = {
                "ok": False,
                "error": str(exc),
                "action": [0],
                "action_e": 0,
                "guard_g": 0,
                "revisable_tail": [],
                "action_sequence": [0],
                "terminate": True,
            }
            if session is not None:
                if request_id is None:
                    request_id = session.next_request_id()
                error_payload["session_id"] = session.session_id
                error_payload["session_dir"] = str(session.session_dir)
                error_payload["request_id"] = request_id
                try:
                    saved_artifacts["response"] = str(session.request_dir(request_id) / "response.json")
                    error_payload["saved"] = saved_artifacts
                    write_session_response(session, request_id, error_payload)
                    session.append_history_event(
                        "request_failed",
                        {
                            "error": str(exc),
                            "saved": saved_artifacts,
                        },
                        request_id=request_id,
                    )
                    session.write_latest_state()
                except Exception as log_exc:
                    print("ERROR writing failed request history:", repr(log_exc), flush=True)
            streamvln_server.json_response(self, 500, error_payload)

    def _handle_vqa_post(self, path: str) -> None:
        streamvln_server = self.server.streamvln_server
        try:
            payload = read_json_body(self)
            if path == "/hitl/vqa/start":
                result = self.server.runtime.start_vqa_checkpoint(
                    reason=str(payload.get("reason") or "operator requested visual QA")
                )
                streamvln_server.json_response(self, 200, {"ok": True, **result})
                return
            if path == "/hitl/vqa/ask":
                result = self.server.runtime.ask_vqa(
                    checkpoint_id=str(payload.get("checkpoint_id") or ""),
                    question=str(payload.get("question") or ""),
                )
                streamvln_server.json_response(self, 200, result)
                return
            if path == "/hitl/vqa/decide":
                result = self.server.runtime.decide_vqa(
                    checkpoint_id=str(payload.get("checkpoint_id") or ""),
                    decision=str(payload.get("decision") or ""),
                    reason=str(payload.get("reason") or ""),
                )
                streamvln_server.json_response(self, 200, result)
                return
            streamvln_server.json_response(self, 404, {"ok": False, "error": "not found"})
        except Exception as exc:
            print("ERROR handling HITL VQA endpoint:", repr(exc), flush=True)
            streamvln_server.json_response(self, 500, {"ok": False, "error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Subgoal-aware Limo RGB-D LiveVLN server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6010)
    parser.add_argument("--streamvln-deploy-root", type=Path, default=DEFAULT_STREAMVLN_DEPLOY_ROOT)
    parser.add_argument("--model-path", dest="model_path", default=str(DEFAULT_STREAMVLN_MODEL_PATH))
    parser.add_argument("--save-dir", default=str(DEFAULT_SAVE_DIR))
    parser.add_argument("--default-instruction", default="Walk forward and stop when the task is complete.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--rgb-height", type=float, default=1.25)
    parser.add_argument("--steps-per-request", type=int, default=4)
    parser.add_argument("--run-every", type=int, default=4)
    parser.add_argument("--stop-confirmations", type=int, default=1)
    parser.add_argument("--include-llm-output", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)

    parser.add_argument("--planner", choices=("qwen", "rule"), default="qwen")
    parser.add_argument("--strict-planner", action="store_true")
    parser.add_argument("--intent-json-path", default="")
    parser.add_argument("--qwen-parser-mode", choices=("subprocess", "inprocess"), default="subprocess")
    parser.add_argument("--qwen-python", default="/root/miniconda3/bin/python")
    parser.add_argument("--qwen-timeout", type=int, default=600)
    parser.add_argument("--qwen-cuda-visible-devices", default="")
    parser.add_argument("--qwen-model-path", default=str(DEFAULT_QWEN_MODEL_PATH))
    parser.add_argument("--qwen-device-map", default="auto")
    parser.add_argument(
        "--qwen-torch-dtype",
        default="bfloat16",
        choices=("auto", "bfloat16", "bf16", "float16", "fp16", "float32", "fp32"),
    )
    parser.add_argument("--qwen-max-new-tokens", type=int, default=768)
    parser.add_argument("--qwen-enable-thinking", action="store_true")
    parser.add_argument("--allow-hf-network", action="store_true")

    parser.add_argument("--vqa-backend", choices=("mock", "streamvln"), default="mock")
    parser.add_argument("--vqa-model-path", default=str(DEFAULT_STREAMVLN_MODEL_PATH))
    parser.add_argument("--vqa-device", default="cuda:0")
    parser.add_argument("--vqa-attn-implementation", default="eager")
    parser.add_argument("--vqa-max-new-tokens", type=int, default=256)
    parser.add_argument("--vqa-warmup", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_huggingface_cache(offline=not args.allow_hf_network)
    streamvln_server = load_streamvln_server_module(args.streamvln_deploy_root)
    server = ThreadingHTTPServer((args.host, args.port), SubgoalLiveVLNHandler)
    server.args = args
    server.streamvln_server = streamvln_server
    server.evaluator = None
    server.runtime = SubgoalRuntime(args, streamvln_server, evaluator=None)
    server.infer_lock = threading.Lock()

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print("Listening on http://%s:%d" % (args.host, args.port), flush=True)
    print("POST endpoint: /eval_vln", flush=True)
    print("Service: subgoal_livevln_server", flush=True)
    print("Intent parser:", args.planner, f"({args.qwen_parser_mode})" if args.planner == "qwen" else "", flush=True)
    print("VQA backend:", args.vqa_backend, flush=True)
    print("StreamVLN model loading: lazy, after first instruction is parsed", flush=True)
    print("HF_HOME:", DEFAULT_HF_HOME, flush=True)
    print("HF_HUB_CACHE:", DEFAULT_HF_HUB_CACHE, flush=True)
    print("HF offline mode:", not args.allow_hf_network, flush=True)
    print("Saving session history to", args.save_dir, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
