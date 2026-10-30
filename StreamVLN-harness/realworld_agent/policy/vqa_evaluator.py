"""Independent visual question answering evaluator for HITL checkpoints.

This module deliberately does not reuse the navigation StreamVLN evaluator.
The VQA evaluator owns its own model state so human questions cannot mutate the
navigation evaluator's KV cache, memory tokens, step id, or action sequence.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from realworld_agent.hitl.vqa_types import VQAAnswer


DEFAULT_STREAMVLN_DEPLOY_ROOT = Path("/root/autodl-tmp/yc_code/StreamVLN-harness")
DEFAULT_STREAMVLN_MODEL_PATH = Path(
    "/root/autodl-tmp/yc_code/StreamVLN/data/pretrained_models/"
    "StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"
)


@dataclass
class VQAEvaluatorConfig:
    backend: str = "mock"
    model_path: Path | str = DEFAULT_STREAMVLN_MODEL_PATH
    streamvln_deploy_root: Path | str = DEFAULT_STREAMVLN_DEPLOY_ROOT
    device: str = "cuda:0"
    attn_implementation: str = "eager"
    model_max_length: int = 4096
    max_new_tokens: int = 256
    num_future_steps: int = 4
    num_frames: int = 32
    num_history: int = 0
    rgb_height: float = 1.25
    warmup: bool = False


class VQAEvaluator:
    """Ask visual-language questions through a separate evaluator instance."""

    def __init__(self, config: VQAEvaluatorConfig | None = None):
        self.config = config or VQAEvaluatorConfig()
        self._loaded = False
        self._server_module = None
        self._evaluator = None

    @property
    def backend(self) -> str:
        return self.config.backend

    def ask(
        self,
        rgb_path: Path | str,
        question: str,
        instruction: str,
        active_subgoal: dict[str, Any] | None,
        completion_condition: str,
        recent_actions: list[int],
        recent_events: list[dict[str, Any]],
        depth_path: Path | str | None = None,
    ) -> VQAAnswer:
        question = (question or "").strip()
        if not question:
            raise ValueError("question must not be empty")

        if self.backend == "mock":
            return self._mock_answer(
                rgb_path=Path(rgb_path),
                question=question,
                instruction=instruction,
                active_subgoal=active_subgoal,
                completion_condition=completion_condition,
                recent_actions=recent_actions,
            )
        if self.backend == "streamvln":
            return self._streamvln_answer(
                rgb_path=Path(rgb_path),
                depth_path=Path(depth_path) if depth_path else None,
                question=question,
                instruction=instruction,
                active_subgoal=active_subgoal,
                completion_condition=completion_condition,
                recent_actions=recent_actions,
                recent_events=recent_events,
            )
        raise ValueError(f"unsupported VQA backend: {self.backend}")

    def _mock_answer(
        self,
        rgb_path: Path,
        question: str,
        instruction: str,
        active_subgoal: dict[str, Any] | None,
        completion_condition: str,
        recent_actions: list[int],
    ) -> VQAAnswer:
        subgoal_text = ""
        if active_subgoal:
            subgoal_text = str(active_subgoal.get("description") or active_subgoal.get("instruction") or "")
        answer = (
            "[mock VQA] 已进入人工视觉问答检查点。"
            f"当前图像路径: {rgb_path}; "
            f"问题: {question}; "
            f"当前任务: {instruction}; "
            f"当前子目标: {subgoal_text or 'unknown'}; "
            f"完成条件: {completion_condition or 'unknown'}; "
            f"最近动作: {recent_actions}. "
            "mock 后端不会读取图像内容，只用于验证 HITL 流程和 session 记录。"
        )
        return VQAAnswer(answer=answer, backend="mock", raw_output=answer)

    def _streamvln_answer(
        self,
        rgb_path: Path,
        depth_path: Path | None,
        question: str,
        instruction: str,
        active_subgoal: dict[str, Any] | None,
        completion_condition: str,
        recent_actions: list[int],
        recent_events: list[dict[str, Any]],
    ) -> VQAAnswer:
        self._ensure_loaded()
        prompt = self._build_prompt(
            question=question,
            instruction=instruction,
            active_subgoal=active_subgoal,
            completion_condition=completion_condition,
            recent_actions=recent_actions,
            recent_events=recent_events,
        )
        started = time.time()
        raw = self._generate_with_image(rgb_path, depth_path, prompt)
        elapsed = time.time() - started
        answer = self._strip_generated_prompt(raw).strip() or raw.strip()
        return VQAAnswer(
            answer=answer,
            backend="streamvln",
            raw_output=f"generate_time={elapsed:.3f}s\n{raw}",
        )

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        model_path = Path(self.config.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"VQA StreamVLN model path does not exist: {model_path}")

        root = Path(self.config.streamvln_deploy_root)
        server_path = root / "limo" / "limo_realworld_server.py"
        if not server_path.exists():
            raise FileNotFoundError(f"missing StreamVLN real-world server: {server_path}")

        for path in (root / "limo", root, root / "streamvln"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        spec = importlib.util.spec_from_file_location("vqa_limo_realworld_server", server_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot import StreamVLN server module from {server_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        args = Namespace(
            model_path=str(model_path),
            device=self.config.device,
            attn_implementation=self.config.attn_implementation,
            num_future_steps=self.config.num_future_steps,
            num_frames=self.config.num_frames,
            num_history=self.config.num_history,
            model_max_length=self.config.model_max_length,
            rgb_height=self.config.rgb_height,
            warmup=self.config.warmup,
            default_instruction="answer a visual question",
            max_new_tokens=self.config.max_new_tokens,
        )
        self._server_module = module
        self._evaluator = module.load_streamvln(args)
        self._loaded = True

    def _build_prompt(
        self,
        question: str,
        instruction: str,
        active_subgoal: dict[str, Any] | None,
        completion_condition: str,
        recent_actions: list[int],
        recent_events: list[dict[str, Any]],
    ) -> str:
        subgoal_text = "unknown"
        if active_subgoal:
            subgoal_text = str(
                active_subgoal.get("instruction")
                or active_subgoal.get("description")
                or active_subgoal.get("title")
                or "unknown"
            )
        event_text = "; ".join(str(item.get("message") or item) for item in recent_events[-4:])
        return f"""You are a visual question answering assistant for a real-world robot navigation task.

Original instruction:
{instruction}

Current semantic subgoal:
{subgoal_text}

Completion condition:
{completion_condition or 'unknown'}

Recent actions:
{recent_actions}

Recent runtime events:
{event_text or 'none'}

Human question:
{question}

Answer in natural language. Do not output only navigation symbols. Do not output ↑, ←, →, or STOP as the whole answer. If unsure, say unsure. Mention visible evidence from the image. If asked whether the current subgoal is complete, answer yes, no, or unsure and explain why."""

    def _generate_with_image(self, rgb_path: Path, depth_path: Path | None, prompt: str) -> str:
        evaluator = self._evaluator
        module = self._server_module
        assert evaluator is not None
        assert module is not None

        import numpy as np
        import torch
        from PIL import Image
        from streamvln.utils.utils import dict_to_cuda

        evaluator.reset_memory()
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        if depth_path and depth_path.exists():
            depth_img = Image.open(depth_path)
            depth = np.asarray(depth_img)
            if depth.ndim == 3:
                depth = depth[..., 0]
            depth = depth.astype(np.float32)
            if depth.size and float(np.nanmax(depth)) > 50.0:
                depth = depth / 1000.0
            depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)[..., None]
        else:
            depth = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
        camera_pose = np.eye(4, dtype=np.float32)
        intrinsic = getattr(module, "DEFAULT_INTRINSIC").copy()

        image = evaluator.image_processor.preprocess(
            images=Image.fromarray(rgb).convert("RGB"),
            return_tensors="pt",
        )["pixel_values"][0]
        input_ids, _ = evaluator.preprocess_qwen(
            [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]],
            evaluator.tokenizer,
            True,
            add_system=True,
        )

        input_dict_raw = {
            "images": torch.stack([image]).unsqueeze(0),
            "depths": torch.stack([torch.from_numpy(depth).float()]).unsqueeze(0),
            "poses": torch.stack([torch.from_numpy(camera_pose).float()]).unsqueeze(0),
            "intrinsics": torch.stack([torch.from_numpy(intrinsic).float()]).unsqueeze(0),
            "inputs": input_ids,
            "env_id": 0,
            "time_ids": [[0]],
        }
        evaluator.model.model.num_history = 0
        input_dict = dict_to_cuda(input_dict_raw.copy(), evaluator.device)
        for key in ("images", "depths", "poses", "intrinsics"):
            input_dict[key] = input_dict[key].to(torch.bfloat16)

        with torch.no_grad():
            outputs = evaluator.model.generate(
                **input_dict,
                task_ids=[0],
                do_sample=False,
                num_beams=1,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,
                return_dict_in_generate=True,
            )
        decoded = evaluator.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)[0].strip()
        evaluator.reset_memory()
        return decoded

    def _strip_generated_prompt(self, raw: str) -> str:
        marker = "<|im_start|>assistant"
        if marker in raw:
            return raw.rsplit(marker, 1)[-1]
        return raw
