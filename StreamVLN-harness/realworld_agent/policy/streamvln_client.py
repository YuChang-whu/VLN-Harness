"""Lazy StreamVLN model client used by the intent planner.

This client intentionally follows the original real-world deployment path in
``/root/autodl-tmp/yc_code/StreamVLN/limo/limo_realworld_server_liveVLN.py``:
model loading goes through ``limo_realworld_server.load_streamvln()``, and
inference goes through a StreamVLN-style multimodal ``generate`` call with
RGB-D, pose, intrinsic, time ids, and task ids.

The intent planner is text-first, so this first version supplies a neutral
black RGB-D frame. That keeps the call compatible with StreamVLN's multimodal
forward path and avoids treating ``StreamVLNForCausalLM`` as a plain text-only
Qwen model.
"""

from __future__ import annotations

import importlib.util
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path


DEFAULT_STREAMVLN_ROOT = Path("/root/autodl-tmp/yc_code/StreamVLN")
DEFAULT_MODEL_PATH = Path(
    "/root/autodl-tmp/yc_code/StreamVLN/data/pretrained_models/"
    "StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"
)


@dataclass
class StreamVLNGenerationConfig:
    model_path: Path = DEFAULT_MODEL_PATH
    streamvln_root: Path = DEFAULT_STREAMVLN_ROOT
    device: str = "cuda:0"
    attn_implementation: str = "eager"
    model_max_length: int = 4096
    max_new_tokens: int = 768
    num_future_steps: int = 4
    num_frames: int = 32
    num_history: int = 8
    rgb_height: float = 1.25
    warmup: bool = False
    verify_realworld_step: bool = True


@dataclass
class StreamVLNActionResult:
    actions: list[int]
    raw_output: str
    generate_time: float


class StreamVLNModelClient:
    """Wrapper around the original StreamVLN real-world evaluator."""

    def __init__(self, config: StreamVLNGenerationConfig | None = None):
        self.config = config or StreamVLNGenerationConfig()
        self._loaded = False
        self._server_module = None
        self._evaluator = None
        self._verified_realworld_step = False

    @property
    def model_path(self) -> Path:
        return Path(self.config.model_path)

    @property
    def streamvln_root(self) -> Path:
        return Path(self.config.streamvln_root)

    @property
    def model_label(self) -> str:
        return f"StreamVLN real-world checkpoint ({self.model_path.name})"

    def generate_text(self, prompt: str) -> str:
        """Generate text through StreamVLN's multimodal real-world path."""
        self._ensure_loaded()
        return self._generate_with_dummy_observation(prompt)

    def generate_action_plan(self, instruction: str) -> StreamVLNActionResult:
        """Run the original real-world evaluator.step path for an instruction."""
        self._ensure_loaded()
        evaluator = self._evaluator
        module = self._server_module
        assert evaluator is not None
        assert module is not None

        import numpy as np

        evaluator.reset_memory()
        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640, 1), dtype=np.float32)
        actions, generate_time, raw_output = evaluator.step(
            0,
            rgb,
            instruction,
            run_model=True,
            depth=depth,
            intrinsic=module.DEFAULT_INTRINSIC.copy(),
        )
        evaluator.reset_memory()
        return StreamVLNActionResult(
            actions=[int(action) for action in (actions or [])],
            raw_output=raw_output or "",
            generate_time=float(generate_time or 0.0),
        )

    def predict_actions(
        self,
        instruction: str,
        rgb=None,
        depth=None,
        camera_pose=None,
        intrinsic=None,
        reset: bool = False,
        run_model: bool = True,
    ) -> StreamVLNActionResult:
        """Predict actions for the current subgoal using StreamVLN's real-world step path.

        Unlike ``generate_action_plan``, this method preserves evaluator memory
        across calls so a subgoal can be advanced frame by frame, matching the
        original real-world server style where ``evaluator.step_id`` is updated
        outside ``VLNEvaluator.step``.
        """
        self._ensure_loaded()
        evaluator = self._evaluator
        module = self._server_module
        assert evaluator is not None
        assert module is not None

        import numpy as np

        if reset:
            evaluator.reset_memory()
        if rgb is None:
            rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        if depth is None:
            depth = np.zeros((rgb.shape[0], rgb.shape[1], 1), dtype=np.float32)
        if camera_pose is None:
            camera_pose = np.eye(4, dtype=np.float32)
        if intrinsic is None:
            intrinsic = module.DEFAULT_INTRINSIC.copy()

        actions, generate_time, raw_output = evaluator.step(
            0,
            rgb,
            instruction,
            run_model=run_model,
            depth=depth,
            camera_pose=camera_pose,
            intrinsic=intrinsic,
        )
        evaluator.step_id += 1
        return StreamVLNActionResult(
            actions=[int(action) for action in (actions or [])],
            raw_output=raw_output or "",
            generate_time=float(generate_time or 0.0),
        )

    def reset_policy_memory(self) -> None:
        """Reset StreamVLN recurrent memory when switching to a new subgoal."""
        self._ensure_loaded()
        evaluator = self._evaluator
        assert evaluator is not None
        evaluator.reset_memory()

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        model_path = self.model_path
        if not model_path.exists():
            raise FileNotFoundError(f"StreamVLN model path does not exist: {model_path}")

        streamvln_root = self.streamvln_root
        server_path = streamvln_root / "limo" / "limo_realworld_server.py"
        if not server_path.exists():
            raise FileNotFoundError(
                "Expected original StreamVLN real-world server at "
                f"{server_path}. Pass --streamvln-root /root/autodl-tmp/yc_code/StreamVLN"
            )

        for path in (streamvln_root, streamvln_root / "streamvln", streamvln_root / "limo"):
            value = str(path)
            if value not in sys.path:
                sys.path.insert(0, value)

        spec = importlib.util.spec_from_file_location("original_limo_realworld_server", server_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import StreamVLN server module from {server_path}")
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
            default_instruction="move forward 25 cm",
            max_new_tokens=self.config.max_new_tokens,
        )
        evaluator = module.load_streamvln(args)

        self._server_module = module
        self._evaluator = evaluator
        self._loaded = True
        if self.config.verify_realworld_step:
            self._verify_realworld_step()

    def _verify_realworld_step(self) -> None:
        if self._verified_realworld_step:
            return
        evaluator = self._evaluator
        assert evaluator is not None

        import numpy as np

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640, 1), dtype=np.float32)
        _, _, llm_output = evaluator.step(
            0,
            rgb,
            "move forward 25 cm",
            run_model=True,
            depth=depth,
        )
        if not llm_output:
            raise RuntimeError("StreamVLN real-world evaluator.step did not return model output")
        evaluator.reset_memory()
        self._verified_realworld_step = True

    def _generate_with_dummy_observation(self, prompt: str) -> str:
        evaluator = self._evaluator
        module = self._server_module
        assert evaluator is not None
        assert module is not None

        import numpy as np
        import torch
        from realworld_agent.action.semantics import parse_action_ids
        from streamvln.utils.utils import dict_to_cuda

        evaluator.reset_memory()

        rgb = np.zeros((480, 640, 3), dtype=np.uint8)
        depth = np.zeros((480, 640, 1), dtype=np.float32)
        camera_pose = np.eye(4, dtype=np.float32)
        intrinsic = module.DEFAULT_INTRINSIC.copy()

        image = evaluator.image_processor.preprocess(
            images=module.Image.fromarray(rgb).convert("RGB"),
            return_tensors="pt",
        )["pixel_values"][0]
        input_ids, _ = evaluator.preprocess_qwen(
            [[{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]],
            evaluator.tokenizer,
            True,
            add_system=True,
        )

        images = [image]
        depths = [torch.from_numpy(depth).float()]
        poses = [torch.from_numpy(camera_pose).float()]
        intrinsics = [torch.from_numpy(intrinsic).float()]
        time_ids = [[0]]
        evaluator.model.model.num_history = 0

        input_dict_raw = {
            "images": torch.stack(images).unsqueeze(0),
            "depths": torch.stack(depths).unsqueeze(0),
            "poses": torch.stack(poses).unsqueeze(0),
            "intrinsics": torch.stack(intrinsics).unsqueeze(0),
            "inputs": input_ids,
            "env_id": 0,
            "time_ids": time_ids,
        }
        input_dict = dict_to_cuda(input_dict_raw.copy(), evaluator.device)
        for key in ("images", "depths", "poses", "intrinsics"):
            input_dict[key] = input_dict[key].to(torch.bfloat16)

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

        # StreamVLN may still emit navigation actions because it was fine-tuned
        # for action generation. Surface that as raw model output; the intent
        # parser will decide whether it contains valid JSON or must fallback.
        action_ids = parse_action_ids(decoded)
        if action_ids and "{" not in decoded:
            return decoded
        return decoded
