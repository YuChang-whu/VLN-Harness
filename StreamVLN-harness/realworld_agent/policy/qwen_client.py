"""Qwen text model client for VLN semantic intent parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_QWEN_MODEL_PATH = Path(
    "/root/autodl-tmp/yc_code/StreamVLN-harness/StreamVLN-harness/models/Qwen3-8B"
)


@dataclass
class QwenGenerationConfig:
    model_path: Path = DEFAULT_QWEN_MODEL_PATH
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 768
    trust_remote_code: bool = True
    enable_thinking: bool = False


class QwenIntentClient:
    """Lazy wrapper around a local Qwen model used as a JSON intent parser."""

    def __init__(self, config: QwenGenerationConfig | None = None):
        self.config = config or QwenGenerationConfig()
        self._loaded = False
        self._tokenizer = None
        self._model = None

    @property
    def model_path(self) -> Path:
        return Path(self.config.model_path)

    @property
    def model_label(self) -> str:
        return f"Qwen intent parser ({self.model_path.name})"

    def generate_text(self, instruction: str, observation: Any | None = None) -> str:
        self._ensure_loaded()
        prompt = self._build_messages(instruction, observation)
        return self._generate(prompt)

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"Qwen model path does not exist: {self.model_path}")

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = self._resolve_dtype(torch)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=self.config.trust_remote_code,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=dtype,
            device_map=self.config.device_map,
            trust_remote_code=self.config.trust_remote_code,
        ).eval()
        self._loaded = True

    def _resolve_dtype(self, torch_module):
        value = self.config.torch_dtype.lower()
        if value in {"auto", "none"}:
            return "auto"
        if value in {"bf16", "bfloat16"}:
            return torch_module.bfloat16
        if value in {"fp16", "float16", "half"}:
            return torch_module.float16
        if value in {"fp32", "float32"}:
            return torch_module.float32
        raise ValueError(f"Unsupported torch dtype: {self.config.torch_dtype}")

    def _build_messages(self, instruction: str, observation: Any | None = None) -> list[dict[str, str]]:
        observation_note = "No visual observation is provided; parse the instruction text only."
        if observation is not None:
            observation_note = "A runtime observation exists, but this parser currently uses instruction text only."

        return [
            {
                "role": "system",
                "content": (
                    "You are a real-world VLN intent parser. Return JSON only. "
                    "Do not include markdown, comments, or extra explanation."
                ),
            },
            {
                "role": "user",
                "content": f"""
Parse this robot navigation instruction into structured JSON.

Instruction:
{instruction}

Observation note:
{observation_note}

Return this exact schema:
{{
  "goal": "final target place or object",
  "stop_condition": "when the robot should stop",
  "landmarks": ["semantic landmarks mentioned in order"],
  "constraints": ["safety, timing, or route constraints"],
  "route_hints": ["ordered route hints"],
  "subgoals": [
    {{
      "type": "move | turn | enter_space | pass_landmark | approach | stop | other",
      "description": "semantic subgoal, not a raw comma fragment",
      "completion_signal": "observable condition that means this subgoal is complete",
      "termination_condition": "exact condition where this subgoal should be considered done and StreamVLN should output STOP",
      "streamvln_instruction": "single executable VLN instruction containing the subgoal action and the termination condition"
    }}
  ]
}}

Rules:
- Use semantic understanding, not comma splitting.
- Treat malformed punctuation as noise.
- Preserve timing constraints such as "when very close to the plants".
- Mention landmarks such as plants, door, and corridor when present.
- Split the instruction into fine-grained semantic subgoals that can each be executed by a robot policy.
- Every subgoal must include a concrete termination_condition.
- Every streamvln_instruction must say both what to do and when to stop, for example: "Move toward the plants. Stop when you are very close to the plants."
- If a condition controls an action, keep it with that subgoal, for example turning left should terminate after the left turn is complete and the robot is aligned toward the door/corridor.
- If the instruction has an explicit stop condition, make the final subgoal type "stop".
- Return valid JSON only.
""".strip(),
            },
        ]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None
        assert model is not None

        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=self.config.enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )
        generated = outputs[0][inputs.input_ids.shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True).strip()
