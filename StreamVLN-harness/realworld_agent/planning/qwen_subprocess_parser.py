"""Qwen intent parser executed in a separate Python environment.

This keeps the StreamVLN server process free from Qwen/transformers version
conflicts. The model is loaded only when a real robot instruction arrives, and
only inside the subprocess used for that parse.
"""

from __future__ import annotations

import json
import os
import subprocess
import uuid
from pathlib import Path
from typing import Any

from realworld_agent.planning.intent_parser import IntentPlan, intent_from_payload


HARNESS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QWEN_EXPORT_SCRIPT = HARNESS_ROOT / "clients" / "operator_client" / "qwen_intent_export.py"
DEFAULT_RUNTIME_INTENT_DIR = HARNESS_ROOT / "outputs" / "runtime_intents"


class QwenSubprocessIntentParser:
    """Run the Qwen3 parser through another Python executable on demand."""

    def __init__(
        self,
        python_bin: str = "/root/miniconda3/bin/python",
        model_path: Path | str = HARNESS_ROOT / "models" / "Qwen3-8B",
        device_map: str = "auto",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 768,
        timeout: int = 600,
        output_dir: Path | str = DEFAULT_RUNTIME_INTENT_DIR,
        export_script: Path | str = DEFAULT_QWEN_EXPORT_SCRIPT,
        cuda_visible_devices: str = "",
    ):
        self.python_bin = str(python_bin)
        self.model_path = Path(model_path)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = int(max_new_tokens)
        self.timeout = int(timeout)
        self.output_dir = Path(output_dir)
        self.export_script = Path(export_script)
        self.cuda_visible_devices = cuda_visible_devices

    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        if not self.export_script.exists():
            raise FileNotFoundError(f"missing Qwen export script: {self.export_script}")
        if not self.model_path.exists():
            raise FileNotFoundError(f"missing Qwen model path: {self.model_path}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / f"intent_{uuid.uuid4().hex}.json"
        cmd = [
            self.python_bin,
            str(self.export_script),
            "--qwen-model-path",
            str(self.model_path),
            "--qwen-device-map",
            self.device_map,
            "--qwen-torch-dtype",
            self.torch_dtype,
            "--max-new-tokens",
            str(self.max_new_tokens),
            "--output",
            str(output_path),
            instruction,
        ]
        env = os.environ.copy()
        env.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
        env.setdefault("HF_HOME", "/root/autodl-tmp/system_cache/huggingface")
        env.setdefault("HF_HUB_CACHE", "/root/autodl-tmp/system_cache/huggingface/hub")
        if self.cuda_visible_devices:
            env["CUDA_VISIBLE_DEVICES"] = self.cuda_visible_devices

        result = subprocess.run(
            cmd,
            cwd=str(HARNESS_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Qwen subprocess parser failed "
                f"with exit code {result.returncode}:\n"
                f"STDOUT:\n{result.stdout}\n"
                f"STDERR:\n{result.stderr}"
            )
        if not output_path.exists():
            raise FileNotFoundError(f"Qwen subprocess did not write intent JSON: {output_path}")

        payload = json.loads(output_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"Qwen subprocess output must be a JSON object: {output_path}")
        intent_instruction = str(payload.get("instruction") or instruction)
        intent = intent_from_payload(intent_instruction, payload)
        intent.parser_name = f"Qwen subprocess parser ({self.model_path.name})"
        intent.raw_output = str(payload.get("raw_output") or result.stdout or "")
        return intent
