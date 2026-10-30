"""GPT-6 Astra client for intent parsing and one-step RGB navigation."""

from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from typing import Any, List, Literal

from pydantic import BaseModel, Field


DEFAULT_ASTRA_MODEL = "gpt-6-astra"


class AstraSubgoal(BaseModel):
    type: str = "other"
    description: str
    completion_signal: str = ""
    termination_condition: str = ""
    streamvln_instruction: str = ""


class AstraIntentOutput(BaseModel):
    goal: str = ""
    stop_condition: str = ""
    landmarks: List[str] = Field(default_factory=list)
    constraints: List[str] = Field(default_factory=list)
    route_hints: List[str] = Field(default_factory=list)
    subgoals: List[AstraSubgoal] = Field(default_factory=list)


class AstraActionOutput(BaseModel):
    action: Literal[0, 1, 2, 3]
    reason: str = ""


@dataclass(frozen=True)
class AstraConfig:
    """Runtime configuration. Secrets are read from the environment."""

    model: str = DEFAULT_ASTRA_MODEL
    base_url: str = ""
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 30.0
    max_retries: int = 2
    reasoning_effort: str = "low"
    image_detail: str = "high"
    max_output_tokens: int = 1024


@dataclass(frozen=True)
class AstraIntentResult:
    payload: dict[str, Any]
    raw_output: str
    generate_time: float
    response_id: str = ""


@dataclass(frozen=True)
class AstraActionResult:
    actions: list[int]
    raw_output: str
    generate_time: float
    response_id: str = ""


def _model_to_dict(value: BaseModel) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump()
    return value.dict()


def _model_to_json(value: BaseModel) -> str:
    if hasattr(value, "model_dump_json"):
        return value.model_dump_json()
    return value.json(ensure_ascii=False)


class AstraClient:
    """Small Responses API adapter with structured intent/action outputs."""

    INTENT_SYSTEM_PROMPT = """You are the intent planner for a real-world vision-language navigation robot.
Convert the user's complete instruction into short, ordered, executable semantic subgoals.
Preserve landmarks, route order, negations, safety constraints, and the final stopping condition.
Every subgoal must include an observable termination_condition and a short streamvln_instruction
that tells a downstream navigation policy what to do and when that subgoal is complete.
Do not split mechanically on punctuation. Do not invent measurements or unseen landmarks."""

    ACTION_SYSTEM_PROMPT = """You are the one-step navigation policy for a mobile robot.
Use the current RGB image and the active subgoal to choose exactly one action:
0 = STOP, 1 = MOVE_FORWARD, 2 = TURN_LEFT, 3 = TURN_RIGHT.
STOP means the active subgoal is complete. Choose STOP when the completion condition is satisfied.
Choose only one immediate action; do not output a future action sequence.
Depth is unavailable. Base the decision only on the RGB image and supplied text context."""

    def __init__(self, config: AstraConfig | None = None, client: Any | None = None):
        self.config = config or AstraConfig()
        self._client = client

    @property
    def model_label(self) -> str:
        return f"OpenAI Responses API ({self.config.model})"

    def _ensure_client(self):
        if self._client is not None:
            return self._client

        api_key = os.getenv(self.config.api_key_env, "").strip()
        if not api_key:
            raise RuntimeError(
                f"missing Astra API key: set environment variable {self.config.api_key_env}"
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise RuntimeError(
                "the Astra backend requires the openai package; "
                "install requirements-gpt6-astra.txt"
            ) from exc

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.timeout,
            "max_retries": self.config.max_retries,
        }
        if self.config.base_url.strip():
            client_kwargs["base_url"] = self.config.base_url.strip()
        self._client = OpenAI(**client_kwargs)
        return self._client

    def parse_intent(self, instruction: str) -> AstraIntentResult:
        clean_instruction = " ".join((instruction or "").split())
        if not clean_instruction:
            raise ValueError("instruction must not be empty")

        started = time.perf_counter()
        response = self._ensure_client().responses.parse(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
            input=[
                {"role": "system", "content": self.INTENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Parse this navigation instruction into semantic subgoals:\n"
                        f"{clean_instruction}"
                    ),
                },
            ],
            text_format=AstraIntentOutput,
        )
        parsed = self._require_parsed(response, AstraIntentOutput)
        return AstraIntentResult(
            payload=_model_to_dict(parsed),
            raw_output=_model_to_json(parsed),
            generate_time=time.perf_counter() - started,
            response_id=str(getattr(response, "id", "") or ""),
        )

    def predict_action(
        self,
        *,
        image_bytes: bytes,
        full_instruction: str,
        active_instruction: str,
        completion_condition: str,
        previous_actions: list[int] | None = None,
        step_id: str = "",
    ) -> AstraActionResult:
        if not image_bytes:
            raise ValueError("Astra action prediction requires a non-empty RGB image")

        context = {
            "full_instruction": full_instruction,
            "active_subgoal": active_instruction,
            "completion_condition": completion_condition,
            "previous_actions": list(previous_actions or []),
            "step_id": step_id,
        }
        image_url = self._data_url(image_bytes)
        started = time.perf_counter()
        response = self._ensure_client().responses.parse(
            model=self.config.model,
            reasoning={"effort": self.config.reasoning_effort},
            max_output_tokens=self.config.max_output_tokens,
            input=[
                {"role": "system", "content": self.ACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": "Navigation state:\n" + json.dumps(
                                context,
                                ensure_ascii=False,
                                separators=(",", ":"),
                            ),
                        },
                        {
                            "type": "input_image",
                            "image_url": image_url,
                            "detail": self.config.image_detail,
                        },
                    ],
                },
            ],
            text_format=AstraActionOutput,
        )
        parsed = self._require_parsed(response, AstraActionOutput)
        return AstraActionResult(
            actions=[int(parsed.action)],
            raw_output=_model_to_json(parsed),
            generate_time=time.perf_counter() - started,
            response_id=str(getattr(response, "id", "") or ""),
        )

    @staticmethod
    def _require_parsed(response: Any, expected_type):
        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            refusal = getattr(response, "refusal", None)
            suffix = f": {refusal}" if refusal else ""
            raise RuntimeError(f"Astra returned no structured output{suffix}")
        if not isinstance(parsed, expected_type):
            raise TypeError(
                f"unexpected structured output type: {type(parsed).__name__}; "
                f"expected {expected_type.__name__}"
            )
        return parsed

    @staticmethod
    def _data_url(image_bytes: bytes) -> str:
        mime_type = "image/png" if image_bytes.startswith(b"\x89PNG\r\n\x1a\n") else "image/jpeg"
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
