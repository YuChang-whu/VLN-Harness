"""Model-driven intent parsing for real-world VLN plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from realworld_agent.action.semantics import ACTION_SPECS
from realworld_agent.policy.qwen_client import QwenIntentClient
from realworld_agent.policy.streamvln_client import StreamVLNActionResult, StreamVLNModelClient


@dataclass
class Subgoal:
    type: str
    description: str
    completion_signal: str = ""
    termination_condition: str = ""
    streamvln_instruction: str = ""


@dataclass
class IntentPlan:
    instruction: str
    goal: str = ""
    stop_condition: str = ""
    landmarks: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    route_hints: list[str] = field(default_factory=list)
    subgoals: list[Subgoal] = field(default_factory=list)
    parser_name: str = "unknown"
    fallback: bool = False
    fallback_reason: str = ""
    raw_output: str = ""


class IntentParser(Protocol):
    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        ...


class QwenIntentParser:
    """Use a local Qwen text model to parse navigation intent into JSON."""

    def __init__(
        self,
        model_client: QwenIntentClient | None = None,
        fallback_parser: IntentParser | None = None,
        allow_fallback: bool = True,
    ):
        self.model_client = model_client or QwenIntentClient()
        self.fallback_parser = fallback_parser or RuleFallbackIntentParser()
        self.allow_fallback = allow_fallback

    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        try:
            raw_output = self.model_client.generate_text(instruction, observation=observation)
            payload = extract_json_object(raw_output)
            intent = intent_from_payload(instruction, payload)
            intent.parser_name = self.model_client.model_label
            intent.raw_output = raw_output
            return intent
        except Exception as exc:
            if not self.allow_fallback:
                raise RuntimeError(f"Qwen intent parsing failed: {exc}") from exc
            fallback = self.fallback_parser.parse(instruction, observation)
            fallback.fallback = True
            fallback.fallback_reason = str(exc)
            fallback.raw_output = fallback.raw_output or ""
            return fallback


class StreamVLNIntentParser:
    """Ask StreamVLN to convert a navigation instruction into structured JSON."""

    def __init__(
        self,
        model_client: StreamVLNModelClient | None = None,
        fallback_parser: IntentParser | None = None,
        allow_fallback: bool = True,
    ):
        self.model_client = model_client or StreamVLNModelClient()
        self.fallback_parser = fallback_parser or RuleFallbackIntentParser()
        self.allow_fallback = allow_fallback

    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        prompt = self._build_prompt(instruction, observation)
        json_error: Exception | None = None
        try:
            raw_output = self.model_client.generate_text(prompt)
            payload = extract_json_object(raw_output)
            intent = intent_from_payload(instruction, payload)
            intent.parser_name = self.model_client.model_label
            intent.raw_output = raw_output
            return intent
        except Exception as exc:
            json_error = exc

        try:
            action_result = self.model_client.generate_action_plan(instruction)
            return intent_from_action_result(
                instruction,
                action_result,
                parser_name=f"{self.model_client.model_label} action-plan interpreter",
            )
        except Exception as action_exc:
            reason = f"json intent failed: {json_error}; action-plan failed: {action_exc}"
            if not self.allow_fallback:
                raise RuntimeError(f"StreamVLN intent parsing failed: {reason}") from action_exc
            fallback = self.fallback_parser.parse(instruction, observation)
            fallback.fallback = True
            fallback.fallback_reason = reason
            fallback.raw_output = fallback.raw_output or ""
            return fallback

    def _build_prompt(self, instruction: str, observation: Any | None = None) -> str:
        observation_note = "No current visual observation is provided in this first version."
        if observation is not None:
            observation_note = (
                "A current observation object is available to the runtime, but this "
                "text-only intent planner version does not encode images yet."
            )

        return f"""You are a real-world VLN intent planner.
Given a robot navigation instruction, extract the semantic navigation intent.
Return JSON only. Do not include markdown. Do not include explanations.

JSON schema:
{{
  "goal": "final target place or object",
  "stop_condition": "when the robot should stop",
  "landmarks": ["semantic landmarks mentioned by the instruction"],
  "constraints": ["safety or route constraints, including negative constraints"],
  "route_hints": ["ordered route hints from the instruction"],
  "subgoals": [
    {{
      "type": "move | turn | enter_space | pass_landmark | approach | stop | other",
      "description": "one semantic subgoal, not a raw text fragment",
      "completion_signal": "observable condition that means this subgoal is complete",
      "termination_condition": "the exact condition under which StreamVLN should output STOP for this subgoal",
      "streamvln_instruction": "short executable navigation command including both the subgoal and its STOP condition"
    }}
  ]
}}

Rules:
- Use semantic understanding, not comma splitting.
- Preserve negations such as "do not approach the sofa" as constraints.
- If the instruction has an explicit stop condition, make the final subgoal type "stop".
- If the stop condition is implicit, create a final stop subgoal after the final target is reached.
- Keep subgoals short and executable in the real world.
- Every subgoal must have a concrete termination_condition.
- Every streamvln_instruction must tell the robot what to do and when to stop.

Observation note: {observation_note}
Instruction: {instruction}
"""

    def _extract_json_object(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
            stripped = re.sub(r"\s*```$", "", stripped)

        try:
            value = json.loads(stripped)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
            if not match:
                raise
            value = json.loads(match.group(0))

        if not isinstance(value, dict):
            raise ValueError("intent output must be a JSON object")
        return value


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))

    if not isinstance(value, dict):
        raise ValueError("intent output must be a JSON object")
    return value


class RuleFallbackIntentParser:
    """Conservative fallback used only when StreamVLN intent parsing fails."""

    STOP_WORDS = ("stop", "wait", "停止", "停下", "等待", "终点", "到达")

    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        clean_instruction = re.sub(r"\s+", " ", instruction or "").strip()
        clean_instruction = clean_instruction or "move forward and stop when the task is complete"

        stop_condition = clean_instruction if self._has_stop(clean_instruction) else "stop when the task is complete"
        subgoals = [
            Subgoal(
                type="other",
                description=f"follow the navigation instruction: {clean_instruction}",
                completion_signal="operator or downstream verifier confirms progress",
                termination_condition="operator or downstream verifier confirms progress",
                streamvln_instruction=f"Follow the navigation instruction: {clean_instruction}. Stop when operator or downstream verifier confirms this subgoal is complete.",
            ),
            Subgoal(
                type="stop",
                description=stop_condition,
                completion_signal="final target or stop condition is confirmed",
                termination_condition="final target or stop condition is confirmed",
                streamvln_instruction=f"Stop when {stop_condition}.",
            ),
        ]
        return IntentPlan(
            instruction=clean_instruction,
            goal=self._guess_goal(clean_instruction),
            stop_condition=stop_condition,
            subgoals=subgoals,
            parser_name="rule_fallback",
            fallback=True,
            fallback_reason="StreamVLN intent parser unavailable",
        )

    def _has_stop(self, instruction: str) -> bool:
        lowered = instruction.lower()
        return any(word in lowered or word in instruction for word in self.STOP_WORDS)

    def _guess_goal(self, instruction: str) -> str:
        lowered = instruction.lower()
        for marker in ("near ", "beside ", "in front of ", "at ", "到", "在"):
            if marker in lowered:
                return lowered.split(marker, 1)[-1].strip(" .。")
        return "task completion point"


def intent_from_payload(instruction: str, payload: dict[str, Any]) -> IntentPlan:
    subgoals = []
    for item in _list(payload.get("subgoals")):
        if isinstance(item, dict):
            description = str(item.get("description") or item.get("title") or "").strip()
            subgoal_type = str(item.get("type") or "other").strip() or "other"
            completion_signal = str(item.get("completion_signal") or "").strip()
            termination_condition = str(
                item.get("termination_condition")
                or item.get("stop_condition")
                or item.get("completion_condition")
                or completion_signal
                or ""
            ).strip()
            streamvln_instruction = str(
                item.get("streamvln_instruction")
                or item.get("execution_instruction")
                or item.get("instruction")
                or ""
            ).strip()
        else:
            description = str(item).strip()
            subgoal_type = "other"
            completion_signal = ""
            termination_condition = ""
            streamvln_instruction = ""
        if description:
            if not termination_condition:
                termination_condition = _default_subgoal_termination(subgoal_type, description)
            if not completion_signal:
                completion_signal = termination_condition
            subgoals.append(
                Subgoal(
                    type=subgoal_type,
                    description=description,
                    completion_signal=completion_signal,
                    termination_condition=termination_condition,
                    streamvln_instruction=streamvln_instruction,
                )
            )

    if not subgoals:
        raise ValueError("intent output did not include any usable subgoals")

    return IntentPlan(
        instruction=instruction,
        goal=str(payload.get("goal") or "").strip(),
        stop_condition=str(payload.get("stop_condition") or "").strip(),
        landmarks=[str(item).strip() for item in _list(payload.get("landmarks")) if str(item).strip()],
        constraints=[str(item).strip() for item in _list(payload.get("constraints")) if str(item).strip()],
        route_hints=[str(item).strip() for item in _list(payload.get("route_hints")) if str(item).strip()],
        subgoals=subgoals,
    )


ACTION_TO_SUBGOAL = {
    0: ("stop", "stop at the current location"),
    1: ("move", "move forward one StreamVLN step"),
    2: ("turn", "turn left one StreamVLN step"),
    3: ("turn", "turn right one StreamVLN step"),
}


def intent_from_action_result(
    instruction: str,
    action_result: StreamVLNActionResult,
    parser_name: str,
) -> IntentPlan:
    if not action_result.actions:
        raise ValueError("StreamVLN action-plan output did not include any actions")

    subgoals: list[Subgoal] = []
    route_hints: list[str] = []
    for action_id, count in _compress_actions(action_result.actions):
        spec = ACTION_SPECS.get(action_id)
        name = spec.name if spec is not None else f"ACTION_{action_id}"
        route_hints.append(f"{name} x{count}")
        subgoal_type, base_description = ACTION_TO_SUBGOAL.get(action_id, ("other", f"execute action {action_id}"))
        if action_id == 1:
            distance = 0.25 * count
            description = f"{base_description} for {count} step(s), approximately {distance:.2f} meters"
            completion = f"robot odometry advances by approximately {distance:.2f} meters or local verifier confirms progress"
        elif action_id in (2, 3):
            degrees = 15 * count
            direction = "left" if action_id == 2 else "right"
            description = f"turn {direction} for {count} step(s), approximately {degrees} degrees"
            completion = f"robot heading changes {direction} by approximately {degrees} degrees"
        elif action_id == 0:
            description = _derive_stop_condition(instruction)
            completion = "robot issues STOP and task session terminates"
        else:
            description = f"{base_description} x{count}"
            completion = "downstream action executor confirms completion"
        subgoals.append(
            Subgoal(
                type=subgoal_type,
                description=description,
                completion_signal=completion,
                termination_condition=completion,
                streamvln_instruction=f"{description}. Stop when {completion}.",
            )
        )

    return IntentPlan(
        instruction=instruction,
        goal="task completion point described by the instruction",
        stop_condition=_derive_stop_condition(instruction),
        landmarks=[],
        constraints=[],
        route_hints=route_hints,
        subgoals=subgoals,
        parser_name=parser_name,
        fallback=False,
        fallback_reason="",
        raw_output=action_result.raw_output,
    )


def _compress_actions(actions: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    for action_id in actions:
        if groups and groups[-1][0] == action_id:
            groups[-1] = (action_id, groups[-1][1] + 1)
        else:
            groups.append((action_id, 1))
    return groups


def _derive_stop_condition(instruction: str) -> str:
    match = re.search(r"\bstop\b.*", instruction, flags=re.IGNORECASE)
    if match:
        return match.group(0).strip(" .")
    return "stop when the StreamVLN action plan reaches task completion"


def _default_subgoal_termination(subgoal_type: str, description: str) -> str:
    lowered_type = (subgoal_type or "").lower()
    lowered_description = description.lower()
    if lowered_type == "stop":
        return description
    if lowered_type == "turn" or "turn left" in lowered_description or "turn right" in lowered_description:
        return "the requested turn is complete and the robot is aligned with the next route segment"
    if lowered_type == "enter_space" or "enter" in lowered_description:
        return "the robot has entered the target space"
    if lowered_type in {"approach", "pass_landmark"} or "near" in lowered_description or "close" in lowered_description:
        return "the referenced landmark or target condition is visually reached"
    return "the subgoal has been visually achieved"


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str) and value.strip():
        return [value]
    return []
