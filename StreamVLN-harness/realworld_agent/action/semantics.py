"""Shared action semantics compatible with StreamVLN real-world outputs."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class ActionSpec:
    action_id: int
    name: str
    short: str
    description: str


# Keep the same action ids used by StreamVLN:
# 0 STOP, 1 MOVE_FORWARD, 2 TURN_LEFT, 3 TURN_RIGHT.
ACTION_SPECS: dict[int, ActionSpec] = {
    0: ActionSpec(0, "STOP", "STOP", "stop robot and finish or pause"),
    1: ActionSpec(1, "MOVE_FORWARD", "FORWARD", "move forward by one discrete step"),
    2: ActionSpec(2, "TURN_LEFT", "LEFT", "turn left by one discrete step"),
    3: ActionSpec(3, "TURN_RIGHT", "RIGHT", "turn right by one discrete step"),
}

ACTION_NAME_TO_ID = {
    "stop": 0,
    "forward": 1,
    "move_forward": 1,
    "move forward": 1,
    "straight": 1,
    "left": 2,
    "turn_left": 2,
    "turn left": 2,
    "right": 3,
    "turn_right": 3,
    "turn right": 3,
    "停止": 0,
    "停": 0,
    "前进": 1,
    "向前": 1,
    "左转": 2,
    "向左": 2,
    "右转": 3,
    "向右": 3,
}

ARROW_TO_ID = {
    "↑": 1,
    "←": 2,
    "→": 3,
}


def action_name(action_id: int) -> str:
    return ACTION_SPECS.get(int(action_id), ActionSpec(-1, "UNKNOWN", "UNKNOWN", "")).name


def format_action_ids(action_ids: Iterable[int]) -> str:
    names = [action_name(action_id) for action_id in action_ids]
    return "[" + ", ".join(names) + "]"


def parse_action_ids(value: object) -> list[int]:
    """Parse StreamVLN-style action ids, arrows, or natural-language actions."""
    if value is None:
        return []

    if isinstance(value, int):
        return [value] if value in ACTION_SPECS else []

    if isinstance(value, float) and value.is_integer():
        action_id = int(value)
        return [action_id] if action_id in ACTION_SPECS else []

    if isinstance(value, (list, tuple)):
        action_ids: list[int] = []
        for item in value:
            action_ids.extend(parse_action_ids(item))
        return action_ids

    text = str(value).strip()
    if not text:
        return []

    action_ids: list[int] = []
    for arrow, action_id in ARROW_TO_ID.items():
        if arrow in text:
            action_ids.extend([action_id] * text.count(arrow))

    tokens = [
        item.strip()
        for item in re.split(r"[,;\n]+|\band\b|\bthen\b|然后|再|接着", text, flags=re.IGNORECASE)
        if item.strip()
    ]
    if not tokens:
        tokens = [text]

    for token in tokens:
        lowered = token.lower()
        if re.fullmatch(r"[0-3]", lowered):
            action_ids.append(int(lowered))
            continue
        for keyword, action_id in ACTION_NAME_TO_ID.items():
            if keyword in lowered or keyword in token:
                action_ids.append(action_id)
                break

    deduped: list[int] = []
    for action_id in action_ids:
        if action_id in ACTION_SPECS:
            deduped.append(action_id)
    return deduped

