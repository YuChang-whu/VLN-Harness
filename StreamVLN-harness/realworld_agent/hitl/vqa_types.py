"""Typed state for human-in-the-loop visual QA checkpoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ACTIVE = "active"
CLOSED = "closed"

CONTINUE = "continue"
COMPLETE_SUBGOAL = "complete_subgoal"
STOP = "stop"
SUPPORTED_DECISIONS = {CONTINUE, COMPLETE_SUBGOAL, STOP}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def checkpoint_id_from_time() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    return f"vqa_{stamp}"


@dataclass
class VQACheckpoint:
    checkpoint_id: str
    session_id: str
    request_id: str
    reason: str
    rgb_path: str
    depth_path: str = ""
    instruction: str = ""
    active_subgoal: dict[str, Any] | None = None
    completion_condition: str = ""
    recent_actions: list[int] = field(default_factory=list)
    recent_events: list[dict[str, Any]] = field(default_factory=list)
    last_llm_output: str = ""
    status: str = ACTIVE
    created_at: str = field(default_factory=utc_now_iso)
    closed_at: str = ""

    @property
    def is_active(self) -> bool:
        return self.status == ACTIVE


@dataclass
class VQAMessage:
    role: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    backend: str = ""
    raw_output: str = ""


@dataclass
class VQAAnswer:
    answer: str
    backend: str
    raw_output: str = ""
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class VQADecision:
    decision: str
    reason: str
    operator: str = "human"
    created_at: str = field(default_factory=utc_now_iso)


def checkpoint_dir(session_dir: Path, checkpoint_id: str) -> Path:
    return session_dir / "vqa" / checkpoint_id
