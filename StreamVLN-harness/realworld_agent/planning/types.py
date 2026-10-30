"""Typed planning state for the terminal task-flow prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from realworld_agent.planning.intent_parser import IntentPlan


PENDING = "pending"
IN_PROGRESS = "in_progress"
COMPLETED = "completed"
BLOCKED = "blocked"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class PlanStep:
    step_id: str
    title: str
    detail: str = ""
    status: str = PENDING
    evidence: str = ""
    subgoal_type: str = ""
    completion_condition: str = ""
    streamvln_instruction: str = ""
    started_at: str | None = None
    completed_at: str | None = None

    def start(self) -> None:
        if self.status == PENDING:
            self.status = IN_PROGRESS
            self.started_at = utc_now_iso()

    def complete(self, evidence: str = "") -> None:
        self.status = COMPLETED
        self.evidence = evidence or self.evidence
        self.completed_at = utc_now_iso()


@dataclass
class PlanEvent:
    kind: str
    message: str
    created_at: str = field(default_factory=utc_now_iso)


@dataclass
class TaskPlan:
    instruction: str
    steps: list[PlanStep]
    intent: IntentPlan | None = None
    events: list[PlanEvent] = field(default_factory=list)
    stopped: bool = False
    stop_reason: str = ""

    def add_event(self, kind: str, message: str) -> None:
        self.events.append(PlanEvent(kind=kind, message=message))
        self.events = self.events[-8:]

    @property
    def completed_count(self) -> int:
        return sum(1 for step in self.steps if step.status == COMPLETED)

    @property
    def progress_ratio(self) -> float:
        if not self.steps:
            return 1.0
        return self.completed_count / len(self.steps)

    @property
    def current_step(self) -> PlanStep | None:
        for step in self.steps:
            if step.status == IN_PROGRESS:
                return step
        for step in self.steps:
            if step.status == PENDING:
                return step
        return None
