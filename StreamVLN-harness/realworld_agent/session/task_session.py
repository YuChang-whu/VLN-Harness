"""Task session manager for the real-world VLN planning prototype."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from realworld_agent.action.semantics import format_action_ids, parse_action_ids
from realworld_agent.planning.manager import PlanningManager
from realworld_agent.planning.types import TaskPlan


HARNESS_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SESSION_HISTORY_ROOT = HARNESS_ROOT / "outputs" / "sessions"


def default_session_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"rw_{stamp}"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class TaskSessionManager:
    instruction: str
    session_id: str = field(default_factory=default_session_id)
    planner: PlanningManager = field(default_factory=PlanningManager)
    history_root: Path | str = DEFAULT_SESSION_HISTORY_ROOT
    plan: TaskPlan | None = None
    last_actions: list[int] = field(default_factory=list)
    observation_count: int = 0
    request_count: int = 0

    def __post_init__(self) -> None:
        self.history_root = Path(self.history_root)

    def start(self, observation: Any | None = None) -> TaskPlan:
        self.ensure_history_dir()
        self.append_history_event(
            "session_started",
            {
                "instruction": self.instruction,
            },
        )
        self.plan = self.planner.create_plan(self.instruction, observation=observation)
        self.plan.add_event("session", f"Session {self.session_id} started.")
        self.planner.start(self.plan)
        self._append_intent_events()
        self._write_instruction_file()
        self.append_history_event(
            "instruction_received",
            {
                "instruction": self.instruction,
                "total_steps": len(self.plan.steps),
                "intent": None if self.plan.intent is None else asdict(self.plan.intent),
            },
        )
        self.write_latest_state()
        return self.plan

    def _append_intent_events(self) -> None:
        assert self.plan is not None
        intent = self.plan.intent
        if intent is not None:
            self.append_history_event(
                "intent_interpretation_completed",
                {
                    "goal": intent.goal,
                    "stop_condition": intent.stop_condition,
                    "landmarks": intent.landmarks,
                    "constraints": intent.constraints,
                    "route_hints": intent.route_hints,
                    "parser": intent.parser_name,
                    "fallback": intent.fallback,
                    "fallback_reason": intent.fallback_reason,
                },
            )

        subgoals = [
            asdict(step)
            for step in self.plan.steps
            if self._is_navigation_step(step.step_id)
        ]
        self.append_history_event(
            "subgoal_plan_completed",
            {
                "subgoals": subgoals,
                "count": len(subgoals),
            },
        )

    def require_plan(self) -> TaskPlan:
        if self.plan is None:
            return self.start()
        return self.plan

    def observe(self, note: str = "收到新的真实世界观测。") -> None:
        plan = self.require_plan()
        self.observation_count += 1
        plan.add_event("observe", f"Observation #{self.observation_count}: {note}")
        self.append_history_event(
            "observation_recorded",
            {
                "observation_count": self.observation_count,
                "note": note,
            },
        )
        self.write_latest_state()

    def complete_current_step(self, evidence: str = "", request_id: str | None = None) -> None:
        plan = self.require_plan()
        completed = self.planner.complete_current(plan, evidence)
        next_step = plan.current_step
        self.append_history_event(
            "todo_completed",
            {
                "step": None if completed is None else asdict(completed),
                "evidence": evidence,
                "next_step": None if next_step is None else asdict(next_step),
            },
            request_id=request_id,
        )
        if completed is not None and self._is_navigation_step(completed.step_id):
            self.append_history_event(
                "subgoal_transition",
                {
                    "from_step": asdict(completed),
                    "to_step": None if next_step is None else asdict(next_step),
                    "reason": evidence,
                },
                request_id=request_id,
            )
        if next_step is not None and self._is_navigation_step(next_step.step_id):
            self.append_history_event(
                "subgoal_started",
                {
                    "step": asdict(next_step),
                    "previous_step_id": None if completed is None else completed.step_id,
                },
                request_id=request_id,
            )
        self.write_latest_state()

    def apply_actions(self, actions: object, source: str = "model") -> list[int]:
        plan = self.require_plan()
        action_ids = parse_action_ids(actions)
        self.last_actions = action_ids

        if not action_ids:
            plan.add_event("action", f"{source} action parse failed; wait for safe STOP/replan.")
            self.append_history_event(
                "action_parse_failed",
                {
                    "source": source,
                    "raw_actions": actions,
                },
            )
            self.write_latest_state()
            return []

        plan.add_event("action", f"{source} proposed {format_action_ids(action_ids)}")
        self.append_history_event(
            "action_proposed",
            {
                "source": source,
                "actions": action_ids,
                "action_text": format_action_ids(action_ids),
            },
        )
        if action_ids[0] == 0:
            self.stop("模型或操作员请求 STOP。")
        else:
            self.write_latest_state()
        return action_ids

    def apply_subgoal_policy_actions(
        self,
        actions: object,
        source: str = "streamvln_subgoal_policy",
        request_id: str | None = None,
    ) -> list[int]:
        """Apply policy actions where STOP completes the current subgoal.

        This is different from ``apply_actions``: here a STOP from StreamVLN
        means the current subgoal is complete. The whole task stops only after
        all plan steps are completed.
        """
        plan = self.require_plan()
        action_ids = parse_action_ids(actions)
        self.last_actions = action_ids

        if not action_ids:
            plan.add_event("action", f"{source} returned no action; keeping current subgoal.")
            self.append_history_event(
                "action_parse_failed",
                {
                    "source": source,
                    "raw_actions": actions,
                    "reason": "empty parsed action list",
                },
                request_id=request_id,
            )
            self.write_latest_state()
            return []

        plan.add_event("action", f"{source} proposed {format_action_ids(action_ids)}")
        self.append_history_event(
            "action_proposed",
            {
                "source": source,
                "actions": action_ids,
                "action_text": format_action_ids(action_ids),
                "current_step": None if plan.current_step is None else asdict(plan.current_step),
            },
            request_id=request_id,
        )
        if action_ids[0] != 0:
            self.write_latest_state()
            return action_ids

        current = plan.current_step
        if current is None:
            self.stop("所有子目标已完成，执行全局 STOP。")
            return action_ids

        evidence = f"{source} 输出 STOP，判定当前子目标完成。"
        self.complete_current_step(evidence, request_id=request_id)
        self.append_history_event(
            "subgoal_completed",
            {
                "source": source,
                "completed_step": asdict(current),
            },
            request_id=request_id,
        )
        if plan.current_step is None:
            self.stop("所有子目标已完成，执行全局 STOP。")
        else:
            self.write_latest_state()
        return action_ids

    def complete_current_subgoal_by_human_vqa(
        self,
        evidence: str = "Human VQA confirmed current subgoal is complete.",
        request_id: str | None = None,
    ) -> None:
        plan = self.require_plan()
        current = plan.current_step
        self.complete_current_step(evidence, request_id=request_id)
        self.append_history_event(
            "subgoal_completed_by_human_vqa",
            {
                "evidence": evidence,
                "completed_step": None if current is None else asdict(current),
                "next_step": None if plan.current_step is None else asdict(plan.current_step),
            },
            request_id=request_id,
        )
        self.write_latest_state()

    def stop(self, reason: str = "到达终点，任务停止。") -> None:
        plan = self.require_plan()
        self.planner.stop(plan, reason)
        self.append_history_event(
            "session_stopped",
            {
                "reason": reason,
                "last_actions": self.last_actions,
                "progress_ratio": plan.progress_ratio,
            },
        )
        self.append_history_event(
            "task_completed",
            {
                "reason": reason,
                "last_actions": self.last_actions,
                "progress_ratio": plan.progress_ratio,
            },
        )
        self.write_latest_state()

    def is_done(self) -> bool:
        plan = self.require_plan()
        return plan.stopped

    @property
    def session_dir(self) -> Path:
        return Path(self.history_root) / self.session_id

    @property
    def requests_dir(self) -> Path:
        return self.session_dir / "requests"

    @property
    def event_log_path(self) -> Path:
        return self.session_dir / "events.jsonl"

    @property
    def latest_state_path(self) -> Path:
        return self.session_dir / "latest_state.json"

    def ensure_history_dir(self) -> None:
        self.requests_dir.mkdir(parents=True, exist_ok=True)

    def request_dir(self, request_id: str) -> Path:
        path = self.requests_dir / request_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def next_request_id(self) -> str:
        self.ensure_history_dir()
        self.request_count += 1
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"{stamp}_{self.request_count:03d}"

    def append_history_event(
        self,
        event: str,
        payload: dict[str, Any] | None = None,
        request_id: str | None = None,
    ) -> None:
        self.ensure_history_dir()
        record = {
            "time": utc_now_iso(),
            "session_id": self.session_id,
            "request_id": request_id,
            "event": event,
            "payload": payload or {},
        }
        with self.event_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(_jsonable(record), ensure_ascii=False) + "\n")

    def write_latest_state(self) -> None:
        self.ensure_history_dir()
        state = {
            "time": utc_now_iso(),
            "session_id": self.session_id,
            "instruction": self.instruction,
            "observation_count": self.observation_count,
            "request_count": self.request_count,
            "last_actions": self.last_actions,
            "progress": self._progress_snapshot(),
            "plan": None if self.plan is None else asdict(self.plan),
        }
        self.latest_state_path.write_text(
            json.dumps(_jsonable(state), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _progress_snapshot(self) -> dict[str, Any]:
        if self.plan is None:
            return {
                "current_step_id": None,
                "in_progress_step_ids": [],
                "completed_step_ids": [],
                "pending_step_ids": [],
            }
        current = self.plan.current_step
        return {
            "current_step_id": None if current is None else current.step_id,
            "in_progress_step_ids": [
                step.step_id for step in self.plan.steps if step.status == "in_progress"
            ],
            "completed_step_ids": [
                step.step_id for step in self.plan.steps if step.status == "completed"
            ],
            "pending_step_ids": [
                step.step_id for step in self.plan.steps if step.status == "pending"
            ],
        }

    def _is_navigation_step(self, step_id: str) -> bool:
        return step_id.startswith("subgoal_") or step_id == "final_stop"

    def _write_instruction_file(self) -> None:
        self.ensure_history_dir()
        payload = {
            "session_id": self.session_id,
            "instruction": self.instruction,
            "created_at": utc_now_iso(),
            "plan": None if self.plan is None else asdict(self.plan),
        }
        (self.session_dir / "instruction.json").write_text(
            json.dumps(_jsonable(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)
