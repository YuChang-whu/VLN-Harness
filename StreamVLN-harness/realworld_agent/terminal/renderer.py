"""Terminal renderer for the real-world VLN task plan."""

from __future__ import annotations

import os
from dataclasses import dataclass

from realworld_agent.action.semantics import format_action_ids
from realworld_agent.planning.types import BLOCKED, COMPLETED, IN_PROGRESS, PENDING, TaskPlan
from realworld_agent.session.task_session import TaskSessionManager


@dataclass
class TerminalRenderer:
    use_color: bool = True
    width: int = 96

    COLORS = {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "cyan": "\033[36m",
        "blue": "\033[34m",
    }

    def clear(self) -> None:
        os.system("clear")

    def render(self, session: TaskSessionManager) -> str:
        plan = session.require_plan()
        lines: list[str] = []
        lines.extend(self._header(session, plan))
        lines.extend(self._progress(plan))
        lines.extend(self._intent(plan))
        lines.extend(self._steps(plan))
        lines.extend(self._runtime(session, plan))
        lines.extend(self._events(plan))
        return "\n".join(lines)

    def print(self, session: TaskSessionManager, clear: bool = False) -> None:
        if clear:
            self.clear()
        print(self.render(session))

    def _header(self, session: TaskSessionManager, plan: TaskPlan) -> list[str]:
        status = "STOPPED" if plan.stopped else "RUNNING"
        status_color = "green" if plan.stopped else "cyan"
        return [
            self._line(),
            self._color("bold", "Real-World VLN Agent Task Flow"),
            self._line(),
            f"Session     : {session.session_id}",
            f"Status      : {self._color(status_color, status)}",
            f"Instruction : {plan.instruction}",
            self._line(),
        ]

    def _progress(self, plan: TaskPlan) -> list[str]:
        total_slots = 28
        filled = int(total_slots * plan.progress_ratio)
        bar = "#" * filled + "-" * (total_slots - filled)
        pct = int(plan.progress_ratio * 100)
        return [
            f"Progress    : [{bar}] {pct:3d}% ({plan.completed_count}/{len(plan.steps)})",
            "",
        ]

    def _intent(self, plan: TaskPlan) -> list[str]:
        intent = plan.intent
        lines = [self._color("bold", "Intent Understanding")]
        if intent is None:
            lines.append("  Intent parser : not available")
            lines.append("")
            return lines

        parser_color = "yellow" if intent.fallback else "green"
        lines.append(f"  Goal           : {intent.goal or 'unknown'}")
        lines.append(f"  Stop condition : {intent.stop_condition or 'implicit task completion'}")
        lines.append(f"  Landmarks      : {', '.join(intent.landmarks) if intent.landmarks else 'none'}")
        lines.append(f"  Constraints    : {', '.join(intent.constraints) if intent.constraints else 'none'}")
        lines.append(f"  Route hints    : {', '.join(intent.route_hints) if intent.route_hints else 'none'}")
        lines.append(f"  Parser         : {self._color(parser_color, intent.parser_name)}")
        lines.append(f"  Fallback       : {'yes' if intent.fallback else 'no'}")
        if intent.fallback_reason:
            lines.append(f"  Reason         : {intent.fallback_reason}")
        lines.append("")
        return lines

    def _steps(self, plan: TaskPlan) -> list[str]:
        lines = [self._color("bold", "Plan")]
        for index, step in enumerate(plan.steps, start=1):
            mark = self._mark(step.status)
            title = step.title
            status = self._status_text(step.status)
            lines.append(f"  {index:02d}. {mark} {title}  {status}")
            if step.status == IN_PROGRESS and step.detail:
                lines.append(self._color("dim", f"      -> {step.detail}"))
            if step.status == COMPLETED and step.evidence:
                lines.append(self._color("dim", f"      evidence: {step.evidence}"))
        lines.append("")
        return lines

    def _runtime(self, session: TaskSessionManager, plan: TaskPlan) -> list[str]:
        current = plan.current_step.title if plan.current_step is not None else "无待处理步骤"
        lines = [self._color("bold", "Runtime")]
        lines.append(f"  Current step : {current}")
        lines.append(f"  Observations : {session.observation_count}")
        lines.append(f"  Last actions : {format_action_ids(session.last_actions) if session.last_actions else '[]'}")
        if plan.stopped:
            lines.append(f"  Stop reason  : {plan.stop_reason}")
        lines.append("")
        return lines

    def _events(self, plan: TaskPlan) -> list[str]:
        lines = [self._color("bold", "Recent Events")]
        if not plan.events:
            lines.append("  No events yet.")
            return lines
        for event in plan.events[-6:]:
            lines.append(f"  - {event.created_at} [{event.kind}] {event.message}")
        lines.append(self._line())
        return lines

    def _mark(self, status: str) -> str:
        if status == COMPLETED:
            return self._color("green", "[x]")
        if status == IN_PROGRESS:
            return self._color("yellow", "[>]")
        if status == BLOCKED:
            return self._color("red", "[!]")
        return "[ ]"

    def _status_text(self, status: str) -> str:
        if status == COMPLETED:
            return self._color("green", "completed")
        if status == IN_PROGRESS:
            return self._color("yellow", "in progress")
        if status == PENDING:
            return self._color("dim", "pending")
        if status == BLOCKED:
            return self._color("red", "blocked")
        return status

    def _line(self) -> str:
        return "-" * self.width

    def _color(self, name: str, text: str) -> str:
        if not self.use_color:
            return text
        return f"{self.COLORS[name]}{text}{self.COLORS['reset']}"
