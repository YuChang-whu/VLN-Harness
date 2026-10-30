"""Human-in-the-loop visual QA checkpoint manager."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

from realworld_agent.hitl.vqa_types import (
    ACTIVE,
    CLOSED,
    COMPLETE_SUBGOAL,
    CONTINUE,
    STOP,
    SUPPORTED_DECISIONS,
    VQAAnswer,
    VQACheckpoint,
    VQADecision,
    VQAMessage,
    checkpoint_dir,
    checkpoint_id_from_time,
    utc_now_iso,
)
from realworld_agent.policy.vqa_evaluator import VQAEvaluator
from realworld_agent.session.task_session import TaskSessionManager


class VQAManager:
    """Owns HITL VQA checkpoint state and persistent transcript files."""

    def __init__(self, evaluator: VQAEvaluator):
        self.evaluator = evaluator
        self.active_checkpoint: VQACheckpoint | None = None

    def status(self) -> dict[str, Any]:
        checkpoint = self.active_checkpoint
        return {
            "active": bool(checkpoint and checkpoint.is_active),
            "checkpoint": None if checkpoint is None else asdict(checkpoint),
            "backend": self.evaluator.backend,
        }

    def start_checkpoint(
        self,
        session: TaskSessionManager,
        request_id: str,
        saved_artifacts: dict[str, str],
        reason: str,
        plan_payload: dict[str, Any],
        last_llm_output: str = "",
    ) -> VQACheckpoint:
        if self.active_checkpoint is not None and self.active_checkpoint.is_active:
            return self.active_checkpoint

        rgb_path = saved_artifacts.get("image") or saved_artifacts.get("rgb") or ""
        if not rgb_path:
            raise ValueError("cannot start VQA before an RGB observation has been saved")
        if not Path(rgb_path).exists():
            raise FileNotFoundError(f"latest RGB image does not exist: {rgb_path}")

        checkpoint_id = checkpoint_id_from_time()
        dst_dir = checkpoint_dir(session.session_dir, checkpoint_id)
        dst_dir.mkdir(parents=True, exist_ok=True)

        copied_rgb = dst_dir / "rgb.jpg"
        shutil.copy2(rgb_path, copied_rgb)
        depth_path = saved_artifacts.get("depth") or ""
        copied_depth = ""
        if depth_path and Path(depth_path).exists():
            copied_depth_path = dst_dir / "depth.png"
            shutil.copy2(depth_path, copied_depth_path)
            copied_depth = str(copied_depth_path)

        active_subgoal = plan_payload.get("active_subgoal") if plan_payload else None
        checkpoint = VQACheckpoint(
            checkpoint_id=checkpoint_id,
            session_id=session.session_id,
            request_id=request_id,
            reason=reason or "operator requested visual QA",
            rgb_path=str(copied_rgb),
            depth_path=copied_depth,
            instruction=session.instruction,
            active_subgoal=active_subgoal,
            completion_condition=(active_subgoal or {}).get("completion_condition", "") if active_subgoal else "",
            recent_actions=session.last_actions,
            recent_events=plan_payload.get("recent_events", []) if plan_payload else [],
            last_llm_output=last_llm_output,
            status=ACTIVE,
        )
        self.active_checkpoint = checkpoint
        self._write_context(session, checkpoint)
        session.append_history_event(
            "hitl_vqa_started",
            {
                "checkpoint": asdict(checkpoint),
            },
            request_id=request_id,
        )
        session.write_latest_state()
        return checkpoint

    def ask_question(
        self,
        session: TaskSessionManager,
        checkpoint_id: str,
        question: str,
    ) -> VQAAnswer:
        checkpoint = self._require_active(checkpoint_id)
        self._append_message(session, checkpoint, VQAMessage(role="human", content=question))
        session.append_history_event(
            "hitl_vqa_question_asked",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "question": question,
            },
            request_id=checkpoint.request_id,
        )
        answer = self.evaluator.ask(
            rgb_path=checkpoint.rgb_path,
            depth_path=checkpoint.depth_path or None,
            question=question,
            instruction=checkpoint.instruction,
            active_subgoal=checkpoint.active_subgoal,
            completion_condition=checkpoint.completion_condition,
            recent_actions=checkpoint.recent_actions,
            recent_events=checkpoint.recent_events,
        )
        self._append_message(
            session,
            checkpoint,
            VQAMessage(
                role="model",
                content=answer.answer,
                backend=answer.backend,
                raw_output=answer.raw_output,
            ),
        )
        session.append_history_event(
            "hitl_vqa_answered",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "backend": answer.backend,
                "answer": answer.answer,
            },
            request_id=checkpoint.request_id,
        )
        session.write_latest_state()
        return answer

    def record_decision(
        self,
        session: TaskSessionManager,
        checkpoint_id: str,
        decision: str,
        reason: str,
        operator: str = "human",
    ) -> VQADecision:
        checkpoint = self._require_active(checkpoint_id)
        if decision not in SUPPORTED_DECISIONS:
            raise ValueError(f"unsupported VQA decision: {decision}")
        payload = VQADecision(decision=decision, reason=reason or decision, operator=operator)
        decision_path = checkpoint_dir(session.session_dir, checkpoint.checkpoint_id) / "decision.json"
        decision_path.write_text(json.dumps(asdict(payload), ensure_ascii=False, indent=2), encoding="utf-8")
        session.append_history_event(
            "hitl_vqa_decision",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "decision": asdict(payload),
            },
            request_id=checkpoint.request_id,
        )
        checkpoint.status = CLOSED
        checkpoint.closed_at = utc_now_iso()
        self._write_context(session, checkpoint)
        self.active_checkpoint = None
        session.append_history_event(
            "hitl_vqa_closed",
            {
                "checkpoint_id": checkpoint.checkpoint_id,
                "decision": decision,
            },
            request_id=checkpoint.request_id,
        )
        session.write_latest_state()
        return payload

    def _require_active(self, checkpoint_id: str = "") -> VQACheckpoint:
        checkpoint = self.active_checkpoint
        if checkpoint is None or not checkpoint.is_active:
            raise RuntimeError("no active VQA checkpoint")
        requested_id = self._normalize_checkpoint_id(checkpoint_id)
        if requested_id and checkpoint.checkpoint_id != requested_id:
            raise ValueError(
                f"active checkpoint is {checkpoint.checkpoint_id}, got {requested_id}. "
                "Use the active checkpoint_id returned by /hitl/vqa/start, "
                "or omit checkpoint_id to use the active checkpoint."
            )
        return checkpoint

    def _normalize_checkpoint_id(self, checkpoint_id: str = "") -> str:
        value = str(checkpoint_id or "").strip()
        if value.lower() in {"", "active", "latest", "current", "vqa_xxx"}:
            return ""
        return value

    def _write_context(self, session: TaskSessionManager, checkpoint: VQACheckpoint) -> None:
        path = checkpoint_dir(session.session_dir, checkpoint.checkpoint_id) / "context.json"
        path.write_text(json.dumps(asdict(checkpoint), ensure_ascii=False, indent=2), encoding="utf-8")

    def _append_message(self, session: TaskSessionManager, checkpoint: VQACheckpoint, message: VQAMessage) -> None:
        path = checkpoint_dir(session.session_dir, checkpoint.checkpoint_id) / "messages.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")


__all__ = [
    "COMPLETE_SUBGOAL",
    "CONTINUE",
    "STOP",
    "VQAManager",
]
