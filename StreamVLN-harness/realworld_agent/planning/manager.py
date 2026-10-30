"""Task planning manager for real-world VLN terminal visualization."""

from __future__ import annotations

import re

from realworld_agent.planning.intent_parser import IntentParser, IntentPlan, QwenIntentParser
from realworld_agent.planning.types import IN_PROGRESS, PENDING, PlanStep, TaskPlan


class PlanningManager:
    """Create and update a semantic task plan from model-parsed intent.

    QwenIntentParser is the primary path. It asks the local Qwen text
    model to produce structured intent JSON, then this manager converts that
    structure into visible plan steps.
    """

    def __init__(self, intent_parser: IntentParser | None = None):
        self.intent_parser = intent_parser or QwenIntentParser()

    def create_plan(self, instruction: str, observation=None) -> TaskPlan:
        clean_instruction = self._clean_text(instruction)
        intent = self.intent_parser.parse(clean_instruction, observation=observation)
        steps = [
            PlanStep(
                step_id="understand_instruction",
                title="理解目标语义",
                detail=self._intent_summary(intent),
            ),
            PlanStep(
                step_id="stabilize_observation",
                title="稳定初始观测",
                detail="确认 RGB-D / odom / 当前朝向可用，然后再进入子目标推进。",
            ),
        ]

        for index, subgoal in enumerate(intent.subgoals, start=1):
            completion_condition = (
                subgoal.termination_condition
                or subgoal.completion_signal
                or "当前子目标已经由视觉导航策略确认完成。"
            )
            steps.append(
                PlanStep(
                    step_id=f"subgoal_{index:02d}",
                    title=f"子目标 {index}: {self._short_title(subgoal.description)}",
                    detail=subgoal.description,
                    subgoal_type=subgoal.type,
                    completion_condition=completion_condition,
                    streamvln_instruction=self._build_streamvln_instruction(
                        subgoal.description,
                        completion_condition,
                        subgoal.streamvln_instruction,
                    ),
                )
            )

        if not intent.subgoals or intent.subgoals[-1].type.lower() != "stop":
            stop_detail = intent.stop_condition or "当最后一个语义子目标完成后，执行 STOP 并结束任务。"
            steps.append(
                PlanStep(
                    step_id="final_stop",
                    title="到达终点后停止",
                    detail=stop_detail,
                    subgoal_type="stop",
                    completion_condition=stop_detail,
                    streamvln_instruction=f"Stop when {stop_detail}.",
                )
            )

        plan = TaskPlan(instruction=clean_instruction, steps=steps, intent=intent)
        plan.add_event("session", "创建任务计划，等待开始。")
        if intent.fallback:
            plan.add_event("intent", f"意图解析失败，启用 fallback: {intent.fallback_reason}")
        else:
            plan.add_event("intent", f"已使用 {intent.parser_name} 完成语义意图解析。")
        return plan

    def start(self, plan: TaskPlan) -> PlanStep | None:
        step = plan.current_step
        if step is not None:
            step.start()
            plan.add_event("planning", f"开始: {step.title}")
        return step

    def complete_current(self, plan: TaskPlan, evidence: str = "") -> PlanStep | None:
        step = plan.current_step
        if step is None:
            return None

        if step.status == PENDING:
            step.start()
        step.complete(evidence)
        plan.add_event("planning", f"完成: {step.title}" + (f" ({evidence})" if evidence else ""))

        next_step = self.start_next_pending(plan)
        if next_step is None and not plan.stopped:
            plan.add_event("planning", "所有计划步骤已完成，等待 STOP。")
        return step

    def start_next_pending(self, plan: TaskPlan) -> PlanStep | None:
        if any(step.status == IN_PROGRESS for step in plan.steps):
            return plan.current_step
        for step in plan.steps:
            if step.status == PENDING:
                step.start()
                plan.add_event("planning", f"进入: {step.title}")
                return step
        return None

    def stop(self, plan: TaskPlan, reason: str = "任务到达终点，执行 STOP。") -> None:
        for step in plan.steps:
            if step.status == PENDING:
                step.start()
                step.complete("任务结束时自动收束。")
            elif step.status == IN_PROGRESS:
                step.complete("STOP 前完成当前阶段。")
        plan.stopped = True
        plan.stop_reason = reason
        plan.add_event("stop", reason)

    def _intent_summary(self, intent: IntentPlan) -> str:
        parts = []
        if intent.goal:
            parts.append(f"目标: {intent.goal}")
        if intent.stop_condition:
            parts.append(f"停止条件: {intent.stop_condition}")
        if intent.landmarks:
            parts.append("地标: " + ", ".join(intent.landmarks))
        if intent.constraints:
            parts.append("约束: " + ", ".join(intent.constraints))
        parser = intent.parser_name
        if intent.fallback:
            parser += f" (fallback: {intent.fallback_reason})"
        parts.append(f"解析器: {parser}")
        return "；".join(parts)

    def _build_streamvln_instruction(
        self,
        description: str,
        completion_condition: str,
        model_instruction: str = "",
    ) -> str:
        if model_instruction:
            return model_instruction.strip()
        description = description.strip()
        completion_condition = completion_condition.strip()
        if completion_condition:
            return f"{description}. Stop when {completion_condition}."
        return description

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text or "").strip()
        return text or "move forward and stop when the task is complete"

    def _short_title(self, text: str) -> str:
        text = text.strip()
        if len(text) <= 32:
            return text
        return text[:29] + "..."
