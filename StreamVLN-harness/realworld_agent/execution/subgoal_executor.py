"""Subgoal execution loop driven by StreamVLN STOP signals."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from realworld_agent.action.semantics import format_action_ids
from realworld_agent.policy.streamvln_client import StreamVLNModelClient
from realworld_agent.session.task_session import TaskSessionManager
from realworld_agent.terminal.renderer import TerminalRenderer


ObservationProvider = Callable[[], dict]


@dataclass
class SubgoalExecutorConfig:
    max_steps_per_subgoal: int = 0
    delay: float = 0.2
    run_model_every: int = 1
    stop_action_id: int = 0


class StreamVLNSubgoalExecutor:
    """Execute Qwen-planned subgoals with StreamVLN actions.

    The core policy is intentionally simple for the first version:
    current subgoal description -> StreamVLN instruction -> actions. If the
    first action is STOP, mark the current subgoal complete and switch to the
    next one. Otherwise keep the same subgoal and continue with the next frame.
    """

    def __init__(
        self,
        session: TaskSessionManager,
        streamvln_client: StreamVLNModelClient,
        renderer: TerminalRenderer | None = None,
        config: SubgoalExecutorConfig | None = None,
        observation_provider: ObservationProvider | None = None,
    ):
        self.session = session
        self.streamvln_client = streamvln_client
        self.renderer = renderer
        self.config = config or SubgoalExecutorConfig()
        self.observation_provider = observation_provider or self._dummy_observation
        self._active_subgoal_id = ""
        self._steps_on_subgoal = 0

    def run(self) -> None:
        plan = self.session.require_plan()
        if plan.current_step is None:
            self.session.stop("没有可执行的计划步骤。")
            return

        self._complete_preflight_steps()
        if self.renderer is not None:
            self.renderer.print(self.session, clear=True)

        while not self.session.is_done():
            plan = self.session.require_plan()
            current = plan.current_step
            if current is None:
                self.session.stop("所有子目标已完成，执行全局 STOP。")
                break

            if self._active_subgoal_id != current.step_id:
                self._active_subgoal_id = current.step_id
                self._steps_on_subgoal = 0
                self.streamvln_client.reset_policy_memory()
                plan.add_event("subgoal", f"切换到 {current.title}; 使用子目标文本作为 StreamVLN instruction。")

            if (
                self.config.max_steps_per_subgoal > 0
                and self._steps_on_subgoal >= self.config.max_steps_per_subgoal
            ):
                plan.add_event(
                    "subgoal",
                    f"达到 max_steps_per_subgoal={self.config.max_steps_per_subgoal}，调试模式停止执行循环。",
                )
                self.session.stop("调试模式达到子目标步数上限，停止而不是强制切换子目标。")
                break

            observation = self.observation_provider()
            self.session.observe(f"执行 {current.title}: 获取 RGB-D/pose 观测。")
            run_model = self._steps_on_subgoal % max(1, self.config.run_model_every) == 0
            result = self.streamvln_client.predict_actions(
                current.detail,
                rgb=observation.get("rgb"),
                depth=observation.get("depth"),
                camera_pose=observation.get("camera_pose"),
                intrinsic=observation.get("intrinsic"),
                run_model=run_model,
            )
            self._steps_on_subgoal += 1

            plan.add_event(
                "streamvln",
                f"当前子目标 instruction='{current.detail}' -> {format_action_ids(result.actions)}",
            )
            self.session.apply_subgoal_policy_actions(result.actions, source="streamvln_subgoal_policy")

            if self.renderer is not None:
                self.renderer.print(self.session, clear=True)
            if self.config.delay > 0:
                time.sleep(self.config.delay)

    def _complete_preflight_steps(self) -> None:
        while True:
            plan = self.session.require_plan()
            current = plan.current_step
            if current is None:
                return
            if current.step_id.startswith("subgoal_") or current.step_id == "final_stop":
                return
            if current.step_id == "understand_instruction":
                evidence = "意图解析器已输出 goal / landmarks / constraints / subgoals。"
            elif current.step_id == "stabilize_observation":
                evidence = "初始 RGB-D / pose 观测已准备，进入 StreamVLN 子目标执行循环。"
            else:
                evidence = "进入子目标执行循环前完成准备步骤。"
            self.session.complete_current_step(evidence)

    def _dummy_observation(self) -> dict:
        import numpy as np

        return {
            "rgb": np.zeros((480, 640, 3), dtype=np.uint8),
            "depth": np.zeros((480, 640, 1), dtype=np.float32),
            "camera_pose": np.eye(4, dtype=np.float32),
            "intrinsic": None,
        }
