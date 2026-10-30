from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from realworld_agent.policy.astra_client import AstraActionResult, AstraIntentResult
from realworld_agent.server.gpt6_astra_server import AstraSubgoalRuntime, parse_args


class FakeAstraPolicy:
    model_label = "fake-gpt-6-astra"

    def parse_intent(self, instruction: str) -> AstraIntentResult:
        return AstraIntentResult(
            payload={
                "goal": "room",
                "stop_condition": "inside the room",
                "landmarks": ["door"],
                "constraints": [],
                "route_hints": ["go through the door"],
                "subgoals": [
                    {
                        "type": "enter_space",
                        "description": "enter the room",
                        "completion_signal": "robot is inside the room",
                        "termination_condition": "robot is inside the room",
                        "streamvln_instruction": "Enter the room and stop once inside.",
                    }
                ],
            },
            raw_output="{}",
            generate_time=0.01,
            response_id="intent_test",
        )

    def predict_action(self, **kwargs) -> AstraActionResult:
        if "depth" in kwargs:
            raise AssertionError("depth must not be sent to Astra")
        return AstraActionResult(
            actions=[0],
            raw_output='{"action":0,"reason":"done"}',
            generate_time=0.02,
            response_id="action_test",
        )


class AstraRuntimeTest(unittest.TestCase):
    def test_any_astra_stop_completes_current_subgoal_immediately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            original_argv = sys.argv
            try:
                sys.argv = [
                    "server",
                    "--GPT",
                    "--save-dir",
                    temp_dir,
                    "--stop-confirmations",
                    "99",
                ]
                args = parse_args()
            finally:
                sys.argv = original_argv

            runtime = AstraSubgoalRuntime(args, streamvln_server_module=object())
            runtime.astra_client = FakeAstraPolicy()
            runtime.reset("Enter the room", {})
            image_path = Path(temp_dir) / "rgb.jpg"
            image_path.write_bytes(b"\xff\xd8\xfftest")
            runtime.latest_saved_artifacts = {"image": str(image_path)}
            runtime.latest_request_id = "request_test"

            result = runtime.step(None, object(), object(), {})

            self.assertTrue(result["subgoal_completed"])
            self.assertEqual(result["action_sequence"], [0])
            self.assertFalse(result["terminate"])
            self.assertEqual(result["plan"]["active_subgoal"]["step_id"], "final_stop")


if __name__ == "__main__":
    unittest.main()
