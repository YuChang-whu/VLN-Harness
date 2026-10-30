from __future__ import annotations

import unittest

from realworld_agent.planning.astra_intent_parser import AstraIntentParser
from realworld_agent.policy.astra_client import (
    AstraActionOutput,
    AstraClient,
    AstraConfig,
    AstraIntentOutput,
    AstraSubgoal,
)


class FakeResponses:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(kwargs)
        parsed = self.outputs.pop(0)
        return type("FakeResponse", (), {"id": "resp_test", "output_parsed": parsed})()


class FakeOpenAI:
    def __init__(self, outputs):
        self.responses = FakeResponses(outputs)


class AstraClientTest(unittest.TestCase):
    def test_intent_parser_maps_structured_output(self):
        fake = FakeOpenAI(
            [
                AstraIntentOutput(
                    goal="table",
                    stop_condition="near the table",
                    landmarks=["door", "table"],
                    subgoals=[
                        AstraSubgoal(
                            type="enter_space",
                            description="enter the room",
                            termination_condition="the robot is inside the room",
                            streamvln_instruction="Enter the room and stop once inside.",
                        )
                    ],
                )
            ]
        )
        client = AstraClient(AstraConfig(), client=fake)
        intent = AstraIntentParser(client).parse("Enter the room and stop near the table")

        self.assertEqual(intent.goal, "table")
        self.assertEqual(len(intent.subgoals), 1)
        self.assertIn("gpt-6-astra", intent.parser_name)
        self.assertEqual(fake.responses.calls[0]["model"], "gpt-6-astra")

    def test_action_request_contains_rgb_only_and_one_action(self):
        fake = FakeOpenAI([AstraActionOutput(action=2, reason="doorway is to the left")])
        client = AstraClient(AstraConfig(), client=fake)
        result = client.predict_action(
            image_bytes=b"\xff\xd8\xfftest-jpeg",
            full_instruction="Enter the room",
            active_instruction="Turn toward the doorway",
            completion_condition="doorway is centered",
            previous_actions=[1],
            step_id="subgoal_01",
        )

        self.assertEqual(result.actions, [2])
        content = fake.responses.calls[0]["input"][1]["content"]
        image_parts = [part for part in content if part["type"] == "input_image"]
        self.assertEqual(len(image_parts), 1)
        self.assertTrue(image_parts[0]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertNotIn("depth", str(content).lower())

    def test_stop_is_returned_as_action_zero(self):
        fake = FakeOpenAI([AstraActionOutput(action=0, reason="subgoal complete")])
        client = AstraClient(AstraConfig(), client=fake)
        result = client.predict_action(
            image_bytes=b"\x89PNG\r\n\x1a\nimage",
            full_instruction="Stop near the chair",
            active_instruction="Approach the chair",
            completion_condition="the chair is directly ahead and nearby",
        )
        self.assertEqual(result.actions, [0])


if __name__ == "__main__":
    unittest.main()
