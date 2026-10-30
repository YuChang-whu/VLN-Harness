"""IntentParser adapter backed by the GPT-6 Astra Responses API client."""

from __future__ import annotations

from typing import Any

from realworld_agent.planning.intent_parser import IntentPlan, intent_from_payload
from realworld_agent.policy.astra_client import AstraClient


class AstraIntentParser:
    def __init__(self, model_client: AstraClient):
        self.model_client = model_client

    def parse(self, instruction: str, observation: Any | None = None) -> IntentPlan:
        # Intent decomposition is text-only in this integration. RGB is used by
        # the subsequent per-step action policy, not by the initial planner.
        del observation
        result = self.model_client.parse_intent(instruction)
        intent = intent_from_payload(instruction, result.payload)
        intent.parser_name = self.model_client.model_label
        intent.raw_output = result.raw_output
        return intent
