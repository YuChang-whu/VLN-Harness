#!/usr/bin/env python3
"""Operator console for human-in-the-loop visual QA checkpoints."""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request


HELP = """Commands:
  status                         show current session/VQA status
  start [reason]                 pause robot with a VQA checkpoint
  ask <question>                 ask the VQA evaluator about latest image
  continue [reason]              close VQA and continue current subgoal
  complete [reason]              mark current subgoal complete by human VQA
  stop [reason]                  stop the whole task
  quit                           exit console
"""


class VQAConsole:
    def __init__(self, server: str):
        self.server = server.rstrip("/")
        self.checkpoint_id = ""

    def get(self, path: str) -> dict:
        with urllib.request.urlopen(self.server + path, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    def post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.server + path,
            data=data,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            try:
                return json.loads(body)
            except Exception:
                return {"ok": False, "error": body or str(exc)}

    def run(self) -> None:
        print("HITL VQA console connected to", self.server)
        print(HELP)
        while True:
            try:
                line = input("hitl-vqa> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not line:
                continue
            command, _, rest = line.partition(" ")
            command = command.lower()
            try:
                if command in {"quit", "exit", "q"}:
                    return
                if command in {"help", "h", "?"}:
                    print(HELP)
                    continue
                if command == "status":
                    self.print_json(self.get("/hitl/vqa/status"))
                    continue
                if command == "start":
                    result = self.post("/hitl/vqa/start", {"reason": rest or "operator requested visual QA"})
                    checkpoint = result.get("checkpoint") or result.get("vqa", {}).get("checkpoint") or {}
                    self.checkpoint_id = result.get("checkpoint_id") or checkpoint.get("checkpoint_id") or self.checkpoint_id
                    self.print_json(result)
                    continue
                if command == "ask":
                    result = self.post("/hitl/vqa/ask", {"checkpoint_id": self.checkpoint_id, "question": rest})
                    print(result.get("answer") or result)
                    continue
                if command in {"continue", "complete", "stop"}:
                    decision = {"continue": "continue", "complete": "complete_subgoal", "stop": "stop"}[command]
                    result = self.post(
                        "/hitl/vqa/decide",
                        {"checkpoint_id": self.checkpoint_id, "decision": decision, "reason": rest or decision},
                    )
                    if result.get("ok"):
                        self.checkpoint_id = ""
                    self.print_json(result)
                    continue
                print("Unknown command. Type help.")
            except Exception as exc:
                print("ERROR:", exc)

    def print_json(self, payload: dict) -> None:
        print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> int:
    parser = argparse.ArgumentParser(description="HITL VQA operator console")
    parser.add_argument("--server", default="http://127.0.0.1:6010")
    args = parser.parse_args()
    VQAConsole(args.server).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
