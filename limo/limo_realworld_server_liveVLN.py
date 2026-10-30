#!/usr/bin/env python
"""LiveVLN-style HTTP inference server for the Limo StreamVLN deployment.

This server keeps the original StreamVLN inference path and RGB-D request
format, but exposes the predicted action chunk as an immediate action, a guard
action, and a revisable tail.  The robot client can execute the guard while it
requests the next action chunk.
"""

import argparse
import cgi
import threading
import time
from http.server import ThreadingHTTPServer
from pathlib import Path

import limo_realworld_server as streamvln_server


def split_livevln_actions(actions):
    """Return a JSON-safe LiveVLN action split.

    A missing second action is padded with STOP so the client never invents an
    unsafe guard action.
    """
    full_actions = [int(action) for action in actions]
    if not full_actions:
        full_actions = [0]

    action_e = full_actions[0]
    guard_g = full_actions[1] if len(full_actions) > 1 else 0
    revisable_tail = full_actions[2:]
    return full_actions, action_e, guard_g, revisable_tail


class LimoStreamVLNLiveVLNHandler(streamvln_server.LimoStreamVLNHandler):
    server_version = "LimoStreamVLNLiveVLN/1.0"

    def do_GET(self):
        if self.path not in ("/", "/eval_vln"):
            streamvln_server.json_response(self, 404, {"ok": False, "error": "not found"})
            return
        streamvln_server.json_response(self, 200, {
            "ok": True,
            "service": "limo_streamvln_server_liveVLN",
            "endpoint": "/eval_vln",
            "model_path": self.server.args.model_path,
            "save_dir": self.server.args.save_dir,
            "action_semantics": streamvln_server.ACTION_NAMES,
            "response_fields": [
                "action_e",
                "guard_g",
                "revisable_tail",
                "action_sequence",
            ],
        })

    def do_POST(self):
        if self.path != "/eval_vln":
            streamvln_server.json_response(self, 404, {"ok": False, "error": "not found"})
            return

        started = time.time()
        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            streamvln_server.json_response(
                self,
                400,
                {"ok": False, "error": "expected multipart/form-data"},
            )
            return

        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("content-length", "0"),
                },
            )
            metadata = streamvln_server.parse_json_field(form)
            image_field, image_bytes = streamvln_server.read_file_field(form, "image")
            depth_field, depth_bytes = streamvln_server.read_file_field(form, "depth")
            if image_field is None or not image_bytes:
                streamvln_server.json_response(
                    self,
                    400,
                    {"ok": False, "error": "missing image file field"},
                )
                return

            instruction = (
                metadata.get("prompt")
                or metadata.get("instruction")
                or metadata.get("task")
                or self.server.args.default_instruction
            )
            if not instruction:
                streamvln_server.json_response(
                    self,
                    400,
                    {"ok": False, "error": "missing prompt/instruction in json"},
                )
                return

            rgb = streamvln_server.decode_rgb(image_bytes)
            depth = streamvln_server.decode_depth(depth_bytes, metadata, rgb.shape)
            intrinsic = streamvln_server.parse_intrinsic(metadata)
            reset = bool(metadata.get("reset", False))

            with self.server.infer_lock:
                if reset:
                    print("Reset StreamVLN LiveVLN memory for new Limo request stream", flush=True)
                    self.server.evaluator.reset_memory()
                    self.server.terminate = False
                    self.server.action_seq = []
                    self.server.last_llm_output = ""

                if self.server.terminate:
                    raw_actions = [0]
                    generate_time = 0.0
                    llm_output = self.server.last_llm_output
                else:
                    generate_time = 0.0
                    llm_output = self.server.last_llm_output
                    for _ in range(self.server.args.steps_per_request):
                        run_model = self.server.evaluator.step_id % self.server.args.run_every == 0
                        return_action, step_generate_time, return_llm_output = self.server.evaluator.step(
                            0,
                            rgb,
                            instruction,
                            run_model=run_model,
                            depth=depth,
                            intrinsic=intrinsic,
                        )
                        if step_generate_time > 0:
                            generate_time = float(step_generate_time)
                        if return_llm_output is not None:
                            llm_output = return_llm_output
                            self.server.last_llm_output = return_llm_output
                        if return_action is not None:
                            self.server.action_seq = [int(action) for action in return_action]
                        self.server.evaluator.step_id += 1
                    raw_actions = self.server.action_seq

                full_actions, action_e, guard_g, revisable_tail = split_livevln_actions(raw_actions)

                # STOP in a revisable tail is only a prediction.  Latch the
                # episode as terminated only when STOP is the immediate action.
                if action_e == 0:
                    self.server.terminate = True

            payload = {
                "ok": True,
                # Keep `action` for clients that already read this field.  The
                # LiveVLN client should use the explicit fields below.
                "action": full_actions,
                "action_names": streamvln_server.action_names(full_actions),
                "action_e": action_e,
                "action_e_name": streamvln_server.action_names([action_e])[0],
                "guard_g": guard_g,
                "guard_g_name": streamvln_server.action_names([guard_g])[0],
                "revisable_tail": revisable_tail,
                "revisable_tail_names": streamvln_server.action_names(revisable_tail),
                "action_sequence": full_actions,
                "action_sequence_names": streamvln_server.action_names(full_actions),
                "instruction": instruction,
                "reset": reset,
                "terminate": self.server.terminate,
                "generate_time": generate_time,
                "elapsed_ms": round((time.time() - started) * 1000.0, 2),
                "received": {
                    "image": {
                        "filename": getattr(image_field, "filename", None),
                        "bytes": len(image_bytes),
                        "shape": list(rgb.shape),
                    },
                    "depth": {
                        "present": bool(depth_bytes),
                        "filename": getattr(depth_field, "filename", None) if depth_field is not None else None,
                        "bytes": len(depth_bytes),
                        "shape": list(depth.shape) if depth is not None else None,
                    },
                },
            }
            if self.server.args.include_llm_output:
                payload["llm_output"] = llm_output

            if self.server.args.save_dir:
                payload["saved"] = streamvln_server.save_upload(
                    self.server.args.save_dir,
                    image_bytes,
                    depth_bytes,
                    metadata,
                    response_payload=payload,
                )

            print("=" * 72, flush=True)
            print("Received Limo LiveVLN POST /eval_vln", flush=True)
            print("instruction:", instruction, flush=True)
            print(
                "rgb_shape:",
                rgb.shape,
                "depth_shape:",
                None if depth is None else depth.shape,
                flush=True,
            )
            print("action_e:", action_e, "guard_g:", guard_g, flush=True)
            print("revisable_tail:", revisable_tail, "sequence:", full_actions, flush=True)
            print(
                "generate_time:",
                generate_time,
                "elapsed_ms:",
                payload["elapsed_ms"],
                flush=True,
            )
            streamvln_server.json_response(self, 200, payload)
        except Exception as exc:
            print("ERROR handling LiveVLN /eval_vln:", repr(exc), flush=True)
            streamvln_server.json_response(self, 500, {
                "ok": False,
                "error": str(exc),
                "action": [0],
                "action_e": 0,
                "guard_g": 0,
                "revisable_tail": [],
                "action_sequence": [0],
                "terminate": True,
            })


def parse_args():
    parser = argparse.ArgumentParser(description="Limo RGB-D StreamVLN LiveVLN HTTP server")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(
            streamvln_server.STREAMVLN_ROOT
            / "data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"
        ),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6009)
    parser.add_argument(
        "--save-dir",
        default=str(streamvln_server.STREAMVLN_ROOT / "limo/uploads_liveVLN"),
    )
    parser.add_argument(
        "--default-instruction",
        default="Walk forward and stop when the task is complete.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--rgb-height", type=float, default=1.25)
    parser.add_argument("--steps-per-request", type=int, default=4)
    parser.add_argument("--run-every", type=int, default=4)
    parser.add_argument("--include-llm-output", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading StreamVLN LiveVLN model from", args.model_path, flush=True)
    evaluator = streamvln_server.load_streamvln(args)

    server = ThreadingHTTPServer((args.host, args.port), LimoStreamVLNLiveVLNHandler)
    server.args = args
    server.evaluator = evaluator
    server.infer_lock = threading.Lock()
    server.terminate = False
    server.action_seq = []
    server.last_llm_output = ""

    if args.save_dir:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    print("Listening on http://%s:%d" % (args.host, args.port), flush=True)
    print("POST endpoint: /eval_vln", flush=True)
    print("Saving uploads to", args.save_dir, flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
