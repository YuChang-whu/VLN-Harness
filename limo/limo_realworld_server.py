#!/usr/bin/env python
import argparse
import cgi
import json
import os
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
import transformers
from PIL import Image

STREAMVLN_ROOT = Path(__file__).resolve().parents[1]
STREAMVLN_PKG = STREAMVLN_ROOT / "streamvln"
sys.path.insert(0, str(STREAMVLN_ROOT))
sys.path.insert(0, str(STREAMVLN_PKG))

from streamvln.streamvln_agent import VLNEvaluator
from model.stream_video_vln import StreamVLNForCausalLM


ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD_0.25M",
    2: "TURN_LEFT_15_DEG",
    3: "TURN_RIGHT_15_DEG",
}


DEFAULT_INTRINSIC = np.array([
    [192.0, 0.0, 191.42857143, 0.0],
    [0.0, 192.0, 191.42857143, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
], dtype=np.float32)


def json_response(handler, status, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def parse_json_field(form):
    if "json" not in form:
        return {}
    value = form["json"].value
    if not value:
        return {}
    return json.loads(value)


def read_file_field(form, name):
    if name not in form:
        return None, b""
    field = form[name]
    data = field.file.read()
    return field, data


def decode_rgb(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def resize_depth(depth, height, width):
    if depth.shape[:2] == (height, width):
        return depth
    depth_img = Image.fromarray(depth.squeeze())
    depth_img = depth_img.resize((width, height), Image.NEAREST)
    return np.asarray(depth_img)[..., None]


def decode_depth(depth_bytes, metadata, rgb_shape):
    if not depth_bytes:
        return None
    depth_img = Image.open(BytesIO(depth_bytes))
    depth = np.asarray(depth_img)
    if depth.ndim == 3:
        depth = depth[..., 0]

    raw_dtype = str(depth.dtype)
    depth = depth.astype(np.float32)
    depth_unit = str(metadata.get("depth_unit", "")).lower()

    # The current Limo test client sends depth.png as uint16 millimeters.
    # Float depth is treated as meters unless metadata says otherwise.
    if "mm" in depth_unit or (raw_dtype.startswith("uint") and depth.size and float(np.nanmax(depth)) > 50.0):
        depth = depth / 1000.0

    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth = np.maximum(depth, 0.0)[..., None]
    return resize_depth(depth, rgb_shape[0], rgb_shape[1]).astype(np.float32)


def parse_intrinsic(metadata):
    intrinsic = metadata.get("camera_intrinsic") or metadata.get("intrinsic")
    if intrinsic is None:
        return DEFAULT_INTRINSIC.copy()
    intrinsic = np.asarray(intrinsic, dtype=np.float32)
    if intrinsic.shape == (3, 3):
        expanded = np.eye(4, dtype=np.float32)
        expanded[:3, :3] = intrinsic
        return expanded
    if intrinsic.shape == (4, 4):
        return intrinsic
    raise ValueError("camera_intrinsic must be 3x3 or 4x4")


def action_names(actions):
    return [ACTION_NAMES.get(int(action), "UNKNOWN") for action in actions]


def save_upload(save_dir, image_bytes, depth_bytes, metadata, response_payload=None):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    upload_dir = Path(save_dir) / stamp
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved = {}
    if image_bytes:
        image_path = upload_dir / "rgb.jpg"
        image_path.write_bytes(image_bytes)
        saved["image"] = str(image_path)
    if depth_bytes:
        depth_path = upload_dir / "depth.png"
        depth_path.write_bytes(depth_bytes)
        saved["depth"] = str(depth_path)

    metadata_path = upload_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    saved["metadata"] = str(metadata_path)

    if response_payload is not None:
        response_path = upload_dir / "response.json"
        response_path.write_text(json.dumps(response_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        saved["response"] = str(response_path)
    return saved


class LimoStreamVLNHandler(BaseHTTPRequestHandler):
    server_version = "LimoStreamVLN/1.0"

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def do_GET(self):
        if self.path not in ("/", "/eval_vln"):
            json_response(self, 404, {"ok": False, "error": "not found"})
            return
        json_response(self, 200, {
            "ok": True,
            "service": "limo_streamvln_server",
            "endpoint": "/eval_vln",
            "model_path": self.server.args.model_path,
            "save_dir": self.server.args.save_dir,
            "action_semantics": ACTION_NAMES,
        })

    def do_POST(self):
        if self.path != "/eval_vln":
            json_response(self, 404, {"ok": False, "error": "not found"})
            return

        started = time.time()
        content_type = self.headers.get("content-type", "")
        if "multipart/form-data" not in content_type:
            json_response(self, 400, {"ok": False, "error": "expected multipart/form-data"})
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
            metadata = parse_json_field(form)
            image_field, image_bytes = read_file_field(form, "image")
            depth_field, depth_bytes = read_file_field(form, "depth")
            if image_field is None or not image_bytes:
                json_response(self, 400, {"ok": False, "error": "missing image file field"})
                return

            instruction = (
                metadata.get("prompt")
                or metadata.get("instruction")
                or metadata.get("task")
                or self.server.args.default_instruction
            )
            if not instruction:
                json_response(self, 400, {"ok": False, "error": "missing prompt/instruction in json"})
                return

            rgb = decode_rgb(image_bytes)
            depth = decode_depth(depth_bytes, metadata, rgb.shape)
            intrinsic = parse_intrinsic(metadata)
            reset = bool(metadata.get("reset", False))

            with self.server.infer_lock:
                if reset:
                    print("Reset StreamVLN memory for new Limo request stream", flush=True)
                    self.server.evaluator.reset_memory()
                    self.server.idx = 0
                    self.server.terminate = False
                    self.server.action_seq = []
                    self.server.last_llm_output = ""

                if self.server.terminate:
                    full_actions = [0]
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
                            generate_time = step_generate_time
                        if return_llm_output is not None:
                            llm_output = return_llm_output
                            self.server.last_llm_output = return_llm_output
                        if return_action is not None:
                            self.server.action_seq = [int(action) for action in return_action]
                        if 0 in self.server.action_seq:
                            self.server.terminate = True
                        self.server.evaluator.step_id += 1
                    full_actions = self.server.action_seq or [0]

            returned_actions = full_actions[:1] if self.server.args.return_single_action else full_actions
            payload = {
                "ok": True,
                "action": returned_actions,
                "action_names": action_names(returned_actions),
                "action_sequence": full_actions,
                "action_sequence_names": action_names(full_actions),
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
                payload["saved"] = save_upload(
                    self.server.args.save_dir,
                    image_bytes,
                    depth_bytes,
                    metadata,
                    response_payload=payload,
                )

            print("=" * 72, flush=True)
            print("Received Limo POST /eval_vln", flush=True)
            print("instruction:", instruction, flush=True)
            print("rgb_shape:", rgb.shape, "depth_shape:", None if depth is None else depth.shape, flush=True)
            print("action:", returned_actions, "sequence:", full_actions, flush=True)
            print("elapsed_ms:", payload["elapsed_ms"], flush=True)
            json_response(self, 200, payload)
        except Exception as exc:
            print("ERROR handling /eval_vln:", repr(exc), flush=True)
            json_response(self, 500, {"ok": False, "error": str(exc), "action": [0]})


def load_streamvln(args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=args.model_max_length,
        padding_side="right",
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        attn_implementation=args.attn_implementation,
        torch_dtype=torch.bfloat16,
        config=config,
        low_cpu_mem_usage=False,
    )
    model.model.num_history = args.num_history
    model.reset(1)
    model.requires_grad_(False)
    model.to(args.device)
    model.eval()

    evaluator = VLNEvaluator(
        {
            "rgb_height": args.rgb_height,
            "camera_intrinsic": DEFAULT_INTRINSIC.copy(),
        },
        model=model,
        tokenizer=tokenizer,
        args=args,
    )

    if args.warmup:
        evaluator.step(
            0,
            np.zeros((480, 640, 3), dtype=np.uint8),
            args.default_instruction or "move forward 25 cm",
            run_model=True,
        )
        evaluator.reset_memory()
    return evaluator


def parse_args():
    parser = argparse.ArgumentParser(description="Limo RGB-D StreamVLN HTTP server")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(STREAMVLN_ROOT / "data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world"),
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=6006)
    parser.add_argument("--save-dir", default=str(STREAMVLN_ROOT / "limo/uploads"))
    parser.add_argument("--default-instruction", default="Walk forward and stop when the task is complete.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attn-implementation", default="eager")
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--model_max_length", type=int, default=4096)
    parser.add_argument("--rgb-height", type=float, default=1.25)
    parser.add_argument("--steps-per-request", type=int, default=4)
    parser.add_argument("--run-every", type=int, default=4)
    parser.add_argument("--return-full-sequence", action="store_true")
    parser.add_argument("--include-llm-output", action="store_true")
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.set_defaults(warmup=True)
    args = parser.parse_args()
    args.return_single_action = not args.return_full_sequence
    return args


def main():
    args = parse_args()
    print("Loading StreamVLN model from", args.model_path, flush=True)
    evaluator = load_streamvln(args)

    server = ThreadingHTTPServer((args.host, args.port), LimoStreamVLNHandler)
    server.args = args
    server.evaluator = evaluator
    server.infer_lock = threading.Lock()
    server.idx = 0
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
