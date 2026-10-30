#!/usr/bin/env python3

import argparse
import copy
import itertools
import json
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import transformers
from PIL import Image, ImageDraw, ImageFont
from transformers.image_utils import to_numpy_array

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
STREAMVLN_DIR = os.path.join(PROJECT_ROOT, "streamvln")
for path in (PROJECT_ROOT, STREAMVLN_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import (
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IMAGE_TOKEN_INDEX,
    MEMORY_TOKEN_INDEX,
    dict_to_cuda,
)


ACTION_SYMBOLS = OrderedDict(
    {
        "STOP": 0,
        "↑": 1,
        "←": 2,
        "→": 3,
    }
)
ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}
ACTION_SHORT_NAMES = {
    0: "STOP",
    1: "FWD",
    2: "LEFT",
    3: "RIGHT",
}


def resolve_device(device_arg: str) -> torch.device:
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[name]


def auto_depth_video_path(video_path: str) -> Optional[str]:
    stem, ext = os.path.splitext(video_path)
    candidate = f"{stem}_depth{ext}"
    return candidate if os.path.exists(candidate) else None


def open_video(path: str):
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    return cap


def preprocess_depth_frame(
    depth_bgr: Optional[np.ndarray],
    target_size: Tuple[int, int],
) -> torch.Tensor:
    import cv2

    target_width, target_height = target_size
    if depth_bgr is None:
        arr = np.zeros((target_height, target_width), dtype=np.float32)
    else:
        if depth_bgr.ndim == 3:
            gray = cv2.cvtColor(depth_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = depth_bgr
        pil_depth = Image.fromarray(gray.astype(np.uint8)).resize(
            (target_width, target_height), Image.NEAREST
        )
        arr = to_numpy_array(pil_depth).astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    return torch.from_numpy(arr).float()


def make_intrinsic(width: int, height: int, hfov_degrees: float = 79.0) -> np.ndarray:
    fx = (width / 2.0) / np.tan(np.deg2rad(hfov_degrees / 2.0))
    fy = fx
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    return np.array(
        [
            [fx, 0.0, cx, 0.0],
            [0.0, fy, cy, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def make_fake_pose(step_id: int, forward_step: float = 0.25) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[2, 3] = -step_id * forward_step
    return pose


def preprocess_qwen(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    system_message: str = "You are a helpful assistant.",
    add_system: bool = False,
):
    roles = {"human": "user", "gpt": "assistant"}
    tokenizer = copy.deepcopy(tokenizer)
    if has_image:
        tokenizer.add_tokens(["<image>"], special_tokens=True)
        tokenizer.add_tokens(["<memory>"], special_tokens=True)

    image_token_index = tokenizer.convert_tokens_to_ids("<image>")
    memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    input_ids = []
    conversations = []
    for source in sources:
        prompt = "you can see " + DEFAULT_IMAGE_TOKEN
        if len(source[0]["value"]) != 0:
            source[0]["value"] += f" {prompt}."
        else:
            source[0]["value"] = f"{prompt}."

        if roles[source[0]["from"]] != roles["human"]:
            source = source[1:]

        input_id = []
        if add_system:
            input_id += tokenizer.apply_chat_template(
                [{"role": "system", "content": system_message}]
            )

        for conv in source:
            role = roles.get(conv.get("role", conv.get("from")), conv.get("role", conv.get("from")))
            content = conv.get("content", conv.get("value"))
            conversations.append(content)
            input_id += tokenizer.apply_chat_template(
                [{"role": role, "content": content}]
            )

        for idx, encode_id in enumerate(input_id):
            if encode_id == image_token_index:
                input_id[idx] = IMAGE_TOKEN_INDEX
            if encode_id == memory_token_index:
                input_id[idx] = MEMORY_TOKEN_INDEX
        input_ids.append(input_id)

    return torch.tensor(input_ids, dtype=torch.long), conversations


def parse_actions(text: str) -> List[int]:
    symbol_patterns = "|".join(re.escape(action) for action in ACTION_SYMBOLS)
    symbol_matches = re.findall(symbol_patterns, text)
    actions = [ACTION_SYMBOLS[item] for item in symbol_matches]
    if actions:
        return actions

    word_patterns = [
        (r"\bSTOP\b", 0),
        (r"\bMOVE\s+FORWARD\b|\bFORWARD\b", 1),
        (r"\bTURN\s+LEFT\b|\bLEFT\b", 2),
        (r"\bTURN\s+RIGHT\b|\bRIGHT\b", 3),
    ]
    matches = []
    upper = text.upper()
    for pattern, action in word_patterns:
        for match in re.finditer(pattern, upper):
            matches.append((match.start(), action))
    matches.sort(key=lambda item: item[0])
    return [action for _, action in matches]


def action_from_token_text(token_text: str) -> Optional[int]:
    upper = token_text.upper()
    if "STOP" in upper:
        return 0
    if "↑" in token_text or "FORWARD" in upper:
        return 1
    if "←" in token_text or "LEFT" in upper:
        return 2
    if "→" in token_text or "RIGHT" in upper:
        return 3
    return None


def first_token_ids(tokenizer, variants: List[str]) -> List[int]:
    ids = []
    for variant in variants:
        token_ids = tokenizer(variant, add_special_tokens=False).input_ids
        if token_ids:
            ids.append(token_ids[0])
    return sorted(set(ids))


def action_distribution_from_score(score: torch.Tensor, tokenizer) -> Dict[int, float]:
    if score is None:
        return {action_id: 0.0 for action_id in ACTION_NAMES}

    logits = score[0].float()
    variants = {
        0: ["STOP", " stop", "Stop"],
        1: ["↑", " MOVE", " move", " forward", " Forward"],
        2: ["←", " LEFT", " left", " Left"],
        3: ["→", " RIGHT", " right", " Right"],
    }
    action_logits = {}
    for action_id, action_variants in variants.items():
        ids = first_token_ids(tokenizer, action_variants)
        if ids:
            action_logits[action_id] = torch.logsumexp(logits[ids], dim=0)
        else:
            action_logits[action_id] = torch.tensor(-1e9, device=logits.device)

    stacked = torch.stack([action_logits[action_id] for action_id in [0, 1, 2, 3]])
    probs = torch.softmax(stacked, dim=0)
    return {action_id: float(probs[idx].item()) for idx, action_id in enumerate([0, 1, 2, 3])}


def action_distribution_from_scores(outputs, tokenizer) -> Dict[int, float]:
    if not getattr(outputs, "scores", None):
        return {action_id: 0.0 for action_id in ACTION_NAMES}
    return action_distribution_from_score(outputs.scores[0], tokenizer)


def generated_action_confidences(outputs, tokenizer) -> List[Tuple[int, float, Dict[int, float]]]:
    if not getattr(outputs, "scores", None):
        return []

    gen_len = len(outputs.scores)
    gen_ids = outputs.sequences[0, -gen_len:].detach().cpu().tolist()
    action_confidences = []
    for token_id, score in zip(gen_ids, outputs.scores):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        action_id = action_from_token_text(token_text)
        if action_id is None:
            continue
        dist = action_distribution_from_score(score, tokenizer)
        chosen_prob = dist.get(action_id, 0.0)
        action_confidences.append((action_id, chosen_prob, dist))
    return action_confidences


def resolve_font_path() -> str:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.otf",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/opentype/source-han-sans/SourceHanSansCN-Regular.otf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "fonts/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return ""


def load_font(size: int) -> ImageFont.ImageFont:
    font_path = resolve_font_path()
    if font_path:
        return ImageFont.truetype(font_path, size)
    return ImageFont.load_default()


def wrap_text(text: str, max_chars: int) -> List[str]:
    words = text.split()
    lines, line = [], ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if len(candidate) <= max_chars:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def annotate_frame(
    frame_rgb: np.ndarray,
    action_name: str,
    action_confidence: float,
    action_distribution: Dict[int, float],
    step_id: int,
    instruction: str,
    model_text: str,
    width: int,
    height: int,
) -> np.ndarray:
    import cv2

    resized = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_CUBIC)
    image = Image.fromarray(resized)
    draw = ImageDraw.Draw(image)
    panel_h = 156
    draw.rectangle((0, 0, width, panel_h), fill=(0, 0, 0))
    draw.rectangle((0, panel_h - 2, width, panel_h), fill=(80, 210, 130))

    font_large = load_font(24)
    font_mid = load_font(15)
    font_small = load_font(13)
    font_tiny = load_font(12)

    draw.text(
        (12, 8),
        f"ACTION: {action_name}   CONF: {action_confidence:.3f}",
        font=font_large,
        fill=(70, 255, 140),
    )
    draw.text(
        (width - 120, 14),
        f"step {step_id:04d}",
        font=font_mid,
        fill=(230, 230, 230),
    )

    dist_label = "REL PROB: " + "  ".join(
        f"{ACTION_SHORT_NAMES[action_id]}:{action_distribution.get(action_id, 0.0):.3f}"
        for action_id in [0, 1, 2, 3]
    )
    draw.text((12, 41), dist_label, font=font_mid, fill=(255, 235, 170))

    for idx, line in enumerate(wrap_text(instruction, 100)[:2]):
        draw.text(
            (12, 66 + idx * 20),
            line,
            font=font_small,
            fill=(210, 235, 255),
        )
    clean_text = re.sub(r"\s+", " ", model_text.replace("<|im_end|>", " ")).strip()
    for idx, line in enumerate(wrap_text(f"model: {clean_text}", 112)[:1]):
        draw.text(
            (12, 128 + idx * 18),
            line,
            font=font_tiny,
            fill=(235, 235, 235),
        )
    return np.array(image)


def build_prompt(instruction: str) -> List[Dict[str, str]]:
    prompt = (
        "<video>\n"
        "You are an autonomous navigation assistant. Your task is to "
        "<instruction>. Please devise an action sequence to follow the "
        "instruction which may include turning left or right by a certain "
        "degree, moving forward by a certain distance or stopping once the "
        "task is complete. Use the four actions: TURN LEFT (←), TURN RIGHT "
        "(→), MOVE FORWARD (↑), or STOP."
    )
    prompt = prompt.replace(DEFAULT_VIDEO_TOKEN + "\n", "")
    prompt = prompt.replace("<instruction>.", instruction)
    return [{"from": "human", "value": prompt}, {"from": "gpt", "value": ""}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_v1_3",
    )
    parser.add_argument("--video", default="data/my_test_data/test1.mp4")
    parser.add_argument("--depth_video", default=None)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output_dir", default="outputs/video_demo/test1")
    parser.add_argument("--frame_stride", type=int, default=5)
    parser.add_argument("--max_frames", type=int, default=120)
    parser.add_argument("--num_frames", type=int, default=4)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--num_future_steps", type=int, default=2)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--cpu_dtype",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
    )
    parser.add_argument("--output_fps", type=float, default=0.0)
    parser.add_argument("--display_width", type=int, default=768)
    parser.add_argument("--display_height", type=int, default=448)
    args = parser.parse_args()

    import cv2

    device = resolve_device(args.device)
    model_dtype = dtype_from_name(args.cpu_dtype) if device.type == "cpu" else torch.float16
    if args.depth_video is None:
        args.depth_video = auto_depth_video_path(args.video)

    os.makedirs(args.output_dir, exist_ok=True)
    actions_path = os.path.join(args.output_dir, "actions.json")
    video_path = os.path.join(args.output_dir, "annotated.avi")

    print(f"Using device: {device}")
    print(f"Using model dtype: {model_dtype}")
    print(f"Using video: {args.video}")
    print(f"Using depth video: {args.depth_video or 'zero depth'}")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_path,
        model_max_length=4096,
        padding_side="right",
    )
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    model = StreamVLNForCausalLM.from_pretrained(
        args.model_path,
        config=config,
        low_cpu_mem_usage=True,
        torch_dtype=model_dtype,
    )
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    model.to(device)
    model.eval()
    model.reset(1)

    image_processor = model.get_vision_tower().image_processor
    crop_size = image_processor.crop_size
    depth_size = (crop_size["width"], crop_size["height"])

    cap = open_video(args.video)
    depth_cap = open_video(args.depth_video) if args.depth_video else None
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    if args.output_fps > 0:
        fps_out = args.output_fps
    else:
        fps_out = fps_in / max(args.frame_stride, 1) if fps_in > 0 else 8
    writer = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps_out,
        (args.display_width, args.display_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {video_path}")

    rgb_list: List[torch.Tensor] = []
    depth_list: List[torch.Tensor] = []
    pose_list: List[torch.Tensor] = []
    intrinsic_list: List[torch.Tensor] = []
    time_ids: List[int] = []
    pending_actions: List[Tuple[int, float, Dict[int, float]]] = []
    predictions = []
    output_ids = None
    past_key_values = None
    latest_model_text = ""
    latest_action_distribution = {action_id: 0.0 for action_id in ACTION_NAMES}
    raw_frame_idx = 0
    used_frame_count = 0

    with torch.no_grad():
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            depth_bgr = None
            if depth_cap is not None:
                depth_ok, depth_bgr = depth_cap.read()
                if not depth_ok:
                    break

            if raw_frame_idx % args.frame_stride != 0:
                raw_frame_idx += 1
                continue

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb).convert("RGB")
            image = image_processor.preprocess(
                images=pil_image, return_tensors="pt"
            )["pixel_values"][0]
            image_size = pil_image.size
            depth = preprocess_depth_frame(depth_bgr, depth_size)
            pose = torch.from_numpy(make_fake_pose(used_frame_count)).float()
            intrinsic = torch.from_numpy(make_intrinsic(*image_size)).float()

            time_ids.append(used_frame_count)
            rgb_list.append(image)
            depth_list.append(depth)
            pose_list.append(pose)
            intrinsic_list.append(intrinsic)

            if not pending_actions:
                if output_ids is None:
                    sources = build_prompt(args.instruction)
                    add_system = True
                else:
                    sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                    add_system = False

                input_ids, _ = preprocess_qwen(
                    [sources], tokenizer, True, add_system=add_system
                )
                if output_ids is not None:
                    input_ids = torch.cat(
                        [output_ids, input_ids.to(output_ids.device)], dim=1
                    )

                images = rgb_list[-1:]
                depths = depth_list[-1:]
                poses = pose_list[-1:]
                intrinsics = intrinsic_list[-1:]
                if (
                    len(time_ids) > 1
                    and used_frame_count != 0
                    and used_frame_count % args.num_frames == 0
                ):
                    if args.num_history <= 0:
                        history_step = max(1, args.num_future_steps)
                    else:
                        history_step = max(1, time_ids[0] // args.num_history)
                    history_ids = slice(0, time_ids[0], history_step)
                    images = rgb_list[history_ids] + images
                    depths = depth_list[history_ids] + depths
                    poses = pose_list[history_ids] + poses
                    intrinsics = intrinsic_list[history_ids] + intrinsics

                input_dict = {
                    "images": torch.stack(images).unsqueeze(0),
                    "depths": torch.stack(depths).unsqueeze(0),
                    "poses": torch.stack(poses).unsqueeze(0),
                    "intrinsics": torch.stack(intrinsics).unsqueeze(0),
                    "inputs": input_ids,
                    "env_id": 0,
                    "time_ids": [time_ids],
                    "task_type": [0],
                }
                input_dict = dict_to_cuda(input_dict, device)
                for key in ["images", "depths", "poses", "intrinsics"]:
                    input_dict[key] = input_dict[key].to(model_dtype)

                # StreamVLN assumes the first `num_history` views are memory
                # frames and the remaining views are current observations.
                # In this offline demo the available history can be shorter
                # than the requested value, especially near the beginning.
                model.model.num_history = max(0, len(images) - 1)

                outputs = model.generate(
                    **input_dict,
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_scores=True,
                    past_key_values=past_key_values,
                )
                output_ids = outputs.sequences
                past_key_values = outputs.past_key_values
                latest_model_text = tokenizer.batch_decode(
                    output_ids, skip_special_tokens=False
                )[0].strip()
                latest_action_distribution = action_distribution_from_scores(
                    outputs, tokenizer
                )
                parsed_actions = parse_actions(latest_model_text)
                action_confidences = generated_action_confidences(
                    outputs, tokenizer
                )
                pending_actions = []
                for action_idx, action_id in enumerate(parsed_actions):
                    if action_idx < len(action_confidences):
                        matched_action_id, confidence, dist = action_confidences[action_idx]
                    else:
                        matched_action_id, confidence, dist = (
                            action_id,
                            latest_action_distribution.get(action_id, 0.0),
                            latest_action_distribution,
                        )
                    if matched_action_id == action_id:
                        action_dist = dist
                    else:
                        confidence = latest_action_distribution.get(action_id, 0.0)
                        action_dist = latest_action_distribution
                    pending_actions.append((action_id, confidence, action_dist))
                if not pending_actions:
                    fallback_action = 0
                    pending_actions = [
                        (
                            fallback_action,
                            latest_action_distribution.get(fallback_action, 0.0),
                            latest_action_distribution,
                        )
                    ]

            action_id, action_confidence, action_distribution = pending_actions.pop(0)
            action_name = ACTION_NAMES[action_id]
            predictions.append(
                {
                    "raw_frame": raw_frame_idx,
                    "step": used_frame_count,
                    "action_id": action_id,
                    "action": action_name,
                    "action_confidence": action_confidence,
                    "next_action_distribution": {
                        ACTION_NAMES[k]: v
                        for k, v in action_distribution.items()
                    },
                    "model_text": latest_model_text,
                }
            )

            annotated = annotate_frame(
                frame_rgb,
                action_name,
                action_confidence,
                action_distribution,
                used_frame_count,
                args.instruction,
                latest_model_text,
                args.display_width,
                args.display_height,
            )
            writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

            used_frame_count += 1
            raw_frame_idx += 1

            if used_frame_count % args.num_frames == 0:
                model.reset_for_env(0)
                output_ids = None
                past_key_values = None
                time_ids = []
                pending_actions = []

            if args.max_frames > 0 and used_frame_count >= args.max_frames:
                break

    cap.release()
    if depth_cap is not None:
        depth_cap.release()
    writer.release()

    with open(actions_path, "w") as f:
        json.dump(
            {
                "video": args.video,
                "depth_video": args.depth_video,
                "instruction": args.instruction,
                "frame_stride": args.frame_stride,
                "actions": predictions,
            },
            f,
            indent=2,
        )

    print(f"Processed frames: {used_frame_count}")
    print(f"Actions: {actions_path}")
    print(f"Annotated video: {video_path}")


if __name__ == "__main__":
    main()
