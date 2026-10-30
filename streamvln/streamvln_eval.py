import sys
import os
STREAMVLN_PACKAGE_DIR = os.path.abspath(os.path.dirname(__file__))
STREAMVLN_ROOT = os.path.abspath(os.path.join(STREAMVLN_PACKAGE_DIR, ".."))
sys.path.insert(0, STREAMVLN_ROOT)
sys.path.insert(0, STREAMVLN_PACKAGE_DIR)
import re
import tqdm
import torch
import copy
import json
import random
import argparse
import itertools
import transformers
import numpy as np

from typing import Any
from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFile, ImageFont
from collections import OrderedDict
from torch.nn.utils.rnn import pad_sequence
from depth_camera_filtering import filter_depth
from transformers.image_utils import to_numpy_array

import habitat
from habitat import logger, Env
from habitat_extensions import measures
from habitat_baselines.config.default import get_config as get_habitat_config

try:
    from habitat.config.default_structured_configs import (
        CollisionsMeasurementConfig,
        FogOfWarConfig,
        TopDownMapMeasurementConfig,
    )
except ImportError:
    CollisionsMeasurementConfig = None
    FogOfWarConfig = None
    TopDownMapMeasurementConfig = None

from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video, observations_to_image
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from model.stream_video_vln import StreamVLNForCausalLM
from utils.utils import dict_to_cuda
from utils.dist import *
from utils.utils import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, DEFAULT_VIDEO_TOKEN


def quaternion_to_rotation_matrix(rotation):
    if isinstance(rotation, torch.Tensor):
        rotation = rotation.detach().cpu().numpy()
    if hasattr(rotation, 'w') and hasattr(rotation, 'x') and hasattr(rotation, 'y') and hasattr(rotation, 'z'):
        qx, qy, qz, qw = rotation.x, rotation.y, rotation.z, rotation.w
    else:
        rotation = np.asarray(rotation, dtype=np.float32)
        if rotation.shape == (4,):
            qx, qy, qz, qw = rotation
        else:
            qx, qy, qz, qw = rotation[0], rotation[1], rotation[2], rotation[3]
    return np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ], dtype=np.float32)


def build_navigation_prompt(instruction: str) -> str:
    return (
        "You are an autonomous navigation assistant. Your task is to "
        f"{instruction.strip()} Devise an action sequence to follow the "
        "instruction using the four actions: TURN LEFT (left arrow) or "
        "TURN RIGHT (right arrow) by 15 degrees, MOVE FORWARD (up arrow) "
        "by 25 centimeters, or STOP."
    )


def normalize_prompt_action_language(instruction: str):
    normalized = instruction
    rewrites = []

    replacements = [
        (
            r"\bturn left\b",
            "adjust your heading left until the path is aligned, then move forward",
            "turn left -> discrete heading adjustment",
        ),
        (
            r"\bturn right\b",
            "adjust your heading right until the path is aligned, then move forward",
            "turn right -> discrete heading adjustment",
        ),
        (
            r"\bgo around\b",
            "navigate around the obstacle or furniture while continuing along the open path",
            "go around -> obstacle/furniture circumnavigation",
        ),
        (
            r"\baround the bed\b|\baround bed\b",
            "around the bed by following the open floor next to the bed, turning only as needed, then continue forward",
            "around bed -> open-floor circumnavigation",
        ),
        (
            r"\bdoorway\b",
            "doorway threshold",
            "doorway -> doorway threshold",
        ),
        (
            r"\bwait near\b",
            "stop when you are within the goal area near",
            "wait near -> explicit STOP near goal",
        ),
        (
            r"\bstop beside\b",
            "stop when you are beside",
            "stop beside -> explicit STOP beside goal",
        ),
        (
            r"\bstairs\b",
            "stairs area; align with the navigable stair path before moving forward",
            "stairs -> align before moving",
        ),
        (
            r"\bhallway\b",
            "hallway corridor",
            "hallway -> corridor grounding",
        ),
    ]

    for pattern, replacement, label in replacements:
        updated = re.sub(pattern, replacement, normalized, flags=re.IGNORECASE)
        if updated != normalized:
            rewrites.append(label)
            normalized = updated

    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized, rewrites


def load_video_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        os.path.join(os.environ.get("CONDA_PREFIX", ""), "fonts/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def wrap_text_to_width(text: str, font: ImageFont.ImageFont, max_width: int) -> list:
    words = text.split()
    if not words:
        return [""]
    measure = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        bbox = measure.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def annotate_video_frames(frames, prompt: str, demo_index: int, metrics: dict):
    if not frames:
        return frames
    font_title = load_video_font(18)
    font_prompt = load_video_font(16)
    annotated_frames = []
    title = (
        f"Demo {demo_index} | SPL: {metrics['spl']:.3f} | "
        f"Success: {metrics['success']:.1f} | NE: {metrics['distance_to_goal']:.3f}"
    )
    for frame in frames:
        image = Image.fromarray(frame).convert("RGB")
        width, height = image.size
        prompt_lines = wrap_text_to_width(f"Prompt: {prompt}", font_prompt, width - 24)[:4]
        panel_height = 14 + 24 + 6 + len(prompt_lines) * 21 + 12
        canvas = Image.new("RGB", (width, height + panel_height), (0, 0, 0))
        canvas.paste(image, (0, panel_height))
        draw = ImageDraw.Draw(canvas)
        draw.text((12, 10), title, font=font_title, fill=(120, 255, 160))
        y = 40
        for line in prompt_lines:
            draw.text((12, y), line, font=font_prompt, fill=(235, 235, 235))
            y += 21
        annotated_frames.append(np.asarray(canvas))
    return annotated_frames


ACTION_ID_TO_NAME = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}


def load_episode_filter(path):
    if not path:
        return None
    with open(path, "r") as f:
        data = json.load(f)
    if isinstance(data, dict):
        if "episodes" in data:
            rows = data["episodes"]
        elif "selected_episodes" in data:
            rows = data["selected_episodes"]
        else:
            rows = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else []
    elif isinstance(data, list):
        rows = data
    else:
        rows = []

    allowed = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        scene_id = row.get("scene_id")
        episode_id = row.get("episode_id")
        if scene_id is not None and episode_id is not None:
            allowed.add((str(scene_id), str(episode_id)))
    return allowed


def get_config_value(config, path, default=None):
    value = config
    for part in path.split("."):
        try:
            value = getattr(value, part)
        except Exception:
            try:
                value = value[part]
            except Exception:
                return default
    return value


def recent_distance_stagnated(distance_history, window, min_delta):
    valid = [float(value) for value in distance_history if value is not None and np.isfinite(value)]
    if len(valid) < window:
        return False, None
    recent = valid[-window:]
    improvement = recent[0] - recent[-1]
    return improvement <= min_delta, improvement


def parse_action_list(text_value, default):
    if not text_value:
        return list(default)
    name_to_action = {
        "STOP": 0,
        "MOVE_FORWARD": 1,
        "FORWARD": 1,
        "TURN_LEFT": 2,
        "LEFT": 2,
        "TURN_RIGHT": 3,
        "RIGHT": 3,
    }
    actions = []
    for item in str(text_value).split(','):
        item = item.strip().upper()
        if not item:
            continue
        if item in name_to_action:
            actions.append(name_to_action[item])
        else:
            actions.append(int(item))
    return actions or list(default)


def get_episode_goal_position(episode):
    goals = getattr(episode, "goals", None)
    if goals:
        position = getattr(goals[0], "position", None)
        if position is not None:
            return np.asarray(position, dtype=np.float32)
    reference_path = getattr(episode, "reference_path", None)
    if reference_path:
        return np.asarray(reference_path[-1], dtype=np.float32)
    return None


def get_oracle_action_record(follower, episode):
    goal_position = get_episode_goal_position(episode)
    if follower is None or goal_position is None:
        return {
            "oracle_action": None,
            "oracle_action_name": None,
            "oracle_goal_position": to_jsonable(goal_position),
            "oracle_error": "missing_follower_or_goal",
        }
    try:
        oracle_action = follower.get_next_action(goal_position)
    except Exception as exc:
        return {
            "oracle_action": None,
            "oracle_action_name": None,
            "oracle_goal_position": to_jsonable(goal_position),
            "oracle_error": repr(exc),
        }
    if oracle_action is not None and not isinstance(oracle_action, (int, np.integer)):
        oracle_action = int(np.asarray(oracle_action).argmax())
    if oracle_action is not None:
        oracle_action = int(oracle_action)
    return {
        "oracle_action": oracle_action,
        "oracle_action_name": ACTION_ID_TO_NAME.get(oracle_action, str(oracle_action)),
        "oracle_goal_position": to_jsonable(goal_position),
        "oracle_error": None,
    }


def to_jsonable(value):
    if isinstance(value, torch.Tensor):
        return to_jsonable(value.detach().cpu().tolist())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def compact_metrics(metrics):
    compact = {}
    for key in [
        "success",
        "spl",
        "oracle_success",
        "distance_to_goal",
        "oracle_navigation_error",
    ]:
        if key in metrics:
            compact[key] = to_jsonable(metrics[key])
    collisions = metrics.get("collisions")
    if collisions is not None:
        if isinstance(collisions, dict):
            compact["collisions"] = to_jsonable(collisions)
        else:
            compact["collisions"] = to_jsonable(collisions)
    return compact


def action_ids_to_names(actions):
    return [ACTION_ID_TO_NAME.get(int(action), str(action)) for action in actions]


def write_episode_log(output_path, episode_log):
    log_dir = os.path.join(output_path, "episode_logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{episode_log['episode_index']:04d}_{episode_log['scene_id']}_{episode_log['episode_id']}.json"
    log_path = os.path.join(log_dir, filename)
    with open(log_path, "w") as f:
        json.dump(to_jsonable(episode_log), f, indent=2)
    return log_path


class VLNEvaluator:
    def __init__(
        self,
        config_path: str,
        split: str = "val_seen",
        env_num: int = 8,
        output_path: str = None,
        model: Any = None,
        tokenizer: Any = None,
        epoch: int = 0,
        args: argparse.Namespace = None,
    ):
        self.args = args
        requested_device = getattr(args, 'device', 'cpu')
        if requested_device == 'cuda' and not torch.cuda.is_available():
            requested_device = 'cpu'
        self.device = torch.device(requested_device)
        self.split = split
        self.env_num = env_num
        self.save_video = args.save_video
        self.output_path = output_path
        self.epoch = epoch
        self.max_episodes = getattr(args, 'max_episodes', None)
        self.skip_episodes = getattr(args, 'skip_episodes', 0)
        self.record_oracle_actions = getattr(args, 'record_oracle_actions', False)
        self.limit_consecutive_turns = getattr(args, 'limit_consecutive_turns', False)
        self.max_consecutive_turns = getattr(args, 'max_consecutive_turns', 6)
        self.turn_limit_replacement_action = getattr(args, 'turn_limit_replacement_action', 1)
        self.enable_stuck_detector = getattr(args, 'enable_stuck_detector', False)
        self.stuck_forward_steps = getattr(args, 'stuck_forward_steps', 3)
        self.stuck_min_displacement = getattr(args, 'stuck_min_displacement', 0.03)
        self.stuck_recovery_actions = parse_action_list(getattr(args, 'stuck_recovery_actions', None), [2, 2, 1])
        self.stop_verifier = getattr(args, 'stop_verifier', False)
        self.stop_verifier_window = getattr(args, 'stop_verifier_window', 5)
        self.stop_verifier_min_delta = getattr(args, 'stop_verifier_min_delta', 0.05)
        self.stop_verifier_replacement_action = getattr(args, 'stop_verifier_replacement_action', 1)
        self.prompt_action_normalization = getattr(args, 'prompt_action_normalization', False)
        self.episode_filter_file = getattr(args, 'episode_filter_file', None)
        self.episode_filter = load_episode_filter(self.episode_filter_file)
        self.config_path = config_path
        self.config = get_habitat_config(config_path)
        config_success_distance = get_config_value(
            self.config, 'habitat.task.measurements.success.success_distance', 3.0
        )
        self.oracle_success_distance = getattr(args, 'oracle_success_distance', None)
        if self.oracle_success_distance is None:
            self.oracle_success_distance = float(config_success_distance)
        self.agent_config = None
        self.sim_sensors_config = self.config.habitat.simulator.agents.main_agent.sim_sensors

        if hasattr(self.config, "defrost"):
            self.config.defrost()
        else:
            self.config = OmegaConf.create(OmegaConf.to_container(self.config, resolve=True))
        self.config.habitat.dataset.split = self.split
        if TopDownMapMeasurementConfig is not None and FogOfWarConfig is not None and CollisionsMeasurementConfig is not None:
            self.config.habitat.task.measurements.update(
                {
                    "top_down_map": TopDownMapMeasurementConfig(
                        map_padding=3,
                        map_resolution=1024,
                        draw_source=True,
                        draw_border=True,
                        draw_shortest_path=True,
                        draw_view_points=True,
                        draw_goal_positions=True,
                        draw_goal_aabbs=True,
                        fog_of_war=FogOfWarConfig(
                            draw=True,
                            visibility_dist=5.0,
                            fov=90,
                        ),
                    ),
                    "collisions": CollisionsMeasurementConfig(),
                }
            )

        print(f"config = {type(self.config)}")
        try:
            print(OmegaConf.to_yaml(self.config))
        except Exception as e:
            print(f"OmegaConf serialization skipped: {e}")
            print(self.config)

        rgb_sensor_cfg = self.sim_sensors_config.rgb_sensor
        depth_sensor_cfg = self.sim_sensors_config.depth_sensor

        if hasattr(rgb_sensor_cfg, "position"):
            try:
                self._camera_height = rgb_sensor_cfg.position[1]
            except Exception:
                self._camera_height = getattr(rgb_sensor_cfg, "position", [0, 0, 0])[1]
        else:
            self._camera_height = getattr(rgb_sensor_cfg, "height", 0) / 2.0

        self._min_depth = depth_sensor_cfg.min_depth
        self._max_depth = depth_sensor_cfg.max_depth

        camera_fov_rad = np.deg2rad(depth_sensor_cfg.hfov)
        self._camera_fov = camera_fov_rad
        self._fx = self._fy = self.sim_sensors_config.depth_sensor.width / (2 * np.tan(camera_fov_rad / 2))
        self.image_processor = model.get_vision_tower().image_processor
        self.model = model
        self.tokenizer = tokenizer
        prompt = f"<video>\nYou are an autonomous navigation assistant. Your task is to <instruction>. Devise an action sequence to follow the instruction using the four actions: TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, MOVE FORWARD (↑) by 25 centimeters, or STOP."
        answer = ""
        self.conversation = [{"from": "human", "value": prompt}, {"from": "gpt", "value": answer}]
        self.actions2idx = OrderedDict({
            'STOP': [0],
            "↑": [1],
            "←": [2],
            "→": [3]
        })
        self.conjunctions = [
                                'you can see ',
                                'in front of you is ',
                                'there is ',
                                'you can spot ',
                                'you are toward the ',
                                'ahead of you is ',
                                'in your sight is '
                            ]
        self.num_frames = args.num_frames
        self.num_future_steps = args.num_future_steps
        self.num_history = args.num_history
        self.max_new_tokens = getattr(args, 'max_new_tokens', 512)
        
    
    def preprocess_depth_image(self, depth_image, do_depth_scale=True, depth_scale=1000):
        target_height = self.image_processor.crop_size['height']  # 384
        target_width  = self.image_processor.crop_size['width']  # 384
        resized_depth_image = depth_image.resize((target_width, target_height), Image.NEAREST)
        
        img = to_numpy_array(resized_depth_image)
        if do_depth_scale:
            img = img / depth_scale
    
        return img, (target_width, target_height)
    
    def get_intrinsic_matrix(self, sensor_cfg) -> np.ndarray:
        width = sensor_cfg.width
        height = sensor_cfg.height
        fov = sensor_cfg.hfov
        fx = (width / 2.0) / np.tan(np.deg2rad(fov / 2.0))
        fy = fx  # Assuming square pixels (fx = fy)
        cx = (width - 1.0) / 2.0
        cy = (height - 1.0) / 2.0

        intrinsic_matrix = np.array([
            [fx,  0.0, cx, 0.0],
            [ 0.0, fy, cy, 0.0],
            [ 0.0,  0.0,  1.0, 0.0],
            [ 0.0,  0.0,  0.0, 1.0]
        ])
        return intrinsic_matrix
    
    def preprocess_instrinsic(self, intrinsic, ori_size, target_size):  # (V, 4, 4) (resize_shape) (h, w)
        intrinsic = copy.deepcopy(intrinsic)
        if len(intrinsic.shape) == 2:
            intrinsic = intrinsic[None, :, :]  # (1, 4, 4) or (B, 4, 4)
        
        intrinsic[:, 0] /= ori_size[0] / target_size[0]  # width
        intrinsic[:, 1] /= ori_size[1] / target_size[1]  # height

        # for crop transform
        intrinsic[:, 0, 2] -= (target_size[0] - target_size[1]) / 2

        if intrinsic.shape[0] == 1:
            intrinsic = intrinsic.squeeze(0)

        return intrinsic
    
    def get_axis_align_matrix(self):
        # ma = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        ma = torch.tensor([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 0, 1]]).double()
        return ma
    
    def xyz_yaw_to_tf_matrix(self, xyz: np.ndarray, yaw: float) -> np.ndarray:
        x, y, z = xyz
        transformation_matrix = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0, x],
                [np.sin(yaw), np.cos(yaw), 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )
        return transformation_matrix

    def config_env(self) -> Env:
        env = Env(config=self.config)
        # env.episodes = env.episodes[0:1]
        return env

    def eval_action(self, idx) -> None:
        env = self.config_env()
        oracle_follower = None
        if self.record_oracle_actions:
            oracle_follower = ShortestPathFollower(
                env.sim,
                goal_radius=self.oracle_success_distance,
                return_one_hot=False,
                stop_on_error=True,
            )
        scene_episode_dict = {}
        for episode in env.episodes:
            if episode.scene_id not in scene_episode_dict:
                scene_episode_dict[episode.scene_id] = []
            scene_episode_dict[episode.scene_id].append(episode)

        intrinsic_matrix = self.get_intrinsic_matrix(self.config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor)
        sucs, spls, oss, ones = [], [], [], []
        done_res = []
        if os.path.exists(os.path.join(self.output_path, f'result.json')):
            with open(os.path.join(self.output_path, f'result.json'),'r') as f:
                for line in f.readlines():
                    res = json.loads(line)
                    if "scene_id" not in res or "episode_id" not in res:
                        continue
                    done_res.append([res["scene_id"], res["episode_id"], res["episode_instruction"]])
                    if get_rank() == 0:
                        sucs.append(res['success'])
                        spls.append(res['spl'])
                        oss.append(res['os'])
                        ones.append(res['ne'])
        processed_episodes = 0
        skipped_episodes = 0
        for scene in sorted(scene_episode_dict.keys()):
            episodes = scene_episode_dict[scene]
            scene_id = scene.split('/')[-2]
            print(f"scene_id = {scene_id}")
            # episode_id = 0
            process_bar = tqdm.tqdm(range(len(episodes[idx::self.env_num])), desc=f"scene {scene_id}")
            for episode in episodes[idx::self.env_num]:
                episode_instruction = episode.instruction.instruction_text if 'objectnav' not in self.config_path else episode.object_category
                episode_id = episode.episode_id
                if self.episode_filter is not None and (str(scene_id), str(episode_id)) not in self.episode_filter:
                    continue
                if [scene_id, episode_id, episode_instruction] in done_res:
                    continue
                if skipped_episodes < self.skip_episodes:
                    skipped_episodes += 1
                    continue
                if self.max_episodes is not None and processed_episodes >= self.max_episodes:
                    break
                model_instruction, prompt_action_rewrites = (
                    normalize_prompt_action_language(episode_instruction)
                    if self.prompt_action_normalization
                    else (episode_instruction, [])
                )
                print("episode start", episode_instruction)
                if self.prompt_action_normalization:
                    print("normalized instruction", model_instruction)
                episode_index = self.skip_episodes + processed_episodes + 1
                episode_log = {
                    "episode_index": episode_index,
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "split": self.split,
                    "instruction": episode_instruction,
                    "model_instruction": model_instruction,
                    "prompt_action_normalized": self.prompt_action_normalization,
                    "prompt_action_rewrites": prompt_action_rewrites,
                    "prompt": build_navigation_prompt(model_instruction),
                    "config": {
                        "num_frames": self.num_frames,
                        "num_history": self.num_history,
                        "num_future_steps": self.num_future_steps,
                        "skip_episodes": self.skip_episodes,
                        "max_episodes": self.max_episodes,
                        "record_oracle_actions": self.record_oracle_actions,
                        "oracle_success_distance": self.oracle_success_distance,
                        "limit_consecutive_turns": self.limit_consecutive_turns,
                        "max_consecutive_turns": self.max_consecutive_turns,
                        "turn_limit_replacement_action": self.turn_limit_replacement_action,
                        "enable_stuck_detector": self.enable_stuck_detector,
                        "stuck_forward_steps": self.stuck_forward_steps,
                        "stuck_min_displacement": self.stuck_min_displacement,
                        "stuck_recovery_actions": self.stuck_recovery_actions,
                        "stop_verifier": self.stop_verifier,
                        "stop_verifier_window": self.stop_verifier_window,
                        "stop_verifier_min_delta": self.stop_verifier_min_delta,
                        "stop_verifier_replacement_action": self.stop_verifier_replacement_action,
                        "episode_filter_file": self.episode_filter_file,
                        "prompt_action_normalization": self.prompt_action_normalization,
                    },
                    "generations": [],
                    "steps": [],
                }
                self.model.reset_for_env(idx)
                env.current_episode = episode
                observations = env.reset()
                os.makedirs(os.path.join(self.output_path, f'check_sim_{self.epoch}'), exist_ok=True)
                Image.fromarray(observations['rgb']).save(os.path.join(self.output_path, f'check_sim_{self.epoch}', f'rgb_{idx}.jpg'))
                
                vis_frames = []
                step_id = 0
                
                if self.save_video:
                    os.makedirs(os.path.join(self.output_path, f'vis_{self.epoch}', f'{scene_id}_{episode_id}'), exist_ok=True)
                initial_height = env.sim.get_agent_state().position[1]

                rgb_list = []
                depth_list = []
                depth_images_list = []
                pose_list = []
                intrinsic_list = []
                time_ids = []
                action_seq = []
                recovery_action_seq = []
                past_key_values = None
                output_ids = None
                last_gps_before_action = None
                last_executed_action = None
                small_forward_displacement_count = 0
                consecutive_turn_count = 0
                last_turn_action = None
                distance_history = []
                postprocess_summary = {
                    "limit_turn_interventions": 0,
                    "stuck_recovery_interventions": 0,
                    "stop_verifier_blocks": 0,
                    "stop_verifier_allows": 0,
                    "oracle_stop_rescue_candidates": 0,
                }
                while not env.episode_over:
                    self.model.eval()
                    time_ids.append(step_id)
                    rgb = observations["rgb"]
                    depth = observations["depth"]
                    x, y = observations["gps"]
                    camera_yaw = observations["compass"][0]
                    depth = filter_depth(depth.reshape(depth.shape[:2]), blur_type=None)
                    depth = depth * (self._max_depth - self._min_depth) + self._min_depth
                    depth = depth * 1000

                    agent_state = env.sim.get_agent_state()
                    height = agent_state.position[1] - initial_height # Habitat GPS makes west negative, so flip y
                    camera_position = np.array([x, -y, self._camera_height + height])
                    robot_xy = camera_position[:2]
                    tf_camera_to_episodic = self.xyz_yaw_to_tf_matrix(camera_position, camera_yaw)
                    
                    rotation = agent_state.rotation
                    translation = agent_state.position
                    rotation_matrix = quaternion_to_rotation_matrix(rotation)
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation_matrix
                    transformation_matrix[:3, 3] = translation
                    
                    image = Image.fromarray(rgb).convert('RGB')
                    image_size = image.size
                    # image = self.image_processor.preprocess(images=image, do_rescale=True, do_normalize=True, return_tensors='pt')['pixel_values'][0]
                    image = self.image_processor.preprocess(images=image, return_tensors='pt')['pixel_values'][0]
                    depth_image, resize_shape = self.preprocess_depth_image(Image.fromarray(depth.astype(np.uint16), mode='I;16'), do_depth_scale=True)
                    
                    intrinsic = self.preprocess_instrinsic(intrinsic_matrix, image_size, resize_shape)
                    intrinsic = torch.from_numpy(intrinsic).float()
    
                    rgb_list.append(image)
                    depth_list.append(torch.from_numpy(depth_image).float())
                    pose_list.append(torch.from_numpy(tf_camera_to_episodic) @ self.get_axis_align_matrix())
                    intrinsic_list.append(intrinsic)
                    
                    info = env.get_metrics()
                    step_pre_metrics = compact_metrics(info)
                    current_distance_to_goal = step_pre_metrics.get("distance_to_goal")
                    distance_history.append(current_distance_to_goal)
                    stuck_forward_displacement = None
                    stuck_recovery_started = False
                    if self.enable_stuck_detector and last_executed_action == 1 and last_gps_before_action is not None:
                        current_gps = np.asarray([x, y], dtype=np.float32)
                        stuck_forward_displacement = float(np.linalg.norm(current_gps - last_gps_before_action))
                        if stuck_forward_displacement < self.stuck_min_displacement:
                            small_forward_displacement_count += 1
                        else:
                            small_forward_displacement_count = 0
                        if small_forward_displacement_count >= self.stuck_forward_steps and not recovery_action_seq:
                            recovery_action_seq = list(self.stuck_recovery_actions)
                            small_forward_displacement_count = 0
                            stuck_recovery_started = True
                    if info['top_down_map'] is not None:
                        frame = observations_to_image({'rgb':observations['rgb']}, info)
                        vis_frames.append(frame)
                    # import ipdb; ipdb.set_trace()
                    generation_record = None
                    if len(action_seq) == 0:
                        prompt_uses_history = output_ids is None and step_id != 0 and step_id % self.num_frames == 0
                        if output_ids is None:
                            sources = copy.deepcopy(self.conversation)
                            sources[0]["value"] = sources[0]["value"].replace(' Where should you go next to stay on track?', f' Please devise an action sequence to follow the instruction which may include turning left or right by a certain degree, moving forward by a certain distance or stopping once the task is complete.')
                            if prompt_uses_history:
                                sources[0]["value"] += f' These are your historical observations {DEFAULT_MEMORY_TOKEN}.'
                            sources[0]["value"] = sources[0]["value"].replace(DEFAULT_VIDEO_TOKEN+'\n', '')
                            sources[0]["value"] = sources[0]["value"].replace('<instruction>.', model_instruction)
                            add_system = True
                            print(step_id, sources[0]["value"])
                        else:
                            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
                            add_system = False
                            
                        prompt_for_generation = sources[0]["value"] if sources else ""
                        input_ids, conversations = self.preprocess_qwen([sources], self.tokenizer, True, add_system=add_system)
                        if output_ids is not None:
                            input_ids = torch.cat([output_ids,input_ids.to(output_ids.device)], dim=1)
                        use_history = prompt_uses_history or bool((input_ids == MEMORY_TOKEN_INDEX).any().item())

                        images = rgb_list[-1:]
                        depths = depth_list[-1:]
                        poses = pose_list[-1:]
                        intrinsics = intrinsic_list[-1:]
                        # import ipdb; ipdb.set_trace()
                        if use_history:
                            if self.num_history is None:
                                history_ids = slice(0, time_ids[0], self.num_future_steps)
                            else:
                                history_ids = slice(0, time_ids[0], max(1, time_ids[0] // self.num_history))
                            images = rgb_list[history_ids] + images
                            depths = depth_list[history_ids] + depths
                            poses = pose_list[history_ids] + poses
                            intrinsics = intrinsic_list[history_ids] + intrinsics
                        self.model.model.num_history = max(0, len(images) - 1)

                                
                        input_dict = {'images':torch.stack(images).unsqueeze(0), 'depths':torch.stack(depths).unsqueeze(0), \
                                        'poses':torch.stack(poses).unsqueeze(0), 'intrinsics':torch.stack(intrinsics).unsqueeze(0), 'inputs':input_ids, 'env_id':idx, 'time_ids':[time_ids],'task_type':[0]}
                            
                        input_dict = dict_to_cuda(input_dict, self.device)
                        
                        model_dtype = next(self.model.parameters()).dtype
                        for key, value in input_dict.items():
                            if key in ['images', 'depths', 'poses', 'intrinsics']:
                                input_dict[key] = input_dict[key].to(model_dtype)
                        
                        outputs = self.model.generate(
                            **input_dict,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=self.max_new_tokens,
                            use_cache=True,
                            return_dict_in_generate=True,
                            past_key_values=past_key_values,
                            num_logits_to_keep=1,
                        )
                        
                        output_ids = outputs.sequences
                        past_key_values = outputs.past_key_values
                        llm_outputs = self.tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0].strip()
                        print(llm_outputs, flush=True)
                        action_seq = self.parse_actions(llm_outputs)
                        print('actions', action_seq, flush=True)
                        fallback_to_stop = False
                        if len(action_seq) == 0: ## if generated llm without Specific values
                            action_seq = [0]
                            fallback_to_stop = True
                        generation_record = {
                            "step_id": step_id,
                            "prompt": prompt_for_generation,
                            "conversations": conversations,
                            "llm_output": llm_outputs,
                            "parsed_actions": list(action_seq),
                            "parsed_action_names": action_ids_to_names(action_seq),
                            "fallback_to_stop": fallback_to_stop,
                            "num_input_views": len(images),
                            "actual_num_history": self.model.model.num_history,
                            "time_ids": list(time_ids),
                            "pre_metrics": step_pre_metrics,
                        }
                        episode_log["generations"].append(generation_record)
                    queued_actions_before_pop = list(action_seq)
                    recovery_queue_before_pop = list(recovery_action_seq)
                    postprocess_interventions = []
                    action_source = "model"
                    model_proposed_action = None
                    stop_verifier_decision = None
                    stop_verifier_recent_improvement = None
                    oracle_stop_rescue_candidate = False

                    if recovery_action_seq:
                        model_proposed_action = action_seq[0] if action_seq else None
                        action = recovery_action_seq.pop(0)
                        action_source = "stuck_recovery"
                        postprocess_interventions.append("stuck_recovery")
                        postprocess_summary["stuck_recovery_interventions"] += 1
                    else:
                        action = action_seq.pop(0)
                        model_proposed_action = action

                        if self.stop_verifier and action == 0:
                            stagnated, recent_improvement = recent_distance_stagnated(
                                distance_history, self.stop_verifier_window, self.stop_verifier_min_delta
                            )
                            stop_verifier_recent_improvement = recent_improvement
                            close_enough = (
                                current_distance_to_goal is not None
                                and float(current_distance_to_goal) <= self.oracle_success_distance
                            )
                            if close_enough or stagnated:
                                stop_verifier_decision = "allow_stop"
                                postprocess_summary["stop_verifier_allows"] += 1
                            else:
                                action = self.stop_verifier_replacement_action
                                action_source = "stop_verifier"
                                stop_verifier_decision = "block_stop_continue"
                                postprocess_interventions.append("stop_verifier_block_stop")
                                postprocess_summary["stop_verifier_blocks"] += 1

                        if (
                            self.limit_consecutive_turns
                            and action in (2, 3)
                            and consecutive_turn_count + 1 > self.max_consecutive_turns
                        ):
                            postprocess_interventions.append("limit_consecutive_turns")
                            postprocess_summary["limit_turn_interventions"] += 1
                            action = self.turn_limit_replacement_action
                            action_source = "turn_limit"

                    if (
                        self.stop_verifier
                        and model_proposed_action != 0
                        and current_distance_to_goal is not None
                        and float(current_distance_to_goal) <= self.oracle_success_distance
                    ):
                        oracle_stop_rescue_candidate = True
                        postprocess_summary["oracle_stop_rescue_candidates"] += 1

                    oracle_record = get_oracle_action_record(oracle_follower, episode) if self.record_oracle_actions else {}
                    oracle_action = oracle_record.get("oracle_action")
                    model_matches_oracle = oracle_action is not None and model_proposed_action is not None and int(model_proposed_action) == int(oracle_action)

                    turn_run_before_update = consecutive_turn_count
                    observations = env.step(action)
                    step_post_metrics = compact_metrics(env.get_metrics())
                    if int(action) in (2, 3):
                        if last_turn_action in (2, 3):
                            consecutive_turn_count += 1
                        else:
                            consecutive_turn_count = 1
                        last_turn_action = int(action)
                    else:
                        consecutive_turn_count = 0
                        last_turn_action = None
                    last_gps_before_action = np.asarray([x, y], dtype=np.float32)
                    last_executed_action = int(action)
                    episode_log["steps"].append({
                        "step_id": step_id,
                        "gps": to_jsonable(observations.get("gps")) if isinstance(observations, dict) else None,
                        "compass": to_jsonable(observations.get("compass")) if isinstance(observations, dict) else None,
                        "camera_position": to_jsonable(camera_position),
                        "camera_yaw": to_jsonable(camera_yaw),
                        "action": int(action),
                        "action_name": ACTION_ID_TO_NAME.get(int(action), str(action)),
                        "model_action": int(model_proposed_action) if model_proposed_action is not None else None,
                        "model_action_name": ACTION_ID_TO_NAME.get(int(model_proposed_action), str(model_proposed_action)) if model_proposed_action is not None else None,
                        "action_source": action_source,
                        "postprocess_interventions": postprocess_interventions,
                        "overrode_model_action": model_proposed_action is not None and int(model_proposed_action) != int(action),
                        "recovery_queue_before_pop": recovery_queue_before_pop,
                        "remaining_recovery_actions_after_pop": list(recovery_action_seq),
                        "turn_run_before_update": turn_run_before_update,
                        "turn_run_after_update": consecutive_turn_count,
                        "stuck_forward_displacement": stuck_forward_displacement,
                        "stuck_small_forward_count": small_forward_displacement_count,
                        "stuck_recovery_started": stuck_recovery_started,
                        "stop_verifier_decision": stop_verifier_decision,
                        "stop_verifier_recent_improvement": stop_verifier_recent_improvement,
                        "oracle_stop_rescue_candidate": oracle_stop_rescue_candidate,
                        "oracle_action": oracle_record.get("oracle_action"),
                        "oracle_action_name": oracle_record.get("oracle_action_name"),
                        "oracle_goal_position": oracle_record.get("oracle_goal_position"),
                        "oracle_error": oracle_record.get("oracle_error"),
                        "model_matches_oracle": model_matches_oracle,
                        "queued_actions_before_pop": queued_actions_before_pop,
                        "queued_action_names_before_pop": action_ids_to_names(queued_actions_before_pop),
                        "remaining_actions_after_pop": list(action_seq),
                        "generated_this_step": generation_record is not None,
                        "generation_index": len(episode_log["generations"]) - 1 if generation_record is not None else None,
                        "pre_metrics": step_pre_metrics,
                        "post_metrics": step_post_metrics,
                    })
                    step_id += 1
                    if step_id % self.num_frames == 0:
                        self.model.reset_for_env(idx)
                        output_ids = None
                        past_key_values = None
                        time_ids = []
                        
                process_bar.update(1)
                # episode_id += 1
                metrics = env.get_metrics()
                video_file = None
                if self.save_video:
                    video_index = self.skip_episodes + processed_episodes + 1
                    video_name = f"{video_index}_{scene_id}_{episode_id}_spl_{metrics['spl']:.3f}"
                    prompt_text = build_navigation_prompt(model_instruction)
                    annotated_frames = annotate_video_frames(vis_frames, prompt_text, video_index, metrics)
                    images_to_video(
                        annotated_frames, os.path.join(self.output_path, f'vis_{self.epoch}'), video_name, fps=6, quality=9
                    )
                    video_file = os.path.join(self.output_path, f'vis_{self.epoch}', f'{video_name}.mp4')
                vis_frames.clear()
                sucs.append(metrics['success'])
                spls.append(metrics['spl'])
                oss.append(metrics['oracle_success'])
                ones.append(metrics['distance_to_goal'])
                print(f"scene_episode {scene_id}_{episode_id} success: {metrics['success']}, spl: {metrics['spl']}, os: {metrics['oracle_success']}, ne: {metrics['distance_to_goal']}")
                episode_log["final_metrics"] = compact_metrics(metrics)
                episode_log["video_file"] = video_file
                episode_log["num_steps"] = step_id
                episode_log["postprocess_summary"] = postprocess_summary
                episode_log_path = write_episode_log(self.output_path, episode_log)
                result = {
                    "scene_id": scene_id,
                    "episode_id": episode_id,
                    "success": metrics["success"],
                    "spl": metrics["spl"],
                    "os": metrics['oracle_success'],
                    "ne": metrics["distance_to_goal"],
                    "steps": step_id,
                    "episode_instruction": episode_instruction,
                    "model_instruction": model_instruction,
                    "prompt_action_normalized": self.prompt_action_normalization,
                    "prompt_action_rewrites": prompt_action_rewrites,
                    "postprocess_summary": postprocess_summary,
                    "video_file": video_file,
                    "episode_log": episode_log_path
                }
                
                with open(os.path.join(self.output_path, f'result.json'), 'a') as f:
                    f.write(json.dumps(result) + "\n")
                processed_episodes += 1

            if self.max_episodes is not None and processed_episodes >= self.max_episodes:
                break

        env.close()
        return torch.tensor(sucs).to(self.device), torch.tensor(spls).to(self.device), torch.tensor(oss).to(self.device), torch.tensor(ones).to(self.device), torch.tensor(len(sucs)).to(self.device)     

    def parse_actions(self, output):
        action_patterns = '|'.join(re.escape(action) for action in self.actions2idx)
        # import ipdb; ipdb.set_trace()
        regex = re.compile(action_patterns)
        matches = regex.findall(output)
        actions = [self.actions2idx[match] for match in matches]
        actions = itertools.chain.from_iterable(actions)
        return list(actions)



    def preprocess_qwen(self, sources, tokenizer: transformers.PreTrainedTokenizer, has_image: bool = False, max_len=2048, system_message: str = "You are a helpful assistant.",add_system: bool = False):
        # roles = {"human": "<|im_start|>user", "gpt": "<|im_start|>assistant"}
        roles = {"human": "user", "gpt": "assistant"}
        # import ipdb; ipdb.set_trace()
        # Add image tokens to tokenizer as a special tokens
        # Use a deepcopy of tokenizer so that we don't modify on the tokenizer
        tokenizer = copy.deepcopy(tokenizer)
        # When there is actually an image, we add the image tokens as a special token
        if has_image:
            tokenizer.add_tokens(["<image>"], special_tokens=True)
            tokenizer.add_tokens(["<memory>"], special_tokens=True)

        image_token_index = tokenizer.convert_tokens_to_ids("<image>")
        memory_token_index = tokenizer.convert_tokens_to_ids("<memory>")
        im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
        im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
        # unmask_tokens = ["<|im_start|>", "<|im_start|>", "\n"]
        unmask_tokens_idx =  [198, im_start, im_end]
        nl_tokens = tokenizer("\n").input_ids

        # Reset Qwen chat templates so that it won't include system message every time we apply
        chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        tokenizer.chat_template = chat_template

        # _system = tokenizer("system").input_ids + nl_tokens
        # _user = tokenizer("user").input_ids + nl_tokens
        # _assistant = tokenizer("assistant").input_ids + nl_tokens

        # Apply prompt templates
        conversations = []
        input_ids = []
        for i, source in enumerate(sources):
            prompt = random.choice(self.conjunctions) + DEFAULT_IMAGE_TOKEN
            if len(source[0]["value"]) != 0:
                source[0]["value"] += f" {prompt}."
            else: 
                source[0]["value"] = f"{prompt}."
            if roles[source[0]["from"]] != roles["human"]:
                # Skip the first one if it is not from human
                source = source[1:]

            input_id, target = [], []

            # import ipdb; ipdb.set_trace()
            # New version, use apply chat template
            # Build system message for each sentence
            if add_system:
                input_id += tokenizer.apply_chat_template([{"role" : "system", "content" : system_message}])

            for conv in source:
                # Make sure llava data can load
                try:
                    role = conv["role"]
                    content = conv["content"]
                except:
                    role = conv["from"]
                    content = conv["value"]

                role =  roles.get(role, role)
                
                conv = [{"role" : role, "content" : content}]
                # import ipdb; ipdb.set_trace()
                conversations.append(content)
                encode_id = tokenizer.apply_chat_template(conv)
                input_id += encode_id
            

            # assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
            for idx, encode_id in enumerate(input_id):
                if encode_id == image_token_index:
                    input_id[idx] = IMAGE_TOKEN_INDEX
                if encode_id == memory_token_index:
                    input_id[idx] = MEMORY_TOKEN_INDEX
                    
            input_ids.append(input_id)
        input_ids = torch.tensor(input_ids, dtype=torch.long)

        return input_ids,  conversations # tensor(bs x seq_len)

def pad_tensors(tensors, lens=None, max_len=None, pad=0):
    """B x [T, ...]"""
    if lens is None:
        lens = [t.size(0) for t in tensors]
        if len(lens) == 1 and lens[0] == max_len:
            return tensors
    if max_len is None:
        max_len = max(lens)
    bs = len(tensors)
    hid = tensors[0].shape[1:]
    dtype = tensors[0].dtype
    output = torch.zeros(bs, max_len, *hid, dtype=dtype).to(tensors[0].device)
    if pad:
        output.data.fill_(pad)
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output.data[i, :l, ...] = t.data
    return output
   
def eval():
    global local_rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--habitat_config_path", type=str, default='config/vln_r2r.yaml')
    parser.add_argument("--eval_split", type=str, default='val_unseen')
    parser.add_argument("--output_path", type=str, default='./results/val_unseen/streamvln')
    parser.add_argument("--num_future_steps", type=int, default=4)
    parser.add_argument("--num_frames", type=int, default=32)
    parser.add_argument("--save_video", action="store_true", default=False)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--max_episodes", type=int, default=None, help="Limit evaluation to a small number of episodes for quick demos")
    parser.add_argument("--skip_episodes", type=int, default=0, help="Skip this many episodes before evaluating, useful for batched demos")
    parser.add_argument("--record_oracle_actions", action="store_true", default=False, help="Record Habitat shortest-path expert action at every step")
    parser.add_argument("--oracle_success_distance", type=float, default=None, help="Goal radius for shortest-path follower; defaults to Habitat success_distance")
    parser.add_argument("--limit_consecutive_turns", action="store_true", default=False, help="Replace turn actions once consecutive turns exceed the threshold")
    parser.add_argument("--max_consecutive_turns", type=int, default=6)
    parser.add_argument("--turn_limit_replacement_action", type=int, default=1, help="Action inserted when turn limit is hit; default MOVE_FORWARD")
    parser.add_argument("--enable_stuck_detector", action="store_true", default=False, help="Trigger recovery when repeated MOVE_FORWARD produces tiny GPS displacement")
    parser.add_argument("--stuck_forward_steps", type=int, default=3)
    parser.add_argument("--stuck_min_displacement", type=float, default=0.03)
    parser.add_argument("--stuck_recovery_actions", type=str, default="2,2,1", help="Comma-separated recovery actions, default TURN_LEFT,TURN_LEFT,MOVE_FORWARD")
    parser.add_argument("--stop_verifier", action="store_true", default=False, help="Gate STOP by recent distance trend and record oracle STOP rescue candidates")
    parser.add_argument("--stop_verifier_window", type=int, default=5)
    parser.add_argument("--stop_verifier_min_delta", type=float, default=0.05)
    parser.add_argument("--stop_verifier_replacement_action", type=int, default=1)
    parser.add_argument("--episode_filter_file", type=str, default=None, help="JSON file listing scene_id/episode_id pairs to evaluate")
    parser.add_argument("--prompt_action_normalization", action="store_true", default=False, help="Rewrite action-like language into discrete-control-friendly navigation instructions")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum new tokens per action generation; lower values save GPU memory")
    parser.add_argument("--model_max_length", type=int, default=4096,
                        help= "Maximum sequence length. Sequences will be right padded (and possibly truncated).")
    
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--rank', default=0, type=int,
                        help='rank')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--port', default='1111')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--device', default='cpu',
                        help='device to use for training / testing')
    parser.add_argument(
        '--cpu_dtype',
        default='bfloat16',
        choices=['float32', 'bfloat16', 'float16'],
        help='dtype used when loading the model on CPU. bfloat16 uses much less RAM than float32.'
    )
    
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        args.device = 'cpu'
    init_distributed_mode(args)
    local_rank = args.local_rank

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path,
                                                        model_max_length=args.model_max_length,
                                                        padding_side="right")
    
    config = transformers.AutoConfig.from_pretrained(args.model_path)
    cpu_dtype_map = {
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
        'float16': torch.float16,
    }
    model_dtype = cpu_dtype_map[args.cpu_dtype] if args.device == 'cpu' else torch.float16
    model_kwargs = {
        "config": config,
        "low_cpu_mem_usage": True,
        "torch_dtype": model_dtype,
    }
    if args.device != 'cpu':
        try:
            import flash_attn  # noqa: F401
        except Exception:
            model_kwargs["attn_implementation"] = "sdpa"
        else:
            model_kwargs["attn_implementation"] = "flash_attention_2"
    model = StreamVLNForCausalLM.from_pretrained(args.model_path, **model_kwargs)
    model.model.num_history = args.num_history
    model.requires_grad_(False)
    if args.device == 'cpu':
        model.to(torch.device('cpu'))
    else:
        model.to(torch.device(args.device))
    evaluate(model, tokenizer, args)



def evaluate(model, tokenizer, args):
    model.eval()
    
    world_size = get_world_size()
    model.reset(world_size)
    evaluator = VLNEvaluator(
        config_path=args.habitat_config_path,
        split=args.eval_split,
        env_num=world_size,
        output_path=args.output_path,
        model=model,
        tokenizer=tokenizer,
        epoch=0,
        args=args
    )
    sucs, spls, oss, ones, ep_num = evaluator.eval_action(get_rank())

    if world_size == 1:
        result_all = {
            "sucs_all": sucs.float().mean().item(),
            "spls_all": spls.float().mean().item(),
            "oss_all": oss.float().mean().item(),
            "ones_all": ones.float().mean().item(),
            'length': int(ep_num.item())
        }
    else:
        ep_num_all = [torch.zeros_like(ep_num) for _ in range(world_size)]
        dist.all_gather(ep_num_all, ep_num)
        sucs_all = [torch.zeros(ep_num_all[i], dtype=sucs.dtype).to(sucs.device) for i in range(world_size)]
        spls_all = [torch.zeros(ep_num_all[i], dtype=spls.dtype).to(spls.device) for i in range(world_size)]
        oss_all = [torch.zeros(ep_num_all[i], dtype=oss.dtype).to(oss.device) for i in range(world_size)]
        ones_all = [torch.zeros(ep_num_all[i], dtype=ones.dtype).to(ones.device) for i in range(world_size)]
        dist.barrier()
        dist.all_gather(sucs_all, sucs)
        dist.all_gather(spls_all, spls)
        dist.all_gather(oss_all, oss)
        dist.all_gather(ones_all, ones)
        dist.barrier()
        sucs_all = torch.cat(sucs_all, dim=0)
        spls_all = torch.cat(spls_all, dim=0)
        oss_all = torch.cat(oss_all, dim=0)
        ones_all = torch.cat(ones_all, dim=0)
        result_all = {
                        "sucs_all": (sum(sucs_all)/len(sucs_all)).item(),
                        "spls_all": (sum(spls_all)/len(spls_all)).item(),
                        "oss_all": (sum(oss_all)/len(oss_all)).item(),
                        "ones_all": (sum(ones_all)/len(ones_all)).item(),
                        'length': len(sucs_all)
                    }
    
    print(result_all)
    if get_rank() == 0:
        with open(os.path.join(args.output_path, f'result.json'), 'a') as f:
            f.write(json.dumps(result_all))

if __name__ == "__main__":
    eval()
