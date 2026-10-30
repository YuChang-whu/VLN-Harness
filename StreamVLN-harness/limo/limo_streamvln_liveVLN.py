#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import json
import re
import threading
import time

import cv2
import numpy as np
import requests
import rospy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image


ACTION_STOP = 0
ACTION_FORWARD = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3


class AsyncResult(object):
    def __init__(self):
        self.event = threading.Event()
        self.response = None
        self.error = None
        self.started_at = time.time()
        self.finished_at = None

    def set_response(self, response):
        self.response = response
        self.finished_at = time.time()
        self.event.set()

    def set_error(self, error):
        self.error = error
        self.finished_at = time.time()
        self.event.set()

    def wait(self):
        self.event.wait()
        if self.error is not None:
            raise self.error
        return self.response


class LimoStreamVLNLiveVLN(object):
    def __init__(self, args):
        self.args = args
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.step_id = 0

        self.cmd_pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
        rospy.Subscriber(args.rgb_topic, Image, self.rgb_cb, queue_size=1)
        rospy.Subscriber(args.depth_topic, Image, self.depth_cb, queue_size=1)

    def rgb_cb(self, msg):
        with self.lock:
            self.latest_rgb = msg

    def depth_cb(self, msg):
        with self.lock:
            self.latest_depth = msg

    def get_messages(self):
        with self.lock:
            return self.latest_rgb, self.latest_depth

    def wait_for_images(self):
        rospy.loginfo("Waiting for RGB topic: %s", self.args.rgb_topic)
        rospy.loginfo("Waiting for depth topic: %s", self.args.depth_topic)
        start = time.time()
        while not rospy.is_shutdown():
            rgb_msg, depth_msg = self.get_messages()
            if rgb_msg is not None and depth_msg is not None:
                rospy.loginfo("Got RGB and depth messages.")
                return
            if time.time() - start > self.args.wait_timeout:
                raise RuntimeError("timeout waiting for RGB/depth topics")
            rospy.sleep(0.1)

    def encode_rgb_jpeg(self, msg):
        rgb_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        ok, encoded = cv2.imencode(
            ".jpg", rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), self.args.jpeg_quality]
        )
        if not ok:
            raise RuntimeError("failed to encode RGB image")
        return encoded.tostring()

    def encode_depth_png(self, msg):
        depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_np = np.asarray(depth)

        if depth_np.dtype == np.float32 or depth_np.dtype == np.float64:
            depth_mm = np.nan_to_num(depth_np) * 1000.0
            depth_mm = np.clip(depth_mm, 0, 65535).astype(np.uint16)
            depth_unit = "uint16_mm_from_float_m"
        else:
            depth_mm = np.clip(depth_np, 0, 65535).astype(np.uint16)
            depth_unit = "uint16_original"

        ok, encoded = cv2.imencode(".png", depth_mm)
        if not ok:
            raise RuntimeError("failed to encode depth image")
        return encoded.tostring(), depth_unit, str(depth_np.dtype), list(depth_np.shape)

    def build_post_payload(self, reset, phase, executed_guard=None):
        rgb_msg, depth_msg = self.get_messages()
        if rgb_msg is None or depth_msg is None:
            raise RuntimeError("RGB/depth message is not ready")

        rgb_bytes = self.encode_rgb_jpeg(rgb_msg)
        depth_bytes, depth_unit, depth_dtype, depth_shape = self.encode_depth_png(depth_msg)

        metadata = {
            "reset": reset,
            "phase": phase,
            "step_id": self.step_id,
            "timestamp": time.time(),
            "rgb_topic": self.args.rgb_topic,
            "depth_topic": self.args.depth_topic,
            "depth_unit": depth_unit,
            "depth_dtype": depth_dtype,
            "depth_shape": depth_shape,
        }
        if executed_guard is not None:
            metadata["executed_guard"] = int(executed_guard)
        if self.args.instruction:
            metadata["instruction"] = self.args.instruction

        files = {
            "image": ("rgb.jpg", rgb_bytes, "image/jpeg"),
            "depth": ("depth.png", depth_bytes, "image/png"),
        }
        data = {"json": json.dumps(metadata)}
        return files, data, len(rgb_bytes), len(depth_bytes)

    def post_observation(self, reset, phase, executed_guard=None):
        files, data, rgb_size, depth_size = self.build_post_payload(
            reset=reset, phase=phase, executed_guard=executed_guard
        )
        rospy.loginfo(
            "Step %d POST %s phase=%s (RGB bytes: %d, depth bytes: %d)",
            self.step_id,
            self.args.server,
            phase,
            rgb_size,
            depth_size,
        )
        response = requests.post(
            self.args.server,
            files=files,
            data=data,
            timeout=self.args.timeout,
        )
        response.raise_for_status()
        if self.args.print_response:
            print("HTTP status:", response.status_code)
            print("Response text:")
            print(response.text)
        return response

    def start_background_request(self, reset, phase, executed_guard=None):
        result = AsyncResult()

        def worker():
            try:
                response = self.post_observation(
                    reset=reset, phase=phase, executed_guard=executed_guard
                )
                result.set_response(response)
            except Exception as exc:
                result.set_error(exc)

        thread = threading.Thread(target=worker)
        thread.daemon = True
        thread.start()
        return result

    def parse_action_id(self, value):
        if value is None:
            return ACTION_STOP
        if isinstance(value, (int, np.integer)):
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        if isinstance(value, (list, tuple)):
            if not value:
                return ACTION_STOP
            return self.parse_action_id(value[0])
        if isinstance(value, dict):
            for key in ("action_e", "guard_g", "action"):
                if key in value:
                    return self.parse_action_id(value[key])
            return ACTION_STOP

        text = str(value).strip().lower()
        if not text:
            return ACTION_STOP
        if text[0] in "[{" and text[-1] in "]}":
            try:
                return self.parse_action_id(json.loads(text))
            except Exception:
                pass
        if "stop" in text:
            return ACTION_STOP
        if "forward" in text or "straight" in text:
            return ACTION_FORWARD
        if "left" in text:
            return ACTION_LEFT
        if "right" in text:
            return ACTION_RIGHT
        match = re.search(r"-?\d+", text)
        if match:
            return int(match.group())
        return ACTION_STOP

    def parse_action_pair(self, response):
        payload = response.json()

        if "action_e" in payload or "guard_g" in payload:
            action_e = self.parse_action_id(payload.get("action_e", ACTION_STOP))
            guard_g = self.parse_action_id(payload.get("guard_g", ACTION_STOP))
        else:
            actions = payload.get("action", [ACTION_STOP])
            if not isinstance(actions, list):
                actions = [actions]
            action_e = self.parse_action_id(actions[0] if actions else ACTION_STOP)
            guard_g = self.parse_action_id(actions[1] if len(actions) > 1 else ACTION_STOP)

        rospy.loginfo(
            "Parsed action pair: action_e=%s guard_g=%s", action_e, guard_g
        )
        return action_e, guard_g, payload

    def publish_stop(self):
        self.cmd_pub.publish(Twist())

    def action_to_twist(self, action):
        twist = Twist()
        if action == ACTION_FORWARD:
            twist.linear.x = self.args.forward_speed
        elif action == ACTION_LEFT:
            twist.angular.z = self.args.turn_speed
        elif action == ACTION_RIGHT:
            twist.angular.z = -self.args.turn_speed
        elif action == ACTION_STOP:
            pass
        else:
            rospy.logwarn("Unknown action %s -> stop robot.", action)

        twist.linear.x = max(min(twist.linear.x, self.args.max_linear), -self.args.max_linear)
        twist.angular.z = max(min(twist.angular.z, self.args.max_angular), -self.args.max_angular)
        return twist

    def execute_action(self, action, label):
        twist = self.action_to_twist(action)
        rospy.loginfo(
            "Step %d execute %s action %s for %.2fs: linear.x=%.3f angular.z=%.3f",
            self.step_id,
            label,
            action,
            self.args.action_duration,
            twist.linear.x,
            twist.angular.z,
        )

        if self.args.dry_run:
            rospy.sleep(self.args.action_duration)
            return

        end_time = time.time() + self.args.action_duration
        rate = rospy.Rate(self.args.cmd_rate)
        while not rospy.is_shutdown() and time.time() < end_time:
            self.cmd_pub.publish(twist)
            rate.sleep()

        self.publish_stop()
        if self.args.stop_pause > 0:
            rospy.sleep(self.args.stop_pause)

    def wait_future_and_parse(self, future):
        wait_start = time.time()
        response = future.wait()
        wait_time = time.time() - wait_start
        total_time = time.time() - future.started_at
        rospy.loginfo(
            "Background request finished: total=%.3fs waited_after_guard=%.3fs",
            total_time,
            wait_time,
        )
        return self.parse_action_pair(response)

    def spin(self):
        self.wait_for_images()

        # Bootstrap: no guard is available yet, so this first request is blocking.
        rospy.loginfo("Bootstrap request: waiting for action_e and guard_g.")
        response = self.post_observation(reset=self.args.reset_first, phase="bootstrap")
        action_e, guard_g, _ = self.parse_action_pair(response)

        if action_e == ACTION_STOP:
            rospy.loginfo("Bootstrap action_e is STOP. Stop and exit.")
            self.publish_stop()
            return

        self.execute_action(action_e, "action_e")
        if self.args.once:
            return

        while not rospy.is_shutdown():
            if guard_g == ACTION_STOP:
                rospy.loginfo("guard_g is STOP. Stop and exit.")
                self.publish_stop()
                return

            # LiveVLN core: refresh next plan in the background while executing
            # the committed guard action from the previous plan.
            future = self.start_background_request(
                reset=False, phase="refresh", executed_guard=guard_g
            )
            self.execute_action(guard_g, "guard_g")

            try:
                next_action_e, next_guard_g, _ = self.wait_future_and_parse(future)
            except Exception as exc:
                rospy.logerr("Background refresh failed: %s", exc)
                self.publish_stop()
                rospy.sleep(self.args.error_pause)
                if self.args.stop_on_error:
                    return
                continue

            if self.args.check_prefix and next_action_e != guard_g:
                rospy.logwarn(
                    "Prefix mismatch: executed guard_g=%s but next action_e=%s. "
                    "Stopping for safety.",
                    guard_g,
                    next_action_e,
                )
                self.publish_stop()
                if self.args.stop_on_prefix_mismatch:
                    return

            guard_g = next_guard_g
            self.step_id += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="POST URL, e.g. http://127.0.0.1:6009/eval_vln")
    parser.add_argument("--rgb-topic", default="/camera/rgb/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image")
    parser.add_argument("--instruction", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--wait-timeout", type=float, default=15.0)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    parser.add_argument("--once", action="store_true", default=False)
    parser.add_argument("--reset-first", action="store_true", default=True)
    parser.add_argument("--cmd-topic", default="/cmd_vel")
    parser.add_argument("--action-duration", type=float, default=0.5)
    parser.add_argument("--forward-speed", type=float, default=0.25)
    parser.add_argument("--turn-speed", type=float, default=0.35)
    parser.add_argument("--cmd-rate", type=float, default=10.0)
    parser.add_argument("--max-linear", type=float, default=0.30)
    parser.add_argument("--max-angular", type=float, default=0.6)
    parser.add_argument("--stop-pause", type=float, default=0.05)
    parser.add_argument("--error-pause", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true", default=False)
    parser.add_argument("--print-response", action="store_true", default=False)
    parser.add_argument("--check-prefix", action="store_true", default=False)
    parser.add_argument("--stop-on-prefix-mismatch", action="store_true", default=True)
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    return parser.parse_args(rospy.myargv()[1:])


def main():
    args = parse_args()
    rospy.init_node("limo_streamvln_liveVLN")
    controller = LimoStreamVLNLiveVLN(args)
    rospy.on_shutdown(controller.publish_stop)
    controller.spin()


if __name__ == "__main__":
    main()
