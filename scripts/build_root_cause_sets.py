#!/usr/bin/env python3
"""Build three root-cause diagnostic failure sets for StreamVLN eval outputs.

The sets are intentionally non-exclusive: one failed episode can belong to more
than one diagnostic set if multiple symptoms are present.
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from statistics import mean

ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

KEYWORD_PATTERNS = {
    "left": r"\bleft\b",
    "right": r"\bright\b",
    "turn": r"\bturn\b|\bveer\b|\bpivot\b",
    "straight/forward": r"\bstraight\b|\bforward\b|\bahead\b",
    "back/around": r"\bback\b|\baround\b|\bbehind\b",
    "door/doorway": r"\bdoor\b|\bdoorway\b|\bentryway\b|\bentrance\b",
    "hall/hallway": r"\bhall\b|\bhallway\b|\bcorridor\b",
    "stairs": r"\bstair\b|\bstairs\b|\bstaircase\b",
    "room": r"\broom\b",
    "bathroom": r"\bbathroom\b|\btoilet\b",
    "bedroom/bed": r"\bbedroom\b|\bbed\b",
    "kitchen": r"\bkitchen\b",
    "living room": r"\bliving room\b|\bfamily room\b",
    "dining": r"\bdining\b|\btable\b",
    "office/desk": r"\boffice\b|\bdesk\b",
    "closet": r"\bcloset\b",
    "couch/sofa": r"\bcouch\b|\bsofa\b",
    "chair": r"\bchair\b",
    "cabinet": r"\bcabinet\b|\bcupboard\b",
    "fireplace": r"\bfireplace\b",
    "window": r"\bwindow\b",
    "painting/picture": r"\bpainting\b|\bpicture\b|\bphoto\b",
}

STOPWORDS = {
    "the", "and", "to", "a", "an", "of", "in", "on", "at", "by", "for",
    "from", "then", "with", "your", "you", "are", "is", "be", "go", "walk",
    "move", "towards", "toward", "until", "into", "out", "through", "past",
    "continue", "stop", "near", "next", "after", "before", "as", "it", "this",
    "that", "there", "where", "should", "follow", "instruction", "using",
}


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def safe_mean(values):
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def load_results(output_dir):
    path = os.path.join(output_dir, "result.json")
    if not os.path.exists(path):
        raise SystemExit(f"No result.json found: {path}")
    rows = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scene_id" in item and "episode_id" in item:
                rows.append(item)
    return rows


def load_episode_log(result, output_dir):
    log_path = result.get("episode_log")
    candidates = []
    if log_path:
        candidates.append(log_path)
        candidates.append(os.path.join(output_dir, log_path))
    log_dir = os.path.join(output_dir, "episode_logs")
    if os.path.isdir(log_dir):
        suffix = f"_{result.get('scene_id')}_{result.get('episode_id')}.json"
        candidates.extend(
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.endswith(suffix)
        )
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            with open(candidate, "r") as f:
                return json.load(f), candidate
    return None, None


def actions_from_log(log):
    actions = []
    if not log:
        return actions
    for step in log.get("steps", []):
        action = step.get("model_action", step.get("action"))
        try:
            actions.append(int(action))
        except Exception:
            pass
    return actions


def distances_from_log(log):
    distances = []
    if not log:
        return distances
    for step in log.get("steps", []):
        for source in ("post_metrics", "pre_metrics"):
            metrics = step.get(source) or {}
            distance = as_float(metrics.get("distance_to_goal"))
            if distance is not None:
                distances.append(distance)
                break
    return distances


def collision_count(log):
    if not log:
        return 0
    last_count = None
    inferred = 0
    for step in log.get("steps", []):
        metrics = step.get("post_metrics") or step.get("pre_metrics") or {}
        collisions = metrics.get("collisions")
        if not isinstance(collisions, dict):
            continue
        if collisions.get("count") is not None:
            last_count = as_float(collisions.get("count"))
        elif collisions.get("is_collision"):
            inferred += 1
    if last_count is not None:
        return int(last_count)
    return inferred


def max_run_length(actions, action_set=None):
    best = 0
    cur = 0
    prev = None
    for action in actions:
        if action_set is not None and action not in action_set:
            cur = 0
            prev = None
            continue
        if action == prev:
            cur += 1
        else:
            cur = 1
            prev = action
        best = max(best, cur)
    return best


def action_summary(log):
    actions = actions_from_log(log)
    counts = Counter(actions)
    total = len(actions)
    stop_count = counts[0]
    forward_count = counts[1]
    turn_count = counts[2] + counts[3]
    distances = distances_from_log(log)
    return {
        "action_counts": {ACTION_NAMES.get(key, str(key)): value for key, value in sorted(counts.items())},
        "steps_from_log": total,
        "stop_count": stop_count,
        "has_stop": stop_count > 0,
        "forward_ratio": forward_count / total if total else 0.0,
        "turn_ratio": turn_count / total if total else 0.0,
        "max_turn_run": max_run_length(actions, {2, 3}),
        "max_same_action_run": max_run_length(actions),
        "collision_count": collision_count(log),
        "min_ne": min(distances) if distances else None,
        "final_ne_from_log": distances[-1] if distances else None,
    }


def route_group_key(result):
    try:
        return f"{result.get('scene_id')}::{int(result.get('episode_id')) // 3}"
    except Exception:
        return f"{result.get('scene_id')}::{result.get('episode_id')}"


def extract_keywords(text):
    text = (text or "").lower()
    hits = []
    for name, pattern in KEYWORD_PATTERNS.items():
        if re.search(pattern, text):
            hits.append(name)
    return hits


def extract_top_tokens(text):
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]{2,}", (text or "").lower())
    return [word for word in words if word not in STOPWORDS]


def episode_record(index, result, log, log_path, route_status):
    summary = action_summary(log)
    instruction = result.get("episode_instruction", "")
    ne = as_float(result.get("ne"))
    steps = int(result.get("steps") or summary["steps_from_log"] or 0)
    os_value = as_float(result.get("os")) or 0.0
    success = as_float(result.get("success")) or 0.0
    min_ne = summary["min_ne"] if summary["min_ne"] is not None else ne
    route_key = route_group_key(result)
    route_items = route_status.get(route_key, [])
    route_success_count = sum(1 for item in route_items if (as_float(item.get("success")) or 0.0) >= 1.0)
    route_failure_count = len(route_items) - route_success_count

    return {
        "episode_index": index,
        "scene_id": result.get("scene_id"),
        "episode_id": result.get("episode_id"),
        "route_group": route_key,
        "instruction": instruction,
        "keywords": extract_keywords(instruction),
        "success": success,
        "spl": as_float(result.get("spl")),
        "os": os_value,
        "ne": ne,
        "min_ne": min_ne,
        "steps": steps,
        "video_file": result.get("video_file"),
        "episode_log": log_path or result.get("episode_log"),
        "same_route_partial_success": route_success_count > 0 and route_failure_count > 0,
        "route_success_count": route_success_count,
        "route_failure_count": route_failure_count,
        **summary,
    }


def in_near_miss(record, success_distance):
    return record["success"] < 1.0 and (
        record["os"] >= 1.0 or (record["min_ne"] is not None and record["min_ne"] <= success_distance)
    )


def in_far_off(record, far_ne_threshold):
    return record["success"] < 1.0 and record["os"] < 1.0 and (record["ne"] or 0.0) > far_ne_threshold


def in_stuck_control(record, max_step_threshold, collision_high, turn_ratio_high, max_turn_run_high):
    return record["success"] < 1.0 and (
        record["steps"] >= max_step_threshold
        or record["collision_count"] >= collision_high
        or record["turn_ratio"] >= turn_ratio_high
        or record["max_turn_run"] >= max_turn_run_high
    )


def summarize_set(name, target, records, all_results):
    scene_counter = Counter(record["scene_id"] for record in records)
    keyword_counter = Counter()
    token_counter = Counter()
    route_counter = Counter(record["route_group"] for record in records if record["same_route_partial_success"])
    for record in records:
        keyword_counter.update(record["keywords"])
        token_counter.update(extract_top_tokens(record.get("instruction", "")))

    return {
        "name": name,
        "target": target,
        "count": len(records),
        "avg_ne": safe_mean([record.get("ne") for record in records]),
        "avg_min_ne": safe_mean([record.get("min_ne") for record in records]),
        "avg_steps": safe_mean([record.get("steps") for record in records]),
        "total_stop_count": sum(record.get("stop_count", 0) for record in records),
        "avg_stop_count": safe_mean([record.get("stop_count") for record in records]),
        "avg_max_turn_run": safe_mean([record.get("max_turn_run") for record in records]),
        "max_turn_run": max([record.get("max_turn_run", 0) for record in records], default=0),
        "avg_forward_ratio": safe_mean([record.get("forward_ratio") for record in records]),
        "avg_turn_ratio": safe_mean([record.get("turn_ratio") for record in records]),
        "avg_collision_count": safe_mean([record.get("collision_count") for record in records]),
        "top_prompt_keywords": keyword_counter.most_common(30),
        "top_prompt_tokens": token_counter.most_common(30),
        "top_failed_scenes": scene_counter.most_common(20),
        "same_route_partial_success_episode_count": sum(1 for record in records if record["same_route_partial_success"]),
        "same_route_partial_success_route_count": len(route_counter),
        "same_route_partial_success_routes": route_counter.most_common(30),
    }


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def fmt(value, digits=3):
    value = as_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def write_set_json(path, name, target, records, summary, thresholds):
    write_json(path, {
        "set_name": name,
        "target": target,
        "thresholds": thresholds,
        "summary": summary,
        "episodes": records,
    })


def write_markdown(path, output_dir, set_payloads, overlaps, thresholds, total_failures):
    lines = []
    lines.append("# 阶段 3：三套失败诊断集报告")
    lines.append("")
    lines.append(f"分析目录：`{output_dir}`")
    lines.append(f"失败 episode 总数：{total_failures}")
    lines.append("")
    lines.append("## 1. 集合定义")
    lines.append("")
    lines.append(f"- near_miss：失败且 `OS=1` 或 `min_NE <= {thresholds['success_distance']}m`，用于分析 STOP/停点问题。")
    lines.append(f"- far_off：失败且 `OS=0` 且 `NE > {thresholds['far_ne_threshold']}m`，用于分析高层导航/grounding 问题。")
    lines.append(f"- stuck_control：失败且 `steps >= {thresholds['max_step_threshold']}` 或 `collision_count >= {thresholds['collision_high']}` 或 `turn_ratio >= {thresholds['turn_ratio_high']}` 或 `max_turn_run >= {thresholds['max_turn_run_high']}`，用于分析动作执行、碰撞恢复、连续转向问题。")
    lines.append("- 注意：三个诊断集允许重叠，因为同一个失败 episode 可能同时暴露多个根因信号。")
    lines.append("")

    lines.append("## 2. 总览")
    lines.append("")
    lines.append("| 诊断集 | 数量 | 平均 NE | 平均 min_NE | 平均 steps | STOP 总数 | 平均 STOP | 平均最大连续 TURN | 最大连续 TURN | forward 占比 | turn 占比 | 平均碰撞 | 同路径部分成功 episode |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for payload in set_payloads:
        s = payload["summary"]
        lines.append(
            f"| {payload['name']} | {s['count']} | {fmt(s['avg_ne'])} | {fmt(s['avg_min_ne'])} | "
            f"{fmt(s['avg_steps'])} | {s['total_stop_count']} | {fmt(s['avg_stop_count'])} | "
            f"{fmt(s['avg_max_turn_run'])} | {s['max_turn_run']} | {fmt(s['avg_forward_ratio'])} | "
            f"{fmt(s['avg_turn_ratio'])} | {fmt(s['avg_collision_count'])} | {s['same_route_partial_success_episode_count']} |"
        )
    lines.append("")

    lines.append("## 3. 集合重叠")
    lines.append("")
    for name, count in sorted(overlaps.items()):
        lines.append(f"- {name}: {count}")
    lines.append("")

    for payload in set_payloads:
        s = payload["summary"]
        lines.append(f"## {payload['name']}")
        lines.append("")
        lines.append(f"目标：{payload['target']}")
        lines.append("")
        lines.append("### prompt 关键词")
        if s["top_prompt_keywords"]:
            for key, count in s["top_prompt_keywords"][:20]:
                lines.append(f"- {key}: {count}")
        else:
            lines.append("- 无明显关键词命中")
        lines.append("")
        lines.append("### prompt 高频词")
        if s["top_prompt_tokens"]:
            for key, count in s["top_prompt_tokens"][:20]:
                lines.append(f"- {key}: {count}")
        else:
            lines.append("- 无可统计词")
        lines.append("")
        lines.append("### 最常失败 scene")
        if s["top_failed_scenes"]:
            for scene, count in s["top_failed_scenes"][:15]:
                lines.append(f"- {scene}: {count}")
        else:
            lines.append("- 无")
        lines.append("")
        lines.append("### 同一路径部分成功")
        lines.append(f"- episode 数：{s['same_route_partial_success_episode_count']}")
        lines.append(f"- route group 数：{s['same_route_partial_success_route_count']}")
        for route, count in s["same_route_partial_success_routes"][:15]:
            lines.append(f"- {route}: {count}")
        lines.append("")
        lines.append("### 样例")
        for record in payload["episodes"][:20]:
            lines.append(
                f"- #{record['episode_index']} {record['scene_id']}_{record['episode_id']} "
                f"NE={fmt(record['ne'])} min_NE={fmt(record['min_ne'])} steps={record['steps']} "
                f"STOP={record['stop_count']} maxTURN={record['max_turn_run']} "
                f"forward={fmt(record['forward_ratio'])} collision={record['collision_count']} "
                f"partial_route={record['same_route_partial_success']}"
            )
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def build_sets(output_dir, output_subdir, success_distance, far_ne_threshold, max_step_threshold, collision_high, turn_ratio_high, max_turn_run_high):
    results = load_results(output_dir)
    route_status = defaultdict(list)
    for result in results:
        if "scene_id" in result and "episode_id" in result:
            route_status[route_group_key(result)].append(result)

    records = []
    missing_logs = []
    for index, result in enumerate(results, start=1):
        if (as_float(result.get("success")) or 0.0) >= 1.0:
            continue
        log, log_path = load_episode_log(result, output_dir)
        if log is None:
            missing_logs.append(f"{result.get('scene_id')}_{result.get('episode_id')}")
        records.append(episode_record(index, result, log, log_path, route_status))

    thresholds = {
        "success_distance": success_distance,
        "far_ne_threshold": far_ne_threshold,
        "max_step_threshold": max_step_threshold,
        "collision_high": collision_high,
        "turn_ratio_high": turn_ratio_high,
        "max_turn_run_high": max_turn_run_high,
    }

    sets = {
        "near_miss": {
            "target": "分析 STOP/停点问题",
            "episodes": [record for record in records if in_near_miss(record, success_distance)],
        },
        "far_off": {
            "target": "分析高层导航/grounding 问题",
            "episodes": [record for record in records if in_far_off(record, far_ne_threshold)],
        },
        "stuck_control": {
            "target": "分析动作执行、碰撞恢复、连续转向问题",
            "episodes": [
                record for record in records
                if in_stuck_control(record, max_step_threshold, collision_high, turn_ratio_high, max_turn_run_high)
            ],
        },
    }

    output_dir_sets = os.path.join(output_dir, output_subdir)
    os.makedirs(output_dir_sets, exist_ok=True)

    payloads = []
    membership = defaultdict(list)
    for name, data in sets.items():
        summary = summarize_set(name, data["target"], data["episodes"], results)
        payload = {
            "name": name,
            "target": data["target"],
            "episodes": data["episodes"],
            "summary": summary,
        }
        payloads.append(payload)
        write_set_json(os.path.join(output_dir_sets, f"{name}.json"), name, data["target"], data["episodes"], summary, thresholds)
        for record in data["episodes"]:
            membership[f"{record['scene_id']}_{record['episode_id']}"].append(name)

    overlap_counter = Counter()
    for names in membership.values():
        overlap_counter[" + ".join(sorted(names))] += 1
    not_in_sets = len(records) - len(membership)
    if not_in_sets:
        overlap_counter["未进入三套诊断集"] = not_in_sets

    metadata = {
        "output_dir": output_dir,
        "output_subdir": output_subdir,
        "thresholds": thresholds,
        "total_episodes": len(results),
        "total_failures": len(records),
        "missing_logs": missing_logs,
        "set_counts": {payload["name"]: payload["summary"]["count"] for payload in payloads},
        "overlaps": dict(overlap_counter),
    }
    write_json(os.path.join(output_dir_sets, "metadata.json"), metadata)
    write_markdown(os.path.join(output_dir_sets, "set_summary.md"), output_dir, payloads, overlap_counter, thresholds, len(records))
    return output_dir_sets, metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True, help="StreamVLN eval output dir, e.g. outputs/eval_0730")
    parser.add_argument("--output_subdir", default="root_cause_sets")
    parser.add_argument("--success_distance", type=float, default=3.0)
    parser.add_argument("--far_ne_threshold", type=float, default=10.0)
    parser.add_argument("--max_step_threshold", type=int, default=500)
    parser.add_argument("--collision_high", type=int, default=5)
    parser.add_argument("--turn_ratio_high", type=float, default=0.65)
    parser.add_argument("--max_turn_run_high", type=int, default=8)
    args = parser.parse_args()

    output_dir_sets, metadata = build_sets(
        args.output_dir,
        args.output_subdir,
        args.success_distance,
        args.far_ne_threshold,
        args.max_step_threshold,
        args.collision_high,
        args.turn_ratio_high,
        args.max_turn_run_high,
    )
    print(f"Wrote {output_dir_sets}")
    print(json.dumps(metadata["set_counts"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
