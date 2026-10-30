#!/usr/bin/env python3
"""Prompt-action mismatch experiment utilities for StreamVLN.

Modes:
  select  - build keyword-filtered episode lists from an existing eval output
  compare - compare original-prompt and normalized-prompt eval outputs
"""

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
from statistics import mean

KEYWORD_REGEX = {
    "turn left": r"\bturn\s+left\b",
    "turn right": r"\bturn\s+right\b",
    "around": r"\baround\b",
    "go around": r"\bgo\s+around\b",
    "doorway": r"\bdoorway\b|\bdoor\s+way\b|\bthreshold\b",
    "stairs": r"\bstair\b|\bstairs\b|\bstaircase\b",
    "bed": r"\bbed\b|\bbedroom\b",
    "hallway": r"\bhallway\b|\bhall\b|\bcorridor\b",
    "wait near": r"\bwait\s+near\b|\bwait\s+at\b|\bwait\s+by\b",
    "stop beside": r"\bstop\s+beside\b|\bstop\s+next\s+to\b|\bstop\s+near\b|\bstop\s+at\b|\bwait\s+at\b",
}

ACTION_NAMES = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
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


def fmt(value, digits=3):
    value = as_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


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
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scene_id" in row and "episode_id" in row:
                rows.append(row)
    return rows


def key_of(result):
    return (str(result.get("scene_id")), str(result.get("episode_id")))


def load_episode_log(result, output_dir):
    candidates = []
    log_path = result.get("episode_log")
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


def keyword_hits(instruction):
    text = (instruction or "").lower()
    return [name for name, pattern in KEYWORD_REGEX.items() if re.search(pattern, text, flags=re.IGNORECASE)]


def route_group_key(result):
    try:
        return f"{result.get('scene_id')}::{int(result.get('episode_id')) // 3}"
    except Exception:
        return f"{result.get('scene_id')}::{result.get('episode_id')}"


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


def max_run(actions, target=None):
    best = 0
    cur = 0
    prev = None
    for action in actions:
        if target is not None and action != target:
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


def max_turn_run(actions):
    best = 0
    cur = 0
    prev = None
    for action in actions:
        if action not in {2, 3}:
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
    last = None
    inferred = 0
    for step in log.get("steps", []):
        metrics = step.get("post_metrics") or step.get("pre_metrics") or {}
        collisions = metrics.get("collisions")
        if not isinstance(collisions, dict):
            continue
        if collisions.get("count") is not None:
            last = as_float(collisions.get("count"))
        elif collisions.get("is_collision"):
            inferred += 1
    return int(last) if last is not None else inferred


def action_summary(result, output_dir):
    log, log_path = load_episode_log(result, output_dir)
    actions = actions_from_log(log)
    counts = Counter(actions)
    total = len(actions)
    distances = distances_from_log(log)
    turn_count = counts[2] + counts[3]
    return {
        "episode_log": log_path or result.get("episode_log"),
        "steps": int(result.get("steps") or total or 0),
        "success": as_float(result.get("success")) or 0.0,
        "spl": as_float(result.get("spl")),
        "os": as_float(result.get("os")) or 0.0,
        "ne": as_float(result.get("ne")),
        "min_ne": min(distances) if distances else as_float(result.get("ne")),
        "stop_count": counts[0],
        "forward_ratio": counts[1] / total if total else 0.0,
        "turn_ratio": turn_count / total if total else 0.0,
        "max_turn_run": max_turn_run(actions),
        "max_left_run": max_run(actions, 2),
        "max_right_run": max_run(actions, 3),
        "collision_count": collision_count(log),
        "action_counts": {ACTION_NAMES.get(k, str(k)): v for k, v in sorted(counts.items())},
    }


def select(args):
    results = load_results(args.input_dir)
    selected = []
    failed = []
    keyword_counter = Counter()
    keyword_fail_counter = Counter()
    route_groups = defaultdict(list)
    for result in results:
        route_groups[route_group_key(result)].append(result)

    for index, result in enumerate(results, start=1):
        instruction = result.get("episode_instruction", "")
        hits = keyword_hits(instruction)
        if not hits:
            continue
        success = (as_float(result.get("success")) or 0.0) >= 1.0
        route_items = route_groups[route_group_key(result)]
        route_success_count = sum(1 for item in route_items if (as_float(item.get("success")) or 0.0) >= 1.0)
        record = {
            "episode_index": index,
            "scene_id": result.get("scene_id"),
            "episode_id": result.get("episode_id"),
            "route_group": route_group_key(result),
            "instruction": instruction,
            "keyword_hits": hits,
            "original_success": success,
            "success": as_float(result.get("success")),
            "spl": as_float(result.get("spl")),
            "os": as_float(result.get("os")),
            "ne": as_float(result.get("ne")),
            "steps": result.get("steps"),
            "video_file": result.get("video_file"),
            "episode_log": result.get("episode_log"),
            "same_route_partial_success": route_success_count > 0 and route_success_count < len(route_items),
        }
        selected.append(record)
        keyword_counter.update(hits)
        if not success:
            failed.append(record)
            keyword_fail_counter.update(hits)

    os.makedirs(args.output_dir, exist_ok=True)
    all_path = os.path.join(args.output_dir, "keyword_episodes_all.json")
    failed_path = os.path.join(args.output_dir, "keyword_episodes_failed.json")
    write_json(all_path, {"source_dir": args.input_dir, "episodes": selected})
    write_json(failed_path, {"source_dir": args.input_dir, "episodes": failed})

    lines = []
    lines.append("# Prompt-Action Mismatch 关键词集合")
    lines.append("")
    lines.append(f"来源目录：`{args.input_dir}`")
    lines.append(f"总 episode：{len(results)}")
    lines.append(f"命中关键词 episode：{len(selected)}")
    lines.append(f"命中关键词且失败 episode：{len(failed)}")
    lines.append("")
    lines.append("## 关键词命中")
    for key, count in keyword_counter.most_common():
        fail_count = keyword_fail_counter.get(key, 0)
        lines.append(f"- {key}: {count}，失败 {fail_count}，失败率 {fail_count / count:.3f}")
    lines.append("")
    lines.append("## 输出文件")
    lines.append(f"- 全部关键词 episode：`{all_path}`")
    lines.append(f"- 失败关键词 episode：`{failed_path}`")
    lines.append("")
    with open(os.path.join(args.output_dir, "keyword_selection.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {all_path}")
    print(f"Wrote {failed_path}")
    print(f"Selected all={len(selected)}, failed={len(failed)}")


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_filter_keys(filter_file):
    with open(filter_file, "r") as f:
        data = json.load(f)
    rows = data.get("episodes", data if isinstance(data, list) else [])
    return {str(row.get("scene_id")) + "::" + str(row.get("episode_id")): row for row in rows}


def paired_records(original_dir, normalized_dir, filter_file):
    filter_rows = load_filter_keys(filter_file)
    original = {f"{r.get('scene_id')}::{r.get('episode_id')}": r for r in load_results(original_dir)}
    normalized = {f"{r.get('scene_id')}::{r.get('episode_id')}": r for r in load_results(normalized_dir)}
    records = []
    missing_original = []
    missing_normalized = []
    for key, selected in filter_rows.items():
        if key not in original:
            missing_original.append(key)
            continue
        if key not in normalized:
            missing_normalized.append(key)
            continue
        o = original[key]
        n = normalized[key]
        o_summary = action_summary(o, original_dir)
        n_summary = action_summary(n, normalized_dir)
        hits = selected.get("keyword_hits") or keyword_hits(o.get("episode_instruction", ""))
        record = {
            "key": key,
            "scene_id": o.get("scene_id"),
            "episode_id": o.get("episode_id"),
            "instruction": o.get("episode_instruction", ""),
            "normalized_instruction": n.get("model_instruction"),
            "keyword_hits": hits,
            "original": o_summary,
            "normalized": n_summary,
            "rescued": o_summary["success"] < 1.0 and n_summary["success"] >= 1.0,
            "regressed": o_summary["success"] >= 1.0 and n_summary["success"] < 1.0,
            "delta_ne": (n_summary["ne"] - o_summary["ne"]) if n_summary["ne"] is not None and o_summary["ne"] is not None else None,
            "delta_steps": n_summary["steps"] - o_summary["steps"],
            "delta_turn_ratio": n_summary["turn_ratio"] - o_summary["turn_ratio"],
        }
        records.append(record)
    return records, missing_original, missing_normalized


def aggregate(records):
    return {
        "count": len(records),
        "original_sr": safe_mean([r["original"]["success"] for r in records]),
        "normalized_sr": safe_mean([r["normalized"]["success"] for r in records]),
        "original_avg_ne": safe_mean([r["original"]["ne"] for r in records]),
        "normalized_avg_ne": safe_mean([r["normalized"]["ne"] for r in records]),
        "original_avg_steps": safe_mean([r["original"]["steps"] for r in records]),
        "normalized_avg_steps": safe_mean([r["normalized"]["steps"] for r in records]),
        "original_avg_turn_ratio": safe_mean([r["original"]["turn_ratio"] for r in records]),
        "normalized_avg_turn_ratio": safe_mean([r["normalized"]["turn_ratio"] for r in records]),
        "original_avg_max_turn_run": safe_mean([r["original"]["max_turn_run"] for r in records]),
        "normalized_avg_max_turn_run": safe_mean([r["normalized"]["max_turn_run"] for r in records]),
        "rescued_count": sum(1 for r in records if r["rescued"]),
        "regressed_count": sum(1 for r in records if r["regressed"]),
    }


def mismatch_flags(record):
    hits = set(record["keyword_hits"])
    o = record["original"]
    n = record["normalized"]
    flags = []
    if "turn left" in hits and o["max_left_run"] >= 8:
        flags.append("原 prompt: turn left 后连续 TURN_LEFT >= 8")
    if "turn right" in hits and o["max_right_run"] >= 8:
        flags.append("原 prompt: turn right 后连续 TURN_RIGHT >= 8")
    if ({"around", "go around", "bed"} & hits) and o["success"] < 1.0 and (o["collision_count"] >= 5 or o["max_turn_run"] >= 8 or (o["ne"] or 0) > 3.0):
        flags.append("原 prompt: around/bed 相关绕行失败或卡住")
    if ({"doorway", "wait near", "stop beside"} & hits) and o["min_ne"] is not None and o["min_ne"] <= 3.0 and (o["ne"] or 0) > 3.0:
        flags.append("原 prompt: 门口/附近曾到达但未在目标处 STOP")
    if "stairs" in hits and o["success"] < 1.0:
        flags.append("原 prompt: stairs 相关失败")
    if flags and n["success"] >= 1.0 and o["success"] < 1.0:
        flags.append("规范化 prompt 后救回")
    return flags


def compare(args):
    records, missing_original, missing_normalized = paired_records(args.original_dir, args.normalized_dir, args.filter_file)
    os.makedirs(args.output_dir, exist_ok=True)
    overall = aggregate(records)
    by_keyword = {}
    for keyword in KEYWORD_REGEX:
        subset = [r for r in records if keyword in r["keyword_hits"]]
        if subset:
            by_keyword[keyword] = aggregate(subset)
    originally_failed = [r for r in records if r["original"]["success"] < 1.0]
    originally_success = [r for r in records if r["original"]["success"] >= 1.0]
    flagged = []
    flag_counter = Counter()
    for record in records:
        flags = mismatch_flags(record)
        if flags:
            flagged.append({**record, "flags": flags})
            flag_counter.update(flags)

    result = {
        "original_dir": args.original_dir,
        "normalized_dir": args.normalized_dir,
        "filter_file": args.filter_file,
        "missing_original": missing_original,
        "missing_normalized": missing_normalized,
        "overall": overall,
        "originally_failed": aggregate(originally_failed),
        "originally_success": aggregate(originally_success),
        "by_keyword": by_keyword,
        "flag_counts": dict(flag_counter),
        "records": records,
        "flagged_records": flagged,
    }
    json_path = os.path.join(args.output_dir, "prompt_action_mismatch_comparison.json")
    write_json(json_path, result)
    md_path = os.path.join(args.output_dir, "prompt_action_mismatch_report.md")
    write_compare_md(md_path, result, json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def write_metric_line(lines, label, stats):
    delta_sr = None
    if stats.get("original_sr") is not None and stats.get("normalized_sr") is not None:
        delta_sr = stats["normalized_sr"] - stats["original_sr"]
    delta_ne = None
    if stats.get("original_avg_ne") is not None and stats.get("normalized_avg_ne") is not None:
        delta_ne = stats["normalized_avg_ne"] - stats["original_avg_ne"]
    lines.append(
        f"| {label} | {stats['count']} | {fmt(stats.get('original_sr'))} | {fmt(stats.get('normalized_sr'))} | "
        f"{fmt(delta_sr)} | {fmt(stats.get('original_avg_ne'))} | {fmt(stats.get('normalized_avg_ne'))} | "
        f"{fmt(delta_ne)} | {stats.get('rescued_count', 0)} | {stats.get('regressed_count', 0)} |"
    )


def write_compare_md(path, data, json_path):
    lines = []
    lines.append("# 阶段 4：Prompt-Action Mismatch 对照报告")
    lines.append("")
    lines.append(f"原 prompt 目录：`{data['original_dir']}`")
    lines.append(f"动作规范化 prompt 目录：`{data['normalized_dir']}`")
    lines.append(f"episode 清单：`{data['filter_file']}`")
    lines.append("")
    lines.append("## 1. 总体对照")
    lines.append("")
    lines.append("| 集合 | 数量 | 原 SR | 规范化 SR | SR 变化 | 原平均 NE | 规范化平均 NE | NE 变化 | 救回 | 退化 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    write_metric_line(lines, "全部关键词 episode", data["overall"])
    write_metric_line(lines, "原本失败子集", data["originally_failed"])
    write_metric_line(lines, "原本成功子集", data["originally_success"])
    lines.append("")
    lines.append("判断标准：如果规范化 prompt 在原本失败子集上救回较多、SR 明显提升，同时原本成功子集退化较少，则 prompt-action mismatch 是重要根因之一；如果提升很小，更可能是视觉 grounding、路径规划或局部控制问题。")
    lines.append("")

    lines.append("## 2. 按关键词对照")
    lines.append("")
    lines.append("| 关键词 | 数量 | 原 SR | 规范化 SR | SR 变化 | 原平均 NE | 规范化平均 NE | NE 变化 | 救回 | 退化 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for keyword, stats in data["by_keyword"].items():
        write_metric_line(lines, keyword, stats)
    lines.append("")

    lines.append("## 3. 典型 mismatch 信号")
    lines.append("")
    if data["flag_counts"]:
        for flag, count in Counter(data["flag_counts"]).most_common():
            lines.append(f"- {flag}: {count}")
    else:
        lines.append("- 未检测到预设 mismatch 信号。")
    lines.append("")

    lines.append("## 4. 救回样例")
    rescued = [r for r in data["records"] if r["rescued"]]
    if rescued:
        for r in rescued[:60]:
            lines.append(
                f"- {r['scene_id']}_{r['episode_id']} keywords={','.join(r['keyword_hits'])} "
                f"NE {fmt(r['original']['ne'])}->{fmt(r['normalized']['ne'])} "
                f"steps {r['original']['steps']}->{r['normalized']['steps']} "
                f"turn_ratio {fmt(r['original']['turn_ratio'])}->{fmt(r['normalized']['turn_ratio'])}"
            )
    else:
        lines.append("- 无救回样例。")
    lines.append("")

    lines.append("## 5. 退化样例")
    regressed = [r for r in data["records"] if r["regressed"]]
    if regressed:
        for r in regressed[:60]:
            lines.append(
                f"- {r['scene_id']}_{r['episode_id']} keywords={','.join(r['keyword_hits'])} "
                f"NE {fmt(r['original']['ne'])}->{fmt(r['normalized']['ne'])} "
                f"steps {r['original']['steps']}->{r['normalized']['steps']}"
            )
    else:
        lines.append("- 无退化样例。")
    lines.append("")

    lines.append("## 6. 需人工复核")
    lines.append("")
    lines.append("- 对 `around/bed`、`doorway/wait near`、`stairs` 的结论需要结合视频确认：自动规则只能看到动作、距离和碰撞，不能保证理解具体 landmark 是否被正确识别。")
    lines.append(f"- 完整逐 episode 数据见 `{json_path}`。")
    if data["missing_original"] or data["missing_normalized"]:
        lines.append(f"- 缺少原 prompt 结果：{len(data['missing_original'])}；缺少规范化结果：{len(data['missing_normalized'])}。")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--input_dir", required=True)
    select_parser.add_argument("--output_dir", required=True)
    select_parser.set_defaults(func=select)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--original_dir", required=True)
    compare_parser.add_argument("--normalized_dir", required=True)
    compare_parser.add_argument("--filter_file", required=True)
    compare_parser.add_argument("--output_dir", required=True)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
