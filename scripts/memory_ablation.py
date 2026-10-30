#!/usr/bin/env python3
"""Stage 6 memory/context ablation helpers for StreamVLN.

Modes:
  select  - choose 100 high-failure episodes from a baseline eval output
  compare - compare num_frames/num_history ablation outputs on the same episodes
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean

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
    return f"{result.get('scene_id')}::{result.get('episode_id')}"


def route_group_key(result):
    try:
        return f"{result.get('scene_id')}::{int(result.get('episode_id')) // 3}"
    except Exception:
        return f"{result.get('scene_id')}::{result.get('episode_id')}"


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
    return int(last_count) if last_count is not None else inferred


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


def action_summary(result, output_dir):
    log, log_path = load_episode_log(result, output_dir)
    actions = actions_from_log(log)
    counts = Counter(actions)
    total = len(actions)
    distances = distances_from_log(log)
    return {
        "episode_log": log_path or result.get("episode_log"),
        "steps": int(result.get("steps") or total or 0),
        "success": as_float(result.get("success")) or 0.0,
        "spl": as_float(result.get("spl")),
        "os": as_float(result.get("os")) or 0.0,
        "ne": as_float(result.get("ne")),
        "min_ne": min(distances) if distances else as_float(result.get("ne")),
        "turn_ratio": (counts[2] + counts[3]) / total if total else 0.0,
        "forward_ratio": counts[1] / total if total else 0.0,
        "stop_count": counts[0],
        "max_turn_run": max_turn_run(actions),
        "collision_count": collision_count(log),
        "action_counts": {ACTION_NAMES.get(k, str(k)): v for k, v in sorted(counts.items())},
    }


def classify(record, success_distance=3.0, far_ne_threshold=10.0, max_step_threshold=500, collision_high=5, turn_ratio_high=0.65, max_turn_run_high=8):
    near_miss = record["success"] < 1.0 and (
        record["os"] >= 1.0 or (record["min_ne"] is not None and record["min_ne"] <= success_distance)
    )
    far_off = record["success"] < 1.0 and record["os"] < 1.0 and (record["ne"] or 0.0) > far_ne_threshold
    stuck_control = record["success"] < 1.0 and (
        record["steps"] >= max_step_threshold
        or record["collision_count"] >= collision_high
        or record["turn_ratio"] >= turn_ratio_high
        or record["max_turn_run"] >= max_turn_run_high
    )
    return {
        "near_miss": bool(near_miss),
        "far_off": bool(far_off),
        "stuck_control": bool(stuck_control),
        "max_step_500": bool(record["steps"] >= max_step_threshold),
    }


def make_record(index, result, output_dir, scene_fail_counts, route_success_counts):
    summary = action_summary(result, output_dir)
    classes = classify(summary)
    scene_id = result.get("scene_id")
    route_key = route_group_key(result)
    route_success_count, route_count = route_success_counts.get(route_key, (0, 0))
    record = {
        "episode_index": index,
        "scene_id": scene_id,
        "episode_id": result.get("episode_id"),
        "route_group": route_key,
        "instruction": result.get("episode_instruction", ""),
        "scene_failure_count": scene_fail_counts.get(scene_id, 0),
        "same_route_partial_success": route_success_count > 0 and route_success_count < route_count,
        **summary,
        **classes,
    }
    record["selection_score"] = selection_score(record)
    return record


def selection_score(record):
    score = 0.0
    if record.get("far_off"):
        score += 5.0
    if record.get("stuck_control"):
        score += 3.0
    if record.get("near_miss"):
        score += 2.0
    if record.get("max_step_500"):
        score += 2.0
    if record.get("same_route_partial_success"):
        score += 1.0
    score += min((record.get("ne") or 0.0) / 10.0, 3.0)
    score += min(record.get("turn_ratio", 0.0), 1.0)
    score += min(record.get("collision_count", 0) / 50.0, 2.0)
    score += min(record.get("scene_failure_count", 0) / 10.0, 2.0)
    return score


def select(args):
    results = load_results(args.baseline_dir)
    failures = [r for r in results if (as_float(r.get("success")) or 0.0) < 1.0]
    scene_fail_counts = Counter(r.get("scene_id") for r in failures)
    route_items = defaultdict(list)
    for result in results:
        route_items[route_group_key(result)].append(result)
    route_success_counts = {}
    for route_key, items in route_items.items():
        success_count = sum(1 for item in items if (as_float(item.get("success")) or 0.0) >= 1.0)
        route_success_counts[route_key] = (success_count, len(items))

    records = []
    result_index = {key_of(result): idx for idx, result in enumerate(results, start=1)}
    for result in failures:
        records.append(make_record(result_index[key_of(result)], result, args.baseline_dir, scene_fail_counts, route_success_counts))
    records = sorted(records, key=lambda r: (-r["selection_score"], -(r.get("ne") or 0.0), -r.get("steps", 0)))
    selected = records[:args.limit]

    os.makedirs(args.output_dir, exist_ok=True)
    json_path = os.path.join(args.output_dir, f"selected_{args.limit}_high_failure.json")
    payload = {
        "baseline_dir": args.baseline_dir,
        "selection_rule": "failed episodes ranked by far_off/stuck_control/near_miss/max_step/high_NE/high_turn/high_collision/scene_failure_count",
        "limit": args.limit,
        "episodes": selected,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    md_path = os.path.join(args.output_dir, "selection_summary.md")
    write_selection_md(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"selected={len(selected)} far_off={sum(r['far_off'] for r in selected)} stuck_control={sum(r['stuck_control'] for r in selected)} near_miss={sum(r['near_miss'] for r in selected)} max_step_500={sum(r['max_step_500'] for r in selected)}")


def write_selection_md(path, payload):
    episodes = payload["episodes"]
    lines = []
    lines.append("# 阶段 6：100 条失败高发样本选择")
    lines.append("")
    lines.append(f"Baseline：`{payload['baseline_dir']}`")
    lines.append(f"选择数量：{len(episodes)}")
    lines.append("")
    lines.append("## 1. 样本构成")
    lines.append("")
    lines.append(f"- far_off：{sum(r['far_off'] for r in episodes)}")
    lines.append(f"- stuck_control：{sum(r['stuck_control'] for r in episodes)}")
    lines.append(f"- near_miss：{sum(r['near_miss'] for r in episodes)}")
    lines.append(f"- max_step=500：{sum(r['max_step_500'] for r in episodes)}")
    lines.append(f"- 平均 NE：{fmt(safe_mean([r['ne'] for r in episodes]))}")
    lines.append(f"- 平均 turn_ratio：{fmt(safe_mean([r['turn_ratio'] for r in episodes]))}")
    lines.append("")
    lines.append("## 2. 高频失败 scene")
    for scene, count in Counter(r["scene_id"] for r in episodes).most_common(20):
        lines.append(f"- {scene}: {count}")
    lines.append("")
    lines.append("## 3. 样例")
    for record in episodes[:50]:
        labels = [name for name in ("far_off", "stuck_control", "near_miss", "max_step_500") if record.get(name)]
        lines.append(
            f"- #{record['episode_index']} {record['scene_id']}_{record['episode_id']} "
            f"score={fmt(record['selection_score'])} labels={','.join(labels)} "
            f"NE={fmt(record['ne'])} steps={record['steps']} turn={fmt(record['turn_ratio'])} maxTURN={record['max_turn_run']} collisions={record['collision_count']}"
        )
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def load_filter_keys(filter_file):
    with open(filter_file, "r") as f:
        data = json.load(f)
    rows = data.get("episodes", data if isinstance(data, list) else [])
    return {f"{row.get('scene_id')}::{row.get('episode_id')}" for row in rows if isinstance(row, dict)}


def aggregate(records):
    return {
        "count": len(records),
        "sr": safe_mean([r["success"] for r in records]),
        "spl": safe_mean([r["spl"] for r in records]),
        "ne": safe_mean([r["ne"] for r in records]),
        "steps": safe_mean([r["steps"] for r in records]),
        "turn_ratio": safe_mean([r["turn_ratio"] for r in records]),
        "forward_ratio": safe_mean([r["forward_ratio"] for r in records]),
        "max_turn_run": safe_mean([r["max_turn_run"] for r in records]),
        "collision_count": safe_mean([r["collision_count"] for r in records]),
        "far_off_count": sum(1 for r in records if r["far_off"]),
        "stuck_control_count": sum(1 for r in records if r["stuck_control"]),
        "near_miss_count": sum(1 for r in records if r["near_miss"]),
        "max_step_500_count": sum(1 for r in records if r["max_step_500"]),
    }


def parse_experiment(value):
    if "=" in value:
        return value.split("=", 1)
    if ":" in value:
        return value.split(":", 1)
    path = value
    return os.path.basename(os.path.normpath(path)), path


def compare(args):
    filter_keys = load_filter_keys(args.filter_file)
    baseline_results = {key_of(r): r for r in load_results(args.baseline_dir)}
    baseline_records = []
    for idx, key in enumerate(sorted(filter_keys), start=1):
        result = baseline_results.get(key)
        if result:
            summary = action_summary(result, args.baseline_dir)
            baseline_records.append({"key": key, **summary, **classify(summary)})
    baseline_agg = aggregate(baseline_records)

    experiments = []
    for exp_arg in args.experiment:
        name, exp_dir = parse_experiment(exp_arg)
        exp_results = {key_of(r): r for r in load_results(exp_dir)}
        records = []
        rescued = []
        regressed = []
        missing = []
        for key in sorted(filter_keys):
            result = exp_results.get(key)
            base_result = baseline_results.get(key)
            if result is None:
                missing.append(key)
                continue
            summary = action_summary(result, exp_dir)
            rec = {"key": key, "scene_id": result.get("scene_id"), "episode_id": result.get("episode_id"), **summary, **classify(summary)}
            base_success = as_float(base_result.get("success")) if base_result else None
            rec["baseline_success"] = base_success
            rec["rescued"] = base_success is not None and base_success < 1.0 and rec["success"] >= 1.0
            rec["regressed"] = base_success is not None and base_success >= 1.0 and rec["success"] < 1.0
            records.append(rec)
            if rec["rescued"]:
                rescued.append(rec)
            if rec["regressed"]:
                regressed.append(rec)
        experiments.append({
            "name": name,
            "dir": exp_dir,
            "aggregate": aggregate(records),
            "missing": missing,
            "rescued_count": len(rescued),
            "regressed_count": len(regressed),
            "rescued": rescued,
            "regressed": regressed,
            "records": records,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    data = {
        "baseline_dir": args.baseline_dir,
        "filter_file": args.filter_file,
        "baseline": baseline_agg,
        "experiments": experiments,
    }
    json_path = os.path.join(args.output_dir, "memory_ablation_comparison.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    md_path = os.path.join(args.output_dir, "memory_ablation_report.md")
    write_compare_md(md_path, data, json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def row_delta(current, baseline, key):
    if current.get(key) is None or baseline.get(key) is None:
        return None
    return current[key] - baseline[key]


def write_table_row(lines, name, stats, baseline=None, rescued=None, regressed=None):
    lines.append(
        f"| {name} | {stats['count']} | {fmt(stats.get('sr'))} | {fmt(row_delta(stats, baseline, 'sr') if baseline else None)} | "
        f"{fmt(stats.get('ne'))} | {fmt(row_delta(stats, baseline, 'ne') if baseline else None)} | "
        f"{fmt(stats.get('turn_ratio'))} | {fmt(row_delta(stats, baseline, 'turn_ratio') if baseline else None)} | "
        f"{fmt(stats.get('steps'))} | {stats.get('far_off_count')} | {stats.get('stuck_control_count')} | "
        f"{stats.get('near_miss_count')} | {stats.get('max_step_500_count')} | {rescued if rescued is not None else '-'} | {regressed if regressed is not None else '-'} |"
    )


def write_compare_md(path, data, json_path):
    lines = []
    lines.append("# 阶段 6：模型/记忆机制消融报告")
    lines.append("")
    lines.append(f"Baseline：`{data['baseline_dir']}`")
    lines.append(f"样本清单：`{data['filter_file']}`")
    lines.append("")
    lines.append("## 1. 总体对照")
    lines.append("")
    lines.append("| 实验 | 数量 | SR | SR变化 | NE | NE变化 | turn_ratio | turn变化 | steps | far_off | stuck_control | near_miss | max_step=500 | 救回 | 退化 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    write_table_row(lines, "frames4_history8_baseline", data["baseline"])
    for exp in data["experiments"]:
        write_table_row(lines, exp["name"], exp["aggregate"], data["baseline"], exp["rescued_count"], exp["regressed_count"])
    lines.append("")
    lines.append("## 2. 判断标准")
    lines.append("")
    lines.append("- `num_history` 增大后 far_off 明显减少：长程记忆不足是根因之一。")
    lines.append("- `num_frames` 增大后 turn_ratio/max_step=500 减少：模型需要更长观察窗口来稳定动作。")
    lines.append("- near_miss 增加但 far_off 减少：路线更容易走近目标，但 STOP 校准仍弱。")
    lines.append("- 所有设置变化都不明显：更可能是视觉 grounding、路径规划或动作接口问题，而不是记忆长度单点问题。")
    lines.append("")
    for exp in data["experiments"]:
        if exp.get("missing"):
            lines.append(f"## {exp['name']} 缺失 episode")
            lines.append(f"- 缺失数量：{len(exp['missing'])}")
            lines.append("")
        lines.append(f"## {exp['name']} 救回样例")
        if exp["rescued"]:
            for rec in exp["rescued"][:50]:
                lines.append(
                    f"- {rec['scene_id']}_{rec['episode_id']} NE={fmt(rec['ne'])} steps={rec['steps']} "
                    f"far_off={rec['far_off']} near_miss={rec['near_miss']} turn={fmt(rec['turn_ratio'])}"
                )
        else:
            lines.append("- 无救回样例。")
        lines.append("")
    lines.append(f"完整 JSON：`{json_path}`")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)

    select_parser = subparsers.add_parser("select")
    select_parser.add_argument("--baseline_dir", required=True)
    select_parser.add_argument("--output_dir", required=True)
    select_parser.add_argument("--limit", type=int, default=100)
    select_parser.set_defaults(func=select)

    compare_parser = subparsers.add_parser("compare")
    compare_parser.add_argument("--baseline_dir", required=True)
    compare_parser.add_argument("--filter_file", required=True)
    compare_parser.add_argument("--experiment", action="append", required=True, help="name=path; repeat for multiple experiments")
    compare_parser.add_argument("--output_dir", required=True)
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
