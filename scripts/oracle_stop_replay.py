#!/usr/bin/env python3
"""Oracle STOP replay analysis for StreamVLN evaluation outputs.

This script does not rerun Habitat or the model. It reads result.json and
per-episode logs, finds the closest step to the goal for every failed episode,
and estimates whether forcing STOP at that step would have made the episode
successful under the VLN-CE success-distance threshold.
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean


def load_results(output_dir):
    path = os.path.join(output_dir, "result.json")
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
        suffix = f"_{result['scene_id']}_{result['episode_id']}.json"
        candidates.extend(
            os.path.join(log_dir, name)
            for name in os.listdir(log_dir)
            if name.endswith(suffix)
        )
    for path in candidates:
        if path and os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f), path
    return None, None


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def iter_distance_records(log):
    if not log:
        return
    for step in log.get("steps", []):
        step_id = step.get("step_id")
        action = step.get("action")
        action_name = step.get("action_name")
        for metric_key in ("post_metrics", "pre_metrics"):
            metrics = step.get(metric_key) or {}
            distance = as_float(metrics.get("distance_to_goal"))
            if distance is None:
                continue
            yield {
                "step_id": step_id,
                "action": action,
                "action_name": action_name,
                "metric_source": metric_key,
                "distance_to_goal": distance,
                "success": as_float(metrics.get("success")),
                "spl": as_float(metrics.get("spl")),
                "oracle_success": as_float(metrics.get("oracle_success")),
                "gps": step.get("gps"),
                "compass": step.get("compass"),
            }


def closest_record(result, log):
    records = list(iter_distance_records(log))
    if not records:
        ne = as_float(result.get("ne"))
        if ne is None:
            return None
        return {
            "step_id": None,
            "action": None,
            "action_name": None,
            "metric_source": "result.ne",
            "distance_to_goal": ne,
            "success": as_float(result.get("success")),
            "spl": as_float(result.get("spl")),
            "oracle_success": as_float(result.get("os")),
            "gps": None,
            "compass": None,
        }
    return min(records, key=lambda item: item["distance_to_goal"])


def action_counts(log):
    counts = Counter()
    if not log:
        return counts
    for step in log.get("steps", []):
        name = step.get("action_name") or str(step.get("action"))
        counts[name] += 1
    return counts


def fmt(value, digits=3):
    value = as_float(value)
    if value is None:
        return "NA"
    return f"{value:.{digits}f}"


def bucket_ne(value):
    value = as_float(value)
    if value is None:
        return "unknown"
    if value <= 3.0:
        return "<=3m"
    if value <= 5.0:
        return "3-5m"
    if value <= 10.0:
        return "5-10m"
    return ">10m"


def analyze(output_dir, success_distance):
    results = load_results(output_dir)
    if not results:
        raise SystemExit(f"No episode rows found in {output_dir}/result.json")

    details = []
    original_successes = 0
    oracle_successes = 0
    rescued = []
    unrecovered = []
    failed = []
    os1_failed = []
    missing_logs = []

    for index, result in enumerate(results, start=1):
        log, log_path = load_episode_log(result, output_dir)
        if log is None:
            missing_logs.append(f"{result.get('scene_id')}_{result.get('episode_id')}")
        closest = closest_record(result, log)
        original_success = as_float(result.get("success")) == 1.0
        if original_success:
            original_successes += 1
        min_distance = closest["distance_to_goal"] if closest else as_float(result.get("ne"))
        oracle_stop_success = original_success or (
            min_distance is not None and min_distance <= success_distance
        )
        if oracle_stop_success:
            oracle_successes += 1

        row = {
            "episode_index": index,
            "scene_id": result.get("scene_id"),
            "episode_id": result.get("episode_id"),
            "instruction": result.get("episode_instruction", ""),
            "video_file": result.get("video_file"),
            "episode_log": log_path or result.get("episode_log"),
            "original_success": original_success,
            "oracle_stop_success": bool(oracle_stop_success),
            "rescued_by_oracle_stop": (not original_success) and bool(oracle_stop_success),
            "success": as_float(result.get("success")),
            "spl": as_float(result.get("spl")),
            "os": as_float(result.get("os")),
            "ne": as_float(result.get("ne")),
            "steps": result.get("steps"),
            "min_distance": min_distance,
            "closest_step": closest,
            "action_counts": dict(action_counts(log)),
        }
        details.append(row)

        if not original_success:
            failed.append(row)
            if as_float(result.get("os")) == 1.0:
                os1_failed.append(row)
            if row["rescued_by_oracle_stop"]:
                rescued.append(row)
            else:
                unrecovered.append(row)

    total = len(results)
    original_sr = original_successes / total
    oracle_sr = oracle_successes / total
    gain = oracle_sr - original_sr

    return {
        "output_dir": output_dir,
        "success_distance": success_distance,
        "total_episodes": total,
        "original_successes": original_successes,
        "original_failures": len(failed),
        "original_sr": original_sr,
        "oracle_stop_successes": oracle_successes,
        "oracle_stop_failures": total - oracle_successes,
        "oracle_stop_sr": oracle_sr,
        "oracle_stop_gain": gain,
        "rescued_count": len(rescued),
        "unrecovered_failure_count": len(unrecovered),
        "os1_failed_count": len(os1_failed),
        "os0_failed_count": len(failed) - len(os1_failed),
        "ne_gt_10_failed_count": sum(1 for row in failed if (row["ne"] or 0) > 10.0),
        "max_step_500_failed_count": sum(1 for row in failed if int(row.get("steps") or 0) >= 500),
        "missing_logs": missing_logs,
        "rescued": rescued,
        "unrecovered": unrecovered,
        "failed": failed,
        "details": details,
    }


def write_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_markdown(path, data):
    failed = data["failed"]
    rescued = data["rescued"]
    unrecovered = data["unrecovered"]
    failed_min_buckets = Counter(bucket_ne(row.get("min_distance")) for row in failed)
    unrecovered_ne_buckets = Counter(bucket_ne(row.get("ne")) for row in unrecovered)
    rescued_os1 = sum(1 for row in rescued if row.get("os") == 1.0)
    rescued_os0 = len(rescued) - rescued_os1

    lines = []
    lines.append("# Oracle STOP Replay 报告")
    lines.append("")
    lines.append(f"分析目录：`{data['output_dir']}`")
    lines.append(f"成功半径阈值：`{fmt(data['success_distance'])} m`")
    lines.append("")
    lines.append("## 1. Baseline 固定结果")
    lines.append("")
    lines.append(f"- 总 episode 数：{data['total_episodes']}")
    lines.append(f"- 原始成功数：{data['original_successes']}")
    lines.append(f"- 原始失败数：{data['original_failures']}")
    lines.append(f"- 原始 SR：{data['original_sr']:.3f}")
    lines.append(f"- 失败中 OS=1：{data['os1_failed_count']}")
    lines.append(f"- 失败中 OS=0：{data['os0_failed_count']}")
    lines.append(f"- 失败中 NE>10m：{data['ne_gt_10_failed_count']}")
    lines.append(f"- 失败中 steps>=500：{data['max_step_500_failed_count']}")
    lines.append("")
    lines.append("## 2. Oracle STOP Replay 结果")
    lines.append("")
    lines.append(f"- Oracle STOP 后成功数：{data['oracle_stop_successes']}")
    lines.append(f"- Oracle STOP 后失败数：{data['oracle_stop_failures']}")
    lines.append(f"- Oracle STOP SR：{data['oracle_stop_sr']:.3f}")
    lines.append(f"- SR 提升：{data['oracle_stop_gain']:.3f}")
    lines.append(f"- 可由 Oracle STOP 救回的失败：{data['rescued_count']}")
    lines.append(f"- Oracle STOP 仍救不回的失败：{data['unrecovered_failure_count']}")
    lines.append(f"- 可救回失败中 OS=1：{rescued_os1}")
    lines.append(f"- 可救回失败中 OS=0：{rescued_os0}")
    lines.append("")
    if data["oracle_stop_gain"] >= 0.05:
        lines.append("结论：Oracle STOP 带来明显提升，STOP/目标到达判断是重要根因之一。")
    elif data["oracle_stop_gain"] < 0.03:
        lines.append("结论：Oracle STOP 提升较小，大部分失败不是单纯 STOP 问题，更可能是路线规划/grounding 偏离。")
    else:
        lines.append("结论：Oracle STOP 有一定提升，但不是唯一主因，需要继续做 expert action 对齐。")
    lines.append("")
    lines.append("## 3. 失败轨迹最近距离分布")
    lines.append("")
    for key in ["<=3m", "3-5m", "5-10m", ">10m", "unknown"]:
        if failed_min_buckets.get(key, 0):
            lines.append(f"- min_NE {key}：{failed_min_buckets[key]}")
    lines.append("")
    lines.append("## 4. 可由 Oracle STOP 救回的 Episode")
    lines.append("")
    if not rescued:
        lines.append("没有失败 episode 可由 Oracle STOP 救回。")
    else:
        for row in sorted(rescued, key=lambda item: item.get("min_distance") or 999)[:200]:
            closest = row.get("closest_step") or {}
            lines.append(
                f"- `{row['scene_id']}_{row['episode_id']}` | index={row['episode_index']} | "
                f"min_NE={fmt(row['min_distance'])} at step={closest.get('step_id')} | "
                f"final_NE={fmt(row['ne'])} | OS={fmt(row['os'], 1)} | steps={row.get('steps')} | "
                f"video=`{row.get('video_file')}`"
            )
    lines.append("")
    lines.append("## 5. Oracle STOP 仍救不回的失败")
    lines.append("")
    lines.append("这些 episode 即使在轨迹中最近目标的一步强制 STOP，也不满足成功半径，优先归入路线规划/视觉语言 grounding/控制执行问题。")
    lines.append("")
    for key in ["<=3m", "3-5m", "5-10m", ">10m", "unknown"]:
        if unrecovered_ne_buckets.get(key, 0):
            lines.append(f"- final_NE {key}：{unrecovered_ne_buckets[key]}")
    lines.append("")
    for row in sorted(unrecovered, key=lambda item: item.get("ne") or 0, reverse=True)[:200]:
        closest = row.get("closest_step") or {}
        lines.append(
            f"- `{row['scene_id']}_{row['episode_id']}` | index={row['episode_index']} | "
            f"min_NE={fmt(row['min_distance'])} at step={closest.get('step_id')} | "
            f"final_NE={fmt(row['ne'])} | OS={fmt(row['os'], 1)} | steps={row.get('steps')} | "
            f"video=`{row.get('video_file')}`"
        )
    lines.append("")
    lines.append("## 6. 下一步建议")
    lines.append("")
    lines.append("- 对可救回集合：重点检查 STOP token 何时生成、目标附近视觉帧是否明确，后续可做 STOP verifier 或 near-miss 加权训练。")
    lines.append("- 对救不回集合：继续做 expert action 对齐，定位第一次偏离发生在初始朝向、landmark 转换、碰撞恢复还是长程记忆阶段。")
    lines.append("- 对 OS=1 但未成功集合：优先回看视频，判断是穿过目标点未停、停点半径误差，还是成功判定与语言目标不一致。")
    lines.append("")
    if data["missing_logs"]:
        lines.append("## 7. 缺失日志")
        lines.append("")
        lines.append(f"- 缺失 episode log 数量：{len(data['missing_logs'])}")
        for item in data["missing_logs"][:100]:
            lines.append(f"  - {item}")
        lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--success_distance", type=float, default=3.0)
    parser.add_argument("--report_path", default=None)
    parser.add_argument("--json_path", default=None)
    args = parser.parse_args()

    data = analyze(args.output_dir, args.success_distance)
    report_path = args.report_path or os.path.join(args.output_dir, "oracle_stop_report.md")
    json_path = args.json_path or os.path.join(args.output_dir, "oracle_stop_replay.json")
    write_markdown(report_path, data)
    write_json(json_path, data)
    print(report_path)
    print(json_path)


if __name__ == "__main__":
    main()
