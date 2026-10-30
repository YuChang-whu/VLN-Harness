#!/usr/bin/env python3
"""Compare Stage 5 action-interface ablation experiments against a baseline."""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from statistics import mean


def as_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def safe_mean(values):
    values = [v for v in values if v is not None]
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
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "scene_id" in item and "episode_id" in item:
                rows.append(item)
    return rows


def key_of(result):
    return f"{result.get('scene_id')}::{result.get('episode_id')}"


def load_filter_keys(filter_file):
    if not filter_file:
        return None
    with open(filter_file, "r") as f:
        data = json.load(f)
    rows = data.get("episodes", data if isinstance(data, list) else [])
    return {f"{row.get('scene_id')}::{row.get('episode_id')}" for row in rows if isinstance(row, dict)}


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


def log_action_stats(result, output_dir):
    log, log_path = load_episode_log(result, output_dir)
    steps = log.get("steps", []) if log else []
    postprocess_counter = Counter()
    action_source_counter = Counter()
    override_count = 0
    oracle_stop_rescue_candidates = 0
    max_turn_run = 0
    current_turn_run = 0
    last_turn = None
    collision_count = 0
    for step in steps:
        for intervention in step.get("postprocess_interventions") or []:
            postprocess_counter[intervention] += 1
        action_source_counter[step.get("action_source", "model")] += 1
        if step.get("overrode_model_action"):
            override_count += 1
        if step.get("oracle_stop_rescue_candidate"):
            oracle_stop_rescue_candidates += 1
        action = step.get("action")
        if action in (2, 3):
            if last_turn in (2, 3):
                current_turn_run += 1
            else:
                current_turn_run = 1
            last_turn = action
            max_turn_run = max(max_turn_run, current_turn_run)
        else:
            current_turn_run = 0
            last_turn = None
        metrics = step.get("post_metrics") or {}
        collisions = metrics.get("collisions")
        if isinstance(collisions, dict):
            if collisions.get("count") is not None:
                collision_count = int(collisions.get("count") or 0)
            elif collisions.get("is_collision"):
                collision_count += 1
    summary = result.get("postprocess_summary") or (log.get("postprocess_summary") if log else {}) or {}
    return {
        "episode_log": log_path or result.get("episode_log"),
        "override_count": override_count,
        "postprocess_counter": dict(postprocess_counter),
        "action_source_counter": dict(action_source_counter),
        "oracle_stop_rescue_candidates": oracle_stop_rescue_candidates or int(summary.get("oracle_stop_rescue_candidates") or 0),
        "limit_turn_interventions": int(summary.get("limit_turn_interventions") or postprocess_counter.get("limit_consecutive_turns", 0)),
        "stuck_recovery_interventions": int(summary.get("stuck_recovery_interventions") or postprocess_counter.get("stuck_recovery", 0)),
        "stop_verifier_blocks": int(summary.get("stop_verifier_blocks") or postprocess_counter.get("stop_verifier_block_stop", 0)),
        "stop_verifier_allows": int(summary.get("stop_verifier_allows") or 0),
        "max_turn_run": max_turn_run,
        "collision_count": collision_count,
    }


def result_metrics(result):
    return {
        "success": as_float(result.get("success")) or 0.0,
        "spl": as_float(result.get("spl")),
        "os": as_float(result.get("os")),
        "ne": as_float(result.get("ne")),
        "steps": int(result.get("steps") or 0),
    }


def aggregate(records):
    return {
        "count": len(records),
        "sr": safe_mean([r["metrics"]["success"] for r in records]),
        "spl": safe_mean([r["metrics"]["spl"] for r in records]),
        "os": safe_mean([r["metrics"]["os"] for r in records]),
        "ne": safe_mean([r["metrics"]["ne"] for r in records]),
        "steps": safe_mean([r["metrics"]["steps"] for r in records]),
        "override_count": sum(r["log_stats"]["override_count"] for r in records),
        "limit_turn_interventions": sum(r["log_stats"]["limit_turn_interventions"] for r in records),
        "stuck_recovery_interventions": sum(r["log_stats"]["stuck_recovery_interventions"] for r in records),
        "stop_verifier_blocks": sum(r["log_stats"]["stop_verifier_blocks"] for r in records),
        "stop_verifier_allows": sum(r["log_stats"]["stop_verifier_allows"] for r in records),
        "oracle_stop_rescue_candidates": sum(r["log_stats"]["oracle_stop_rescue_candidates"] for r in records),
        "max_turn_run_avg": safe_mean([r["log_stats"]["max_turn_run"] for r in records]),
        "collision_count_avg": safe_mean([r["log_stats"]["collision_count"] for r in records]),
    }


def parse_experiment(value):
    if "=" in value:
        name, path = value.split("=", 1)
    elif ":" in value:
        name, path = value.split(":", 1)
    else:
        path = value
        name = os.path.basename(os.path.normpath(path))
    return name, path


def compare(args):
    filter_keys = load_filter_keys(args.filter_file)
    baseline_results = {key_of(r): r for r in load_results(args.baseline_dir)}
    if filter_keys is not None:
        baseline_results = {k: v for k, v in baseline_results.items() if k in filter_keys}

    baseline_records = [
        {
            "key": key,
            "result": result,
            "metrics": result_metrics(result),
            "log_stats": log_action_stats(result, args.baseline_dir),
        }
        for key, result in baseline_results.items()
    ]
    baseline_agg = aggregate(baseline_records)

    experiments = []
    for exp_value in args.experiment:
        name, exp_dir = parse_experiment(exp_value)
        exp_results = {key_of(r): r for r in load_results(exp_dir)}
        keys = sorted(set(baseline_results) & set(exp_results))
        missing = sorted(set(baseline_results) - set(exp_results))
        paired = []
        rescued = []
        regressed = []
        for key in keys:
            base = baseline_results[key]
            exp = exp_results[key]
            exp_record = {
                "key": key,
                "scene_id": exp.get("scene_id"),
                "episode_id": exp.get("episode_id"),
                "instruction": exp.get("episode_instruction", base.get("episode_instruction", "")),
                "baseline": result_metrics(base),
                "metrics": result_metrics(exp),
                "log_stats": log_action_stats(exp, exp_dir),
                "video_file": exp.get("video_file"),
                "episode_log": exp.get("episode_log"),
            }
            exp_record["rescued"] = exp_record["baseline"]["success"] < 1.0 and exp_record["metrics"]["success"] >= 1.0
            exp_record["regressed"] = exp_record["baseline"]["success"] >= 1.0 and exp_record["metrics"]["success"] < 1.0
            exp_record["delta_ne"] = (
                exp_record["metrics"]["ne"] - exp_record["baseline"]["ne"]
                if exp_record["metrics"]["ne"] is not None and exp_record["baseline"]["ne"] is not None
                else None
            )
            exp_record["delta_steps"] = exp_record["metrics"]["steps"] - exp_record["baseline"]["steps"]
            paired.append(exp_record)
            if exp_record["rescued"]:
                rescued.append(exp_record)
            if exp_record["regressed"]:
                regressed.append(exp_record)
        exp_agg = aggregate(paired)
        experiments.append({
            "name": name,
            "dir": exp_dir,
            "count": len(paired),
            "missing": missing,
            "aggregate": exp_agg,
            "rescued_count": len(rescued),
            "regressed_count": len(regressed),
            "rescued": rescued,
            "regressed": regressed,
            "records": paired,
        })

    os.makedirs(args.output_dir, exist_ok=True)
    data = {
        "baseline_dir": args.baseline_dir,
        "filter_file": args.filter_file,
        "baseline": baseline_agg,
        "experiments": experiments,
    }
    json_path = os.path.join(args.output_dir, "action_interface_ablation_comparison.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    md_path = os.path.join(args.output_dir, "action_interface_ablation_report.md")
    write_md(md_path, data, json_path)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


def write_row(lines, label, stats, baseline=None, rescued=None, regressed=None):
    delta_sr = None
    delta_ne = None
    if baseline is not None:
        if stats.get("sr") is not None and baseline.get("sr") is not None:
            delta_sr = stats["sr"] - baseline["sr"]
        if stats.get("ne") is not None and baseline.get("ne") is not None:
            delta_ne = stats["ne"] - baseline["ne"]
    lines.append(
        f"| {label} | {stats['count']} | {fmt(stats.get('sr'))} | {fmt(delta_sr)} | "
        f"{fmt(stats.get('spl'))} | {fmt(stats.get('ne'))} | {fmt(delta_ne)} | "
        f"{fmt(stats.get('steps'))} | {rescued if rescued is not None else '-'} | {regressed if regressed is not None else '-'} | "
        f"{stats.get('override_count', 0)} | {stats.get('limit_turn_interventions', 0)} | "
        f"{stats.get('stuck_recovery_interventions', 0)} | {stats.get('stop_verifier_blocks', 0)} | "
        f"{stats.get('oracle_stop_rescue_candidates', 0)} |"
    )


def write_md(path, data, json_path):
    lines = []
    lines.append("# 阶段 5：动作接口消融实验报告")
    lines.append("")
    lines.append(f"Baseline：`{data['baseline_dir']}`")
    if data.get("filter_file"):
        lines.append(f"Filter：`{data['filter_file']}`")
    lines.append("")
    lines.append("## 1. 总体对照")
    lines.append("")
    lines.append("| 实验 | 数量 | SR | SR变化 | SPL | NE | NE变化 | Steps | 救回 | 退化 | 覆盖动作数 | TURN限制 | Stuck恢复 | STOP拦截 | 可救回STOP候选 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    write_row(lines, "baseline", data["baseline"])
    for exp in data["experiments"]:
        write_row(lines, exp["name"], exp["aggregate"], data["baseline"], exp["rescued_count"], exp["regressed_count"])
    lines.append("")
    lines.append("## 2. 如何判断")
    lines.append("")
    lines.append("- TURN限制提升：说明连续转向/动作粒度是系统性问题。")
    lines.append("- Stuck恢复提升：说明局部控制、碰撞恢复或几何反馈闭环不足。")
    lines.append("- STOP verifier 中可救回候选多：说明模型常接近目标但没有 STOP；如果直接提升少，说明仅靠 STOP gate 还不够。")
    lines.append("- Combined 提升最大但单项不明显：说明失败由多个动作接口问题耦合产生。")
    lines.append("- 所有策略无提升：更可能是视觉 grounding 或高层路径规划问题。")
    lines.append("")
    for exp in data["experiments"]:
        lines.append(f"## {exp['name']} 救回样例")
        lines.append("")
        if exp["rescued"]:
            for row in exp["rescued"][:50]:
                lines.append(
                    f"- {row['scene_id']}_{row['episode_id']} NE {fmt(row['baseline']['ne'])}->{fmt(row['metrics']['ne'])} "
                    f"steps {row['baseline']['steps']}->{row['metrics']['steps']} interventions={row['log_stats']}"
                )
        else:
            lines.append("- 无救回样例。")
        lines.append("")
        if exp["regressed"]:
            lines.append(f"### {exp['name']} 退化样例")
            for row in exp["regressed"][:30]:
                lines.append(
                    f"- {row['scene_id']}_{row['episode_id']} NE {fmt(row['baseline']['ne'])}->{fmt(row['metrics']['ne'])} "
                    f"steps {row['baseline']['steps']}->{row['metrics']['steps']}"
                )
            lines.append("")
    lines.append(f"完整逐 episode 数据：`{json_path}`")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_dir", required=True)
    parser.add_argument("--experiment", action="append", required=True, help="name=path; repeat for multiple experiments")
    parser.add_argument("--filter_file", default=None)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    compare(args)


if __name__ == "__main__":
    main()
