#!/usr/bin/env python3
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

MANUAL_REVIEW_REASON = "未覆盖/需人工复核"


def load_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
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
    if log_path and os.path.exists(log_path):
        path = log_path
    elif log_path and os.path.exists(os.path.join(output_dir, log_path)):
        path = os.path.join(output_dir, log_path)
    else:
        matches = []
        log_dir = os.path.join(output_dir, "episode_logs")
        if os.path.isdir(log_dir):
            suffix = f"_{result['scene_id']}_{result['episode_id']}.json"
            matches = [os.path.join(log_dir, name) for name in os.listdir(log_dir) if name.endswith(suffix)]
        if not matches:
            return None
        path = sorted(matches)[0]
    with open(path, "r") as f:
        return json.load(f)


def actions_from_log(log):
    if not log:
        return []
    return [int(step.get("action", -1)) for step in log.get("steps", [])]


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


def distances_from_log(log):
    distances = []
    if not log:
        return distances
    for step in log.get("steps", []):
        metrics = step.get("post_metrics") or step.get("pre_metrics") or {}
        value = metrics.get("distance_to_goal")
        if isinstance(value, (int, float)) and math.isfinite(value):
            distances.append(float(value))
    return distances


def collision_count(log):
    if not log:
        return 0
    last = None
    for step in log.get("steps", []):
        metrics = step.get("post_metrics") or {}
        collisions = metrics.get("collisions")
        if isinstance(collisions, dict):
            if "count" in collisions:
                last = collisions.get("count")
            elif "is_collision" in collisions and collisions.get("is_collision"):
                last = (last or 0) + 1
    return int(last or 0)


def action_summary(log):
    actions = actions_from_log(log)
    counts = Counter(actions)
    total = len(actions)
    turn_count = counts[2] + counts[3]
    forward_count = counts[1]
    stop_positions = [idx for idx, action in enumerate(actions) if action == 0]
    distances = distances_from_log(log)
    generations = log.get("generations", []) if log else []
    return {
        "total": total,
        "counts": counts,
        "turn_ratio": turn_count / total if total else 0.0,
        "forward_ratio": forward_count / total if total else 0.0,
        "stop_positions": stop_positions,
        "has_stop": bool(stop_positions),
        "last_action": actions[-1] if actions else None,
        "max_same_action_run": max_run_length(actions),
        "max_turn_run": max_run_length(actions, {2, 3}),
        "min_distance": min(distances) if distances else None,
        "final_distance_from_log": distances[-1] if distances else None,
        "collisions": collision_count(log),
        "generation_count": len(generations),
        "fallback_count": sum(1 for gen in generations if gen.get("fallback_to_stop")),
        "actions": actions,
    }


def classify_failure(result, log):
    summary = action_summary(log)
    ne = float(result.get("ne", 0.0))
    os_value = float(result.get("os", 0.0))
    steps = int(result.get("steps", summary["total"] or 0))
    min_distance = summary["min_distance"]
    reasons = []

    if summary["fallback_count"] > 0:
        reasons.append("模型输出未解析出有效动作，触发 STOP fallback")
    if summary["has_stop"] and ne > 3.0:
        stop_step = summary["stop_positions"][0]
        if stop_step < max(8, int(steps * 0.5)):
            reasons.append("过早 STOP，距离终点仍较远")
        else:
            reasons.append("STOP 位置不准，未进入成功半径")
    if not summary["has_stop"]:
        reasons.append("未输出 STOP，episode 结束前没有主动停下")
    if os_value >= 1.0 and float(result.get("success", 0.0)) < 1.0:
        reasons.append("曾经接近/经过目标区域，但最终没有在目标处停止")
    if min_distance is not None and min_distance <= 3.0 and ne > 3.0:
        reasons.append("到达目标附近后又离开，终止时距离变大")
    if summary["turn_ratio"] >= 0.65 or summary["max_turn_run"] >= 8:
        reasons.append("转向动作占比过高，存在原地打转或方向振荡")
    if summary["forward_ratio"] <= 0.2 and steps >= 12:
        reasons.append("前进动作不足，探索推进不够")
    if summary["collisions"] >= 5:
        reasons.append("碰撞/受阻次数较多，路径执行被环境几何影响")
    if ne >= 10.0 and os_value < 1.0:
        reasons.append("最终距离很远，整体导航方向明显偏离")
    if not reasons:
        reasons.append(MANUAL_REVIEW_REASON)
    return reasons, summary


def normalize_instruction(text):
    text = re.sub(r"\s+", " ", (text or "").strip().lower())
    return text


def route_group_key(result):
    try:
        episode_id = int(result["episode_id"])
        return f"{result['scene_id']}::{episode_id // 3}"
    except Exception:
        return f"{result.get('scene_id')}::{result.get('episode_id')}"


def fmt_float(value, digits=3):
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "NA"


def build_report(output_dir, report_path):
    results = load_jsonl(os.path.join(output_dir, "result.json"))
    if not results:
        raise SystemExit(f"No episode results found in {output_dir}/result.json")

    logs = {idx: load_episode_log(result, output_dir) for idx, result in enumerate(results)}
    failures = []
    successes = []
    reason_counter = Counter()
    failure_details = []

    for idx, result in enumerate(results):
        success = float(result.get("success", 0.0)) >= 1.0
        if success:
            successes.append(result)
        else:
            reasons, summary = classify_failure(result, logs[idx])
            failures.append(result)
            for reason in reasons:
                reason_counter[reason] += 1
            failure_details.append((result, reasons, summary, logs[idx]))

    instruction_groups = defaultdict(list)
    route_groups = defaultdict(list)
    for result in results:
        instruction_groups[normalize_instruction(result.get("episode_instruction", ""))].append(result)
        route_groups[route_group_key(result)].append(result)

    mixed_instruction_groups = [items for items in instruction_groups.values() if len(items) > 1 and 0 < sum(float(i.get("success", 0.0)) >= 1.0 for i in items) < len(items)]
    mixed_route_groups = [items for items in route_groups.values() if len(items) > 1 and 0 < sum(float(i.get("success", 0.0)) >= 1.0 for i in items) < len(items)]

    avg_spl = mean(float(r.get("spl", 0.0)) for r in results)
    avg_ne = mean(float(r.get("ne", 0.0)) for r in results)
    sr = len(successes) / len(results)

    lines = []
    lines.append("# StreamVLN 失败案例分析报告")
    lines.append("")
    lines.append(f"分析目录：`{output_dir}`")
    lines.append("")
    lines.append("## 1. 总体结果")
    lines.append("")
    lines.append(f"- 总 episode 数：{len(results)}")
    lines.append(f"- 成功数：{len(successes)}")
    lines.append(f"- 失败数：{len(failures)}")
    lines.append(f"- 成功率 SR：{sr:.3f}")
    lines.append(f"- 平均 SPL：{avg_spl:.3f}")
    lines.append(f"- 平均终点距离 NE：{avg_ne:.3f}")
    lines.append("")

    lines.append("## 2. 失败原因统计")
    lines.append("")
    if reason_counter:
        for reason, count in reason_counter.most_common():
            lines.append(f"- {reason}：{count} 个 episode")
    else:
        lines.append("- 没有失败 episode。")
    lines.append("")

    manual_review_details = [
        (result, reasons, summary, log)
        for result, reasons, summary, log in failure_details
        if MANUAL_REVIEW_REASON in reasons
    ]

    lines.append("## 3. 未覆盖/需人工复核")
    lines.append("")
    lines.append(
        f"- 数量：{len(manual_review_details)} 个 episode。"
    )
    if manual_review_details:
        lines.append("- 这些样本没有命中当前规则库中的明确失败类型，报告不强行归因；建议结合视频和 episode JSON 人工复核。")
        for result, reasons, summary, log in manual_review_details[:50]:
            lines.append(
                f"  - {result.get('scene_id')}_{result.get('episode_id')}："
                f"SPL={fmt_float(result.get('spl'))}, NE={fmt_float(result.get('ne'))}, "
                f"steps={result.get('steps')}, video=`{result.get('video_file')}`"
            )
    else:
        lines.append("- 当前失败样本均命中了至少一个自动规则。")
    lines.append("")

    lines.append("## 4. 失败 Episode 明细")
    lines.append("")
    if not failure_details:
        lines.append("没有失败 episode。")
    for result, reasons, summary, log in failure_details:
        actions = summary["actions"]
        preview = " ".join(ACTION_NAMES.get(a, str(a)) for a in actions[:40])
        if len(actions) > 40:
            preview += " ..."
        lines.append(f"### {result.get('scene_id')}_{result.get('episode_id')} | index {result.get('video_file', '').split('/')[-1].split('_')[0] if result.get('video_file') else 'NA'}")
        lines.append("")
        lines.append(f"- 指令：{result.get('episode_instruction', '').strip()}")
        lines.append(f"- 指标：success={fmt_float(result.get('success'), 1)}, SPL={fmt_float(result.get('spl'))}, OS={fmt_float(result.get('os'))}, NE={fmt_float(result.get('ne'))}, steps={result.get('steps')}")
        lines.append(f"- 视频：`{result.get('video_file')}`")
        lines.append(f"- 详细日志：`{result.get('episode_log')}`")
        lines.append(f"- 动作统计：STOP={summary['counts'].get(0, 0)}, FORWARD={summary['counts'].get(1, 0)}, LEFT={summary['counts'].get(2, 0)}, RIGHT={summary['counts'].get(3, 0)}, turn_ratio={summary['turn_ratio']:.2f}, min_NE={fmt_float(summary['min_distance'])}")
        lines.append(f"- 失败原因：{'；'.join(reasons)}")
        lines.append(f"- 动作序列前缀：`{preview}`")
        lines.append("")

    lines.append("## 5. 同一指令/同一路径下的部分成功现象")
    lines.append("")
    if mixed_instruction_groups:
        lines.append("### 完全相同指令的部分成功组")
        for items in mixed_instruction_groups[:20]:
            ok = sum(float(i.get("success", 0.0)) >= 1.0 for i in items)
            lines.append(f"- `{items[0].get('episode_instruction', '').strip()}`：{ok}/{len(items)} 成功")
    else:
        lines.append("- 未发现完全相同指令文本下同时存在成功和失败的 episode。")
    lines.append("")
    if mixed_route_groups:
        lines.append("### 同一路径/相邻三指令组的部分成功组")
        for items in mixed_route_groups[:30]:
            ok = sum(float(i.get("success", 0.0)) >= 1.0 for i in items)
            ids = ", ".join(f"{i.get('scene_id')}_{i.get('episode_id')}:SPL={fmt_float(i.get('spl'))}" for i in items)
            lines.append(f"- {route_group_key(items[0])}：{ok}/{len(items)} 成功；{ids}")
    else:
        lines.append("- 未发现同一路径/相邻三指令组中同时存在成功和失败的 episode。")
    lines.append("")

    lines.append("## 6. 结论")
    lines.append("")
    if reason_counter:
        top_reasons = [reason for reason, _ in reason_counter.most_common(3)]
        lines.append("本轮失败主要集中在：" + "；".join(top_reasons) + "。")
        lines.append("建议优先回看上述类别对应的视频和 episode JSON，重点检查模型在生成动作批次切换时是否发生方向漂移、是否接近目标后未及时 STOP，以及转向动作是否出现连续振荡。")
    else:
        lines.append("本轮没有失败样本，建议扩大评测规模或切换 val_unseen 做更严格压力测试。")
    lines.append("")

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    return report_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--report_path", default=None)
    args = parser.parse_args()
    report_path = args.report_path or os.path.join(args.output_dir, "failure_report.md")
    path = build_report(args.output_dir, report_path)
    print(path)


if __name__ == "__main__":
    main()
