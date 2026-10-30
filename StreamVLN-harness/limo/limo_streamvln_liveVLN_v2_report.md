# limo_streamvln_liveVLN_v2 修改报告

## 修改目标

适配服务器端最新的 StreamVLN-harness 子目标切换逻辑：服务器现在会将 `action_e == STOP` 或显式 `guard_g == STOP` 判定为当前子目标完成，并切换到下一个子目标。小车端因此不能再把 `guard_g == STOP` 直接当作整个任务结束。

## 新增文件

- `limo_streamvln_liveVLN_v2.py`

原文件 `limo_streamvln_liveVLN.py` 保持不动，v2 文件用于小车端部署测试。

## 核心修改

1. `guard_g == ACTION_STOP` 不再 `return` 退出程序。

   原逻辑：

   ```python
   if guard_g == ACTION_STOP:
       rospy.loginfo("guard_g is STOP. Stop and exit.")
       self.publish_stop()
       return
   ```

   v2 逻辑：

   ```text
   guard_g == STOP
     -> publish_stop
     -> blocking refresh /eval_vln
     -> 如果 terminate=true 才退出
     -> 否则继续等待下一个 subgoal/action
   ```

2. `action_e == ACTION_STOP` 也不再无条件退出。

   v2 会先检查服务器响应里的：

   ```json
   {"terminate": true}
   ```

   只有 `terminate=true` 才结束整个任务。否则认为这是子目标切换、hold 或服务器等待下一轮观测。

3. 支持服务器 VQA hold 状态。

   如果服务器返回：

   ```json
   {"hold": true, "hold_reason": "hitl_vqa_active"}
   ```

   小车端会：

   ```text
   publish_stop
   不退出
   继续低频刷新 /eval_vln
   ```

4. 新增辅助函数：

   - `should_terminate(payload)`
   - `is_hold_response(payload)`
   - `hold_before_refresh(reason)`
   - `blocking_refresh_after_stop(phase, executed_guard=ACTION_STOP)`

5. 新增启动参数：

   ```bash
   --hold-refresh-pause 0.2
   ```

   用于 VQA hold 或 guard STOP 后刷新前的短暂停顿。

6. 将 OpenCV buffer 转换从 `encoded.tostring()` 改为 `encoded.tobytes()`，避免 deprecated 用法。

## 推荐运行命令

```bash
python ~/hxdl/limo_streamvln_liveVLN_v2.py   --server http://127.0.0.1:6010/eval_vln   --forward-speed 0.25   --max-linear 0.30   --action-duration 0.5   --hold-refresh-pause 0.2   --instruction "Move toward the yellow box. Turn left when you are close to the yellow box. Stop in front of the green trash can."
```

## 当前语义

```text
action_e == STOP && terminate == true
  -> 全局任务结束，小车退出

action_e == STOP && terminate == false
  -> hold，继续请求服务器

guard_g == STOP
  -> 当前 committed guard 是 STOP
  -> 小车停住一拍
  -> 继续请求服务器，让服务器返回下一个子目标动作

hold == true
  -> 服务器正在 HITL/VQA 人工介入
  -> 小车停住
  -> 不退出，继续刷新
```

## 注意事项

- 如果服务器端已经切换到下一个子目标，小车端需要继续 POST 才能拿到新子目标对应的动作。
- 只有服务器返回 `terminate=true` 时，小车端才应该真正退出导航循环。
- 如果小车端仍使用旧版脚本，遇到 `guard_g == STOP` 会直接停止并退出，导致服务器虽然完成了子目标但小车不再请求下一步。
