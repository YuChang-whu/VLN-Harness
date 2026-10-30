# Real-World VLN Agent Harness Design

## 0. 文档定位

这份设计只面向真实世界部署，不纳入 Habitat / VLN-CE 的离线测评闭环。

目标是把现有 StreamVLN 真机推理代码升级为一套可部署、可恢复、可审计、可人工介入的 VLN Agent Harness。底层模型仍负责根据 RGB-D 历史和语言指令生成导航动作；Harness 负责把模型接入真实机器人运行时，处理任务规划、上下文管理、动作缓冲、安全监督、日志持久化和异常恢复。

当前代码中的相关基础：

- `streamvln/streamvln_agent.py`：StreamVLN 模型封装，维护图像、深度、位姿、KV cache、memory token，并输出离散动作序列。
- `limo/limo_realworld_server.py`：LIMO/真实机器人 HTTP 推理服务，接收 RGB-D multipart 请求，返回动作序列。
- `limo/limo_realworld_server_liveVLN.py`：LiveVLN 风格服务端，把模型动作拆成 `action_e`、`guard_g`、`revisable_tail`。
- `realworld/go2_vln_client.py`：Go2 端 ROS 客户端，订阅 RGB/odom，调用远端推理服务，把离散动作转成目标位姿，再由 PID 控制执行。
- `LiveVLN/real_world_demo/*`：提供双线程连续动作执行范式，即执行 guard buffer 的同时后台刷新下一段动作。

## 1. 真实世界部署的核心判断

真实机器人 VLN 不应该设计成：

```text
observe -> wait model inference -> execute all actions -> observe again
```

这种阻塞式链路会暴露模型推理延迟，使机器人产生明显的 stop-and-go 行为。更合理的真机架构应是：

```text
observe + state update
  -> background inference refresh
  -> committed safe prefix
  -> guarded continuous execution
  -> runtime safety supervision
  -> revise tail / recover / stop
```

结合已有根因分析和真机经验，关键约束是：

1. 真机没有 `distance_to_goal`，不能使用仿真指标做 STOP verifier。
2. 真机底盘存在打滑、漂移、延迟、控制误差、深度噪声和网络抖动。
3. 模型输出的小错误会被连续执行放大，因此模型外必须有动作安全层。
4. 增加短时观测质量和近期帧数，通常比盲目拉长历史更重要。
5. 自然语言里的 `turn left`、`doorway`、`wait near`、`stop beside` 等表达和离散动作接口之间存在落差，应在 prompt / action interface 层显式处理。
6. 真机部署必须 fail-closed：解析失败、通信失败、深度异常、控制异常时默认输出 STOP。

## 2. 总体架构

推荐架构如下：

```text
User / Operator
      |
      v
Task Session Manager
      |
      +-- Planning Manager
      +-- Runtime Context Manager
      +-- Virtual Filesystem / Event Log
      +-- Skill Loader
      +-- Subagent Router
      |
      v
Real-World Runtime Orchestrator
      |
      +-- Perception Adapter
      +-- State Estimator
      +-- Prompt & Action Interface Adapter
      +-- StreamVLN Policy Server
      +-- Live Action Buffer
      +-- Action Verifier
      +-- Safety Supervisor
      +-- Motion Executor
      +-- Human Approval Gate
      |
      v
Robot Platform
  RGB-D Camera / Odometry / Chassis Controller / E-stop
```

这套架构分成三层：

```text
Agent Harness Layer
  管任务、上下文、文件、技能、子 agent、人工介入。

Policy Runtime Layer
  管模型调用、prompt、动作解析、动作缓冲、运行时状态。

Robot Control Layer
  管 ROS 传感器、里程计、速度控制、底盘安全和急停。
```

其中最重要的设计原则是：模型只给出候选动作，最终能不能执行由 Harness 和机器人安全层共同决定。

## 3. 组件职责与耦合关系

### 3.1 Task Session Manager

负责一个真实导航任务的生命周期。

输入：

```json
{
  "session_id": "rw_20260902_001",
  "instruction": "Go forward to enter the room and stop near the table.",
  "robot": "limo",
  "mode": "realworld",
  "operator": "user"
}
```

职责：

- 创建任务 session。
- 初始化 todos。
- 管理 reset / pause / resume / terminate。
- 写入不可变事件日志。
- 给各模块分配 session scoped state。
- 在高风险操作前调用 Human Approval Gate。

与其他组件的耦合：

- 调用 Planning Manager 生成任务级待办。
- 调用 Runtime Orchestrator 执行每一轮导航。
- 调用 Virtual Filesystem 保存观测、响应和日志。
- 调用 Skill Loader 按需加载真实机器人、安全、失败分析等技能。

### 3.2 Planning Manager

Planning 在真机 VLN 中不负责生成全局最短路，而是把自然语言任务拆成可追踪、可恢复的运行时目标。

推荐 todos：

```json
[
  {
    "id": "understand_instruction",
    "content": "提取目标地点、路径约束、停止条件和危险词",
    "status": "completed"
  },
  {
    "id": "stabilize_initial_observation",
    "content": "采集初始 RGB-D 和 odom，确认传感器可用",
    "status": "completed"
  },
  {
    "id": "navigate_to_subgoal",
    "content": "根据当前视觉线索执行下一段短动作",
    "status": "in_progress"
  },
  {
    "id": "verify_stop_condition",
    "content": "用视觉、深度、位移和指令语义共同判断是否可以停止",
    "status": "pending"
  }
]
```

规划分三层：

```text
Mission Plan
  用户指令中的最终目标与停止条件。

Semantic Waypoint Plan
  走廊、门口、房间入口、桌子、床、沙发、楼梯等语义中间点。

Execution Plan
  当前最多 1-2 个 committed 动作，加上可修订 tail。
```

注意：真实世界里不假设有地图或目标坐标，因此 Planning Manager 不做几何 oracle，只做语义进度和执行意图管理。

### 3.3 Perception Adapter

负责把真实机器人传感器变成模型和安全层都能消费的标准观测。

输入来源：

- RGB 图像。
- Depth 图。
- Camera intrinsic。
- Camera extrinsic / camera height。
- Odometry / IMU yaw / velocity。
- 时间戳和 frame id。

输出：

```json
{
  "frame_id": 42,
  "timestamp": 1798840920.123,
  "rgb_path": "sessions/rw_001/observations/rgb_0042.jpg",
  "depth_path": "sessions/rw_001/observations/depth_0042.png",
  "odom": {
    "x": 1.24,
    "y": -0.31,
    "yaw": 1.57,
    "linear_velocity": 0.2,
    "angular_velocity": 0.0
  },
  "health": {
    "rgb_ok": true,
    "depth_ok": true,
    "odom_ok": true,
    "latency_ms": 38
  }
}
```

与其他组件的耦合：

- 给 StreamVLN Policy Server 提供 RGB-D 和 intrinsic。
- 给 Safety Supervisor 提供近距离障碍、深度有效率、传感器延迟。
- 给 State Estimator 提供 odom 和运动反馈。
- 给 VFS 保存原始观测，避免把图片长期塞进上下文。

### 3.4 State Estimator

真机不能依赖仿真 `distance_to_goal`，但可以维护局部运行时状态：

```json
{
  "recent_displacement_m": 0.31,
  "recent_yaw_change_deg": 14.8,
  "stuck_score": 0.2,
  "turn_oscillation_score": 0.0,
  "loop_score": 0.1,
  "front_clearance_m": 0.78,
  "last_successful_motion_time": 1798840921.8
}
```

核心职责：

- 用 odom 判断执行动作是否真的生效。
- 检测连续 forward 但位移很小。
- 检测连续 turn、左右振荡、长时间空转。
- 根据深度估计前方通行风险。
- 形成给 STOP verifier 和 Safety Supervisor 使用的状态摘要。

### 3.5 Prompt & Action Interface Adapter

负责把用户自然语言转成更适合离散动作接口的模型输入，同时把模型输出转成统一动作对象。

Prompt 侧职责：

- 规范化容易出错的短语。
- 明确动作粒度：前进 25cm，左/右转 15 度。
- 把 `wait near`、`stop beside`、`doorway` 之类停点词转成显式 stop condition。
- 在必要时注入最近运行时状态摘要，而不是注入全部历史。

Action 侧职责：

- 解析 `STOP / ↑ / ← / →`。
- 兼容自然语言动作：`move forward`、`turn left 15 degrees`。
- 限制单次返回动作数量。
- 将动作切成 `action_e`、`guard_g`、`revisable_tail`。
- 解析失败时返回 STOP，并记录 `parse_error`。

统一动作格式：

```json
{
  "action_id": 1,
  "name": "MOVE_FORWARD",
  "distance_m": 0.25,
  "angle_deg": null,
  "source": "model",
  "committed": false
}
```

### 3.6 StreamVLN Policy Server

保留现有重模型服务端形态：模型跑在 GPU 服务器上，机器人端通过 HTTP 请求动作。

服务接口建议从当前 `/eval_vln` 扩展为：

```text
POST /sessions/{session_id}/infer
POST /sessions/{session_id}/reset
GET  /sessions/{session_id}/health
```

请求：

```json
{
  "instruction": "Go forward to enter the room.",
  "reset": false,
  "frame_id": 42,
  "run_model": true,
  "runtime_state": {
    "front_clearance_m": 0.78,
    "recent_actions": ["MOVE_FORWARD", "TURN_LEFT"]
  }
}
```

响应：

```json
{
  "ok": true,
  "action_e": 1,
  "guard_g": 2,
  "revisable_tail": [1, 0],
  "action_sequence": [1, 2, 1, 0],
  "llm_output": "↑ ← ↑ STOP",
  "generate_time_ms": 1420,
  "model_context": {
    "num_frames": 16,
    "num_history": 8,
    "step_id": 44
  }
}
```

耦合要求：

- Policy Server 只负责模型推理和基础动作解析。
- 不直接控制机器人。
- 不拥有最终执行权。
- 每次响应必须写入 session event log。

### 3.7 Live Action Buffer

这是新版架构的关键组件，用于解决真机 stop-and-go。

动作被分为三段：

```text
action_e
  立即执行动作，通常只有 1 个。

guard_g
  推理刷新期间允许继续执行的安全保护动作，通常 1 个。

revisable_tail
  模型预测的后续动作，只作为候选，下一轮观测回来后可被覆盖。
```

运行方式：

```text
Round 0:
  observe_0 -> infer -> action_e_0 + guard_g_0 + tail_0
  execute action_e_0

Round 1:
  while executing guard_g_0:
    observe_1 -> infer in background
  when new result returns:
    replace revisable tail
    choose next action_e / guard_g
```

核心原则：

- 只执行 committed prefix。
- tail 永远可修订。
- STOP 出现在 tail 中不立刻终止，只有 immediate action 或 verifier 确认后才 latch terminate。
- guard 动作必须经过 Safety Supervisor。
- 网络超时或后台推理失败时，guard 用尽后 STOP。

### 3.8 Action Verifier

Action Verifier 是模型和安全层之间的语义动作检查器。

检查项：

- 动作是否可解析。
- 是否连续 TURN 过多。
- 是否左右转振荡。
- STOP 是否过早。
- 是否长时间没有 forward 进展。
- 是否动作序列过长。
- 是否和当前指令停点词冲突。
- 是否在深度风险较高时仍前进。

输出：

```json
{
  "verdict": "modify",
  "original_actions": [2, 2, 2, 2],
  "verified_actions": [2, 1],
  "reason": "连续 TURN 达到阈值，插入一次短前进以打破原地旋转",
  "risk_level": "medium"
}
```

Action Verifier 只修改候选动作，不直接发速度命令。

### 3.9 Safety Supervisor

Safety Supervisor 是真机部署的最高优先级运行时模块。

它必须在每个动作执行前检查：

- 前方最小深度是否低于阈值。
- depth valid ratio 是否异常。
- odom 是否长时间无更新。
- 上一个动作是否执行失败。
- 当前速度是否超过限制。
- 是否收到人工暂停/急停。
- 是否网络延迟过高。
- 是否进入 recovery 模式。

决策：

```text
allow
  动作可以执行。

modify
  降速、缩短距离、缩小角度，或替换为 recovery 动作。

hold
  暂停等待新观测或人工确认。

stop
  立即停止并 latch 安全状态。
```

真实机器人动作链路必须是：

```text
model action -> Action Verifier -> Safety Supervisor -> Motion Executor
```

而不能是：

```text
model action -> Motion Executor
```

### 3.10 Motion Executor

负责把离散动作变成机器人底盘可执行控制。

对 LIMO / Go2 可以抽象为：

```text
MOVE_FORWARD 0.25m -> local target pose + PID / velocity command
TURN_LEFT 15deg    -> local yaw target + PID / angular velocity
TURN_RIGHT 15deg   -> local yaw target + PID / angular velocity
STOP               -> zero velocity + terminate/pause
```

职责：

- 维护当前目标位姿。
- 根据 odom 闭环控制。
- 执行动作完成判定。
- 超时判定。
- 发布 zero velocity。
- 回报执行结果。

执行结果：

```json
{
  "action_id": 1,
  "status": "completed",
  "duration_ms": 920,
  "odom_delta_m": 0.23,
  "yaw_delta_deg": 1.2,
  "collision_risk": false
}
```

### 3.11 Human Approval Gate

真实世界下，人工介入不是附加功能，而是上线边界。

必须人工确认：

- 开始真实机器人任务。
- 从 pause 恢复运动。
- 修改安全阈值。
- 连续 recovery 失败后继续尝试。
- 覆盖默认 STOP 决策。
- 启动长时间自主运行。

可自动执行但要记录：

- 单步低速 forward。
- 单步 15 度 turn。
- 保存观测日志。
- 后台推理刷新。

审批记录进入 event log：

```json
{
  "event": "approval_decided",
  "tool": "robot_start",
  "risk": "high",
  "outcome": "approved",
  "operator": "user",
  "time": 1798840922.1
}
```

## 4. 七个 Agent 能力在真机 VLN 中的重新映射

### 4.1 Planning

从“列任务”升级为“运行时任务板”：

- 记录 instruction understanding。
- 记录当前语义子目标。
- 记录执行状态：observing / inferring / executing / recovering / stopped。
- 记录未完成原因：等待模型、等待人工、等待传感器、等待恢复。

### 4.2 Virtual Filesystem

从“普通文件访问”升级为“真实机器人黑匣子”：

```text
sessions/
└── rw_20260902_001/
    ├── instruction.json
    ├── todos.json
    ├── events.jsonl
    ├── observations/
    │   ├── rgb_000001.jpg
    │   ├── depth_000001.png
    │   └── odom_000001.json
    ├── inference/
    │   ├── request_000001.json
    │   └── response_000001.json
    ├── actions/
    │   ├── proposed.jsonl
    │   ├── verified.jsonl
    │   └── executed.jsonl
    ├── safety/
    │   ├── decisions.jsonl
    │   └── incidents.jsonl
    └── summaries/
        ├── latest_state.md
        └── post_run_report.md
```

这套目录比聊天上下文更权威。恢复任务时，Agent 从 `events.jsonl` 和 `latest_state.md` 重建状态。

### 4.3 Subagents

真机部署中不建议让多个子 agent 同时控制机器人。推荐主 agent 单写执行权，子 agent 只做诊断和建议。

推荐子 agent：

- `PerceptionDiagnosticAgent`：检查 RGB-D 是否可用、是否曝光异常、深度是否大面积无效。
- `StopVerifierAgent`：结合图像、指令停点词和运动状态判断是否 STOP。
- `SafetyAuditAgent`：审查即将执行的动作是否越过安全规则。
- `FailureReviewAgent`：任务结束后读取日志，生成失败原因报告。
- `PromptRefinementAgent`：根据任务和历史失败经验给出 prompt normalization 建议。

关键约束：

```text
Only Main Runtime Orchestrator can commit actions.
Subagents can recommend, but cannot publish robot velocity commands.
```

### 4.4 Context Management

上下文分四层：

```text
Immediate Context
  当前帧、最近动作、当前安全状态、当前待执行动作。

Short-Horizon Context
  最近 8-16 帧摘要、最近位移、最近转向、是否卡住。

Episode Context
  当前 session 的完整事件日志和观测文件。

Long-Term Memory
  跨任务失败经验、用户偏好、场地注意事项、安全阈值记录。
```

策略：

- 图片、深度、完整模型输出落盘。
- 模型请求只携带必要最近帧和短摘要。
- 长历史通过摘要进入 prompt。
- 超过阈值时压缩为 `latest_state.md`。
- 不把未验证的模型推理当成事实记忆。

### 4.5 Code Execution & Sandbox

真机场景下的 execute 工具分成两类：

安全可自动运行：

- 查看日志。
- 生成报告。
- 检查服务健康。
- 编译检查。
- 启动本地 mock server。

需要审批：

- 启动真实机器人运动节点。
- 改 ROS topic。
- 改速度、距离、角度阈值。
- 清理真机日志。
- 安装依赖或改系统服务。

禁止自动执行：

- 删除原始真机数据。
- 覆盖模型 checkpoint。
- 绕过 Safety Supervisor 直接发底盘速度。
- 在未知场地长时间自主运行。

### 4.6 HITL

HITL 设计为三个等级：

```text
Level 0: Audit Only
  自动执行，完整记录。

Level 1: Confirm Before Risk
  中风险动作前确认，比如恢复运动、连续 recovery。

Level 2: Manual Control Required
  开始真机任务、急停后恢复、覆盖 STOP、修改安全阈值。
```

### 4.7 Skills

技能包按需加载，不在启动时全部塞入上下文。

推荐技能：

- `realworld-vln-runtime`：真机 session、传感器、动作闭环流程。
- `limo-safety`：LIMO 底盘速度、深度阈值、急停策略。
- `go2-safety`：Go2 ROS topic、SportMode 控制、安全限制。
- `live-action-buffer`：`action_e / guard_g / revisable_tail` 动作缓冲策略。
- `vln-stop-verifier`：真实世界 STOP 判断。
- `vln-failure-review`：读取 session log 生成复盘。
- `prompt-action-normalizer`：处理语言和离散动作接口错配。

## 5. 运行状态机

```text
BOOT
  加载配置、模型服务地址、安全阈值和机器人适配器。

WAIT_OPERATOR
  等待用户输入任务，并请求真机启动确认。

RESET_SESSION
  清空模型 memory、动作 buffer、runtime state，创建 session 目录。

OBSERVE
  获取 RGB-D、odom、时间戳，进行健康检查。

UPDATE_CONTEXT
  写事件日志，更新近期状态和 todos。

INFER_ASYNC
  后台请求 Policy Server，生成候选动作。

COMMIT_PREFIX
  从候选动作中选择短 committed prefix。

VERIFY_ACTION
  动作解析、STOP 检查、连续转向检查、prompt-action 约束检查。

SAFETY_CHECK
  深度、位移、速度、通信、人工暂停检查。

EXECUTE
  Motion Executor 执行单个动作，并持续监听安全中断。

MONITOR
  检查动作是否完成、是否卡住、是否需要 recovery。

RECOVER
  降速、短后退、转向扫描、重新观测，必要时请求人工。

STOPPED
  发布 zero velocity，锁定 terminate 状态。

REPORT
  生成本次真机任务报告。
```

## 6. 关键数据流

### 6.1 阻塞式基线

```text
Robot Client
  capture rgb
  -> HTTP /eval_vln
  -> wait model
  -> receive full action_sequence
  -> execute
```

缺点：

- 推理时机器人停住。
- full sequence 在真实世界中容易过期。
- 没有把 tail 和 committed action 区分开。
- 网络抖动会直接变成运动停顿。

### 6.2 新版连续部署流

```text
Robot Client / Runtime Orchestrator
  observe_t
  -> send inference request in background
  -> execute committed guard from previous round
  -> monitor safety during execution
  -> receive new action chunk
  -> verify and commit next short prefix
  -> discard stale tail
```

优势：

- 推理延迟被 guard buffer 吸收。
- 每轮只承诺很短动作，降低过时风险。
- 新观测可以覆盖旧 tail。
- 安全层随时可以抢占执行。

## 7. STOP Verifier 的真机版本

真机 STOP 判断不能使用 `distance_to_goal`。建议组合以下信号：

视觉信号：

- 目标物或目标区域是否可见。
- 门口、桌子、床、沙发、房间入口等语义地标是否稳定出现。
- 最近多帧目标区域是否变大或接近图像中心。

语言信号：

- 指令是否包含 `stop near`、`wait near`、`stop beside`。
- 当前地标是否匹配停点词。
- 是否已经完成最后一个语义子目标。

运动信号：

- 最近位移是否足够。
- 是否反复经过相似视角。
- 是否在目标附近出现低速调整。

安全信号：

- 前方是否过近。
- 是否继续前进会碰撞。
- 是否已进入窄门/桌边/墙边等需要停止确认的位置。

输出：

```json
{
  "stop_decision": "confirm_stop",
  "confidence": 0.72,
  "evidence": [
    "instruction contains stop near table",
    "table-like object visible for 5 frames",
    "front clearance below slow threshold"
  ],
  "action": "STOP"
}
```

## 8. Recovery 设计

Recovery 不应由模型随意生成，应该是安全层的有限状态策略。

触发条件：

- 连续 forward 但 odom 位移低于阈值。
- 连续 turn 超过阈值。
- 左右转交替振荡。
- 深度显示前方近障碍。
- 动作超时。
- 推理服务超时。

Recovery 动作集合：

```text
STOP_AND_REOBSERVE
  立即停下，重新采集观测。

SLOW_TURN_SCAN
  低速小角度转向，寻找开阔方向。

SHORT_BACKOFF
  如果底盘支持且安全，短距离后退。

ASK_HUMAN
  多次恢复失败后请求人工接管。
```

Recovery 成功后回到 `OBSERVE`，失败后进入 `STOPPED` 或 `WAIT_OPERATOR`。

## 9. 事件日志设计

真机日志必须能回答四个问题：

1. 模型当时看到了什么？
2. 模型建议了什么？
3. Harness 为什么改了或拒绝动作？
4. 机器人最后实际执行了什么？

推荐事件类型：

```text
session_started
instruction_received
todo_written
observation_saved
inference_requested
inference_completed
action_proposed
action_verified
safety_decision
motion_started
motion_completed
motion_interrupted
recovery_started
approval_requested
approval_decided
session_stopped
post_run_report_written
```

事件必须 JSON 可序列化、只追加、不覆盖。图片和深度图通过路径引用。

## 10. 部署形态

推荐使用双进程或三进程部署：

```text
GPU Server
  StreamVLN Policy Server
  Heavy model inference
  Session inference log mirror

Robot Computer
  ROS sensor adapter
  Runtime Orchestrator
  Safety Supervisor
  Motion Executor
  Local event log

Operator Console
  Task input
  Approval / pause / resume / emergency stop
  Live status and post-run report
```

如果设备资源紧张，Operator Console 可以先用 CLI 或简单 HTTP 控制台代替。

## 11. 最小可行版本

第一版不要一次做满。建议 MVP：

1. 保留现有 `limo_realworld_server_liveVLN.py` 的 `action_e / guard_g / revisable_tail` 接口。
2. 在机器人端新增 Runtime Orchestrator，管理 action buffer 和后台请求。
3. 新增 Safety Supervisor：
   - 前方深度阈值；
   - odom stuck 检测；
   - 连续 TURN 限制；
   - 网络超时 STOP；
   - 人工 pause/stop。
4. 新增 session event log：
   - 保存 RGB-D；
   - 保存请求/响应；
   - 保存 proposed / verified / executed actions。
5. 新增简单 STOP verifier：
   - tail 里的 STOP 不立即终止；
   - immediate STOP 才终止；
   - 对低置信 STOP 可请求人工确认。
6. 把 prompt normalization 做成独立模块，而不是散落在 eval 代码里。

## 12. 第二阶段增强

第二阶段再加入：

- 子 agent 诊断，不给子 agent 控制权。
- 任务恢复：从 `events.jsonl` 重建 session 状态。
- 长期记忆：记录场地、失败模式、用户偏好。
- 更强视觉 STOP verifier。
- 运行后自动复盘报告。
- Operator Console。
- 技能包按需加载。

## 13. 推荐模块目录

```text
realworld_agent/
├── session.py
├── planning.py
├── context.py
├── vfs.py
├── events.py
├── perception.py
├── state_estimator.py
├── prompt_adapter.py
├── action_parser.py
├── action_buffer.py
├── action_verifier.py
├── safety_supervisor.py
├── motion_executor.py
├── policy_client.py
├── hitl.py
├── skills.py
├── subagents.py
└── reports.py
```

与现有代码的迁移关系：

```text
limo_realworld_server.py
  保留为 Policy Server 基础，逐步补 session API。

limo_realworld_server_liveVLN.py
  保留 Live Action Buffer 响应格式，并让终止逻辑只 latch immediate STOP。

realworld/go2_vln_client.py
  拆分为 perception.py、motion_executor.py、runtime_orchestrator.py。

streamvln/streamvln_agent.py
  尽量保持为模型策略模块，不塞入真机安全逻辑。
```

## 14. 最终架构结论

真实世界 VLN Agent 的核心不是让模型一次输出更长动作，而是控制哪些动作被承诺、什么时候重看世界、什么时候拒绝模型、什么时候让人介入。

最终闭环应是：

```text
Instruction
  -> Task Session
  -> Observe RGB-D / Odom
  -> Update Runtime Context
  -> Async Policy Inference
  -> Live Action Buffer
  -> Action Verifier
  -> Safety Supervisor
  -> Motion Executor
  -> Event Log
  -> Recover / Stop / Continue
```

这套设计把 StreamVLN 放在正确的位置：它是强大的导航策略模型，但不是机器人运行时的唯一决策者。真正可部署的 VLN Agent 需要 Harness 在外层承担任务管理、上下文恢复、动作安全、人工控制和持续复盘。
