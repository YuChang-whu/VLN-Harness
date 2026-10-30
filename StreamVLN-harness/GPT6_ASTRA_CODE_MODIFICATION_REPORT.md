# GPT-6 Astra 代码修改报告

## 完成结果

已增加一个可选的 GPT-6 Astra 推理路径。原项目文件没有被覆盖或改写；原 Qwen3-8B + StreamVLN 入口、动作语义、HTTP 协议、session 状态机和 VQA 实现均保留。

新增启动器接收 `--GPT`：

- 有 `--GPT`：Astra 负责完整指令的意图解析，以及每次 RGB 请求的单步动作预测。
- 无 `--GPT`：新增启动器转交原服务；原入口也可以照常直接使用。

## 新增文件

### 运行代码

- `realworld_agent/policy/astra_client.py`
  - 使用 OpenAI Responses API。
  - 使用 Pydantic 结构化输出约束意图和动作结果。
  - 只编码并发送 RGB data URL。
  - API key 从指定环境变量延迟读取，不在源码中保存。
- `realworld_agent/planning/astra_intent_parser.py`
  - 把 Astra 的结构化意图输出转换为项目原有 `IntentPlan`。
- `realworld_agent/server/gpt6_astra_server.py`
  - 在原 `SubgoalRuntime` 上增加 Astra overlay。
  - 复用原 HTTP handler、请求保存、session、计划推进和 VQA 管理。
  - Astra `STOP` 无条件、立即完成当前子目标。
  - `--GPT` 模式不加载 Qwen/StreamVLN 推理权重。
- `clients/operator_client/subgoal_livevln_server_gpt.py`
  - 可从原 operator client 目录直接启动新增服务。

### 配置与文档

- `.env.gpt6-astra.example`
  - 明确留空 `OPENAI_API_KEY`、`OPENAI_BASE_URL`、`OPENAI_MODEL`。
- `requirements-gpt6-astra.txt`
  - Astra 模式的可选依赖，不影响原 requirements。
- `GPT6_ASTRA_USAGE.md`
  - 安装、配置、启动、模型切换和测试说明。
- `GPT6_ASTRA_CODE_MODIFICATION_REPORT.md`
  - 本报告。

### 测试

- `tests/test_astra_client.py`
  - 验证结构化意图映射。
  - 验证每次只返回一个动作。
  - 验证请求只包含 RGB，不包含 depth。
  - 验证 STOP 映射为动作 `0`。
- `tests/test_astra_runtime.py`
  - 验证即使 `--stop-confirmations 99`，Astra 单次 STOP 仍立即完成当前子目标并推进到下一子目标。

## 关键兼容性

- 动作编号没有改变：`0 STOP`、`1 MOVE_FORWARD`、`2 TURN_LEFT`、`3 TURN_RIGHT`。
- `/eval_vln` 的请求和响应字段没有改变，因此现有 Limo 客户端无需为 Astra 改协议。
- Astra 每次返回一个动作；原响应中的 `guard_g` 兼容字段仍由原 helper 生成。
- depth 仍按原逻辑接收和落盘，Astra runtime 明确丢弃 depth/intrinsic，只读取已保存的 RGB。
- VQA handler、VQA evaluator 和人工决策逻辑没有修改。
- Astra 正常输出 STOP 与 API 错误严格分开：只有成功解析出的动作 `0` 会调用 `apply_subgoal_policy_actions([0])`。

## 验证记录

语法编译通过：

```text
python -m py_compile <全部新增 Python 文件>
```

参数兼容检查通过：

```text
--GPT -> use_gpt=True
--astra-model gpt-6-astra -> gpt-6-astra
原 --host 与 --planner 参数仍可解析
```

mock 单元测试结果：

```text
Ran 4 tests in 0.014s
OK
```

测试期间未调用网络、未读取真实 API key、未产生模型费用。真实 API 联调需要使用者填入自己的 key 和可选 base URL 后执行。

## 官方 API 约束的落实

- 模型名使用 `gpt-6-astra`。
- 使用 Responses API，而不是旧式 Completions。
- 默认 `reasoning.effort=low`。
- 请求未设置 Astra 不支持的 `temperature`、`top_p` 或 `top_logprobs`。
- 使用模型支持的 image input 和 Structured Outputs。
