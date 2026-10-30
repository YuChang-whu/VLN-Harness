# Planning Terminal Demo

这个 demo 对应设计文档里的 `Task Session Manager`、`Planning Manager`、`StreamVLNIntentParser` 和 `StreamVLNModelClient`。

当前主路径：

```text
TaskSessionManager
  -> PlanningManager
      -> StreamVLNIntentParser
          -> StreamVLNModelClient
              -> StreamVLNForCausalLM
```

边界：

- `TaskSessionManager` 不直接持有模型。
- `PlanningManager` 不再通过逗号、then、然后等规则拆分来冒充语义理解。
- `StreamVLNIntentParser` 使用专门 prompt 要求模型输出结构化 JSON。
- `IntentJSONParser` 逻辑只负责提取 JSON、校验 subgoals，不负责语义拆分。
- `RuleFallbackIntentParser` 只在模型不可用或输出无法解析时兜底。
- 当前版本是 text-only intent planning，后续可以加入初始 RGB-D 观测，升级成 vision-aware intent planning。

快速验证：

```bash
python clients/operator_client/plan_demo.py --fallback-only --no-color --delay 0.01 \
  "go forward, enter the room, then stop near the table"
```

使用真实 StreamVLN checkpoint：

```bash
python clients/operator_client/plan_demo.py \
  --model-path /root/autodl-tmp/yc_code/StreamVLN/data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
  --device cuda:0 \
  "go forward, enter the room, then stop near the table"
```
