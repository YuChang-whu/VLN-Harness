# StreamVLN Real-World Agent Harness Prototype

这是第一版真机 VLN Agent Harness 原型，当前实现的是任务/计划流程的规划与终端展示。

当前版本已经把意图理解主路径改为 StreamVLN real-world checkpoint：

```text
instruction
  -> StreamVLNIntentParser
  -> StreamVLNModelClient
  -> StreamVLNForCausalLM
  -> structured intent JSON
  -> semantic subgoals
  -> terminal task flow
```

规则解析不再作为主逻辑，只在模型不可用、显存不足、输出无法解析成 JSON 等情况下作为 fallback。

动作语义与 StreamVLN 保持一致：

```text
0 = STOP
1 = MOVE_FORWARD
2 = TURN_LEFT
3 = TURN_RIGHT
```

## 默认模型路径

```text
/root/autodl-tmp/yc_code/StreamVLN/data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world
```

## 使用 StreamVLN 意图解析

原始 StreamVLN 真机代码路径：

```text
/root/autodl-tmp/yc_code/StreamVLN/limo/limo_realworld_server_liveVLN.py
```

当前 demo 会复用同目录下 `limo_realworld_server.py` 的 `load_streamvln()` 加载方式。


```bash
cd /root/autodl-tmp/yc_code/StreamVLN-harness/StreamVLN-harness
python clients/operator_client/plan_demo.py \
  --model-path /root/autodl-tmp/yc_code/StreamVLN/data/pretrained_models/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world \
  --streamvln-root /root/autodl-tmp/yc_code/StreamVLN \
  --device cuda:0 \
  --max-new-tokens 768 \
  "go forward, enter the room, then stop near the table"
```

## 快速验证 fallback 展示

如果当前机器不想加载大模型，可以显式使用 fallback：

```bash
cd /root/autodl-tmp/yc_code/StreamVLN-harness/StreamVLN-harness
python clients/operator_client/plan_demo.py --fallback-only \
  "go forward, enter the room, then stop near the table"
```

## 交互演示

```bash
python clients/operator_client/plan_demo.py --interactive \
  "go forward to enter the room"
```

交互命令：

```text
show
observe <note>
actions <ids/text>
done <evidence>
stop [reason]
quit
```

终端会显示：

```text
Intent Understanding
  Goal           : ...
  Stop condition : ...
  Landmarks      : ...
  Constraints    : ...
  Route hints    : ...
  Parser         : StreamVLN real-world checkpoint (...)
  Fallback       : no
```
