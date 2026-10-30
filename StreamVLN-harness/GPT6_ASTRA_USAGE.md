# GPT-6 Astra 接入说明

这套接入是可选增量路径。原来的 Qwen3-8B 意图解析与 StreamVLN 动作预测入口保持不变；使用新增启动器并传入 `--GPT` 时，GPT-6 Astra 同时负责意图解析和每一步动作预测。

## 1. 安装可选依赖

在实际启动服务所使用的 Python 环境中执行：

```bash
cd /root/autodl-tmp/yc_code/StreamVLN-harness/StreamVLN-harness
python -m pip install -r requirements-gpt6-astra.txt
```

## 2. 填写 API 配置

模板文件是 `.env.gpt6-astra.example`。真实配置建议保存到仓库外的私有文件，不要提交 API key。

需要填写的位置：

```bash
# 必填：你的 key
export OPENAI_API_KEY=""

# 自建或兼容服务才填写；官方 OpenAI 直连可保持为空
export OPENAI_BASE_URL=""

# 保持为空时使用 gpt-6-astra
export OPENAI_MODEL=""
```

`OPENAI_BASE_URL` 应当是 SDK base URL（通常以 `/v1` 结尾），不是手工拼接的 `/responses` 完整请求地址。服务不会打印 API key。

如果你的 key 使用其他环境变量名，可在启动时传：

```bash
--astra-api-key-env YOUR_PRIVATE_KEY_ENV
```

## 3. 启动 GPT-6 Astra 模式

```bash
cd /root/autodl-tmp/yc_code/StreamVLN-harness/StreamVLN-harness
/root/miniconda3/envs/streamvln/bin/python \
  clients/operator_client/subgoal_livevln_server_gpt.py \
  --GPT \
  --host 0.0.0.0 \
  --port 6010
```

可选参数：

```text
--astra-model                 默认 gpt-6-astra
--astra-base-url              默认读取 OPENAI_BASE_URL；空值使用 SDK 官方地址
--astra-api-key-env           默认 OPENAI_API_KEY
--astra-timeout               默认 30 秒
--astra-max-retries           默认 2
--astra-reasoning-effort      low/medium/high/xhigh/max，默认 low
--astra-image-detail          low/high/auto，默认 high
--astra-max-output-tokens     默认 1024
```

原有服务参数仍然可用，例如 `--save-dir`、`--default-instruction`、`--include-llm-output` 和 VQA 参数。

## 4. 保持原模型路径

继续使用原来的 Qwen3-8B + StreamVLN 时，运行原入口即可，不添加 `--GPT`：

```bash
/root/miniconda3/envs/streamvln/bin/python \
  clients/operator_client/subgoal_livevln_server.py \
  --host 0.0.0.0 \
  --port 6010
```

也可以运行新增启动器但不传 `--GPT`；它会转交给原服务。实际使用中建议原模式继续使用原入口，使可选 OpenAI 依赖与本地模型路径完全隔离。

## 5. Astra 模式的运行语义

- 意图解析：Astra 将完整指令转换为原项目现有的 `IntentPlan`/子目标结构。
- 动作预测：每个 `/eval_vln` 请求只预测一个动作，动作编号与 StreamVLN 一致：`0 STOP`、`1 MOVE_FORWARD`、`2 TURN_LEFT`、`3 TURN_RIGHT`。
- STOP：Astra 返回 `0` 后立即把当前子目标标记为完成，并切换到下一个子目标；不检查 reason，也不受 `--stop-confirmations` 数值影响。
- 视觉输入：只把 RGB 图像发送给 Astra。已有接口仍可接收、解析并保存 depth，但 depth 不进入 Astra 请求。
- VQA：沿用原有 VQA 代码和接口，本次没有把 Astra 接到 VQA。
- 权重：`--GPT` 模式不加载 Qwen3-8B 或 StreamVLN 推理权重。原 StreamVLN 服务模块仍会作为 HTTP/图像解码兼容层导入。
- API 异常：API 超时、拒绝或结构化输出缺失会进入原服务的错误响应路径，不会伪装成 Astra 的正常 STOP，也不会错误完成子目标。

## 6. 无真实 API 的回归测试

```bash
cd /root/autodl-tmp/yc_code/StreamVLN-harness/StreamVLN-harness
/root/miniconda3/envs/streamvln/bin/python -m unittest discover \
  -s tests -p 'test_astra*.py' -v
```

这些测试使用 fake Responses client，不读取 key，不联网，也不会产生 API 费用。
