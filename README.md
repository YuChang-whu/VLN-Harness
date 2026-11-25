# VLN-Harness

VLN-Harness 是一个基于 StreamVLN 的真机视觉语言导航（Vision-and-Language Navigation，VLN）项目。核心代码位于 [`StreamVLN-harness/`](StreamVLN-harness/)。它把一条自然语言指令拆成可跟踪的导航子目标，用相机观测预测当前子目标的动作，并在终端显示进度。项目可接入 Limo 小车，也可以在普通电脑上演示任务规划。

## 快速体验

规则解析演示只需 Python 3.9 或以上版本。它用脚本模拟观测、动作和子目标推进，无需模型权重、GPU 或机器人。Windows 上可以在 Git Bash 中运行。

```bash
git clone https://github.com/YuChang-whu/VLN-Harness.git
cd VLN-Harness/StreamVLN-harness
python clients/operator_client/plan_demo.py --parser rule --no-color --delay 0 "Go forward to the table."
```

终端会显示任务目标、停止条件、地标、子目标与动作记录。加上 `--interactive` 可以手动更新任务状态，使用 `show`、`observe <note>`、`actions <ids/text>`、`done [evidence]`、`stop [reason]` 和 `quit`。

演示默认用 Qwen 解析指令，也可以指定 `--parser streamvln` 使用 StreamVLN 权重。用 `--strict-model` 可以在模型加载或解析失败时直接看到错误，便于调试。

## 导航流程

1. **解析指令**。默认由 Qwen3-8B 提取任务目标、停止条件、地标和约束，生成有顺序的语义子目标。
2. **预测动作**。Limo 采集 RGB 和深度图像。服务将最新观测和当前子目标交给 StreamVLN 真机模型，返回离散动作。
3. **推进任务**。服务根据策略返回的停止动作推进子目标，客户端依据 `terminate=true` 结束整次导航。操作员也可以暂停任务，基于最近图像提问，再选择继续、确认当前子目标完成或结束任务。
4. **保存记录**。服务默认将会话写入 `StreamVLN-harness/outputs/sessions/`，其中包含事件流 `events.jsonl`、最新状态 `latest_state.json`，以及按请求保存的图像与请求、响应 JSON。可用 `--save-dir` 指定输出位置。

| 动作 ID | 含义 |
| --- | --- |
| `0` | `STOP` |
| `1` | `MOVE_FORWARD` |
| `2` | `TURN_LEFT` |
| `3` | `TURN_RIGHT` |

## 启动本地模型服务

### 准备环境和权重

- **StreamVLN 环境**。参考上游的 Linux、Python 3.9 和 PyTorch 2.1.2 环境，按本机 CUDA 配置准备 GPU 版 PyTorch。仓库根目录提供 [`requirements.txt`](requirements.txt) 和 [`environment_full.yml`](environment_full.yml) 作为依赖参考。后者是 Linux 环境导出，包含原机器的安装前缀和 CPU 版 PyTorch。准备本地的 [StreamVLN 真机权重](https://huggingface.co/mengwei0427/StreamVLN_Video_qwen_1_5_r2r_rxr_envdrop_scalevln_real_world)。服务默认离线加载，视觉编码器所需文件也要提前缓存，或在允许联网的环境中使用 `--allow-hf-network`。
- **Qwen 环境**。单独准备支持 [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) 的本地推理环境和权重。Qwen3-8B 需要 `transformers>=4.51.0`。服务默认通过子进程调用 Qwen，`--qwen-python` 指向该环境的 Python。

### 启动服务

在 `VLN-Harness/StreamVLN-harness/` 中，激活 StreamVLN 环境并将示例中的路径换成本机路径。`--streamvln-deploy-root ..` 指向仓库根目录，服务会从那里加载 `limo/limo_realworld_server.py`。

```bash
python clients/operator_client/subgoal_livevln_server.py \
  --host 127.0.0.1 --port 6010 \
  --streamvln-deploy-root .. \
  --model-path /path/to/StreamVLN_real_world_checkpoint \
  --qwen-python /path/to/qwen-env/bin/python \
  --qwen-model-path /path/to/Qwen3-8B \
  --device cuda:0
```

可用 `curl http://127.0.0.1:6010/status` 查看服务状态。只想跳过 Qwen 模型时，使用 `--planner rule` 并去掉两个 Qwen 路径参数，动作预测仍由 StreamVLN 处理。

### 连接 Limo 和操作台

机器人与服务器分开运行时，将服务的 `--host` 设为机器人能访问的监听地址，并在 Limo 的 ROS 环境中运行以下命令。将 `SERVER_IP` 换成实际地址。客户端默认订阅 `/camera/rgb/image_raw` 和 `/camera/depth/image`，可通过 `--rgb-topic` 和 `--depth-topic` 调整。

```bash
python limo/limo_streamvln_liveVLN_v2.py \
  --server http://SERVER_IP:6010/eval_vln \
  --instruction "Go forward, enter the room, then stop near the table." \
  --dry-run
```

`--dry-run` 跳过移动和转向指令。确认相机和服务联通后，在真机运行时去掉该参数。Limo 通过 `POST /eval_vln` 发送图像和元数据，并根据响应中的 `action_e`、`guard_g` 和任务状态更新执行。

导航运行期间，可在另一个终端打开人工复核操作台。用 `start` 暂停并进入复核，用 `ask <question>` 针对最近的图像提问，再用 `continue`、`complete` 或 `stop` 提交人工决策。服务的 VQA 后端默认是 `mock`，实际模型问答需要在启动服务时指定 `--vqa-backend streamvln` 和对应的 `--vqa-model-path`。

```bash
python clients/operator_client/hitl_vqa_console.py --server http://127.0.0.1:6010
```

## 可选 Astra 模式

使用 `subgoal_livevln_server_gpt.py --GPT` 可将意图解析和逐步动作预测交给 Astra。该模式复用现有 HTTP 接口和会话流程，每个请求基于 RGB 图像返回一个动作。启动时仍需提供 StreamVLN 部署目录以加载兼容接口，导航推理使用 Astra。依赖、API 配置和测试命令见 [Astra 接入说明](StreamVLN-harness/GPT6_ASTRA_USAGE.md)。

## 代码入口

| 路径 | 用途 |
| --- | --- |
| [`clients/operator_client/`](StreamVLN-harness/clients/operator_client/) | 规划演示、服务启动器和人工复核操作台 |
| [`realworld_agent/`](StreamVLN-harness/realworld_agent/) | 意图解析、模型调用、子目标执行、会话和终端渲染 |
| [`limo/`](StreamVLN-harness/limo/) | Limo ROS 客户端和 v2 协议说明 |
| [`tests/`](StreamVLN-harness/tests/) | Astra 客户端与子目标推进的模拟测试 |

## 上游项目

本项目的真机导航策略基于 Meng Wei 等人的 [StreamVLN](https://github.com/OpenRobotLab/StreamVLN)。模型方法见论文 [StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling](https://arxiv.org/abs/2507.05240)。上游项目声明使用 [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/)。
