# WET-OP 线上评测：火山引擎 / OpenAI

本地 RoboRSI Agent 连接 [WET-OP 在线评测平台](https://internrobotics.shlab.org.cn/eval/quick-start)
的 `wet_op_bench_test`，平台运行模拟器、场景和评分。

## 准备

在项目根目录、已激活的 Python 3.11 虚拟环境中运行。首次使用安装依赖：

```bash
python -m pip install -e '.[runtime,wetop]'
```

系统 JPEG 库等安装说明见 [部署指南](docs/wetop-online.zh-CN.md#1-安装客户端依赖)。
从平台已有任务的详情页取得 `task_id` 和对应的 `endpoint`。

## 火山引擎 Kimi-K3.1-Pro

根目录 `eval.sh` 就是以下启动命令。将四个环境变量的占位符替换为实际值：

```bash
export ARK_API_KEY='你的 Agent Plan 专属 Key'
export EBENCH_SUBMIT_TOKEN='你的评测平台 Token'
export EBENCH_TASK_ID='已有 WET-OP 任务的 task_id'
export EBENCH_ONLINE_ENDPOINT='该任务返回的 endpoint 完整地址'

python scripts/evaluation/wetop_online.py run \
  --model-config configs/models/volcengine_agent_plan.yaml \
  --endpoint "$EBENCH_ONLINE_ENDPOINT" \
  --run-id "$EBENCH_TASK_ID" \
  --worker-id 0 \
  --max-episodes 1 \
  --agent-set 'agent.max_tokens=16384'
```

改好后，在项目根目录执行：

```bash
bash eval.sh
```

默认模型是 `kimi-k3.1-pro`，使用火山引擎 Agent Plan 的 Chat Completions 接口。
需要改模型名时，修改 `configs/models/volcengine_agent_plan.yaml` 或设置 `ARK_MODEL_NAME`。

## 原有 OpenAI

使用 OpenAI 时，将 `eval.sh` 内容换成下面这组命令，并填写实际值：

```bash
export OPENAI_API_KEY='你的 OpenAI API Key'
export EBENCH_SUBMIT_TOKEN='你的评测平台 Token'
export EBENCH_TASK_ID='已有 WET-OP 任务的 task_id'
export EBENCH_ONLINE_ENDPOINT='该任务返回的 endpoint 完整地址'

python scripts/evaluation/wetop_online.py run \
  --model-config configs/models/gpt6.yaml \
  --endpoint "$EBENCH_ONLINE_ENDPOINT" \
  --run-id "$EBENCH_TASK_ID" \
  --worker-id 0 \
  --max-episodes 1 \
  --agent-set 'agent.max_tokens=16384'
```

同样执行 `bash eval.sh`。沿用原项目的 `gpt-6-astra` 默认名称和 Responses 接口；
实际模型以 API 账号权限为准，可修改 `configs/models/gpt6.yaml` 或设置 `GPT6_MODEL_NAME`。

## 回合与结果

- `--max-episodes 1`：先测一个回合；确认正常后删除这一行，继续剩余回合。
- `--agent-set 'agent.max_tokens=16384'`：每次模型回复的 token 上限。
- `--run-id`：必须使用平台的 `task_id`，不能自行起实验名称。
- 当前命令直接连接已有任务，不创建新任务，也不自动查询 endpoint。
- 若没有 endpoint，可执行下方查询命令，从结果的 `data.endpoint` 复制地址。
- 一个任务固定使用一个模型，比较另一模型时使用另一个任务 ID。
- 结果自动保存到 `work/wetop-online/`，包含 `summary.json`、动作记录、图像与模型轨迹。

查询已有任务的 endpoint：

```bash
gmp online ready \
  --base_url https://internrobotics.shlab.org.cn/eval \
  --token "$EBENCH_SUBMIT_TOKEN" \
  --task_id "$EBENCH_TASK_ID"
```

只有返回 `data.ready: true` 且带有 endpoint 时才能启动。若提示任务达到上限，
先确认使用的是已有任务；平台配额问题需在平台处理。

不要将填写真实 Key 和 Token 后的 `eval.sh` 提交到 Git 或公开分享。
