# WET-OP 线上评测：火山引擎 / OpenAI

本地 RoboRSI Agent 连接 [WET-OP 在线评测平台](https://internrobotics.shlab.org.cn/eval/quick-start)
的 `wet_op_bench_test`，平台运行模拟器、场景和评分。

## 准备

在项目根目录、已激活的 Python 3.11 虚拟环境中运行。首次使用安装依赖：

```bash
uv pip install --python .venv/bin/python -e '.[runtime,wetop]'
```

使用项目的 `.venv/bin/python` 安装和运行；同时激活 Conda base 时，直接调用 `pip`
可能安装到 base。uv 创建的虚拟环境可以不包含 pip，不影响使用上述安装命令。

系统 JPEG 库等安装说明见 [部署指南](docs/wetop-online.zh-CN.md#1-安装客户端依赖)。
从平台已有任务的详情页复制 `task_id`。`endpoint` 通过本文末尾的 `gmp online ready`
命令查询，再填写到 `eval.sh` 中。

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
export EBENCH_SUBMIT_TOKEN='你的评测平台 Token'
export EBENCH_TASK_ID='网页显示的已有任务 task_id'

gmp online ready \
  --base_url https://internrobotics.shlab.org.cn/eval \
  --token "$EBENCH_SUBMIT_TOKEN" \
  --task_id "$EBENCH_TASK_ID"
```

只有返回 `data.ready: true` 且带有 endpoint 时才能启动。若提示任务达到上限，
先确认使用的是已有任务；平台配额问题需在平台处理。

## 查看运行过程和可视化

`Connected to server` 表示评测服务已连接；`Polling reset result` 表示平台 worker
正在准备场景。此时还没有取得初始观测，Kimi 尚未开始决策。
收到 `Reset complete`、`Environment reset complete` 后，才会处理任务观测。

`Save process: False` 和 `Save result: False` 只关闭官方 SDK 的录制；本项目仍在
`work/wetop-online/` 保存图像、交互轨迹和原始结果。
当前记录的是每次 Agent 决策看到的相机图像，未开启连续视频录制。

开始模型决策后，找出回合的轨迹文件：

```bash
find work/wetop-online -path '*/trace/events.jsonl'
```

将其中一个路径末尾的 `/trace/events.jsonl` 去掉，得到该回合目录，导出 HTML：

```bash
python -m roborsi.cli.trajectory '上面得到的完整回合目录' \
  --output /gaowenbin/roborsi-work/wetop-report-001
```

报告输出目录必须是新目录或空目录。将整个 `wetop-report-001` 文件夹下载到本机，
用浏览器打开其中的 `index.html`，即可查看相机图像、已记录的模型回复、工具动作和 token 统计。
最终原生成绩在本次运行目录的 `summary.json` 中。

## 客户端导入失败

Ubuntu 20.04 的系统 JPEG 库是 2.x，需使用 `PyTurboJPEG<2`。PyTurboJPEG 2.x
要求系统库至少 3.0，不能通过反复安装 Ubuntu 的旧系统库解决。
项目已约束兼容版本。在项目根目录修复并单独验证导入：

```bash
uv pip install --python .venv/bin/python -e '.[runtime,wetop]' 'PyTurboJPEG>=1.7,<2'
.venv/bin/python -c "import sys; print(sys.executable); from genmanip_client import EvalClient; print('EvalClient 导入成功')"
```

这个验证不会连接平台或调用模型。如果仍失败，完整 traceback 会显示缺少的 Python 包、
原生库或符号。导入成功后，再执行 `bash eval.sh`。

## 初始化后出现空动作 `{}` 校验错误

`ValidationError: {} is not valid under any of the given schemas` 表示 Agent 动作参数校验失败，
不是场景初始化失败。旧版动作 schema 只有 `oneOf` 分支内的参数，经过 OpenAI 兼容 SDK
转换后会产生空的顶层参数表，并禁止额外字段，导致有效动作也无法通过。
适配器已补齐顶层参数定义，同时保留两种控制模式各自的完整校验。

同步最新的 `roborsi/evaluation/wetop_online.py` 后，直接重跑 `bash eval.sh`，沿用已有
任务 ID。空动作不会被送往模拟器，也不会被替换成零动作；模型仍需生成通过校验的动作。
错误后的 `Client closed` 只表示客户端已清理连接，不代表评测完成或取得成绩。

## 关节状态校验失败

旧报错 `WET-OP Lift2 requires state.joints with shape (12,)` 同时涵盖维度不符和
`NaN/Inf`，不能仅凭这句话判断原因。新版本区分缺失字段、真实 shape 和非有限数值，
并在报错中显示 `robot_id`、`episode_id`。

像 `(1, 12)`、`(12, 1)` 这样的单样本额外维度会转换为 `(12,)`，不改变关节数量、
顺序或数值。其他布局和 `NaN/Inf` 仍会停止，避免把错误状态送给模型。
每回合首次观测还会生成 `episodes/<回合目录>/observation-structure.json`，
其中只有公开状态的 shape、异常值数量和相机结构，没有图片内容或凭证。
如果仍失败，提供新的完整报错或该结构文件即可进一步定位。

不要将填写真实 Key 和 Token 后的 `eval.sh` 提交到 Git 或公开分享。
