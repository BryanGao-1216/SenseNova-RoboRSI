# RoboRSI + Kimi 在线评测 WET-OP

此入口把本地 RoboRSI Agent 接到 [在线评测平台](https://internrobotics.shlab.org.cn/eval/quick-start)
的 `wet_op_bench_test`。平台运行模拟器、场景和原生评分；本地运行 Agent 和官方
`genmanip-client`，通过火山引擎 API 调用 Kimi-K3.1-Pro。
无需在本地安装 Isaac Sim、启动 Docker 或下载 WET-OP 场景与相机配置。

当前集成使用通用 RoboRSI API Agent，保留持续工具对话、最近 8 轮上下文和模型调用轨迹。
它提供 Lift2 双臂直接控制，没有接入 WET-OP 训练的 VLA，也没有自动 RSI 训练循环。
能运行评测接口不代表具备已验证的任务成功率。

## 1. 安装客户端依赖

在服务器的项目 Python 3.11 环境中执行，建议沿用已安装 RoboRSI 的虚拟环境。
Ubuntu 20.04 容器中可以直接以 root 安装 JPEG 解码库：

```bash
cd /gaowenbin/projects/SenseNova-RoboRSI
apt-get update
apt-get install -y libturbojpeg0
uv pip install --python .venv/bin/python -e '.[runtime,wetop]'
```

明确指定项目 `.venv/bin/python`，避免同时激活 Conda base 时把包装入另一个环境。
uv 虚拟环境没有 pip 时，直接使用 `uv pip`，不需要切换到 base 的 pip。

`wetop` extra 安装固定版本的
[官方客户端](https://github.com/Sage-PhysicalAI/genmanip-client/tree/fbfcd776639db82be3becdc6e39b5401f994da5c)
及其实际使用的 CPU 图像依赖。不会安装本地模拟器或本地策略检查点。
默认关闭 SDK 自带的视频录制，由本适配器保存每次决策看到的 PNG 图像和动作记录。
其中 `PyTurboJPEG>=1.7,<2` 兼容 Ubuntu 20.04 的 libjpeg-turbo 2.x；
PyTurboJPEG 2.x 则要求系统库至少 3.0。客户端导入失败时，先在同一解释器验证：

```bash
.venv/bin/python -c "import sys; print(sys.executable); from genmanip_client import EvalClient; print('EvalClient 导入成功')"
```

## 2. 配置两套凭证并离线检查

登录评测平台，在 API key 管理页面创建平台 Token。火山引擎仍使用已有 Agent Plan Key。
在运行终端设置，替换下面的占位符；不要把真实凭证写入配置或代码库。

```bash
export EBENCH_SUBMIT_TOKEN='你的平台Token'
export ARK_API_KEY='你的火山引擎AgentPlanKey'
python scripts/evaluation/wetop_online.py check
```

`check` 只校验配置并显示凭证是否存在，不请求平台或模型，不消耗评测回合。
`client_version: null` 表示客户端尚未安装；它不会实际加载 JPEG 原生库，库是否可用会在
`run` 导入客户端时确认。

默认复用 `configs/models/volcengine_agent_plan.yaml`：模型 `kimi-k3.1-pro`、
`https://ark.cn-beijing.volces.com/api/plan/v3`、Chat Completions 接口。
`ARK_MODEL_NAME`、`ARK_BASE_URL` 可覆盖模型名和地址。
Agent 参数位于 `configs/agents/wetop/online.yaml`，可用 `--agent-config`、
`--model-config` 或 `--agent-set 'agent.max_tokens=8192'` 指定。

## 3. 向平台申请评测任务

以下命令会实际创建评测任务。提交者信息请替换为自己的信息：

```bash
gmp online submit \
  --base_url https://internrobotics.shlab.org.cn/eval \
  --token "$EBENCH_SUBMIT_TOKEN" \
  --benchmark_set wet_op_bench_test \
  --model_name RoboRSI-Kimi-K3.1-Pro \
  --model_type LLM-Agent \
  --submitter_name 'gwb' \
  --submitter_homepage 'https://your-homepage.example' \
  --is_public 0
```

`model_type` 是客户端提交的展示信息；最终是否接受该类型以平台响应和挑战规则为准。
不要把该方法标成已训练的 VLA。文档的自动提取示例与参数表在 benchmark 名称上有出入，
这里采用其参数表和主命令中的 `wet_op_bench_test`。

等待分配资源后，从返回结果中取出 **endpoint** 和 **task_id**：

```bash
export EBENCH_ONLINE_ENDPOINT='返回的endpoint完整地址'
export EBENCH_TASK_ID='返回的task_id'
```

`EBENCH_ONLINE_ENDPOINT` 是分配后的远端模拟器地址，不能填平台首页或模型 API 地址。
`EBENCH_TASK_ID` 用作 SDK 的 `run_id`，不自行生成另一个值。

## 4. 先验证一个回合

```bash
python scripts/evaluation/wetop_online.py run \
  --worker-id 0 \
  --max-episodes 1
```

这个命令会实际调用 Kimi 并驱动线上模拟器。它在收到一个原生回合结果后停止；
平台创建的完整任务可能仍未完成。默认输出位于 `work/wetop-online/` 下按 run、worker
和启动时间区分的目录，终端会打印实际路径。

不要使用 `gmp eval` 代替这个命令：官方 `gmp eval` 默认运行示例假动作，没有调用本项目的 Agent。

确认图像、动作和回合结果正确后，运行完整任务：

```bash
python scripts/evaluation/wetop_online.py run --worker-id 0
```

平台保存已完成回合，后续连接处理剩余任务。指定输出目录时，必须使用新的空目录，
避免覆盖旧轨迹。平台长时间无请求可能回收连接；按官方快速开始文档使用相同 `task_id`
重新提交、取得新的 endpoint 后再运行。本地会建立新对话，不恢复中途对话，平台负责剩余回合分配。

单个进程只占用一个 worker。平台文档允许 0–7 共 8 个 worker；先验证单 worker，
增加并发时每个进程选择不同的 `--worker-id` 和输出目录，避免争用同一个远端 worker。
并发会增加模型请求量。进程退出会关闭其拥有的 worker，不调用平台的整任务停止接口。

查询任务进度：

```bash
gmp status \
  --url "$EBENCH_ONLINE_ENDPOINT" \
  --token "$EBENCH_SUBMIT_TOKEN" \
  --run_id "$EBENCH_TASK_ID"
```

## 动作、观测与结束语义

- 接收任务指令、所有 `video.*` RGB 图像以及公开 `state.*` 状态；不向模型传入 evaluator
  字段、隐藏目标位置或平台汇总指标。图像名称与顺序在 `media_fields` 中对应。
- `ee_pose` 工具指定左右手的 `position`、WXYZ `quaternion`、两个手指关节的 `gripper`。
  XYZ 使用与 `state.ee_pose` 相同的坐标系，单位米；手指使用原生关节单位。
  非零四元数会归一化。平台负责 IK，适配器不访问隐藏物体位姿。
- `joint_position` 的 `positions` 是 16 维：左臂 6 维、左手指 2 维、右臂 6 维、右手指 2 维。
  不是简单把 12 维 `state.joints` 后接 4 维 `state.gripper`。两种控制均为绝对目标，底座增量为零。
- `steps` 将同一绝对目标重复提交若干原生仿真步，默认上限 60，可通过 `--max-chunk-steps`
  改为 1–120。它不是插值、秒数或模型决策次数；平台可在中途结束回合。
- 每个 episode 创建一个全新的 RoboRSI 对话。SDK 的 `done` 表示整个 worker 的任务队列结束，
  回合切换通过原生 `episode_id`、`reset` 和 `episode_result` 识别。
- 平台自动重置时，响应中的图像属于下一回合。旧对话只收到该回合的原生结果，并明确标注没有
  终止图像。原生 `score` 和 `sr` 分别保存，不把非零部分分数当成成功。
- 传输重试交给官方 SDK 的可靠请求机制，适配器不会在 SDK 外重发物理动作。
  未带原生成绩的异常回合不会被补成 0 分或成功。
- `--max-decisions` 默认每回合 1000 次模型动作调用。达到本地预算会停止并记为错误，
  不修改平台的原生时限或伪造评分。

## 输出与验证范围

### 分阶段耗时

沿用原来的 `bash eval.sh`，无需新增参数。终端默认打印配置准备、SDK 加载、客户端连接、
环境重置和 Agent 创建耗时。每次决策打印当前回合、决策序号、原生 `timestep`，以及：

```text
[计时] Agent 决策等待开始…
[计时] Agent 决策等待完成: 20.000s
[计时] 云端动作请求开始…
  ⟳ Chunk 35 steps: 5.338s
[计时] 云端动作请求完成: 5.340s
[本轮耗时] 回合=1 决策=1 | 创建=0.100s | 观测=0.020s | Agent=20.000s | 动作转换=0.001s | 云端=5.340s | 落盘=0.010s | 本轮总计=25.480s | 累计=100.000s | 原生步请求/执行=35/35
```

以上数字仅为示例。`Agent` 包含上下文与图片组装、模型 API 等待及重试、工具调用处理和
动作格式修正；首次调用还可能包含模型运行时的延迟加载。`云端` 是整个 `client.step`
请求，包括网络、服务端执行、SDK 重试、返回解码，以及回合结束后的自动重置等待；
客户端不能将它解释为纯物理仿真时间。`观测` 包含公开状态转换、PNG 图像落盘与校验。
`原生步执行` 以 SDK 计数增量为准，可能小于请求步数；没有计数时显示 `unknown`。

正常结束、回合数限制、异常或 Ctrl+C 都会输出最终统计：实际总耗时、各阶段累计耗时与
占比、调用次数、平均/最长耗时和失败/中断次数，包括清理资源的耗时。各阶段互不重叠，
未归入阶段的控制逻辑等开销计入“其它本地开销”。计时从入口配置准备开始，至清理与轨迹
统计结束，不包含解释器启动和最终统计打印/写入本身。强制杀进程无法执行退出统计。

最终还会读取 Agent 轨迹，汇总已记录的模型 API 请求尝试、失败、重试次数及请求耗时；
这是上述阶段内的子统计，不能再与实际总耗时相加。请求耗时不含重试间隔；没有相关轨迹
事件时 `model_api_timing.available=false`，不把缺失数据解释成模型耗时为零。

`summary.json` 的 `timing` 保存阶段统计，`model_api_timing` 保存 API 子统计。
根目录 `events.jsonl` 的 `decision_timing` 保存每次已收到云端响应的决策耗时。
异常或中断中的阶段仍计入最终统计；没有收到响应的动作执行状态继续以
`last-action.json` 的 `delivery_state` 为准，不推断其已执行步数。

每次运行包括：

| 文件 | 内容 |
| --- | --- |
| `run.json` | 远端 run/worker、SDK 版本和预期 commit、模型配置哈希、Agent 配置哈希、控制预算 |
| `summary.json` | `completed` / `limited` / `interrupted` / `error`、平台原始指标、原生回合结果和耗时统计 |
| `events.jsonl` | 公开观测引用、动作请求、每轮耗时、回合切换和原始结果 |
| `episodes/*/task.json` | 原生 episode 标识、指令、动作 schema；seed 从 ID 数字尾段取值，未暴露时标记占位值 |
| `episodes/*/agent-config.json` | 实际使用的 Agent 配置，不含 API Key |
| `episodes/*/media/` | Agent 每次决策实际看到的 PNG 图像 |
| `episodes/*/trace/` | 现有 RoboRSI 模型请求、工具反馈、token 用量等轨迹 |

本地 `step_index` 统计模型决策，原生仿真步在观测的 `timestep` 中，两者不能混用。
`limited` 表示只运行了指定数量的回合，不表示平台完整任务已完成。

离线测试覆盖真实 EmbodiedAgent 工具循环与模拟平台响应、多相机转换、双臂动作、自动换回合、
部分分数、预算、传输异常和清理：

```bash
python -m pytest tests/unit_tests/roborsi/evaluation/test_wetop_online.py -q
```

离线测试不需要凭证，也不证明平台已接受线上提交或模型已完成物理任务。
安装 `wetop` extra 后还会运行官方 SDK 的本地 HTTP 测试，覆盖可靠请求重放、异步重置和
动作序列提前结束。该测试用 Pillow 替代原生 JPEG 解码库；Ubuntu 的 libturbojpeg 和真实
平台连接仍需在服务器上验证。
