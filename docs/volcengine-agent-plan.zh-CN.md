# 火山引擎 Agent Plan：Kimi-K3.1-Pro

[English](volcengine-agent-plan.md) · [全部指南](README.md)

通过现有 `evaluate_model.py` 和独立的
[`volcengine_agent_plan.yaml`](../configs/models/volcengine_agent_plan.yaml)
调用 `kimi-k3.1-pro`，无需修改模型客户端或全局 OpenAI/Codex 配置。

## 模型与认证

按[官方接入说明](https://docs.volcengine.com/docs/ark/agent-plan-personal-other-tools?lang=en)，
使用 Agent Plan 专属 API Key 和 `https://ark.cn-beijing.volces.com/api/plan/v3`。
普通方舟 API Key、按量接口 `/api/v3` 与套餐配置不同，不要混用。
在评测终端设置密钥，不要将真实密钥写入 YAML：

```bash
export ARK_API_KEY='YOUR_AGENT_PLAN_API_KEY'
```

可选的 `ARK_MODEL_NAME`、`ARK_BASE_URL` 覆盖 YAML 中的模型名和地址；命令行优先级更高。
模型需支持图片和工具调用。[官方模型配置](https://docs.volcengine.com/docs/ark/agent-plan-personal-opencode?lang=en)
列出 `kimi-k3.1-pro` 支持文本和图片输入；`glm-5.3` 仅支持文本，无法直接接收机器人相机画面。
实际可用性仍需通过账号权限及接口调用验证。

## 离线计划与单任务评测

先按[快速开始](getting-started.zh-CN.md)安装 Python 3.10–3.12 环境。
在仓库根目录查看单任务计划，无需密钥，不下载资产、不调用模型：

```bash
python scripts/evaluation/evaluate_model.py \
  --benchmark libero_pro \
  --benchmark-config configs/benchmarks/examples/libero_pro_smoke.yaml \
  --model-config configs/models/volcengine_agent_plan.yaml \
  --max-tokens 16384 \
  --run-id kimi-k31-pro-smoke-001 \
  --dry-run
```

准备好 Linux/WSL、Docker、NVIDIA GPU、所需资产访问权限及 `ARK_API_KEY` 后，
去掉 `--dry-run` 执行单任务评测。这会消耗套餐额度，首次执行还可能下载较大的镜像和资产。
每次实际运行使用新的 `--run-id`。

`--max-tokens 16384` 覆盖历史 Agent 的大输出预算，可按模型限制和任务调整。
运行器在下载环境前发送带图片与工具定义的连通性请求；该请求不要求真正调用工具，
不代表已经验证工具执行或任务成功。`--dry-run` 只检查配置与计划。

## 适用范围

该配置用于 `evaluate_model.py` 当前支持的 LIBERO 系列和 RoboCasa365。
多 benchmark 时重复 `--model-config NAME=PATH`。已有 HTTP 服务、续跑和结果位置见
[模型评测指南](model-evaluation.zh-CN.md)。此文件不包含 VLA 部署参数；上述 PRO 示例使用无 VLA 预设。
启用 VLA 仍需按 [VLA 指南](libero-vla.zh-CN.md)准备模型配置与服务。

`evaluate_gpt6.py` 会覆盖模型名，部分 benchmark 的 Agent 还使用独立的 Codex CLI
provider 和认证配置。仅替换其 `model_config` 无法切换全部 benchmark；这些 CLI Agent 需要另行适配。
