# Volcengine Agent Plan: Kimi-K3.1-Pro

[简体中文](volcengine-agent-plan.zh-CN.md) · [All guides](README.md)

Use the existing `evaluate_model.py` runner with the dedicated
[`volcengine_agent_plan.yaml`](../configs/models/volcengine_agent_plan.yaml)
to call `kimi-k3.1-pro`. No model client or global OpenAI/Codex changes are needed.

## Model and authentication

Follow the [official integration guide](https://docs.volcengine.com/docs/ark/agent-plan-personal-other-tools?lang=en):
use the dedicated Agent Plan key and `https://ark.cn-beijing.volces.com/api/plan/v3`.
Standard Ark API keys and the pay-as-you-go `/api/v3` endpoint are separate.
Set the key in the evaluation terminal, not in YAML:

```bash
export ARK_API_KEY='YOUR_AGENT_PLAN_API_KEY'
```

Optional `ARK_MODEL_NAME` and `ARK_BASE_URL` override the YAML defaults; explicit
CLI options have higher priority. The model needs image and tool support.
The [official model configuration](https://docs.volcengine.com/docs/ark/agent-plan-personal-opencode?lang=en)
lists text and image input for `kimi-k3.1-pro`, but only text for `glm-5.3`, which
cannot directly receive robot camera images. Account permissions and real API
checks determine actual availability.

## Offline plan and one-task evaluation

Install a Python 3.10–3.12 environment using [Getting started](getting-started.md).
From the repository root, inspect a one-task plan without a key, downloads or
model requests:

```bash
python scripts/evaluation/evaluate_model.py \
  --benchmark libero_pro \
  --benchmark-config configs/benchmarks/examples/libero_pro_smoke.yaml \
  --model-config configs/models/volcengine_agent_plan.yaml \
  --max-tokens 16384 \
  --run-id kimi-k31-pro-smoke-001 \
  --dry-run
```

With Linux/WSL, Docker, an NVIDIA GPU, required asset access and `ARK_API_KEY`
ready, remove `--dry-run` to evaluate one task. This consumes plan credits and
may download large images and assets on the first run. Use a new `--run-id` for
each actual execution.

`--max-tokens 16384` replaces the historical Agent's larger output budget;
adjust it to model limits and task needs. Before environment downloads, the
runner sends a connectivity probe with an image and tool definitions. The probe
does not require an actual tool call and does not establish tool execution or
task success. `--dry-run` validates only configuration and planning.

## Scope

This configuration supports the current `evaluate_model.py` LIBERO-family and
RoboCasa365 benchmarks. For multiple benchmarks, repeat `--model-config NAME=PATH`.
See [Model evaluation](model-evaluation.md) for HTTP services, resume and outputs.
The file contains no VLA deployment profile; the PRO example uses a no-VLA
preset. Enabling VLA still requires its model configuration and service as
described in the [VLA guide](libero-vla.md).

`evaluate_gpt6.py` overrides the model name, and some benchmark Agents have
independent Codex CLI providers and authentication. Changing only its
`model_config` does not switch the entire suite. Those CLI Agents need separate
adaptation.
