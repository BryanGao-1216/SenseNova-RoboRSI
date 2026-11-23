# 在已有云容器中运行标准 LIBERO

[English](../benchmarks/libero/README.md#run-inside-an-existing-linux-container) · [全部指南](README.md)

已有 Ubuntu 容器可以直接安装仿真环境，再以普通 Python 进程提供 Benchmark HTTP 服务。
评测程序连接 `127.0.0.1`，不需要在容器中启动 Docker daemon 或挂载宿主机 Docker socket。
此流程仅针对标准 `libero`，使用 CPU 渲染，不启用 VLA。

## 系统和 Python

需要 Linux x86_64、Python 3.11，以及安装以下容器内系统依赖的权限。
在 Ubuntu 20.04 的 root 终端执行：

```bash
apt-get update
apt-get install -y --no-install-recommends \
  gcc libc6-dev linux-libc-dev libosmesa6 libgl1 libglib2.0-0 ca-certificates
python3.11 --version
```

若没有 Python 3.11，可使用已有 Conda 环境提供的 3.11 解释器，并在下一步传入
`--python /absolute/path/to/python3.11`。不要将 Ubuntu 20.04 默认的 Python 3.8 用于该锁定环境。
Agent 所在的 `.venv` 继续按[快速开始](getting-started.zh-CN.md)安装 `.[runtime]`。

## 安装仿真环境和资产

在同步了新脚本的项目根目录执行，工作目录放在云平台的持久化存储中：

```bash
python scripts/evaluation/libero_native.py prepare \
  --python python3.11 \
  --work-dir /gaowenbin/roborsi-work/libero-native
```

脚本创建独立的 `venv/`，安装标准 LIBERO 的哈希锁定依赖，使用原 Dockerfile 中的
RLinf wheel、LIBERO 源码版本及 SHA-256，并下载、校验固定版本的仿真资产。
PyTorch 使用锁定的 CPU wheel，不会替换 Agent `.venv` 中的包。
此步骤会联网，不调用模型，也不启动仿真任务。重新执行可复用已下载资产；它仍会检查依赖并重新安装源码。

`assets/` 保存资产快照，`config/` 保存 LIBERO 路径，`native-runtime.json` 记录源码固定版本及路径。
整个准备和评测过程中不要修改资产或源码。该流程沿用原生依赖和代码，但没有 Docker 镜像摘要，
不能据此宣称与发布镜像的操作系统环境完全相同。

## 启动服务并验证

保持第一个终端运行：

```bash
python scripts/evaluation/libero_native.py serve \
  --work-dir /gaowenbin/roborsi-work/libero-native \
  --suite libero_spatial --seed 0 --port 18800
```

默认只监听本容器的 `127.0.0.1`，物理步数上限为 600；可用 `--max-steps` 调整，
改变此值会改变评测条件。第二个终端先检查服务，再使用固定动作 Agent 验证真实仿真链路：

```bash
curl --fail http://127.0.0.1:18800/v1/health
bench eval --benchmark-url libero=http://127.0.0.1:18800 \
  --results-root /gaowenbin/roborsi-work/libero-results --run-id hold-native-001 \
  --benchmark libero --task libero:0 --seed 0 \
  --agent-factory examples.roborsi.hold_agent:make_agent \
  --agent-version hold-example-v1 --timeout-s 180 --case-timeout-s 900
```

固定动作 Agent 预期无法完成操作任务。它用于检查图像、动作、原生评分和产物保存，
`task_failure` 是预期结果；`agent_error`、`infrastructure_error` 需要先排查。
健康检查通过本身不代表已经成功执行 MuJoCo 仿真。

## 调用 Kimi

在安装了框架的 Agent `.venv` 中设置套餐密钥，然后连接上述服务：

```bash
export ARK_API_KEY='YOUR_AGENT_PLAN_API_KEY'
python scripts/evaluation/evaluate_model.py \
  --benchmark libero --benchmark-url libero=http://127.0.0.1:18800 \
  --model-config configs/models/volcengine_agent_plan.yaml \
  --work-dir /gaowenbin/roborsi-work/libero-eval \
  --task libero:0 --seed 0 --max-tokens 16384 --run-id kimi-native-001
```

此处必须传 `--benchmark-url`，否则现有一键入口会使用 Docker 自动准备模式。
不再执行前面的 Docker `--prepare-only` 命令。HTTP 模式下资产由服务端持有，
suite 和 seed 以服务端 catalog 为准；`ROBORSI_LIBERO_SUITE` 不会改变已有服务。
实际评测消耗 Agent Plan 额度，每次使用新的 run ID。

离线测试只验证安装编排、校验和处理、路径配置及 HTTP 服务命令。Ubuntu 20.04 的原生安装、
OSMesa 渲染和模型任务成功仍需在目标服务器执行上面的检查后确认。
