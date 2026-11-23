# Online WET-OP with RoboRSI and Kimi

This adapter connects the local generic RoboRSI API Agent to the platform's
`wet_op_bench_test` service. The platform owns simulation, assets, cameras and
scoring. The local machine runs the Agent and the official GenManip client;
Kimi-K3.1-Pro inference uses the existing Volcengine Agent Plan profile.
No local Isaac Sim, Docker, scene download or camera YAML is required.

See the [complete Chinese guide](wetop-online.zh-CN.md) and the
[official quick start](https://internrobotics.shlab.org.cn/eval/quick-start).

## Install and configure

From the checkout, in the project's Python 3.11 environment:

```bash
apt-get update
apt-get install -y libturbojpeg0
uv pip install --python .venv/bin/python -e '.[runtime,wetop]'
export EBENCH_SUBMIT_TOKEN='your-platform-token'
export ARK_API_KEY='your-volcengine-agent-plan-key'
python scripts/evaluation/wetop_online.py check
```

The extra pins the official SDK to commit
`fbfcd776639db82be3becdc6e39b5401f994da5c`. The offline check validates config and
reports credential presence; it does not allocate a worker or query a model.
`ARK_MODEL_NAME` and `ARK_BASE_URL` override the existing model profile defaults.
Use the same interpreter for installation and evaluation, especially when a
Conda base environment is also active. The extra restricts PyTurboJPEG to 1.x
for Ubuntu 20.04's native libjpeg-turbo 2.x; PyTurboJPEG 2.x requires native 3.0+.
To see an unwrapped client import failure without contacting the platform, run
`.venv/bin/python -c "from genmanip_client import EvalClient"`.

Create a private online job using `gmp online submit`, following the official
quick start. Use `--benchmark_set wet_op_bench_test`, an accurate model name/type
(this is an LLM Agent, not a trained VLA), and your submitter details. Platform
acceptance of model metadata and control settings is determined by its service
and challenge rules. Record the returned endpoint and task ID:

```bash
export EBENCH_ONLINE_ENDPOINT='returned-evaluation-endpoint'
export EBENCH_TASK_ID='returned-task-id'
python scripts/evaluation/wetop_online.py run --worker-id 0 --max-episodes 1
```

This executes real model calls and remote actions. For the remaining benchmark,
omit `--max-episodes`. Do not substitute `gmp eval`: its default policy sends fake
actions rather than running RoboRSI. If the service expires, resubmit with the
same task ID, refresh the endpoint and run again; the platform preserves completed
episodes. Each invocation uses a fresh output directory and starts new dialogues.

## Controls and records

The adapter forwards public state and all RGB cameras, supplies absolute Lift2
dual-arm `ee_pose` or 16-dimensional `joint_position` targets, and holds the base
stationary. XYZ uses meters in the `state.ee_pose` frame; quaternions use WXYZ;
finger targets use native joint units. Joint targets interleave left arm (6),
left fingers (2), right arm (6), right fingers (2).

`steps` repeats the same absolute target for native simulation ticks, without
interpolation. Its default maximum is 60; `--max-chunk-steps` accepts 1–120.
`--max-decisions` limits model calls per episode (default 1000), separately from
the platform's native simulation budget.

One process owns one worker (0–7). Different processes must use different worker
IDs and output directories. Only the SDK retries physical requests, using its
reliable transport. Closing a process closes its owned worker, not the whole job.

Each native episode gets a fresh dialogue. Auto-reset images are never used as
terminal images for the previous episode. Raw `score` and `sr` remain separate;
missing native results and local limits do not become fabricated benchmark scores.

Outputs under `work/wetop-online/` include `run.json`, `summary.json`,
`events.jsonl`, per-episode task/config files, camera PNGs and normal RoboRSI
traces. Credentials are supplied in memory, not saved in configuration records.
`completed` means the remote worker has no remaining tasks; `limited` means this
client stopped after the requested number of scored episodes.

The integration reuses the generic Agent's recent-eight-turn context and trace
support. It does not supply a WET-OP-trained VLA or an automatic RSI training loop.
Offline coverage uses simulated platform responses and the real EmbodiedAgent
tool loop; live platform compatibility and task success require a credentialed run.

```bash
python -m pytest tests/unit_tests/roborsi/evaluation/test_wetop_online.py -q
```
