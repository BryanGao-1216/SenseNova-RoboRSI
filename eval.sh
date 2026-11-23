#!/usr/bin/env bash

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
