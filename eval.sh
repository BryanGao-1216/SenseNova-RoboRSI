#!/usr/bin/env bash

export ARK_API_KEY='ARK_API_KEY'
export EBENCH_SUBMIT_TOKEN='EBENCH_SUBMIT_TOKEN'
export EBENCH_TASK_ID='EBENCH_TASK_ID'
export EBENCH_ONLINE_ENDPOINT='EBENCH_ONLINE_ENDPOINT'

python scripts/evaluation/wetop_online.py run \
  --model-config configs/models/volcengine_agent_plan.yaml \
  --endpoint "$EBENCH_ONLINE_ENDPOINT" \
  --run-id "$EBENCH_TASK_ID" \
  --worker-id 0 \
  --max-episodes 1 \
  --agent-set 'agent.max_tokens=16384'
