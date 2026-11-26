#!/usr/bin/env bash

export ARK_API_KEY='ARK_API_KEY'
export EBENCH_SUBMIT_TOKEN='EBENCH_SUBMIT_TOKEN'
export EBENCH_TASK_ID='EBENCH_TASK_ID'
export EBENCH_ONLINE_ENDPOINT='EBENCH_ONLINE_ENDPOINT'

# enabled / disabled；Kimi-K3.1-Pro 只支持 enabled。
THINKING='enabled'
# Kimi 支持 low / high / max；low 减少思考开销，但仍然开启思考。
REASONING_EFFORT='low'

python scripts/evaluation/wetop_online.py run \
  --model-config configs/models/volcengine_agent_plan.yaml \
  --thinking "$THINKING" \
  --reasoning-effort "$REASONING_EFFORT" \
  --endpoint "$EBENCH_ONLINE_ENDPOINT" \
  --run-id "$EBENCH_TASK_ID" \
  --worker-id 0 \
  --max-episodes 1 \
  --agent-set 'agent.max_tokens=16384'
