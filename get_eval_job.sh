#!/usr/bin/env bash

export EBENCH_SUBMIT_TOKEN='sk-pRlYogu2W9Cu0ImFeQL9R98V0egQbeKewytF4rDeGTBWD7BG'

gmp online submit \
  --base_url https://internrobotics.shlab.org.cn/eval \
  --token "$EBENCH_SUBMIT_TOKEN" \
  --benchmark_set wet_op_bench_test \
  --model_name AgentVLA \
  --model_type agentVLA \
  --submitter_name test \
  --submitter_homepage test \
  --is_public 0