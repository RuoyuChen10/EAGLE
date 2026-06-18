#!/usr/bin/env bash
python text_attribution_atman.py \
  --model-name Qwen/Qwen3-8B \
  --system-prompt $'Answer the given task.\n\nYou must first reason inside <think> and </think>.\n\nThen give the final answer inside <answer> and </answer>.\n\nDo not use external tools.' \
  --user-prompt $'# Current Task\n\nAlice needs to buy milk before going home.\n\nThe supermarket closes at 8 PM.\n\nIt is now 7:30 PM.\n\nWalking to the supermarket takes 20 minutes, and walking from the supermarket to home takes 15 minutes.\n\nCan Alice buy milk before the supermarket closes?' \
  --max-new-tokens 512 \
  --target-token-limit 512 \
  --input-granularity readable \
  --suppression-factor 0.1 \
  --score-reduction sum \
  --batch-size 16 \
  --output-dir ./text_attribution_outputs/atman_text_attribution_outputs_refined
