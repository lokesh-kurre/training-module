#!/usr/bin/env bash
set -euo pipefail

# Minimal 2-process DDP smoke test on CPU.
uv run torchrun \
  --standalone \
  --nproc_per_node=2 \
  train.py \
  --task classification \
  --dataset dataset_a \
  --experiment ddp_smoke \
  --distributed \
  --distributed-backend gloo
