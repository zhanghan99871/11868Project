#!/usr/bin/env bash
# Fine-tune the H-Spec drafter on WikiText-2 against the frozen MLA verifier, then export it for
# MiniTorch inference (project/minitorch_drafter.py).
#
#   bash train.sh                              # full run
#   bash train.sh --max-steps 200 --eval-every 100   # extra flags go to train_drafter.py (last wins)
#   VERIFIER=path/to/best.pt OUT=runs/my-drafter bash train.sh
#
# Needs a GPU and the verifier trained by torch_train/train.py. The drafter's trunk starts from
# distilgpt2 (downloaded once from Hugging Face); its kv_up and frozen LM head come from the
# verifier. WikiText-2 tokens are cached in ./data (downloaded and tokenized on first use).
set -euo pipefail
cd "$(dirname "$0")"

VERIFIER=${VERIFIER:-runs/mla-gpt2-medium/best.pt}
OUT=${OUT:-runs/hspec-drafter}

python torch_train/train_drafter.py \
  --verifier-checkpoint "$VERIFIER" \
  --init distilgpt2 \
  --output-dir "$OUT" \
  --data-cache ./data \
  --seq-len 128 \
  --batch-size 16 \
  --anchors-per-seq 32 \
  --epochs 5 \
  --lr 3e-4 \
  --warmup-ratio 0.01 \
  --eval-every 500 \
  --max-eval-batches 50 \
  "$@"

# torch-free export: config.json + model.safetensors (the frozen head is taken from the verifier)
python torch_train/export_hspec_drafter.py "$OUT/best.pt" "$OUT/minitorch"
