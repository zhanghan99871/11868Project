#!/usr/bin/env bash
# Train the H-Spec drafter against the frozen MLA verifier, then export it for MiniTorch inference
# (project/minitorch_drafter.py).
#
#   bash train.sh                                     # verifier-generated data (default)
#   bash train.sh --max-steps 200 --eval-every 100    # extra flags go to train_drafter.py (last wins)
#   DATA=wikitext bash train.sh                       # the earlier setup: raw WikiText-2 text
#   VERIFIER=path/to/best.pt OUT=runs/my-drafter GEN=data/my_gen bash train.sh
#
# Default (DATA=generated):
#   1. the verifier continues 64-token WikiText-2 prompts with 64 greedy tokens (as in the
#      benchmark) -> $GEN/train.pt, $GEN/validation.pt; skipped if they exist;
#   2. the drafter trains on those rows with every block inside the generated part, so its
#      context and targets are what decoding produces;
#   3. export to $OUT/minitorch.
# Needs a GPU and the verifier trained by torch_train/train.py. The trunk starts from distilgpt2
# (downloaded once); kv_up and the frozen LM head come from the verifier.
set -euo pipefail
cd "$(dirname "$0")"

VERIFIER=${VERIFIER:-runs/mla-gpt2-medium/best.pt}
DATA=${DATA:-generated}

if [ "$DATA" = generated ]; then
  OUT=${OUT:-runs/hspec-drafter-gen}
  GEN=${GEN:-data/verifier_gen}
  python torch_train/generate_verifier_data.py \
    --verifier-checkpoint "$VERIFIER" \
    --data-cache ./data \
    --source wikitext-2 \
    --out-dir "$GEN" \
    --prompt-len 64 \
    --gen-len 64 \
    --batch-size 128

  python torch_train/train_drafter.py \
    --verifier-checkpoint "$VERIFIER" \
    --init distilgpt2 \
    --output-dir "$OUT" \
    --train-data "$GEN/train.pt" \
    --val-data "$GEN/validation.pt" \
    --seq-len 128 \
    --batch-size 16 \
    --anchors-per-seq 48 \
    --epochs 5 \
    --lr 6e-4 \
    --warmup-ratio 0.01 \
    --eval-every 500 \
    --max-eval-batches 50 \
    "$@"
elif [ "$DATA" = wikitext ]; then
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
else
  echo "DATA must be 'generated' or 'wikitext', got '$DATA'" >&2
  exit 1
fi

# torch-free export: config.json + model.safetensors (the frozen head is taken from the verifier)
python torch_train/export_hspec_drafter.py "$OUT/best.pt" "$OUT/minitorch"
