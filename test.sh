python torch_train/eval_compare_mla.py \
  --checkpoint runs/mla-gpt2-medium/best.pt \
  --gpt2-checkpoint runs/gpt2-medium-original/best.pt \
  --data-cache ./data \
  --seq-len 16 \
  --batch-size 16 \
  --max-batches 1 \
  --attention fused \
  --verify-weights