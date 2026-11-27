#!/usr/bin/env bash
# Drafter ablation against the frozen MLA verifier: drafter architecture x drafter training data.
#
#                               WikiText-2 text      verifier-generated text ($GEN)
#   distilgpt2, classic AR      $DISTIL_WIKI         $DISTIL_GEN
#   H-Spec block drafter        $HSPEC_WIKI          $HSPEC_GEN
#
# All four start from distilgpt2 and train with the same verifier-teacher loss (0.1 CE + 0.9 TV),
# AdamW with 1% warmup and cosine decay, batch 16 x 128 tokens, $EPOCHS epochs, on the same data
# files; only the architecture and the data differ. The H-Spec defaults are the two runs already
# trained this way (runs/hspec-drafter-a64-lr6e-4-e10, runs/hspec-drafter-gen-s32).
#
#   bash eval.sh                          # train whatever is missing, then benchmark and summarize
#   STAGES=train bash eval.sh             # train only (e.g. one GPU allocation) ...
#   STAGES=bench bash eval.sh             # ... benchmark later; configs already in $RESULTS are skipped
#   N_PROMPTS=20 TEMPS=1.0 bash eval.sh   # more prompts; also sample at temperature 1
#   DRY_RUN=1 bash eval.sh                # print the commands without running them
#
# Stage train: generates $GEN if missing (the verifier continues 64-token WikiText-2 prompts taken
# every $GEN_STRIDE tokens with 64 greedy tokens), then trains each drafter whose checkpoint is
# missing ($OUT/best/ for distilgpt2, $OUT/best.pt for H-Spec; delete it to retrain).
# Stage bench: exports the verifier and the H-Spec drafters for MiniTorch if needed ($OUT/minitorch;
# the distilgpt2 drafters' best/ loads as is); every config decodes the same $N_PROMPTS WikiText-2 prompts with the MiniTorch MLA
# verifier: ar_target (baseline), each fine-tuned drafter at several draft lengths, and, with
# REFERENCE=1, pretrained distilgpt2 (no fine-tuning). Rows go to $RESULTS (labelled by drafter);
# the tables to $SUMMARY. Needs a GPU, the verifier trained by torch_train/train.py and the
# compiled MiniTorch kernels (compile_cuda.sh).
set -euo pipefail
cd "$(dirname "$0")"

VERIFIER=${VERIFIER:-runs/mla-gpt2-medium/best.pt}
VERIFIER_EXPORT=${VERIFIER_EXPORT:-runs/mla-gpt2-medium/minitorch}
GEN=${GEN:-data/verifier_gen_s32}
GEN_STRIDE=${GEN_STRIDE:-32}
DISTIL_WIKI=${DISTIL_WIKI:-runs/distil-drafter-wikitext}
DISTIL_GEN=${DISTIL_GEN:-runs/distil-drafter-gen-s32}
HSPEC_WIKI=${HSPEC_WIKI:-runs/hspec-drafter-a64-lr6e-4-e10}
HSPEC_GEN=${HSPEC_GEN:-runs/hspec-drafter-gen-s32}
EPOCHS=${EPOCHS:-10}
DISTIL_LR=${DISTIL_LR:-2e-4}
HSPEC_LR=${HSPEC_LR:-6e-4}
RESULTS=${RESULTS:-results/eval.jsonl}
SUMMARY=${SUMMARY:-results/eval.md}
N_PROMPTS=${N_PROMPTS:-10}
DISTIL_CONFIGS=${DISTIL_CONFIGS:-spec_k2 spec_k3 spec_k4 spec_k5}
HSPEC_CONFIGS=${HSPEC_CONFIGS:-hspec_k3 hspec_k5 hspec}
TEMPS=${TEMPS:-}
REFERENCE=${REFERENCE:-1}
STAGES=${STAGES:-train bench}
DRY_RUN=${DRY_RUN:-0}
export MLA_VERIFIER_DIR="$VERIFIER_EXPORT"  # read by specdec.common

run() {  # print a command (shell-quoted, to stderr), then run it unless DRY_RUN=1
  printf '+' >&2
  printf ' %q' "$@" >&2
  printf '\n' >&2
  if [ "$DRY_RUN" != 1 ]; then "$@"; fi
}
stage() { case " $STAGES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }
with_temps() {  # configs, each followed by its _t{T} variants for T in $TEMPS
  local c t
  for c in "$@"; do
    printf '%s ' "$c"
    for t in $TEMPS; do printf '%s_t%s ' "$c" "$t"; done
  done
}

if stage train; then
  if [ ! -f "$GEN/train.pt" ] || [ ! -f "$GEN/validation.pt" ]; then
    run python torch_train/generate_verifier_data.py \
      --verifier-checkpoint "$VERIFIER" \
      --data-cache ./data \
      --source wikitext-2 \
      --out-dir "$GEN" \
      --prompt-len 64 \
      --gen-len 64 \
      --prompt-stride "$GEN_STRIDE" \
      --batch-size 128
  fi

  # 1. distilgpt2, classic drafter, WikiText-2
  if [ ! -f "$DISTIL_WIKI/best/model.safetensors" ]; then
    run python torch_train/finetune_distilgpt2.py \
      --verifier-checkpoint "$VERIFIER" \
      --init distilgpt2 \
      --output-dir "$DISTIL_WIKI" \
      --data-cache ./data \
      --seq-len 128 \
      --batch-size 16 \
      --epochs "$EPOCHS" \
      --lr "$DISTIL_LR" \
      --warmup-ratio 0.01 \
      --eval-every 500 \
      --max-eval-batches 50
  fi

  # 2. distilgpt2, classic drafter, verifier-generated text (loss on the generated tokens only)
  if [ ! -f "$DISTIL_GEN/best/model.safetensors" ]; then
    run python torch_train/finetune_distilgpt2.py \
      --verifier-checkpoint "$VERIFIER" \
      --init distilgpt2 \
      --output-dir "$DISTIL_GEN" \
      --train-data "$GEN/train.pt" \
      --val-data "$GEN/validation.pt" \
      --seq-len 128 \
      --batch-size 16 \
      --epochs "$EPOCHS" \
      --lr "$DISTIL_LR" \
      --warmup-ratio 0.01 \
      --eval-every 500 \
      --max-eval-batches 50
  fi

  # 3. H-Spec drafter, WikiText-2
  if [ ! -f "$HSPEC_WIKI/best.pt" ]; then
    run python torch_train/train_drafter.py \
      --verifier-checkpoint "$VERIFIER" \
      --init distilgpt2 \
      --output-dir "$HSPEC_WIKI" \
      --data-cache ./data \
      --seq-len 128 \
      --batch-size 16 \
      --anchors-per-seq 64 \
      --epochs "$EPOCHS" \
      --lr "$HSPEC_LR" \
      --warmup-ratio 0.01 \
      --eval-every 500 \
      --max-eval-batches 50
  fi

  # 4. H-Spec drafter, verifier-generated text (blocks inside the generated part)
  if [ ! -f "$HSPEC_GEN/best.pt" ]; then
    run python torch_train/train_drafter.py \
      --verifier-checkpoint "$VERIFIER" \
      --init distilgpt2 \
      --output-dir "$HSPEC_GEN" \
      --train-data "$GEN/train.pt" \
      --val-data "$GEN/validation.pt" \
      --seq-len 128 \
      --batch-size 16 \
      --anchors-per-seq 48 \
      --epochs "$EPOCHS" \
      --lr "$HSPEC_LR" \
      --warmup-ratio 0.01 \
      --eval-every 500 \
      --max-eval-batches 50
  fi
fi

if stage bench; then
  # torch-free exports for MiniTorch (the distilgpt2 drafters' best/ is loaded as is)
  if [ ! -f "$VERIFIER_EXPORT/config.json" ]; then
    run python torch_train/export_mla_verifier.py "$VERIFIER" "$VERIFIER_EXPORT"
  fi
  for OUT in "$HSPEC_WIKI" "$HSPEC_GEN"; do
    if [ ! -f "$OUT/minitorch/config.json" ]; then
      run python torch_train/export_hspec_drafter.py "$OUT/best.pt" "$OUT/minitorch"
    fi
  done
  if [ "$DRY_RUN" != 1 ]; then
    for d in "$VERIFIER_EXPORT" "$DISTIL_WIKI/best" "$DISTIL_GEN/best" "$HSPEC_WIKI/minitorch" "$HSPEC_GEN/minitorch"; do
      if [ ! -f "$d/config.json" ]; then
        echo "missing $d/config.json: run the train stage first (STAGES=train bash eval.sh)" >&2
        exit 1
      fi
    done
  fi
  BENCH=(python -m specdec.bench --target mla-gpt2-medium --n-prompts "$N_PROMPTS" --prompt-len 64 --n-new 64
         --out "$RESULTS" --skip-done)
  run "${BENCH[@]}" --configs ar_target
  if [ "$REFERENCE" = 1 ]; then
    # shellcheck disable=SC2046  # word-split the config list
    run "${BENCH[@]}" --draft distilgpt2 --draft-label distilgpt2-pretrained --configs $(with_temps $DISTIL_CONFIGS)
  fi
  # shellcheck disable=SC2046
  run "${BENCH[@]}" --draft-dir "$DISTIL_WIKI/best" --draft-label distil-wikitext --configs $(with_temps $DISTIL_CONFIGS)
  # shellcheck disable=SC2046
  run "${BENCH[@]}" --draft-dir "$DISTIL_GEN/best" --draft-label distil-gen --configs $(with_temps $DISTIL_CONFIGS)
  # shellcheck disable=SC2046
  run "${BENCH[@]}" --hspec-drafter "$HSPEC_WIKI/minitorch" --hspec-label hspec-wikitext --configs $(with_temps $HSPEC_CONFIGS)
  # shellcheck disable=SC2046
  run "${BENCH[@]}" --hspec-drafter "$HSPEC_GEN/minitorch" --hspec-label hspec-gen --configs $(with_temps $HSPEC_CONFIGS)

  if [ "$DRY_RUN" = 1 ]; then
    run python -m specdec.summarize "$RESULTS"
  else
    python -m specdec.summarize "$RESULTS" | tee "$SUMMARY"
    echo "summary written to $SUMMARY"
  fi
fi
