#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/travix3/pplus-log-ft}"
PYTHON_BIN="${PYTHON_BIN:-/home/travix3/vllm-install/.vllm/bin/python3}"
CONFIG="${CONFIG:-configs/training.yaml}"
DATA_DIR="${DATA_DIR:-./datasets}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/qwen3-8b-ft-v1-lots}"
LOG_DIR="${LOG_DIR:-./logs}"

LOT_SIZE="${LOT_SIZE:-20000}"
VAL_SIZE="${VAL_SIZE:-2000}"
SHUFFLE_SEED="${SHUFFLE_SEED-20260621}"
START_LOT="${START_LOT:-0}"
START_OFFSET="${START_OFFSET:-}"
PREVIOUS_ADAPTER="${PREVIOUS_ADAPTER:-}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-}"
MAX_STEPS="${MAX_STEPS:-}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

TOTAL_TRAIN="${TOTAL_TRAIN:-$(wc -l < "${DATA_DIR}/train.jsonl")}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d-%H%M%S)}"
STATE_FILE="${OUTPUT_ROOT}/lot-state-${RUN_ID}.tsv"

echo -e "lot\toffset\tcount\tadapter\tstarted_at\tfinished_at" > "$STATE_FILE"
echo "Lot training run: ${RUN_ID}"
echo "Total train rows: ${TOTAL_TRAIN}"
echo "Lot size: ${LOT_SIZE}"
if [ -n "$SHUFFLE_SEED" ]; then
  echo "Shuffle seed: ${SHUFFLE_SEED}"
else
  echo "Shuffle seed: disabled"
fi
echo "Output root: ${OUTPUT_ROOT}"
echo "State file: ${STATE_FILE}"
if [ -n "$TRAIN_BATCH_SIZE" ]; then
  echo "Batch size override: ${TRAIN_BATCH_SIZE}"
fi
if [ -n "$GRADIENT_ACCUMULATION_STEPS" ]; then
  echo "Gradient accumulation override: ${GRADIENT_ACCUMULATION_STEPS}"
fi
if [ -n "$MAX_STEPS" ]; then
  echo "Max steps override: ${MAX_STEPS}"
fi

if [ -n "$START_OFFSET" ]; then
  offset="$START_OFFSET"
  lot="$START_LOT"
else
  offset=$((START_LOT * LOT_SIZE))
  lot="$START_LOT"
fi
previous_adapter="$PREVIOUS_ADAPTER"

while [ "$offset" -lt "$TOTAL_TRAIN" ]; do
  remaining=$((TOTAL_TRAIN - offset))
  count="$LOT_SIZE"
  if [ "$remaining" -lt "$LOT_SIZE" ]; then
    count="$remaining"
  fi

  lot_name="$(printf 'lot-%03d' "$lot")"
  lot_dir="${OUTPUT_ROOT}/${lot_name}"
  log_file="${LOG_DIR}/train-${RUN_ID}-${lot_name}.log"
  started_at="$(date --iso-8601=seconds)"

  mkdir -p "$lot_dir"

  cmd=(
    "$PYTHON_BIN" training/train_lora.py
    --config "$CONFIG"
    --data-dir "$DATA_DIR"
    --output-dir "$lot_dir"
    --train-offset "$offset"
    --max-train "$count"
    --max-val "$VAL_SIZE"
  )

  if [ -n "$SHUFFLE_SEED" ]; then
    cmd+=(--shuffle-seed "$SHUFFLE_SEED")
  fi

  if [ -n "$TRAIN_BATCH_SIZE" ]; then
    cmd+=(--batch-size "$TRAIN_BATCH_SIZE")
  fi

  if [ -n "$GRADIENT_ACCUMULATION_STEPS" ]; then
    cmd+=(--gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS")
  fi

  if [ -n "$MAX_STEPS" ]; then
    cmd+=(--max-steps "$MAX_STEPS")
  fi

  if [ -n "$previous_adapter" ]; then
    cmd+=(--resume-adapter "$previous_adapter")
  fi

  echo
  echo "=== Starting ${lot_name}: offset=${offset}, count=${count} ==="
  echo "Output: ${lot_dir}"
  echo "Log: ${log_file}"
  if [ -n "$previous_adapter" ]; then
    echo "Resume adapter: ${previous_adapter}"
  fi

  "${cmd[@]}" 2>&1 | tee "$log_file"

  finished_at="$(date --iso-8601=seconds)"
  echo -e "${lot}\t${offset}\t${count}\t${lot_dir}\t${started_at}\t${finished_at}" >> "$STATE_FILE"

  previous_adapter="$lot_dir"
  offset=$((offset + count))
  lot=$((lot + 1))
done

echo
echo "=== All lots complete ==="
echo "Final adapter: ${previous_adapter}"
echo "State file: ${STATE_FILE}"
