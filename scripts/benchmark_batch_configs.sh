#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/travix3/pplus-log-ft}"
PILOT_ADAPTER="${PILOT_ADAPTER:-./outputs/qwen3-8b-ft-v1-pilot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/qwen3-8b-ft-v1-batch-benchmarks}"
LOG_DIR="${LOG_DIR:-./logs}"
RUN_ID="${RUN_ID:-batch-benchmark-$(date +%Y%m%d-%H%M%S)}"
BENCHMARK_STEPS="${BENCHMARK_STEPS:-30}"
BENCHMARK_TRAIN_ROWS="${BENCHMARK_TRAIN_ROWS:-2000}"
BENCHMARK_VAL_ROWS="${BENCHMARK_VAL_ROWS:-500}"
BENCHMARK_OFFSET="${BENCHMARK_OFFSET:-20000}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"

if [ ! -f "${PILOT_ADAPTER}/adapter_model.safetensors" ] && [ ! -f "${PILOT_ADAPTER}/adapter_model.bin" ]; then
  echo "Pilot adapter not found at ${PILOT_ADAPTER}" >&2
  exit 1
fi

echo "Batch benchmark run: ${RUN_ID}"
echo "Pilot adapter: ${PILOT_ADAPTER}"
echo "Steps per config: ${BENCHMARK_STEPS}"
echo "Rows per config: ${BENCHMARK_TRAIN_ROWS}"
echo "Output root: ${OUTPUT_ROOT}"

IFS=' ' read -r -a configs <<< "${BENCHMARK_CONFIGS:-batch12_accum3:12:3}"

for item in "${configs[@]}"; do
  IFS=: read -r name batch_size accum_steps <<< "$item"
  echo
  echo "=== Benchmark ${name} ==="

  SHUFFLE_SEED= \
  START_LOT=0 \
  START_OFFSET="$BENCHMARK_OFFSET" \
  TOTAL_TRAIN="$((BENCHMARK_OFFSET + BENCHMARK_TRAIN_ROWS))" \
  LOT_SIZE="$BENCHMARK_TRAIN_ROWS" \
  VAL_SIZE="$BENCHMARK_VAL_ROWS" \
  PREVIOUS_ADAPTER="$PILOT_ADAPTER" \
  OUTPUT_ROOT="${OUTPUT_ROOT}/${RUN_ID}-${name}" \
  RUN_ID="${RUN_ID}-${name}" \
  TRAIN_BATCH_SIZE="$batch_size" \
  GRADIENT_ACCUMULATION_STEPS="$accum_steps" \
  MAX_STEPS="$BENCHMARK_STEPS" \
  ./scripts/run_lot_training.sh
done

echo
echo "=== Benchmarks complete ==="
echo "Compare logs with:"
echo "  grep -h \"it/s\\|s/it\\|train_runtime\\|train_samples_per_second\" ${LOG_DIR}/train-${RUN_ID}-*.log"
