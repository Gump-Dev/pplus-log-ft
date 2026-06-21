#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/travix3/pplus-log-ft}"
PILOT_ADAPTER="${PILOT_ADAPTER:-./outputs/qwen3-8b-ft-v1-pilot}"
OUTPUT_ROOT="${OUTPUT_ROOT:-./outputs/qwen3-8b-ft-v1-batch-benchmarks}"
LOG_DIR="${LOG_DIR:-./logs}"
RUN_ID="${RUN_ID:-batch-benchmark-$(date +%Y%m%d-%H%M%S)}"
WINNER_ENV="${WINNER_ENV:-${LOG_DIR}/batch-benchmark-winner-${RUN_ID}.env}"
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
echo "Winner env: ${WINNER_ENV}"

IFS=' ' read -r -a configs <<< "${BENCHMARK_CONFIGS:-batch12_accum3:12:3:true batch16_accum2:16:2:true batch8_accum4_no_gc:8:4:false}"

results_file="${LOG_DIR}/batch-benchmark-results-${RUN_ID}.tsv"
echo -e "name\tbatch_size\taccum_steps\tgradient_checkpointing\tstatus\telapsed_seconds\tlog_file" > "$results_file"
best_name=""
best_batch_size=""
best_accum_steps=""
best_gradient_checkpointing=""
best_elapsed=""

for item in "${configs[@]}"; do
  IFS=: read -r name batch_size accum_steps gradient_checkpointing <<< "$item"
  gradient_checkpointing="${gradient_checkpointing:-true}"
  log_file="${LOG_DIR}/benchmark-${RUN_ID}-${name}.log"
  echo
  echo "=== Benchmark ${name} batch=${batch_size} accum=${accum_steps} gradient_checkpointing=${gradient_checkpointing} ==="

  started_at="$(date +%s)"
  set +e
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
  GRADIENT_CHECKPOINTING="$gradient_checkpointing" \
  MAX_STEPS="$BENCHMARK_STEPS" \
  ./scripts/run_lot_training.sh 2>&1 | tee "$log_file"
  status="${PIPESTATUS[0]}"
  set -e
  finished_at="$(date +%s)"
  elapsed="$((finished_at - started_at))"

  if [ "$status" -eq 0 ]; then
    echo -e "${name}\t${batch_size}\t${accum_steps}\t${gradient_checkpointing}\tstable\t${elapsed}\t${log_file}" >> "$results_file"
    if [ -z "$best_elapsed" ] || [ "$elapsed" -lt "$best_elapsed" ]; then
      best_name="$name"
      best_batch_size="$batch_size"
      best_accum_steps="$accum_steps"
      best_gradient_checkpointing="$gradient_checkpointing"
      best_elapsed="$elapsed"
    fi
  else
    echo "Benchmark ${name} failed with exit code ${status}; keeping log ${log_file}" >&2
    echo -e "${name}\t${batch_size}\t${accum_steps}\t${gradient_checkpointing}\tfailed:${status}\t${elapsed}\t${log_file}" >> "$results_file"
  fi
done

echo
echo "=== Benchmarks complete ==="
echo "Results: ${results_file}"
if [ -z "$best_name" ]; then
  echo "No stable benchmark config found" >&2
  exit 1
fi

cat > "$WINNER_ENV" <<EOF
TRAIN_BATCH_SIZE=${best_batch_size}
GRADIENT_ACCUMULATION_STEPS=${best_accum_steps}
GRADIENT_CHECKPOINTING=${best_gradient_checkpointing}
BENCHMARK_WINNER=${best_name}
BENCHMARK_WINNER_SECONDS=${best_elapsed}
BENCHMARK_RESULTS=${results_file}
EOF

echo "Winner: ${best_name} (${best_elapsed}s)"
echo "Winner env: ${WINNER_ENV}"
echo "Compare logs with:"
echo "  cat ${results_file}"
