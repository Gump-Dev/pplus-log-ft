#!/usr/bin/env bash
set -euo pipefail

PILOT_PID="${PILOT_PID:?Set PILOT_PID to the running pilot process id}"
PILOT_LOG="${PILOT_LOG:-logs/train-pilot-20260621-142710.log}"
PILOT_ADAPTER="${PILOT_ADAPTER:-./outputs/qwen3-8b-ft-v1-pilot}"
CHECK_INTERVAL="${CHECK_INTERVAL:-300}"

echo "[$(date --iso-8601=seconds)] waiting for pilot PID ${PILOT_PID}"

while kill -0 "$PILOT_PID" 2>/dev/null; do
  echo "[$(date --iso-8601=seconds)] pilot still running"
  sleep "$CHECK_INTERVAL"
done

echo "[$(date --iso-8601=seconds)] pilot process ended"

if grep -q "=== Done" "$PILOT_LOG" \
  && { [ -f "${PILOT_ADAPTER}/adapter_model.safetensors" ] || [ -f "${PILOT_ADAPTER}/adapter_model.bin" ]; }; then
  echo "[$(date --iso-8601=seconds)] pilot adapter found; starting remaining lots"
  if [ "${RUN_QUALITY_GATE:-0}" = "1" ]; then
    echo "[$(date --iso-8601=seconds)] running dataset quality gate before remaining lots"
    PYTHON_BIN="${PYTHON_BIN:-/home/travix3/vllm-install/.vllm/bin/python3}" ./scripts/run_quality_gate.sh
    echo "[$(date --iso-8601=seconds)] dataset quality gate passed"
  fi
  if [ "${RUN_BATCH_BENCHMARKS:-0}" = "1" ]; then
    echo "[$(date --iso-8601=seconds)] running batch benchmarks before remaining lots"
    PILOT_ADAPTER="$PILOT_ADAPTER" ./scripts/benchmark_batch_configs.sh
    winner_env="$(ls -t logs/batch-benchmark-winner-*.env 2>/dev/null | head -n 1 || true)"
    if [ -n "$winner_env" ]; then
      echo "[$(date --iso-8601=seconds)] loading benchmark winner from ${winner_env}"
      # shellcheck disable=SC1090
      source "$winner_env"
      export TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS GRADIENT_CHECKPOINTING
      echo "[$(date --iso-8601=seconds)] selected benchmark winner ${BENCHMARK_WINNER:-unknown}: batch=${TRAIN_BATCH_SIZE:-}, accum=${GRADIENT_ACCUMULATION_STEPS:-}, gradient_checkpointing=${GRADIENT_CHECKPOINTING:-}"
    else
      echo "[$(date --iso-8601=seconds)] benchmark winner env not found; using existing lot overrides"
    fi
    echo "[$(date --iso-8601=seconds)] batch benchmarks finished; starting remaining lots"
  fi
  SHUFFLE_SEED= \
  START_LOT="${START_LOT:-1}" \
  PREVIOUS_ADAPTER="$PILOT_ADAPTER" \
  ./scripts/run_lot_training.sh
else
  echo "[$(date --iso-8601=seconds)] pilot did not finish cleanly; not starting lots" >&2
  exit 1
fi
