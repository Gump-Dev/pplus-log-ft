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
  if [ "${RUN_BATCH_BENCHMARKS:-0}" = "1" ]; then
    echo "[$(date --iso-8601=seconds)] running batch benchmarks before remaining lots"
    PILOT_ADAPTER="$PILOT_ADAPTER" ./scripts/benchmark_batch_configs.sh
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
