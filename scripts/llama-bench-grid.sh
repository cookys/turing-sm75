#!/usr/bin/env bash
# Four-grid protocol used for every speed claim in this repo.
# Requires: llama-bench that prints the GPU name, a GGUF, CUDA sm_75 build.
#
#   source scripts/env.sh
#   scripts/llama-bench-grid.sh /path/to/Qwen3.8-27B-IQ4_XS.gguf
set -euo pipefail

MODEL="${1:?gguf path}"
LLAMA_BENCH="${LLAMA_BENCH:-llama-bench}"
REPS="${REPS:-3}"

if ! command -v "$LLAMA_BENCH" >/dev/null 2>&1 && [ ! -x "$LLAMA_BENCH" ]; then
  echo "set LLAMA_BENCH to your llama-bench binary" >&2
  exit 1
fi

echo "=== four-grid r=${REPS} (dmon in background — trust load rows, not the first idle line) ==="
DMON_LOG="$(mktemp)"
nvidia-smi dmon -s pucvmet -d 1 > "$DMON_LOG" 2>/dev/null &
DMON_PID=$!
sleep 1
"$LLAMA_BENCH" -m "$MODEL" -ngl 99 -fa on -p 512 -n 128 -d 0,4096 -r "$REPS" -o md
kill "$DMON_PID" 2>/dev/null || true
wait "$DMON_PID" 2>/dev/null || true
echo "=== dmon during bench (tail) ==="
tail -n 15 "$DMON_LOG" || true
rm -f "$DMON_LOG"

echo
echo "Claim checklist:"
echo "  - stdout contains RTX 2080 Ti (or your sm_75 card) and compute 7.5"
echo "  - report pp512, tg128, pp512@d4096, tg128@d4096 with stddev"
echo "  - one variable vs the previous grid (clocks XOR quant XOR kernel)"
echo "  - MTP is a separate long-prompt run with the SAME prompt; not this table"
