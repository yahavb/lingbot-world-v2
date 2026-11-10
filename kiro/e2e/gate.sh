#!/usr/bin/env bash
# Accuracy gate of one candidate (tools/validate_gate.py, pixels): CPU fp32 rollout reference R,
# baseline B and candidate C sweep config dirs (each has gate/{frames.pt,forward_c0s0.pt,forward_x0c0.pt}).
#   gate.sh <R dir> <B cfg dir> <C cfg dir> [<C repeat cfg dir>] -> exit 0 = PASS; verdict JSON in C/gate_verdict.json
# G0 floors (calibrated in kiro/e2e/LEDGER.md "Gate calibration"): first 9 frames PSNR vs R >= FLOOR_DB,
# c0s0 rel-RMS vs R <= MAX_REL, sharpness vs R >= MIN_SHARP. G3: forwards within REL_RATIO x B.
set -u
T="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/tools"
R="$1"; B="$2"; C="$3"; CR="${4:-}"
ARGS=(--mode pixels --ref "$R" --base "$B/gate" --cand "$C/gate"
      --floor-early-db "${FLOOR_DB:-20}" --max-early-rel "${MAX_REL:-0.05}" --min-sharpness "${MIN_SHARP:-0.9}"
      --early-frames 9 --rel-ratio "${REL_RATIO:-1.3}" --json-out "$C/gate_verdict.json")
[ -n "$CR" ] && ARGS+=(--cand-rep "$CR/gate")
d1="$(dirname "$B")/vae_fp32_$(basename "$B")"; d2="$(dirname "$C")/vae_fp32_$(basename "$C")"
[ -s "$d1/frames.pt" ] && [ -s "$d2/frames.pt" ] && ARGS+=(--ref-dec-base "$d1" --ref-dec-cand "$d2")
[ "${EXACT:-0}" = 1 ] && ARGS+=(--exact)
python3 "$T/validate_gate.py" "${ARGS[@]}"
