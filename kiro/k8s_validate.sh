#!/usr/bin/env bash
# End-to-end latency + quality validation of loop_engineering against main, in one k8s pod
# (lbw-v2-i2v-validate-job.yaml), with the job's spec: 81 frames 480x768, chunk 3, window 12,
# sink 1, 4 denoise steps, examples/03, seed 42. Each run = generate.py (GEN, steady-state
# pass of LINGBOT_BENCH_RUNS) + decode_latents_width_sharded.py (VAE steady state), exactly
# as kiro/run_job16.sh; steady fps = FRAME_NUM / ((GEN + VAE) / 1000).
#
# Runs (in this order; a failed run is reported and the rest continue):
#   loop       this branch, the loop_engineering flags (TP4 x CP16)     -> latents, frames, c0s0
#   loop_rep   loop again                                                 -> determinism
#   off        this branch, every flag off, the job's env (TP8 x CP8)     -> latents, frames, c0s0
#   main_code  main checkout, the job's env (TP8 x CP8)                   -> latents, mp4
#   exact      this branch, the job's env with ATTENTION_BACKEND=sdpa     -> bf16 reference trajectory
# Then: fp32 CPU VAE decode of loop's and off's latents (each run's own VAE error); latent
# rel-RMS per chunk, frame PSNR / worst frame / max LSB against exact; c0s0 (chunk 0, step 0
# noise prediction) against exact; main == off bit-level; loop == loop_rep; a verdict.
# Usage (repo root, in the pod): bash kiro/k8s_validate.sh   (RUNS="loop off" to run a subset)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT="${OUT:-/tmp/validate}"; mkdir -p "$OUT"
CKPT="${CKPT:?CKPT}"
NPROC="${NPROC:-64}"; FRAME_NUM=81; SIZE="480*768"
RUNS="${RUNS:-loop loop_rep off main_code exact}"
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"

BASE_ENV=(LINGBOT_BENCH_RUNS=2 LINGBOT_CHUNKED_KV=1 LINGBOT_COMPILE_DIT=1 COMPILE_SCOPE=blocks
  LINGBOT_CP_FLAT_SEQ=1 ATTENTION_BACKEND=const_max_ag
  "NEURON_CC_FLAGS=--internal-backend-options=--disable-cb-before-global-cc"
  NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
  NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
  NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728 NEURON_LOAD_CONCURRENCY=8
  NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR TP=8 CP=8)
# kiro/LOOP_ENGINEERING.md, q16_key (the latest measured row)
LOOP_ENV=("${BASE_ENV[@]}" TP=4 CP=16
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1
  NEURON_LAUNCH_BLOCKING=0 LINGBOT_LOAD_CONCURRENCY=16 LINGBOT_SKIP_LAST_COMMIT=1 LINGBOT_NO_SYNC=1
  LINGBOT_TOKEN_PAD=32 LINGBOT_ATTN_EARLY=2 LINGBOT_ATTN_PAIR=1 LINGBOT_PLUCKER_COMPILE=1
  LINGBOT_FILM_BATCH=1 LINGBOT_HEAD_COMPILE=1 LINGBOT_CROSS_NMAJOR=1 LINGBOT_PREUPLOAD=1
  LINGBOT_EMPTY_CACHE_SEQ=1000000000 LINGBOT_COMMIT_NOHEAD=1
  LINGBOT_VAE_FUSED=1 LINGBOT_VAE_BF16=1 LINGBOT_VAE_CC_FLAGS=-O1 LINGBOT_VAE_NORM=mm
  LINGBOT_VAE_T=6 LINGBOT_VAE_2D=1 LINGBOT_FILM_ALL=1 LINGBOT_KEY_THREADS=4 LINGBOT_IMG_BLOCK=1)
EXACT_ENV=("${BASE_ENV[@]}" ATTENTION_BACKEND=sdpa)

run() {  # <name> <dir> <env...>
  local name="$1" dir="$2"; shift 2
  local d="$OUT/$name"; mkdir -p "$d"
  echo "=== run $name ($dir) ==="
  (
    cd "$dir"
    for kv in "$@"; do export "$kv"; done
    export NEURON_LOGICAL_NC_CONFIG=2 LINGBOT_DUMP_PRED="$d/dump:c0s0" LINGBOT_SAVE_FRAMES="$d/frames.pt"
    ARGS=(--task i2v-A14B --infer_mode causal_fast --ckpt_dir "$CKPT" --size "$SIZE" --frame_num "$FRAME_NUM"
          --chunk_size 3 --image examples/03/image.jpg --action_path examples/03 --prompt "$PROMPT"
          --tp_degree "$TP" --cp_size "$CP" --denoise_steps 4 --local_attn_size 12 --sink_size 1 --base_seed 42)
    torchrun --nproc_per_node="$NPROC" --master_port=29740 generate.py "${ARGS[@]}" \
      --save_file "$d/out.mp4" --save_latents_path "$d/latents.pt" > "$d/gen.log" 2>&1
    echo "  gen exit=$?"
    torchrun --nproc_per_node="$NPROC" --master_port=29740 decode_latents_width_sharded.py \
      --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$d/out.mp4" \
      --timing --warmup > "$d/dec.log" 2>&1
    echo "  dec exit=$?"
  )
  local gen vae
  gen=$(grep -oE "\[BENCH\] GEN time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | tail -1)
  vae=$(grep -oE "vae\.decode steady-state \([^)]*\): [0-9.]+" "$d/dec.log" | grep -oE "[0-9.]+$" | tail -1)
  echo "$name GEN_ms=${gen:-NA} VAE_ms=${vae:-NA}" | tee -a "$OUT/timing.txt"
  grep -m3 -E "Traceback|Error" "$d/gen.log" "$d/dec.log" | cut -c1-220
}

MAIN_DIR=/tmp/lbw-main
for r in $RUNS; do
  case "$r" in
    loop|loop_rep) run "$r" "$REPO" "${LOOP_ENV[@]}" ;;
    off)           run off "$REPO" "${BASE_ENV[@]}" ;;
    exact)         run exact "$REPO" "${EXACT_ENV[@]}" ;;
    main_code)
      rm -rf "$MAIN_DIR"; git clone -q -b main "$(git remote get-url origin)" "$MAIN_DIR" \
        && echo "main checkout: $(git -C "$MAIN_DIR" log --oneline -1)" && run main_code "$MAIN_DIR" "${BASE_ENV[@]}" ;;
  esac
done

echo "=== fp32 CPU VAE decode (each run's own latents, untiled) ==="
for r in loop off; do
  [ -s "$OUT/$r/latents.pt" ] || continue
  mkdir -p "$OUT/vae_fp32_$r"
  LINGBOT_SAVE_FRAMES="$OUT/vae_fp32_$r/frames.pt" torchrun --nproc_per_node=1 --master_port=29741 \
    decode_latents_width_sharded.py --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$OUT/$r/latents.pt" \
    --save_file "$OUT/vae_fp32_$r/out.mp4" --device cpu > "$OUT/vae_fp32_$r/dec.log" 2>&1
  echo "  vae_fp32_$r exit=$?"
done

echo "=== c0s0 vs fp32 CPU DiT (upstream WanModelFast, same inputs) ==="
if [ -s "$OUT/off/dump/c0s0_inputs.pt" ]; then
  UP=/tmp/lbw-upstream; [ -d "$UP" ] || git clone -q https://github.com/robbyant/lingbot-world-v2.git "$UP"
  python3 - "$OUT" <<'PY'
import sys, torch, os
o = sys.argv[1]; a = torch.load(f"{o}/off/dump/c0s0_inputs.pt")
for r in ("loop", "exact"):
    p = f"{o}/{r}/dump/c0s0_inputs.pt"
    if os.path.exists(p):
        b = torch.load(p)
        same = all(torch.equal(a[k], b[k]) for k in ("x", "t", "context", "y")) and \
            all(torch.equal(u, v) for u, v in zip(a["cam"], b["cam"]))
        print(f"  c0s0 inputs {r} == off: {same}")
PY
  python3 kiro/ref_fp32_c0s0.py "$OUT/off/dump/c0s0_inputs.pt" "$CKPT" "$UP" "$OUT/ref_fp32_c0s0.pt" \
    "$OUT/off/dump/c0s0.pt" "$OUT/loop/dump/c0s0.pt" "$OUT/exact/dump/c0s0.pt" 2>&1 \
    | grep -E "^\[ref\]|Error|Traceback" | tee "$OUT/step0_fp32.txt"
fi

python3 kiro/validate_cmp.py "$OUT" 2>&1 | tee "$OUT/VERDICT.txt"
tar -czf "$OUT.tar.gz" --exclude='*.pt' -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
echo "ARCHIVE: $OUT.tar.gz"
