#!/usr/bin/env bash
# Layout sweep of the loop_engineering flags (q20_kbnw) on a partial instance, in one k8s pod:
# for each TPxCP in LAYOUTS, generate.py twice in one process (LINGBOT_BENCH_RUNS=2, GEN = the
# steady pass) and the decode twice (VAE = the faster: the first decode after --warmup can still
# pay a compile), plus the quality checks of kiro/k8s_validate.sh on every layout:
#   repeat run (determinism), c0s0 vs an fp32 CPU forward (kiro/ref_fp32_c0s0.py), VAE frames vs
#   an fp32 CPU decode of the same latents.
# fps = FRAME_NUM / ((GEN + VAE) / 1000), as the job computes it.
# Usage (repo root, in the pod): LNC=2 NPROC=32 LAYOUTS="4x8 8x4" REPEAT=4x8 bash kiro/k8s_sweep.sh
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT="${OUT:-/tmp/sweep}"; mkdir -p "$OUT"
CKPT="${CKPT:?CKPT}"; NPROC="${NPROC:?NPROC}"; LNC="${LNC:?LNC}"; LAYOUTS="${LAYOUTS:?LAYOUTS}"
REPEAT="${REPEAT:-}"
FRAME_NUM=81; SIZE="480*768"
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"

BASE_ENV=(LINGBOT_BENCH_RUNS=2 LINGBOT_CHUNKED_KV=1 LINGBOT_COMPILE_DIT=1 COMPILE_SCOPE=blocks
  LINGBOT_CP_FLAT_SEQ=1
  "NEURON_CC_FLAGS=--internal-backend-options=--disable-cb-before-global-cc"
  NKI_COMPILER_FRONTEND=tracer NEURON_FALLBACK_ENABLED=1 TORCH_NEURONX_DISABLE_FALLBACK_EXECUTION=1
  NEURON_RT_EXEC_TIMEOUT=600 NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=0
  NEURON_RT_DBG_INTRA_RDH_CHANNEL_BUFFER_SIZE=134217728 NEURON_LOAD_CONCURRENCY=8
  NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR)
# kiro/LOOP_ENGINEERING.md, q20_kbnw (TP and CP come from the layout)
LOOP_ENV=("${BASE_ENV[@]}"
  ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1
  NEURON_LAUNCH_BLOCKING=0 LINGBOT_LOAD_CONCURRENCY=16 LINGBOT_SKIP_LAST_COMMIT=1 LINGBOT_NO_SYNC=1
  LINGBOT_TOKEN_PAD=32 LINGBOT_ATTN_EARLY=2 LINGBOT_ATTN_PAIR=1 LINGBOT_PLUCKER_COMPILE=1
  LINGBOT_FILM_BATCH=1 LINGBOT_HEAD_COMPILE=1 LINGBOT_CROSS_NMAJOR=1 LINGBOT_PREUPLOAD=1
  LINGBOT_EMPTY_CACHE_SEQ=1000000000 LINGBOT_COMMIT_NOHEAD=1
  LINGBOT_VAE_FUSED=1 LINGBOT_VAE_BF16=1 LINGBOT_VAE_CC_FLAGS=-O1 LINGBOT_VAE_NORM=mm
  LINGBOT_VAE_T=6 LINGBOT_VAE_2D=1 LINGBOT_FILM_ALL=1 LINGBOT_KEY_THREADS=4 LINGBOT_IMG_BLOCK=1
  LINGBOT_CTE_LOCAL=1 LINGBOT_CTE_VGROUP=8 NKI_ENABLE_TRACE_CACHE=0 LINGBOT_K_BLOCKED=1
  LINGBOT_STEP_NOWRITE=1 ${EXTRA_FLAGS:-})

run() {  # <name> <tp> <cp>
  local name="$1" tp="$2" cp="$3" d="$OUT/$1"; mkdir -p "$d"
  echo "=== run $name (TP$tp x CP$cp, LNC$LNC, $NPROC ranks) ==="
  (
    for kv in "${LOOP_ENV[@]}"; do export "$kv"; done
    export NEURON_LOGICAL_NC_CONFIG="$LNC" LINGBOT_DUMP_PRED="$d/dump:c0s0" LINGBOT_SAVE_FRAMES="$d/frames.pt"
    ARGS=(--task i2v-A14B --infer_mode causal_fast --ckpt_dir "$CKPT" --size "$SIZE" --frame_num "$FRAME_NUM"
          --chunk_size 3 --image examples/03/image.jpg --action_path examples/03 --prompt "$PROMPT"
          --tp_degree "$tp" --cp_size "$cp" --denoise_steps 4 --local_attn_size 12 --sink_size 1 --base_seed 42)
    torchrun --nproc_per_node="$NPROC" --master_port=29740 generate.py "${ARGS[@]}" \
      --save_file "$d/out.mp4" --save_latents_path "$d/latents.pt" > "$d/gen.log" 2>&1
    echo "  gen exit=$?"
    for i in 1 2; do
      torchrun --nproc_per_node="$NPROC" --master_port=29740 decode_latents_width_sharded.py \
        --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$d/out.mp4" \
        --timing --warmup > "$d/dec$i.log" 2>&1
      echo "  dec$i exit=$?"
    done
  )
  local gen vae
  gen=$(grep -oE "\[BENCH\] GEN time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | tail -1)
  vae=$(cat "$d/dec1.log" "$d/dec2.log" | grep -oE "vae\.decode steady-state \([^)]*\): [0-9.]+" \
        | grep -oE "[0-9.]+$" | sort -n | head -1)
  local fps; fps=$(python3 -c "print(round($FRAME_NUM / ((${gen:-nan} + ${vae:-nan}) / 1000), 2))" 2>/dev/null)
  echo "RESULT $name TP$tp CP$cp LNC$LNC GEN_ms=${gen:-NA} VAE_ms=${vae:-NA} fps=${fps:-NA}" | tee -a "$OUT/results.txt"
  grep -m3 -hE "Traceback|Error|error:" "$d/gen.log" "$d/dec1.log" | cut -c1-220
}

for L in $LAYOUTS; do run "lnc${LNC}_${L}" "${L%x*}" "${L#*x}"; done
[ -n "$REPEAT" ] && run "lnc${LNC}_${REPEAT}_rep" "${REPEAT%x*}" "${REPEAT#*x}"

echo "=== quality ==="
FIRST=""
for L in $LAYOUTS; do [ -s "$OUT/lnc${LNC}_${L}/dump/c0s0_inputs.pt" ] && { FIRST="lnc${LNC}_${L}"; break; }; done
if [ -n "$FIRST" ]; then
  UP=/tmp/lbw-upstream; [ -d "$UP" ] || git clone -q https://github.com/robbyant/lingbot-world-v2.git "$UP"
  DUMPS=(); for L in $LAYOUTS; do DUMPS+=("$OUT/lnc${LNC}_${L}/dump/c0s0.pt"); done
  python3 kiro/ref_fp32_c0s0.py "$OUT/$FIRST/dump/c0s0_inputs.pt" "$CKPT" "$UP" "$OUT/ref_fp32_c0s0.pt" "${DUMPS[@]}" 2>&1 \
    | grep -E "^\[ref\]|Error|Traceback" | tee "$OUT/step0_fp32.txt"
fi
for L in $LAYOUTS; do
  r="lnc${LNC}_${L}"; [ -s "$OUT/$r/latents.pt" ] || continue
  mkdir -p "$OUT/vae_fp32_$r"
  LINGBOT_SAVE_FRAMES="$OUT/vae_fp32_$r/frames.pt" torchrun --nproc_per_node=1 --master_port=29741 \
    decode_latents_width_sharded.py --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$OUT/$r/latents.pt" \
    --save_file "$OUT/vae_fp32_$r/out.mp4" --device cpu > "$OUT/vae_fp32_$r/dec.log" 2>&1
done
python3 - "$OUT" "$LNC" "$LAYOUTS" "$REPEAT" <<'PY' | tee "$OUT/SUMMARY.txt"
import math, os, re, sys, torch
out, lnc, layouts, rep = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4]
def ld(*p):
    p = os.path.join(out, *p); return torch.load(p, map_location="cpu") if os.path.exists(p) else None
def psnr(a, b):
    m = (a.double() - b.double()).pow(2).mean().item(); return float("inf") if m == 0 else 10 * math.log10(255 ** 2 / m)
s0 = {}
if os.path.exists(f"{out}/step0_fp32.txt"):
    for line in open(f"{out}/step0_fp32.txt"):
        m = re.search(r"/(lnc\w+)/dump/c0s0\.pt: c0s0 rel-RMS vs fp32 CPU = ([0-9.]+)", line)
        if m: s0[m.group(1)] = float(m.group(2))
res = {l.split()[1]: l.strip() for l in open(f"{out}/results.txt")} if os.path.exists(f"{out}/results.txt") else {}
print("════════ SWEEP SUMMARY ════════  (c0s0 vs fp32: 16-device main 0.0257, q16 0.0125)")
for L in layouts:
    r = f"lnc{lnc}_{L}"; f, v = ld(r, "frames.pt"), ld(f"vae_fp32_{r}", "frames.pt")
    vae = f"{psnr(f, v):.2f} dB" if f is not None and v is not None and f.shape == v.shape else "NA"
    print(f"  {res.get(r, r + ' (no result)')}  | c0s0 vs fp32 {s0.get(r, 'NA')} | VAE vs fp32 decode {vae}")
if rep:
    a, b = ld(f"lnc{lnc}_{rep}", "latents.pt"), ld(f"lnc{lnc}_{rep}_rep", "latents.pt")
    print(f"  deterministic ({rep} x2): {a is not None and b is not None and torch.equal(a, b)}")
PY
tar -czf "$OUT.tar.gz" --exclude='*.pt' -C "$(dirname "$OUT")" "$(basename "$OUT")" 2>/dev/null
echo "ARCHIVE: $OUT.tar.gz"
