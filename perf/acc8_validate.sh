#!/usr/bin/env bash
# kiro/k8s_validate.sh on half a trn2.48xlarge (devices 0-7, TP4 x CP8, LNC2, 32 ranks), through the host queue
# (kiro/queue_runner.sh, q8). Same runs, files and verdict (kiro/validate_cmp.py); every run is kiro/run_job16.sh
# (the job's spec and env: GEN = the steady pass of LINGBOT_BENCH_RUNS=2, VAE = the decode after --warmup):
#   loop       this branch, the 8-device production flags (kiro/LOOP_ENGINEERING.md, d8_lean) -> latents, frames, c0s0
#   loop_rep   loop again                                                          -> determinism
#   off        this branch, every flag off, the job's env                          -> latents, frames, c0s0
#   main_code  a main checkout (MAIN_DIR), the job's env                           -> latents, mp4
#   q20        (not in the default RUNS) the q20_kbnw flags, kiro/k8s_sweep.sh's TP4 x CP8 layout: the all-chunk
#              FiLM graph does not load at CP8 on trn2 (NRT_RESOURCE)
# off and main_code use ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte (BASE_ATTN): main's default const_max_ag kernel
# does not trace on this host (8 or 16 devices). Then the fp32 CPU c0s0 reference (kiro/ref_fp32_c0s0.py on off's
# chunk-0 inputs, as k8s_validate.sh, and on loop's own inputs when they differ), the fp32 CPU VAE decode of loop's
# and off's latents, and the verdict. The fp32 references are CPU-heavy: keep timing runs off the host meanwhile.
# Usage (in the container, from the repo): bash kiro/acc8_validate.sh   (RUNS="loop off" for a subset, CPU=0 to
# skip the CPU references; OUT, MAIN_DIR, UP = the robbyant/lingbot-world-v2 checkout, THREADS)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$REPO"
OUT="${OUT:-/work/out/acc8}"; mkdir -p "$OUT"
CKPT=/work/ckpt/lingbot-world-v2-14b-causal-fast
MAIN_DIR="${MAIN_DIR:-/work/repo_main}"; UP="${UP:-/work/lbw_upstream_ref}"; THREADS="${THREADS:-96}"
RUNS="${RUNS:-loop loop_rep off main_code}"
BASE_ATTN=(ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte)
# q20_kbnw = LOOP_ENV of kiro/k8s_sweep.sh
Q20=(ATTENTION_BACKEND=cte LINGBOT_HOT_ATTN=cte LINGBOT_SP=1 LINGBOT_CTE_MERGE=1 LINGBOT_FILM_A2A=1
  NEURON_LAUNCH_BLOCKING=0 LINGBOT_LOAD_CONCURRENCY=16 LINGBOT_SKIP_LAST_COMMIT=1 LINGBOT_NO_SYNC=1
  LINGBOT_TOKEN_PAD=32 LINGBOT_ATTN_EARLY=2 LINGBOT_ATTN_PAIR=1 LINGBOT_PLUCKER_COMPILE=1
  LINGBOT_FILM_BATCH=1 LINGBOT_HEAD_COMPILE=1 LINGBOT_CROSS_NMAJOR=1 LINGBOT_PREUPLOAD=1
  LINGBOT_EMPTY_CACHE_SEQ=1000000000 LINGBOT_COMMIT_NOHEAD=1
  LINGBOT_VAE_FUSED=1 LINGBOT_VAE_BF16=1 LINGBOT_VAE_CC_FLAGS=-O1 LINGBOT_VAE_NORM=mm
  LINGBOT_VAE_T=6 LINGBOT_VAE_2D=1 LINGBOT_FILM_ALL=1 LINGBOT_KEY_THREADS=4 LINGBOT_IMG_BLOCK=1
  LINGBOT_CTE_LOCAL=1 LINGBOT_CTE_VGROUP=8 NKI_ENABLE_TRACE_CACHE=0 LINGBOT_K_BLOCKED=1
  LINGBOT_STEP_NOWRITE=1)
# d8_lean: q33_a with LINGBOT_START_OVERLAP=4, plus the 8-device rows
LOOP=("${Q20[@]}"
  LINGBOT_CTE_CHAIN=1 LINGBOT_CROSS_TPOUT=1 LINGBOT_CTE_QPAD=32 LINGBOT_V_TOKEN_MAJOR=1
  LINGBOT_CTE_PRIOR_FULL=1 LINGBOT_CMAX=2 NEURON_RT_IO_RING_CACHE_SIZE=1024
  LINGBOT_KV_REUSE=1 LINGBOT_START_OVERLAP=4 LINGBOT_SCHED_FUSE=1 LINGBOT_GC_OFF=1
  LINGBOT_FILM_ALL=1 LINGBOT_FILM_GROUP=4 LINGBOT_FILM_SHARD=1 LINGBOT_VAE_T=4 LINGBOT_FILM_ROWS=640
  LINGBOT_KV_COALESCE=1 LINGBOT_RS_BF16=1 LINGBOT_FILM_AR_BF16=1 LINGBOT_CROSS_DEDUP=128
  LINGBOT_COMMIT_Q_FIRST=2 LINGBOT_COMMIT_FROM_GATHER=1 LINGBOT_POST_LEAN=1)
run() {  # <name> <repo> <KEY=VALUE...>
  local name="$1" dir="$2"; shift 2
  local d="$OUT/$name"; rm -rf "$d"; mkdir -p "$d"   # the dumps are written only if absent
  rm -f "/work/out/acc8_${name}_latents.pt" "/work/out/acc8_${name}.mp4"
  [ -f "$OUT/timing.txt" ] && sed -i "/^$name /d" "$OUT/timing.txt"
  echo "=== run $name ($dir) $(date -Is) ==="
  bash "$REPO/kiro/run_job16.sh" "$dir" "$name" "$@" NPROC=32 TP=4 CP=8 PFX=acc8 \
    "LINGBOT_DUMP_PRED=$d/dump:c0s0" "LINGBOT_SAVE_FRAMES=$d/frames.pt"
  cp "/work/out/acc8_${name}_latents.pt" "$d/latents.pt" 2>/dev/null
  cp "/work/out/acc8_${name}.mp4" "$d/out.mp4" 2>/dev/null
  cp "/work/logs/acc8_${name}_gen.log" "$d/gen.log"; cp "/work/logs/acc8_${name}_dec.log" "$d/dec.log"
  cp "/work/logs/acc8_${name}.log" "$d/run.log"
  local gen vae
  gen=$(grep -oE "\[BENCH\] GEN time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | tail -1)
  vae=$(grep -oE "vae\.decode steady-state \([^)]*\): [0-9.]+" "$d/dec.log" | grep -oE "[0-9.]+$" | tail -1)
  echo "$name GEN_ms=${gen:-NA} VAE_ms=${vae:-NA}" | tee -a "$OUT/timing.txt"
  grep -E "^### (gen|dec|smoke) exit" "$d/run.log" | tr '\n' ' '; echo
  grep -m3 -hE "Traceback|Error" "$d/gen.log" "$d/dec.log" | cut -c1-220
}
for r in $RUNS; do
  case "$r" in
    loop|loop_rep) run "$r" "$REPO" "${LOOP[@]}" ;;
    q20)           run q20 "$REPO" "${Q20[@]}" ;;
    off)           run off "$REPO" "${BASE_ATTN[@]}" ;;
    main_code)     echo "main checkout: $(cat "$MAIN_DIR/.commit" 2>/dev/null)"
                   run main_code "$MAIN_DIR" "${BASE_ATTN[@]}" ;;
  esac
done
[ "${CPU:-1}" = "1" ] || exit 0
echo "CPU steps env: $(env | grep -cE '^LINGBOT_') LINGBOT_* variables set"
rm -f "$OUT/loop_inputs_differ"
echo "=== c0s0 inputs (vs off, else loop) $(date -Is) ==="
INP=""; for r in off loop; do [ -s "$OUT/$r/dump/c0s0_inputs.pt" ] && { INP="$OUT/$r/dump/c0s0_inputs.pt"; break; }; done
[ -n "$INP" ] || { echo "no c0s0 inputs"; exit 1; }
python3 - "$OUT" "$INP" <<'PY' | tee "$OUT/inputs_cmp.txt"
import os, sys, torch
out, ref = sys.argv[1], sys.argv[2]
a = torch.load(ref)
for r in ("loop", "loop_rep", "off", "main_code"):
    p = f"{out}/{r}/dump/c0s0_inputs.pt"
    if not os.path.exists(p) or p == ref:
        continue
    b, diff = torch.load(p), []
    for k in ("x", "t", "context", "y"):
        if not torch.equal(a[k], b[k]):
            diff.append(f"{k} max|d| {(a[k] - b[k]).abs().max().item():.3g}")
    for i, (u, v) in enumerate(zip(a["cam"], b["cam"])):
        if not torch.equal(u, v):
            diff.append(f"cam[{i}] max|d| {(u - v).abs().max().item():.3g} on {(u != v).float().mean().item():.2%}")
    diff += [f"{k} {a[k]} vs {b[k]}" for k in ("seq_len", "frame_seqlen", "max_attention_size", "current_start")
             if a[k] != b[k]]
    print(f"  {r} vs {os.path.basename(os.path.dirname(os.path.dirname(ref)))}: "
          + ("identical" if not diff else "; ".join(diff)))
if os.path.exists(f"{out}/loop/dump/c0s0_inputs.pt") and ref != f"{out}/loop/dump/c0s0_inputs.pt":
    b = torch.load(f"{out}/loop/dump/c0s0_inputs.pt")
    same = all(torch.equal(a[k], b[k]) for k in ("x", "t", "context", "y")) and \
        all(torch.equal(u, v) for u, v in zip(a["cam"], b["cam"]))
    if not same:
        open(f"{out}/loop_inputs_differ", "w").close()
PY
DUMPS=(); for r in off loop loop_rep main_code; do DUMPS+=("$OUT/$r/dump/c0s0.pt"); done
echo "=== c0s0 vs fp32 CPU DiT (upstream WanModelFast at $(git -C "$UP" log --oneline -1 2>/dev/null)) $(date -Is) ==="
THREADS=$THREADS python3 kiro/ref_fp32_c0s0.py "$INP" "$CKPT" "$UP" "$OUT/ref_fp32_c0s0.pt" "${DUMPS[@]}" 2>&1 \
  | grep -E "^\[ref\]|Error|Traceback" | tee "$OUT/step0_fp32.txt"
if [ -f "$OUT/loop_inputs_differ" ]; then
  echo "=== c0s0 vs fp32 CPU DiT on loop's own inputs $(date -Is) ==="
  THREADS=$THREADS python3 kiro/ref_fp32_c0s0.py "$OUT/loop/dump/c0s0_inputs.pt" "$CKPT" "$UP" \
    "$OUT/ref_fp32_c0s0_loopin.pt" "$OUT/loop/dump/c0s0.pt" "$OUT/loop_rep/dump/c0s0.pt" 2>&1 \
    | grep -E "^\[ref\]|Error|Traceback" | tee "$OUT/step0_fp32_loopin.txt"
fi
echo "=== fp32 CPU VAE decode (each run's own latents, untiled) $(date -Is) ==="
for r in loop off; do
  [ -s "$OUT/$r/latents.pt" ] || continue
  rm -rf "$OUT/vae_fp32_$r"; mkdir -p "$OUT/vae_fp32_$r"
  OMP_NUM_THREADS=$THREADS LINGBOT_SAVE_FRAMES="$OUT/vae_fp32_$r/frames.pt" torchrun --nproc_per_node=1 \
    --master_port=29741 decode_latents_width_sharded.py --task i2v-A14B --ckpt_dir "$CKPT" \
    --latents_path "$OUT/$r/latents.pt" --save_file "$OUT/vae_fp32_$r/out.mp4" --device cpu \
    > "$OUT/vae_fp32_$r/dec.log" 2>&1
  echo "  vae_fp32_$r exit=$? $(date -Is)"
done
python3 kiro/validate_cmp.py "$OUT" 2>&1 | tee "$OUT/VERDICT.txt"
python3 - "$OUT" <<'PY' | tee -a "$OUT/VERDICT.txt"
import os, sys
out = sys.argv[1]
print("=== c0s0 vs fp32 CPU, both references ===")
for name in ("step0_fp32.txt", "step0_fp32_loopin.txt"):
    if os.path.exists(os.path.join(out, name)):
        print(f"  {name}:"); [print("    " + l.rstrip()) for l in open(os.path.join(out, name)) if "rel-RMS" in l]
PY
echo "=== done $(date -Is) ==="
