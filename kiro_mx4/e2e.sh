#!/usr/bin/env bash
# e2e.sh <name> [K=V ...]: one 81-frame 480x768 video on the pod's 4 devices (TP4 x CP4, LNC2), the
# k8s_sweep.sh measurement: GEN = last of LINGBOT_BENCH_RUNS warm passes in one process, VAE = the
# faster of 2 decodes, fps = 81 / (GEN + VAE). Overrides after the name are exported last.
set -uo pipefail
source kiro_mx4/common.sh
name="${1:?name}"; shift
export LINGBOT_BENCH_RUNS="${LINGBOT_BENCH_RUNS:-3}"
for kv in "$@"; do export "$kv"; done
d="/work/out/e2e/$name"; mkdir -p "$d"; env | grep -E '^(LINGBOT|NEURON|ATTENTION|COMPILE|NKI|FLASH|TP|CP)' | sort > "$d/env.txt"
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"
ARGS=(--task i2v-A14B --infer_mode causal_fast --ckpt_dir "$CKPT" --size "${SIZE:-480*768}" --frame_num 81
      --chunk_size 3 --image examples/03/image.jpg --action_path examples/03 --prompt "$PROMPT"
      --tp_degree "$TP" --cp_size "$CP" --denoise_steps "${DENOISE_STEPS:-4}" --local_attn_size 12 --sink_size 1 --base_seed 42)
t0=$(date +%s)
torchrun --nproc_per_node="$NPROC" --master_port=29740 generate.py "${ARGS[@]}" \
  --save_file "$d/out.mp4" --save_latents_path "$d/latents.pt" 2>&1 | grep -avE "AliasingInfo|Mutated inputs|Detected collective" > "$d/gen.log"
for i in 1 2; do
  LINGBOT_SAVE_FRAMES="$d/frames.pt" torchrun --nproc_per_node="${VAE_NPROC:-$NPROC}" --master_port=29740 decode_latents_width_sharded.py \
    --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$d/out.mp4" \
    --timing --warmup 2>&1 | grep -avE "AliasingInfo|Mutated inputs" > "$d/dec$i.log"
done
gen=$(grep -aoE "\[BENCH\] GEN time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | tail -1)
runs=$(grep -aoE "\[BENCH\] GEN run[0-9]+ time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | paste -sd, -)
vae=$(cat "$d/dec1.log" "$d/dec2.log" | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
fps=$(python3 -c "print(round(81 / ((${gen:-nan} + ${vae:-nan}) / 1000), 2))" 2>/dev/null)
echo "RESULT $name GEN_ms=${gen:-NA} VAE_ms=${vae:-NA} fps=${fps:-NA} runs=${runs:-NA} wall_s=$(( $(date +%s) - t0 )) overrides=$*" | tee -a /work/out/e2e/results.txt
grep -m3 -ahE "Traceback|Error|error:|NRT_RESOURCE|OOM" "$d/gen.log" "$d/dec1.log" | cut -c1-220
