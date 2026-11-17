#!/usr/bin/env bash
# e2e.sh <name> [K=V ...]: one 81-frame 480x768 video on the pod's 4 devices (TP4 x CP4, LNC2), the
# k8s_sweep.sh measurement: GEN = last of LINGBOT_BENCH_RUNS warm passes in one process, VAE = the
# faster of 2 decodes, fps = 81 / (GEN + VAE). Overrides after the name are exported last.
set -uo pipefail
source kiro_mx4/common.sh
name="${1:?name}"; shift
export LINGBOT_BENCH_RUNS="${LINGBOT_BENCH_RUNS:-3}"
for kv in "$@"; do export "$kv"; done
d="/work/out/e2e/$name"; mkdir -p "$d"; [ "${GATE:-0}" = 1 ] && export LINGBOT_DUMP_PRED="$d/dump:c0s0" LINGBOT_DUMP_ROLLOUT="$d/dump"; env | grep -E '^(LINGBOT|NEURON|ATTENTION|COMPILE|NKI|FLASH|TP|CP)' | sort > "$d/env.txt"
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"
ARGS=(--task i2v-A14B --infer_mode causal_fast --ckpt_dir "$CKPT" --size "${SIZE:-480*768}" --frame_num 81
      --chunk_size 3 --image examples/03/image.jpg --action_path examples/03 --prompt "$PROMPT"
      --tp_degree "$TP" --cp_size "$CP" --denoise_steps "${DENOISE_STEPS:-4}" --local_attn_size 12 --sink_size 1 --base_seed 42)
t0=$(date +%s)
# GEN_VIS / VAE_VIS: NEURON_RT_VISIBLE_CORES of the generation / the decodes (two runs side by side on one pod);
# PORT: torchrun master port (default 29740)
env ${GEN_VIS:+NEURON_RT_VISIBLE_CORES=$GEN_VIS} torchrun --nproc_per_node="$NPROC" --master_port="${PORT:-29740}" generate.py "${ARGS[@]}" \
  --save_file "$d/out.mp4" --save_latents_path "$d/latents.pt" 2>&1 | grep -avE "AliasingInfo|Mutated inputs|Detected collective" > "$d/gen.log"
# VAE_LNC=1 VAE_NPROC=16: decode at LNC1 (on 2 devices the LNC2 8-rank 2-D tiling (4 x 2) decodes black frames)
for i in 1 2; do
  env ${VAE_VIS:+NEURON_RT_VISIBLE_CORES=$VAE_VIS} NEURON_LOGICAL_NC_CONFIG="${VAE_LNC:-$NEURON_LOGICAL_NC_CONFIG}" \
  LINGBOT_SAVE_FRAMES="$d/frames.pt" \
  torchrun --nproc_per_node="${VAE_NPROC:-$NPROC}" --master_port="${PORT:-29740}" decode_latents_width_sharded.py \
    --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$d/out.mp4" \
    --timing --warmup 2>&1 | grep -avE "AliasingInfo|Mutated inputs" > "$d/dec$i.log"
done
gen=$(grep -aoE "\[BENCH\] GEN time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | tail -1)
runs=$(grep -aoE "\[BENCH\] GEN run[0-9]+ time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | paste -sd, -)
vae=$(cat "$d/dec1.log" "$d/dec2.log" | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
fps=$(python3 -c "print(round(81 / ((${gen:-nan} + ${vae:-nan}) / 1000), 2))" 2>/dev/null)
echo "RESULT $name GEN_ms=${gen:-NA} VAE_ms=${vae:-NA} fps=${fps:-NA} runs=${runs:-NA} wall_s=$(( $(date +%s) - t0 )) overrides=$*" | tee -a /work/out/e2e/results.txt
grep -m3 -ahE "Traceback|Error|error:|NRT_RESOURCE|OOM" "$d/gen.log" "$d/dec1.log" | cut -c1-220
if [ "${GATE:-0}" = 1 ]; then   # perf/e2e/gate.sh layout: gate/{frames.pt, forward_c0s0.pt, forward_x0c0.pt}
  mkdir -p "$d/gate"; ln -sf ../frames.pt "$d/gate/frames.pt"; ln -sf ../dump/c0s0.pt "$d/gate/forward_c0s0.pt"
  python3 -c "import torch,sys; r=torch.load(sys.argv[1]); torch.save(r['x0'][:, :r['chunk_size']].contiguous(), sys.argv[2])" \
    "$d/dump/rollout_inputs.pt" "$d/gate/forward_x0c0.pt"
  bash perf/e2e/gate.sh /node-cache/e2e/gate/r480_ex03_s42 /node-cache/e2e/mx-lnc1-dge-g/lnc1_4x8 "$d" > "$d/gate.txt" 2>&1
  echo "GATE $name exit=$? $(python3 -c "import json,sys; v=json.load(open(sys.argv[1])); print(v['verdict'])" "$d/gate_verdict.json" 2>/dev/null)" | tee -a /work/out/e2e/results.txt
fi
