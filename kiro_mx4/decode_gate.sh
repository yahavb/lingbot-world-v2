#!/usr/bin/env bash
# decode_gate.sh <e2e name>: redo the VAE decodes and the gate of an e2e.sh run whose generation finished (its
# latents.pt and dump/ exist), e.g. after the decode step failed. Same env knobs as e2e.sh (VAE_LNC, VAE_NPROC,
# VAE_VIS, PORT); prints the RESULT line with the run's GEN time.
set -uo pipefail
source kiro_mx4/common.sh
name="${1:?name}"
d="/work/out/e2e/$name"
for i in 1 2; do
  env ${VAE_VIS:+NEURON_RT_VISIBLE_CORES=$VAE_VIS} NEURON_LOGICAL_NC_CONFIG="${VAE_LNC:-$NEURON_LOGICAL_NC_CONFIG}" \
  LINGBOT_SAVE_FRAMES="$d/frames.pt" \
  torchrun --nproc_per_node="${VAE_NPROC:-$NPROC}" --master_port="${PORT:-29740}" decode_latents_width_sharded.py \
    --task i2v-A14B --ckpt_dir "$CKPT" --latents_path "$d/latents.pt" --save_file "$d/out.mp4" \
    --timing --warmup 2>&1 | grep -avE "AliasingInfo|Mutated inputs" > "$d/dec$i.log"
done
gen=$(grep -aoE "\[BENCH\] GEN time: [0-9.]+" "$d/gen.log" | grep -oE "[0-9.]+$" | tail -1)
vae=$(cat "$d/dec1.log" "$d/dec2.log" | grep -aoE "vae\.decode steady-state \([^)]*\): [0-9.]+" | grep -oE "[0-9.]+$" | sort -n | head -1)
fps=$(python3 -c "print(round(81 / ((${gen:-nan} + ${vae:-nan}) / 1000), 2))" 2>/dev/null)
echo "RESULT $name GEN_ms=${gen:-NA} VAE_ms=${vae:-NA} fps=${fps:-NA} (decode_gate.sh)" | tee -a /work/out/e2e/results.txt
mkdir -p "$d/gate"; ln -sf ../frames.pt "$d/gate/frames.pt"; ln -sf ../dump/c0s0.pt "$d/gate/forward_c0s0.pt"
python3 -c "import torch,sys; r=torch.load(sys.argv[1]); torch.save(r['x0'][:, :r['chunk_size']].contiguous(), sys.argv[2])" \
  "$d/dump/rollout_inputs.pt" "$d/gate/forward_x0c0.pt"
bash perf/e2e/gate.sh /node-cache/e2e/gate/r480_ex03_s42 /node-cache/e2e/mx-lnc1-dge-g/lnc1_4x8 "$d" > "$d/gate.txt" 2>&1
echo "GATE $name exit=$? $(python3 -c "import json,sys; v=json.load(open(sys.argv[1])); print(v['verdict'])" "$d/gate_verdict.json" 2>/dev/null)" | tee -a /work/out/e2e/results.txt
