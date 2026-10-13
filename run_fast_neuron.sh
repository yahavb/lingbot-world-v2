#!/bin/bash
# Trainium (Neuron) bring-up - LingBot-World v2 causal_fast i2v.
#
# ONE torchrun launch (EWORLD ranks), all three stages -- T5 encode,
# TENSOR-PARALLEL DiT denoise, width-sharded VAE decode -- in the same
# process (generate.py's e2e mode: --tp_degree set, neither
# --prompt_embeds_path nor --save_latents_path passed).
#
# Build <ckpt_dir>/transformers-bf16.pth once via
# scripts/inference/convert_dit_to_pth.py -- otherwise the DiT falls back
# to a slower, more host-RAM-hungry from_pretrained load (at any EWORLD).
#
# Usage: ./run_fast_neuron.sh [EWORLD] [LNC] [SAVE_FILE] - runs with defaults (EWORLD=4, LNC=2)
# Example: ./run_fast_neuron.sh 4 2 output/my_video.mp4

mkdir -p output

EWORLD=${1:-4}
LNC="${2:-2}"
SAVE_FILE="${3:-output/lingbot-world-v2_causal_fast_$(date +%Y%m%d_%H%M%S).mp4}"
PROMPT="gentle ripples spreading across the calm lake water, tree leaves swaying softly in the breeze, clouds drifting slowly across the blue sky, cinematic slow camera push-in, serene natural motion"

# NEURON_LAUNCH_BLOCKING works around a trn2-only launch issue; trn3 doesn't need it.
if neuron-ls 2>/dev/null | grep -qi '^instance-type:[[:space:]]*trn2'; then
    echo "trn2 detected -- exporting NEURON_LAUNCH_BLOCKING=1"
    export NEURON_LAUNCH_BLOCKING=1
fi

NEURON_LOGICAL_NC_CONFIG="$LNC" torchrun --nproc_per_node="$EWORLD" generate.py \
    --task i2v-A14B --infer_mode causal_fast \
    --size 480*832 --frame_num 21 --chunk_size 1 \
    --ckpt_dir /shared/huggingface/lingbot-world-v2-14b-causal-fast \
    --image examples/03/image.jpg --action_path examples/03 \
    --prompt "$PROMPT" \
    --tp_degree "$EWORLD" \
    --local_attn_size 6 --sink_size 1 \
    --save_file "$SAVE_FILE"
