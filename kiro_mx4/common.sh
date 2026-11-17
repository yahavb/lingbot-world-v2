# sourced by every kiro_mx4 experiment: LNC2 MX deploy flags (lbw-v2-lnc2-mx-deploy.yaml) + 4-chip layout
cd /work/lingbot
source kiro_mx4/env_lnc2_mx.sh
export CKPT=/node-cache/lingbot-world-v2-14b-causal-fast NPROC=16 TP=4 CP=4 LNC=2 NEURON_LOGICAL_NC_CONFIG=2
export FLASH_ATTN_LAYERS=33,34,38,39 PYTHONUNBUFFERED=1 HF_HOME=/work/hf-cache
export NEURON_COMPILE_CACHE_URL=/work/cache/neuron-compile-cache TORCH_NEURONX_NEFF_CACHE_DIR=/work/cache/neff_cache
export TORCH_NEURONX_HLO_CACHE_DIR=/work/cache/hlo_cache
