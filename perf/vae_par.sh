#!/usr/bin/env bash
# perf/vae_par.sh <tag> "<core>:<K=V ...>:<script>" ...: one-core benches side by side, one per LNC1 core, with the
# decode's compiler flags (common.sh + LINGBOT_VAE_CC_FLAGS=-O1+--model-type=unet-inference); prints each bench's
# result lines. Logs in /work/out/vaepar/<tag>/<i>.log.
tag=$1; shift
export NEURON_LOGICAL_NC_CONFIG=1 PYTHONUNBUFFERED=1 NEURON_RT_LOG_LEVEL=ERROR NEURON_CC_LOG_LEVEL=ERROR
export NEURON_CC_FLAGS="--internal-backend-options=--disable-cb-before-global-cc -O1 --model-type=unet-inference"
o=/work/out/vaepar/$tag; rm -rf $o; mkdir -p $o
i=0
for spec in "$@"; do
  core=${spec%%:*}; rest=${spec#*:}; script=${rest##*:}; kvs=${rest%:*}
  # shellcheck disable=SC2086
  ( env NEURON_RT_VISIBLE_CORES=$core $kvs python3 $script > $o/$i.log 2>&1; echo "EXIT $? core=$core $kvs $script" >> $o/$i.log ) &
  i=$((i + 1))
done
wait
for f in $o/*.log; do grep -aE "BENCH|RBCHECK|Error|EXIT" $f | cut -c1-400; done
