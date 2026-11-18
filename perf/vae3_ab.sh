#!/usr/bin/env bash
# perf/vae3_ab.sh <cores> <port> <spec> ...: decode A/B on the production VAE flag set, one spec after the other.
# spec = <tag>[:K=V,K=V...] runs perf/vae_env_check_here.sh <tag> with the production flags + the extra K=V;
# spec = unit:<tag>[:K=V,...] runs perf/vae3_unit_timing.sh (per-block synced times) instead.
# Run from the code dir (e.g. /work/lingbot_vae3).
PROD="LINGBOT_VAE_T=1 LINGBOT_VAE_HEAD_BF16=1 LINGBOT_VAE_CC_FLAGS=-O1+--model-type=unet-inference LINGBOT_VAE_NKI_HALO=1
LINGBOT_VAE_NKI_CONV=all LINGBOT_VAE_NKI_CONV_SQ=bf16 LINGBOT_VAE_NKI_CONV_LA=1 LINGBOT_VAE_NKI_CONV_CMAX=192"
vis=$1; port=$2; shift 2
for spec in "$@"; do
  mode=env
  if [ "${spec%%:*}" = unit ]; then mode=unit; spec=${spec#unit:}; fi
  tag=${spec%%:*}; extra=""
  [ "$tag" != "$spec" ] && extra=$(echo "${spec#*:}" | tr ',' ' ')
  echo "=== $(date +%T) $mode $tag $extra"
  # shellcheck disable=SC2086
  if [ $mode = unit ]; then bash perf/vae3_unit_timing.sh "$tag" "$vis" "$port" $PROD $extra
  else bash perf/vae_env_check_here.sh "$tag" "$vis" "$port" $PROD $extra; fi
done
echo "=== $(date +%T) done"
