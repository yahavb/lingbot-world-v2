#!/usr/bin/env bash
# Hot-head hybrid A/B (LINGBOT_HOT_HEADS): perf/mx_block_ab.sh on the given layers, 2-chip TP4 x CP2 LNC2.
#   C=<comma flags> LAYERS="38 33" TAG=hh1 VARS="b:LINGBOT_HOT_NOCOPY=1 c:LINGBOT_HOT_NOCOPY=1,LINGBOT_HOT_ORDER=1" \
#     [PROF_VARIANT=b] bash perf/hh_ab.sh
# Variant a is always C + LINGBOT_HOT_HEADS=default (the reference); every VARS entry adds its flags to that.
set -uo pipefail
source <(sed "s#^cd /work/lingbot\$#cd /work/lingbot_hh#" kiro_mx4/common.sh)
export NPROC=8 TP=4 CP=2 NEURON_RT_VISIBLE_CORES=0-7
base="$C,LINGBOT_HOT_HEADS=default"
V="a:$base"
for v in ${VARS:-}; do V="$V ${v%%:*}:$base,${v#*:}"; done
for L in ${LAYERS:-38}; do
  o=/work/out/${TAG:?}_$L; rm -rf "$o"; mkdir -p "$o"
  echo "[hh] layer $L variants: $(echo "$V" | sed "s/$C,//g")"
  BLOCK=$L CTX_REAL=40 REPS=${REPS:-10} VARIANTS="$V" PROF_VARIANT=${PROF_VARIANT:-none} bash perf/hh_block_ab.sh "$o"
  for f in "$o"/bb_*_1.log; do echo "[hh] L$L $(basename "$f"): $(grep -aoE 'rel_rms_vs_ref=[0-9.e+-]+|ref_equal=[A-Za-z]+' "$f" | tr '\n' ' ')"; done
done
echo "HH_AB DONE"
