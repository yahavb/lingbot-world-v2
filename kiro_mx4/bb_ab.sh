#!/usr/bin/env bash
# one-block A/B: bf16 vs the deployed MX config at TP4 x CP4 LNC2, profile of PROF_VARIANT
source kiro_mx4/common.sh
OUT=${1:?out}; shift
VARIANTS="${VARIANTS:-bf16:LINGBOT_FFN_MX=0,LINGBOT_LIN_MX=0 mx:LINGBOT_FFN_MX=1,LINGBOT_LIN_MX=1}" \
PROF_VARIANT="${PROF_VARIANT:-mx}" bash perf/mx_block_ab.sh "$OUT"
