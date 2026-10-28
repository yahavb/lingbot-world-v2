#!/usr/bin/env bash
# torchrun --no-python wrapper: Neuron runtime inspector (system + session device trace) on LOCAL_RANK 0 only.
#   torchrun --no-python ... bash kiro/prof_rank0.sh python3 kiro/block_bench.py      (PROF_DIR required)
unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_SYSTEM_PROFILE NEURON_RT_INSPECT_DEVICE_PROFILE NEURON_RT_INSPECT_OUTPUT_DIR
if [ "${LOCAL_RANK:-0}" = "0" ]; then
  export NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1
  export NEURON_RT_INSPECT_DEVICE_PROFILE=session NEURON_RT_INSPECT_OUTPUT_DIR="${PROF_DIR:?PROF_DIR not set}"
fi
exec "$@"
