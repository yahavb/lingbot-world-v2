#!/usr/bin/env bash
# torchrun --no-python wrapper: Neuron runtime SYSTEM profile (runtime events and one span per NEFF execution, no
# device instruction trace) on global rank 0 only.
#   SYSPROF_DIR=<dir> torchrun --no-python ... bash kiro/sysprof_rank0.sh python3 generate.py ...
unset NEURON_RT_INSPECT_ENABLE NEURON_RT_INSPECT_SYSTEM_PROFILE NEURON_RT_INSPECT_DEVICE_PROFILE NEURON_RT_INSPECT_OUTPUT_DIR
if [ "${RANK:-0}" = "0" ]; then
  export NEURON_RT_INSPECT_ENABLE=1 NEURON_RT_INSPECT_SYSTEM_PROFILE=1 NEURON_RT_INSPECT_DEVICE_PROFILE=0
  export NEURON_RT_INSPECT_OUTPUT_DIR="${SYSPROF_DIR:?SYSPROF_DIR not set}"
fi
exec "$@"
