#!/usr/bin/env bash
# Build NIXL (ai-dynamo/nixl, Neuron support in its LIBFABRIC plugin since v1.4) without CUDA, against the
# image's libfabric (/opt/amazon/efa), once per node: the result lives in the node cache and every later pod
# reuses it (~30 s build). Prints the env to use it:  eval "$(bash tools/nixl/build_nixl.sh)"
set -euo pipefail
VER=${NIXL_VERSION:-v1.5.2}
ROOT=${NIXL_CACHE:-/node-cache/nixl}/$VER
PY=$(python3 -c 'import sys; print(f"python{sys.version_info.major}.{sys.version_info.minor}")')
PREFIX=$ROOT/opt/nvidia/nvda_nixl
if [ ! -f "$ROOT/.complete" ]; then
  mkdir -p "$ROOT"
  exec 9>"$ROOT.lock"; flock 9
  if [ ! -f "$ROOT/.complete" ]; then
    { apt-get update -qq && apt-get install -y -qq libhwloc-dev; } >/dev/null 2>&1
    pip install -q meson ninja pybind11 >/dev/null 2>&1
    W=$(mktemp -d); git clone -q --depth 1 -b "$VER" https://github.com/ai-dynamo/nixl.git "$W/nixl"
    ( cd "$W/nixl" && meson setup build -Dlibfabric_path=/opt/amazon/efa -Ddisable_gds_backend=true \
        -Denable_plugins=LIBFABRIC -Dbuild_tests=false >/dev/null && ninja -C build -j 32 >/dev/null \
      && meson install -C build --destdir "$ROOT" >/dev/null )
    rm -rf "$W"; touch "$ROOT/.complete"
  fi
fi
# hwloc is a runtime dependency of the plugin; install it in this container if missing
ldconfig -p 2>/dev/null | grep -q libhwloc.so.15 || { apt-get update -qq && apt-get install -y -qq libhwloc15; } >/dev/null 2>&1 || true
echo "export PYTHONPATH=$PREFIX/lib/$PY/site-packages\${PYTHONPATH:+:\$PYTHONPATH}"
echo "export LD_LIBRARY_PATH=$PREFIX/lib/x86_64-linux-gnu:/opt/amazon/efa/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
echo "export NIXL_PLUGIN_DIR=$PREFIX/lib/x86_64-linux-gnu/plugins"
