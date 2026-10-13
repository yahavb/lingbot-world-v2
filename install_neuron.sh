#!/bin/bash
set -e

# Add fix for lzma package on Trainium2 instances.
# Version-agnostic: the image's python moved 3.12 -> 3.14 when the container was
# repinned, and both the .so glob and the destination directory were hardcoded to
# 3.12, so this silently stopped matching. Derive both from the running python.
if ! python -c "import lzma" 2>/dev/null; then
    PYTAG="$(python -c 'import sys; print("%d%d" % sys.version_info[:2])')"
    PYMM="$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
    DEST="$(python -c 'import sysconfig; print(sysconfig.get_config_var("DESTSHARED"))')"
    ORIGINAL_FILE="$(ls /usr/lib/python3*/lib-dynload/_lzma.cpython-${PYTAG}-*-linux-gnu.so 2>/dev/null | head -1)"
    if [ -z "$ORIGINAL_FILE" ]; then
        echo "ERROR: this Python ($PYMM) has no _lzma and no _lzma.cpython-${PYTAG}-*.so in /usr/lib. torchvision will not import. Needs a fixed image." >&2
        exit 1
    fi
    cp "$ORIGINAL_FILE" "$DEST/"
    python -c "import lzma"
    echo "note: patched missing stdlib _lzma from $ORIGINAL_FILE -> $DEST"
fi

# Torch, NKI, and Neuron compiler are pre-installed in the Docker image.
pip install -r "$(dirname "$0")/requirements_neuron.txt"

# torchvision: pre-built wheels ABI-mismatch against this container's torch, so
# build from source instead. --no-deps avoids pulling in a newer torch.
#
# THE TAG MUST MATCH THE IMAGE'S TORCH. torchvision releases track torch minor for
# minor; building v0.26.0 (torch 2.11) against torch 2.14 fails at
# "Preparing metadata (pyproject.toml) ... error / metadata-generation-failed",
# which is what the pinned py314 image hit. Derived, not hardcoded, so repinning the
# image does not silently break this again. Override with TORCHVISION_REF=vX.Y.Z.
pip install --no-deps Pillow
TORCH_MM="$(python -c 'import torch; print(".".join(torch.__version__.split("+")[0].split(".")[:2]))')"
if [ -z "${TORCHVISION_REF:-}" ]; then
    case "$TORCH_MM" in
        2.9)  TORCHVISION_REF=v0.24.0 ;;
        2.10) TORCHVISION_REF=v0.25.0 ;;
        2.11) TORCHVISION_REF=v0.26.0 ;;
        2.12) TORCHVISION_REF=v0.27.0 ;;
        2.13) TORCHVISION_REF=v0.28.0 ;;
        2.14) TORCHVISION_REF=v0.29.0 ;;
        *)
            echo "ERROR: no torchvision tag known for torch $TORCH_MM. torchvision tracks" >&2
            echo "       torch minor-for-minor (2.11->0.26, 2.12->0.27, ...). Set" >&2
            echo "       TORCHVISION_REF=vX.Y.Z explicitly and add it to the case above." >&2
            exit 1 ;;
    esac
fi
echo "torchvision: building $TORCHVISION_REF for torch $TORCH_MM"
TORCHVISION_BUILD_DIR="$(mktemp -d)"
git -c advice.detachedHead=false clone -q --depth 1 --branch "$TORCHVISION_REF" https://github.com/pytorch/vision.git "$TORCHVISION_BUILD_DIR"
pip install "$TORCHVISION_BUILD_DIR" --no-build-isolation --no-deps
rm -rf "$TORCHVISION_BUILD_DIR"
