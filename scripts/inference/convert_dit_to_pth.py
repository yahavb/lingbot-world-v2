#!/usr/bin/env python3
"""Convert LingBot's causal_fast DiT (WanModelFast) checkpoint into a
TP-degree-independent, mmap-able bf16 .pth (see
wan/modules/model_fast.py:load_tp_wan_model_fast).

Usage:
    python scripts/inference/convert_dit_to_pth.py \\
        --ckpt_dir /shared/huggingface/lingbot-world-v2-14b-causal-fast

Writes <ckpt_dir>/<subfolder>-bf16.pth (default subfolder: "transformers",
matching wan.configs.wan_i2v_A14B.i2v_A14B.fast_checkpoint).
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from wan.modules.model_fast import WanModelFast  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def convert_dit(checkpoint_dir, subfolder, dtype):
    logger.info("[convert:dit] loading WanModelFast from %s (subfolder=%s)",
                checkpoint_dir, subfolder)
    t0 = time.time()
    model = WanModelFast.from_pretrained(
        checkpoint_dir, subfolder=subfolder,
        torch_dtype=dtype, low_cpu_mem_usage=True,
    ).eval()
    logger.info("[convert:dit] loaded in %.1fs; dumping state dict...", time.time() - t0)

    sd = {k: v.to(dtype).contiguous() for k, v in model.state_dict().items()}

    # Validate against a from_config() skeleton before writing -- catches a
    # mismatched checkpoint layout once here, not on every TP rank later.
    config = WanModelFast.load_config(checkpoint_dir, subfolder=subfolder)
    skeleton = WanModelFast.from_config(config)
    expected = {k: tuple(t.shape) for k, t in skeleton.state_dict().items()}
    got = {k: tuple(t.shape) for k, t in sd.items()}
    missing, extra = sorted(set(expected) - set(got)), sorted(set(got) - set(expected))
    if missing or extra:
        raise SystemExit(f"[convert:dit] key mismatch\n  missing: {missing}\n  extra: {extra}")
    bad = {k: (expected[k], got[k]) for k in expected if expected[k] != got[k]}
    if bad:
        raise SystemExit(f"[convert:dit] shape mismatch: {bad}")
    logger.info("[convert:dit] validated %d keys against a from_config() skeleton", len(sd))
    return sd


def _save(sd, out, dtype_name):
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(sd, str(out))
    size_gb = out.stat().st_size / 1e9
    logger.info("[convert:dit] saved %d tensors (%s) -> %s (%.2f GB)",
                len(sd), dtype_name, out, size_gb)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt_dir", required=True,
                     help="Checkpoint directory (matches generate.py's --ckpt_dir).")
    ap.add_argument("--subfolder", default="transformers",
                     help="Subfolder under ckpt_dir holding the DiT weights.")
    ap.add_argument("--dtype", choices=["bf16"], default="bf16")
    ap.add_argument("--output", default=None,
                     help="Output .pth path. Default: <ckpt_dir>/<subfolder>-bf16.pth.")
    ap.add_argument("--force", action="store_true",
                     help="Reconvert even if the output file already exists.")
    args = ap.parse_args()

    dtype = torch.bfloat16
    ckpt_dir = Path(args.ckpt_dir)
    out = Path(args.output) if args.output else ckpt_dir / f"{args.subfolder}-bf16.pth"

    if out.exists() and not args.force:
        logger.info("[convert:dit] %s already exists, skipping (--force to reconvert)", out)
        return

    sd = convert_dit(str(ckpt_dir), args.subfolder, dtype)
    _save(sd, out, args.dtype)


if __name__ == "__main__":
    main()
