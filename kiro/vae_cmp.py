"""Compare two VAE_SAVE_DIR dumps of kiro/vae_bench.py slice by slice.
    python3 kiro/vae_cmp.py <ref_dir> <test_dir> [n_slices_kept]
n_slices_kept: slices inside the real (unpadded) width; at 64 ranks the latent is padded 96 -> 128,
so slices 48..63 are cropped from the video and are reported separately.
Video values are in [-1, 1]; PSNR is over that range (peak-to-peak 2) and over the 8-bit frames
the saved video would hold.
"""
import math
import os
import sys

import torch

def full_video(d, keep_w):
    """Assemble a dump into [C, T, H, W]: width slices (default) or 2-D tiles (LAYOUT=2d, rank
    P*dcol + crow with P = 4 rows of tiles)."""
    fs = sorted(f for f in os.listdir(d) if f.endswith(".pt"))
    parts = [torch.load(os.path.join(d, f)) for f in fs]
    if os.path.exists(os.path.join(d, "LAYOUT")):
        P = 4
        D = len(parts) // P
        cols = [torch.cat([parts[P * j + i] for i in range(P)], dim=-2) for j in range(D)]
        return torch.cat(cols, dim=-1)
    return torch.cat(parts[:keep_w], dim=-1)


if __name__ == "__main__":
    ref_dir, test_dir = sys.argv[1], sys.argv[2]
    files = sorted(f for f in os.listdir(ref_dir) if f.endswith(".pt"))
    keep = int(sys.argv[3]) if len(sys.argv) > 3 else len(files)





    if os.path.exists(os.path.join(test_dir, "LAYOUT")) or os.path.exists(os.path.join(ref_dir, "LAYOUT")):
        a = full_video(ref_dir, keep).double()
        b = full_video(test_dir, keep).double()
        W = min(a.shape[-1], b.shape[-1])
        a, b = a[..., :W], b[..., :W]
        d = (a - b).abs()
        mse = (d ** 2).mean().item()
        a8 = ((a + 1) * 127.5).round().clamp(0, 255)
        b8 = ((b + 1) * 127.5).round().clamp(0, 255)
        mse8 = ((a8 - b8) ** 2).mean().item()
        print(f"full video {tuple(a.shape)}: max|d|={d.max().item():.3e} "
              f"PSNR={'inf' if mse == 0 else f'{10 * math.log10(4 / mse):.2f}'} dB; 8-bit: "
              f"max|d|={int((a8 - b8).abs().max().item())} "
              f"PSNR={'inf' if mse8 == 0 else f'{10 * math.log10(255 ** 2 / mse8):.2f}'} dB")
        sys.exit(0)
    se = n = 0
    se8 = 0
    mx = 0.0
    neq = tot = 0
    mx8 = 0
    for i, f in enumerate(files[:keep]):
        a = torch.load(os.path.join(ref_dir, f)).double()
        b = torch.load(os.path.join(test_dir, f)).double()
        d = (a - b).abs()
        mx = max(mx, d.max().item())
        se += (d ** 2).sum().item()
        n += d.numel()
        neq += (d > 0).sum().item()
        tot += d.numel()
        a8 = ((a + 1) * 127.5).round().clamp(0, 255)
        b8 = ((b + 1) * 127.5).round().clamp(0, 255)
        d8 = (a8 - b8).abs()
        mx8 = max(mx8, int(d8.max().item()))
        se8 += (d8 ** 2).sum().item()
    mse = se / n
    mse8 = se8 / n
    print(f"slices 0..{keep - 1}: max|d|={mx:.3e} differing={neq}/{tot} ({100 * neq / tot:.4f}%) "
          f"PSNR={'inf' if mse == 0 else f'{10 * math.log10(4 / mse):.2f}'} dB; "
          f"8-bit: max|d|={mx8} PSNR={'inf' if mse8 == 0 else f'{10 * math.log10(255 ** 2 / mse8):.2f}'} dB")
