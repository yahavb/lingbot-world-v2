"""Multi-frame check of kernels/vae_norm_halo_conv.py (LINGBOT_VAE_NKI_CONV): the fused norm + SiLU + halo + cache +
causal 3x3x3 conv on T new frames, against the same computation in fp32 on the CPU, per output frame and per cache
frame; then the T frames as two calls of T / 2 chained through the returned cache, against the one call. One core,
fake halos (the tile's own opposite edges, as perf/vae_head_bench.py).
    NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=16 SHAPES=1,96,4,120,192:96;1,96,8,120,192:96 \
        python3 perf/vae_conv_t_check.py
SHAPES: ';'-separated b,C,T,H,W[:COUT] (COUT default C). The kernel options come from the environment as in the
decode (LINGBOT_VAE_NKI_CONV_SQ / _LA / _R / _EDGE).
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from kernels.vae_norm_halo_conv import norm_halo_conv  # noqa: E402
from kernels.vae_head_conv import head_conv_weights  # noqa: E402


def edges(x):
    """Fake neighbour edges of x [.., T, h, w]: (left, right [.., h, 1], up, down [.., 1, w + 2])."""
    left, right = x[..., -1:], x[..., :1]
    top = torch.cat([left[..., :1, :], x[..., :1, :], right[..., :1, :]], -1)
    bot = torch.cat([left[..., -1:, :], x[..., -1:, :], right[..., -1:, :]], -1)
    return left, right, bot, top


def ref(x, cache, g, s, w, b):
    """fp32: conv3d(cat(cache, silu(rms(halo(x)))), w, b) and the last two frames of that conv input."""
    x, cache, w, b = x.float(), cache.float(), w.float(), b.float()
    l, r, u, d = edges(x)
    xh = torch.cat([u, torch.cat([l, x, r], -1), d], -2)
    n = xh * torch.rsqrt((xh * xh).sum(1, keepdim=True).clamp_min(1e-24)) * (g.float().view(1, -1, 1, 1, 1) * s)
    xc = torch.cat([cache, F.silu(n)], 2)
    return F.conv3d(xc, w, b), xc[:, :, -2:]


def dev_fn(x, cache, g, w, b):
    s = x.shape[1] ** 0.5
    l, r, u, d = edges(x)
    y, nc = norm_halo_conv(x[0], l[0], r[0], u[0], d[0], cache[0], g.float().reshape(-1, 1) * s,
                           head_conv_weights(w), b.float().reshape(-1, 1))
    return y.unsqueeze(0), nc.unsqueeze(0)


def rel(a, r):
    return ((a.float() - r.float()).norm() / r.float().norm().clamp_min(1e-30)).item()


def main():
    dev = torch.device("neuron:0")
    bf = torch.bfloat16
    cf = torch.compile(dev_fn, backend="neuron", fullgraph=True, dynamic=False)
    for spec in os.environ.get("SHAPES", "1,96,4,120,192:96;1,96,8,120,192:96").split(";"):
        shp, _, co = spec.partition(":")
        b_, C, T, H, W = (int(v) for v in shp.split(","))
        co = int(co or C)
        torch.manual_seed(0)
        x = (torch.randn(b_, C, T, H, W) * 2).to(bf)
        cache = (torch.randn(b_, C, 2, H + 2, W + 2) * 0.5).to(bf)
        w = (torch.randn(co, C, 3, 3, 3) * 0.05).to(bf)
        bias = (torch.randn(co) * 0.1).to(bf)
        g = 1 + 0.2 * torch.randn(C, 1, 1, 1)
        yr, cr = ref(x, cache, g, C ** 0.5, w, bias)
        dv = [t.to(dev) for t in (x, cache, g, w, bias)]
        y, nc = (t.cpu() for t in cf(*dv))
        fr = " ".join(f"{rel(y[:, :, t], yr[:, :, t]):.1e}" for t in range(T))
        cfr = " ".join(f"{rel(nc[:, :, t], cr[:, :, t]):.1e}" for t in range(2))
        print(f"CONVT C={C} T={T} H={H} W={W} CO={co} one-call vs fp32: conv {rel(y, yr):.2e} (frames {fr}) "
              f"cache {rel(nc, cr):.2e} (frames {cfr})", flush=True)
        if T >= 2 and T % 2 == 0:
            h = T // 2
            y1, c1 = cf(dv[0][:, :, :h].contiguous(), dv[1], *dv[2:])
            y2, c2 = cf(dv[0][:, :, h:].contiguous(), c1, *dv[2:])
            y12 = torch.cat([y1.cpu(), y2.cpu()], 2)
            fr = " ".join(f"{rel(y12[:, :, t], yr[:, :, t]):.1e}" for t in range(T))
            print(f"CONVT C={C} T={T} two calls of {h} vs fp32: conv {rel(y12, yr):.2e} (frames {fr}) "
                  f"cache {rel(c2.cpu(), cr):.2e}; vs one call: conv exact {(y12 == y).float().mean().item():.4f} "
                  f"cache exact {(c2.cpu() == nc).float().mean().item():.4f}", flush=True)


if __name__ == "__main__":
    main()
