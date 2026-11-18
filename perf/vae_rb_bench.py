"""One-core ablation timing of one VAE decoder ResidualBlock graph (wan/modules/vae2_1_fused._res_unit) at the 2-D
16-rank per-rank shapes, without collectives: the halo all-gathers are replaced by local edge copies with the same
concatenates (VARIANT=full), so what is left is the block's compute and copies.
    NEURON_RT_VISIBLE_CORES=0 VARIANT=full python3 perf/vae_rb_bench.py      SHAPE=1,96,4,120,192 REPS=20
VARIANT: full | nonorm (norm+SiLU replaced by a cast) | nohalo (spatial halos replaced by the conv's zero padding,
         no spatial concatenates) | convonly (just the two convs on prebuilt padded inputs + the residual add)
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LINGBOT_VAE_NORM", "mm")
os.environ.setdefault("LINGBOT_VAE_2D", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from wan.modules import vae2_1_fused as vf  # noqa: E402


def fake_halo(x, sel):
    left, right = x[..., -1:], x[..., :1]
    x = torch.cat([left, x, right], -1)
    up, down = x[..., -1:, :], x[..., :1, :]
    return torch.cat([up, x, down], -2)


def fake_edges(x, sel):
    """vf._raw_edges_2d with fake_halo's neighbours (the tile's own opposite edges)."""
    left, right = x[..., -1:], x[..., :1]
    top = torch.cat([left[..., :1, :], x[..., :1, :], right[..., :1, :]], -1)
    bot = torch.cat([left[..., -1:, :], x[..., -1:, :], right[..., -1:, :]], -1)
    return left, right, bot, top


def main():
    shape = [int(v) for v in os.environ.get("SHAPE", "1,96,4,120,192").split(",")]
    reps = int(os.environ.get("REPS", "20"))
    var = os.environ.get("VARIANT", "full")
    b, C, T, H, W = shape
    torch.manual_seed(0)
    dev = torch.device("neuron:0")
    bf = torch.bfloat16
    x = (torch.randn(shape) * 2).to(bf).to(dev)
    c1 = (torch.randn(b, C, 2, H + 2, W + 2) * 0.5).to(bf).to(dev)
    c2 = (torch.randn(b, C, 2, H + 2, W + 2) * 0.5).to(bf).to(dev)
    w1 = (torch.randn(C, C, 3, 3, 3) * 0.02).to(bf).to(dev)
    w2 = (torch.randn(C, C, 3, 3, 3) * 0.02).to(bf).to(dev)
    b1 = torch.zeros(C).to(bf).to(dev)
    b2 = torch.zeros(C).to(bf).to(dev)
    g1 = torch.ones(C, 1, 1, 1).to(dev)
    g2 = torch.ones(C, 1, 1, 1).to(dev)
    s = C ** 0.5
    vf._halo = fake_halo
    vf._raw_edges_2d = fake_edges
    if var == "check":
        # LINGBOT_VAE_NKI_HALO's conv input (kernels/vae_norm_halo.py) vs norm -> halo -> cache cat (fake halos on both
        # sides; with LINGBOT_VAE_NKI_NORM=1 LINGBOT_VAE_NKI_NORM_SQ=fp32 the norm arithmetic is the same, so the
        # difference is the halo/cache assembly alone).
        from kernels.vae_norm_halo import norm_halo
        vf._nki_norm_halo = norm_halo

        def f_new(x, c1, g1, w1, b1):
            return vf._nh_conv(x, c1, g1, s, w1, b1, None)

        def f_ref(x, c1, g1, w1, b1):
            return vf._cconv(fake_halo(vf._norm_silu(x, g1, s, torch.bfloat16), None), c1, w1, b1)
        g1r = (1 + 0.2 * torch.randn(C, 1, 1, 1)).to(dev)
        outs = []
        for fn in (f_new, f_ref):
            cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
            outs.append([t.cpu().float() for t in cf(x, c1, g1r, w1, b1)])
            torch.neuron.synchronize()
            t0 = time.perf_counter()
            for _ in range(reps):
                cf(x, c1, g1r, w1, b1)
            torch.neuron.synchronize()
            print(f"RBCHECK {fn.__name__} us/call={(time.perf_counter() - t0) / reps * 1e6:.1f}", flush=True)
        for name, a, r in zip(("conv", "cache"), outs[0], outs[1]):
            d = (a - r)
            print(f"RBCHECK {name} max_abs={d.abs().max().item():.3e} rel_l2={(d.norm() / r.norm()).item():.3e} "
                  f"ref_absmax={r.abs().max().item():.3e}", flush=True)
        return
    if var == "nonorm":
        vf._norm_silu = lambda x, g, sc, dt: x.to(dt)
    if var == "nohalo":
        vf._halo = lambda x, sel: x

        def cc(xnew_h, cache_h, w, bb):
            xh = torch.cat([cache_h[..., 1:-1, 1:-1], xnew_h.to(w.dtype)], 2)
            return F.conv3d(xh, w, bb, padding=(0, 1, 1)), F.pad(xh[:, :, -2:], (1, 1, 1, 1))
        vf._cconv = cc
    if var == "convonly":
        xh1 = torch.cat([c1, F.pad(x, (1, 1, 1, 1))], 2)
        xh2 = torch.cat([c2, F.pad(x, (1, 1, 1, 1))], 2)

        def fn(xh1, xh2, w1, b1, w2, b2):
            return F.conv3d(xh1, w1, b1) + F.conv3d(xh2, w2, b2)
        args = (xh1, xh2, w1, b1, w2, b2)
    else:
        def fn(x, c1, c2, g1, w1, b1, g2, w2, b2):
            return vf._res_unit(x, c1, c2, g1, s, w1, b1, g2, s, w2, b2, None, None, None)
        args = (x, c1, c2, g1, w1, b1, g2, w2, b2)
    cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    t0 = time.perf_counter()
    cf(*args)
    torch.neuron.synchronize()
    tc = time.perf_counter() - t0
    for _ in range(3):
        cf(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        cf(*args)
    torch.neuron.synchronize()
    dt = (time.perf_counter() - t0) / reps * 1e6
    print(f"RBBENCH {var} nki_norm={vf._NKI_NORM} shape={shape} us/call={dt:.1f} first={tc:.1f}s", flush=True)


if __name__ == "__main__":
    main()
