"""One-core timing of the VAE decoder head (wan/modules/vae2_1_fused._head_unit: RMS_norm + SiLU + causal 3x3x3 conv
96 -> 3) at the 2-D 16-rank stage-3 shape, without collectives (fake halos: the tile's own opposite edges).
    NEURON_RT_VISIBLE_CORES=0 VARIANT=full python3 perf/vae_head_bench.py      SHAPE=1,96,4,120,192 REPS=20
VARIANT: full (the head unit, LINGBOT_VAE_NKI_HALO path) | halo (kernels/vae_norm_halo.py alone) | conv (F.conv3d alone
         on a prebuilt conv input) | nki (kernels/vae_head_conv.py on the same input, checked against F.conv3d)
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("LINGBOT_VAE_NORM", "mm")
os.environ.setdefault("LINGBOT_VAE_2D", "1")
os.environ.setdefault("LINGBOT_VAE_NKI_HALO", "1")
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: F401,E402
from wan.modules import vae2_1_fused as vf  # noqa: E402


def fake_edges(x, sel):
    left, right = x[..., -1:], x[..., :1]
    top = torch.cat([left[..., :1, :], x[..., :1, :], right[..., :1, :]], -1)
    bot = torch.cat([left[..., -1:, :], x[..., -1:, :], right[..., -1:, :]], -1)
    return left, right, bot, top


def timeit(cf, args, reps):
    t0 = time.perf_counter()
    out = cf(*args)
    torch.neuron.synchronize()
    tc = time.perf_counter() - t0
    for _ in range(3):
        cf(*args)
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        cf(*args)
    torch.neuron.synchronize()
    return out, (time.perf_counter() - t0) / reps * 1e6, tc


def main():
    shape = [int(v) for v in os.environ.get("SHAPE", "1,96,4,120,192").split(",")]
    reps = int(os.environ.get("REPS", "20"))
    var = os.environ.get("VARIANT", "full")
    co = int(os.environ.get("COUT", "3"))
    b, C, T, H, W = shape
    torch.manual_seed(0)
    dev = torch.device("neuron:0")
    bf = torch.bfloat16
    x = (torch.randn(shape) * 2).to(bf).to(dev)
    cache = (torch.randn(b, C, 2, H + 2, W + 2) * 0.5).to(bf).to(dev)
    w = (torch.randn(co, C, 3, 3, 3) * 0.05).to(bf).to(dev)
    bias = (torch.randn(co) * 0.1).to(bf).to(dev)
    g = (1 + 0.2 * torch.randn(C, 1, 1, 1)).to(dev)
    s = C ** 0.5
    vf._raw_edges_2d = fake_edges
    buf = (torch.randn(b, C, T + 2, H + 2, W + 2)).to(bf).to(dev)
    if var == "full":
        def fn(x, cache, g, w, bias):
            return vf._head_unit(x, cache, g, s, w, bias, None)
        args = (x, cache, g, w, bias)
    elif var == "halo":
        def fn(x, cache, g):
            xb = x.to(bf)
            l, r, u, d = fake_edges(xb, None)
            return vf._nki_norm_halo(xb[0], l[0], r[0], u[0], d[0], cache[0], g.float().reshape(C, 1) * s)
        args = (x, cache, g)
    elif var == "conv":
        def fn(buf, w, bias):
            return F.conv3d(buf, w, bias)
        args = (buf, w, bias)
    elif var == "rbres":
        # ResidualBlock unit (LINGBOT_VAE_NKI_CONV=all) with the shortcut add in the second conv's kernel
        # (vf._NKI_CONV_RES, read at trace time) vs without: outputs and caches must match exactly.
        c2 = (torch.randn(b, C, 2, H + 2, W + 2) * 0.5).to(bf).to(dev)
        w2 = (torch.randn(C, C, 3, 3, 3) * 0.02).to(bf).to(dev)
        wr = (torch.randn(C, C, 3, 3, 3) * 0.02).to(bf).to(dev)
        br = (torch.randn(C) * 0.1).to(bf).to(dev)

        def rb(x, cache, c2, g, wr, br, w2):
            return vf._res_unit(x, cache, c2, g, s, wr, br, g, s, w2, br, None, None, None)
        args = (x, cache, c2, g, wr, br, w2)
        res = {}
        for on in (False, True):
            vf._NKI_CONV_RES = on
            cf = torch.compile(lambda *a: rb(*a), backend="neuron", fullgraph=True, dynamic=False)
            out, dt, tc = timeit(cf, args, reps)
            res[on] = ([t.cpu().float() for t in out], dt)
        msg = f"HEADBENCH rbres shape={shape} us/call off={res[False][1]:.1f} on={res[True][1]:.1f}"
        for name, a, r in zip(("y", "c1", "c2"), res[True][0], res[False][0]):
            d = a - r
            msg += f" {name}: max_abs={d.abs().max().item():.3e} exact={(d == 0).float().mean().item():.4f}"
        print(msg, flush=True)
        return
    elif var == "fused":
        # kernels/vae_norm_halo_conv.py vs vf._nh_conv (halo kernel + F.conv3d): conv output and new cache
        from kernels.vae_norm_halo_conv import norm_halo_conv
        from kernels.vae_head_conv import head_conv_weights

        def fn(x, cache, g, w, bias):
            xb = x.to(bf)
            l, r_, u, d = fake_edges(xb, None)
            y, nc = norm_halo_conv(xb[0], l[0], r_[0], u[0], d[0], cache[0], g.float().reshape(C, 1) * s,
                                   head_conv_weights(w), bias.float().reshape(-1, 1))
            return y.unsqueeze(0), nc.unsqueeze(0)

        def ref(x, cache, g, w, bias):
            return vf._nh_conv(x, cache, g, s, w, bias, None)
        args = (x, cache, g, w, bias)
        rf = torch.compile(ref, backend="neuron", fullgraph=True, dynamic=False)
        r2, rdt, _ = timeit(rf, args, reps)
        r2 = [t.cpu().float() for t in r2]
        cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
        out, dt, tc = timeit(cf, args, reps)
        msg = f"HEADBENCH fused shape={shape} cout={co} us/call={dt:.1f} ref_us={rdt:.1f} first={tc:.1f}s"
        outs = [t.cpu().float() for t in out]
        for name, a, r in zip(("conv", "cache"), outs, r2):
            d = a - r
            msg += (f" {name}: max_abs={d.abs().max().item():.3e} rel_l2={(d.norm() / r.norm()).item():.3e} "
                    f"exact={(d == 0).float().mean().item():.4f}")
        print(msg, flush=True)
        if os.environ.get("DIAG", "0") == "1":
            ne = (outs[1] != r2[1]).float()[0]                       # [C, 2, Hp, Wp]
            print("DIAG cache frames", ne.mean((0, 2, 3)).tolist(), flush=True)
            print("DIAG cache rows", [round(v, 3) for v in ne.mean((0, 1, 3)).tolist()], flush=True)
            print("DIAG cache cols", [round(v, 3) for v in ne.mean((0, 1, 2)).tolist()], flush=True)
            print("DIAG cache chans", [round(v, 3) for v in ne.mean((1, 2, 3)).tolist()], flush=True)
            ne = (outs[0] != r2[0]).float()[0]                       # [CO, T, H, W]
            print("DIAG conv frames", ne.mean((0, 2, 3)).tolist(), flush=True)
            print("DIAG conv rows", [round(v, 3) for v in ne.mean((0, 1, 3)).tolist()], flush=True)
            # rerun: deterministic?
            out2 = [t.cpu().float() for t in cf(*args)]
            print("DIAG rerun same", [bool((a == b).all()) for a, b in zip(outs, out2)], flush=True)
        return
    elif var == "nki":
        from kernels.vae_head_conv import head_conv, head_conv_weights
        wt = head_conv_weights(w)

        def fn(buf, wt, bias):
            return head_conv(buf[0], wt, bias.float().reshape(-1, 1)).unsqueeze(0)

        def ref(buf, w, bias):
            return F.conv3d(buf, w, bias)
        args = (buf, wt, bias)
        r = torch.compile(ref, backend="neuron", fullgraph=True, dynamic=False)(buf, w, bias).cpu().float()
    cf = torch.compile(fn, backend="neuron", fullgraph=True, dynamic=False)
    out, dt, tc = timeit(cf, args, reps)
    msg = f"HEADBENCH {var} shape={shape} cout={co} us/call={dt:.1f} first={tc:.1f}s"
    if var == "nki":
        o = out.cpu().float()
        d = o - r
        msg += (f" max_abs={d.abs().max().item():.3e} rel_l2={(d.norm() / r.norm()).item():.3e} "
                f"ref_absmax={r.abs().max().item():.3e} exact_frac={(d == 0).float().mean().item():.4f}")
    print(msg, flush=True)


if __name__ == "__main__":
    main()
