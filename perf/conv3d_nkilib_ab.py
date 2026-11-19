"""One-core A/B of a stride-1 3x3x3 conv at a VAE decoder shape: nkilib's conv3d (experimental/conv/conv3d.py,
K-replication), the compiler's conv (torch.compile of F.conv3d), and kernels/vae_head_conv.py's flat-shift matmuls
(tap weight [C, CO] stationary, offset slice of the padded SBUF band moving). Same pre-padded input for all three.
    NEURON_LOGICAL_NC_CONFIG=1 NEURON_RT_VISIBLE_CORES=0 SHAPE=96,96,4,120,192 python3 perf/conv3d_nkilib_ab.py
SHAPE: C, CO, T (output frames), H, W; the input is [1, C, T + 2, H + 2, W + 2] (temporal cache and spatial halo
already in place), the output [1, CO, T, H, W]. Prints CONVAB lines: us/call and rel_l2 vs an fp32 CPU conv3d.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch_neuronx  # noqa: F401,E402


def timeit(fn, reps):
    out = fn()
    torch.neuron.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        out = fn()
    torch.neuron.synchronize()
    return out, (time.perf_counter() - t0) / reps * 1e6


def main():
    C, CO, T, H, W = (int(v) for v in os.environ.get("SHAPE", "96,96,4,120,192").split(","))
    reps = int(os.environ.get("REPS", "20"))
    dev = torch.device("neuron:0")
    torch.manual_seed(0)
    x = torch.randn(1, C, T + 2, H + 2, W + 2).bfloat16()
    w = (torch.randn(CO, C, 3, 3, 3) * 0.05).bfloat16()
    b = (torch.randn(CO) * 0.1).bfloat16()
    ref = F.conv3d(x.float(), w.float(), b.float())
    xd, wd, bd = x.to(dev), w.to(dev), b.to(dev)

    def rel(o):
        o = o.float().cpu().reshape(ref.shape)
        return ((o - ref).norm() / ref.norm()).item()

    res = []
    cf = torch.compile(lambda a, ww, bb: F.conv3d(a, ww, bb), backend="neuron", fullgraph=True, dynamic=False)
    o, us = timeit(lambda: cf(xd, wd, bd), reps)
    res.append(("compiler F.conv3d", us, rel(o)))

    from kernels.vae_head_conv import head_conv, head_conv_weights
    wt, bt = head_conv_weights(wd), bd.float().reshape(-1, 1)
    hc = torch.compile(lambda a, ww, bb: head_conv(a[0], ww, bb), backend="neuron", fullgraph=True, dynamic=False)
    o, us = timeit(lambda: hc(xd, wt, bt), reps)
    res.append(("flat-shift (vae_head_conv)", us, rel(o)))

    try:
        from nkilib.experimental.conv.conv3d import conv3d
        from torch_neuronx.nki_hop import wrap_nki
        filt = wd.permute(2, 3, 4, 1, 0).contiguous()            # [K_d, K_h, K_w, C_in, C_out]
        k3 = wrap_nki(conv3d)
        nc = torch.compile(lambda a, ff, bb: k3(a, ff, bb), backend="neuron", fullgraph=True, dynamic=False)
        o, us = timeit(lambda: nc(xd, filt, bd), reps)
        res.append(("nkilib conv3d", us, rel(o)))
    except Exception as e:  # report and keep the other rows
        res.append(("nkilib conv3d", float("nan"), f"ERROR {type(e).__name__}: {str(e)[:200]}"))
    for name, us, r in res:
        print(f"CONVAB C={C} CO={CO} T={T} H={H} W={W} {name:28s} us/call={us:9.1f} rel_l2={r}", flush=True)


if __name__ == "__main__":
    main()
